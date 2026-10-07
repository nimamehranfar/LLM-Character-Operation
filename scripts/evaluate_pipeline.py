from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import tomllib
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.character_dataset import load_jsonl, SCALAR_RESULT_OPERATIONS, STRING_RESULT_OPERATIONS
from src.data.tool_policy_dataset import render_tool_policy_example, parse_policy_output, call_is_exact
from src.evaluation.runtime import ResumeLedger, inference_setup_metadata, telemetry_summary
from src.evaluation.parallel import add_shard_arguments, shard_examples
from src.evaluation.pipeline import execute_parsed_call
from src.executor.operations import CharacterExecutor
from src.model.local_model import ensure_local_model_path
from src.model.device import select_device
from src.model.result_injector import (
    LayeredSymbolicResultMapper, SymbolicResultMapper, TrainableLayeredResultInjector,
    TrainableOracleResultInjector, result_to_symbol,
)

DEFAULT_P1 = REPO_ROOT / 'configs/experiments/qwen3_8b/tool_policy.toml'
DEFAULT_P2 = REPO_ROOT / 'configs/experiments/qwen3_8b/result_injection.toml'


def resolve(v: str | Path) -> Path:
    p = Path(v)
    return p if p.is_absolute() else REPO_ROOT / p


def cfgload(p: Path) -> dict[str, Any]:
    with p.open('rb') as h:
        return tomllib.load(h)


def chat(tok, system: str, user: str) -> str:
    msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def answer_prompt(tok, user: str) -> str:
    msgs = [{'role': 'user', 'content': user + '\nAnswer with only the final value.'}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def policy_generate(model, tok, ex, cfg):
    r = render_tool_policy_example(ex, seed=int(cfg['training']['seed']), epoch=0,
                                   mask_probability=float(cfg['data']['function_mask_probability']))
    p = chat(tok, r.system_prompt, r.user_prompt)
    inp = tok(p, return_tensors='pt', add_special_tokens=False)
    input_tokens = int(inp['input_ids'].numel())
    inp = {k: v.to(model.device) for k, v in inp.items()}
    torch.cuda.synchronize()
    started = time.perf_counter()
    out = model.generate(**inp, max_new_tokens=int(cfg['evaluation']['max_new_tokens']), do_sample=False,
                         use_cache=True, pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    generated_ids = out[0, inp['input_ids'].shape[1]:]
    raw = tok.decode(generated_ids, skip_special_tokens=True)
    return raw, parse_policy_output(raw, r.tool_id_to_operation), {
        'input_tokens': input_tokens, 'output_tokens': int(generated_ids.numel()), 'seconds': seconds,
    }


@torch.no_grad()
def phase2_generate(policy, tok, mapper, layer: int, result, prompt: str, max_new: int, arch: dict[str, Any]):
    rendered = answer_prompt(tok, prompt)
    inp = tok(rendered, return_tensors='pt', add_special_tokens=False)
    input_tokens = int(inp['input_ids'].numel())
    inp = {k: v.to(policy.device) for k, v in inp.items()}
    sym = result_to_symbol(result, max_integer=int(arch['max_integer_result']), ascii_vocab_size=int(arch['ascii_vocab_size']))
    base = policy.get_base_model()
    if isinstance(mapper, LayeredSymbolicResultMapper):
        context = TrainableLayeredResultInjector(base, mapper, layer_index=layer, symbol=sym, position_index=-1, prefill_only=True)
    else:
        context = TrainableOracleResultInjector(base, mapper, layer_index=layer, symbol=sym, position_index=-1, prefill_only=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with policy.disable_adapter():
        with context as active:
            out = policy.generate(**inp, max_new_tokens=max_new, do_sample=False, use_cache=True,
                                  pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    generated_ids = out[0, inp['input_ids'].shape[1]:]
    answer = tok.decode(generated_ids, skip_special_tokens=True).strip()
    return answer, active.last_gate, {
        'input_tokens': input_tokens, 'output_tokens': int(generated_ids.numel()), 'seconds': seconds,
    }


def load_all(p1, p2, *, auto_download: bool | None = None, cache_dir: str | None = None):
    device = select_device(announce=True)
    cache_cfg = p1.get('cache', {})
    effective_auto = bool(cache_cfg.get('auto_download', False)) if auto_download is None else bool(auto_download)
    effective_cache = cache_dir if cache_dir is not None else cache_cfg.get('cache_dir')
    cache_started=time.perf_counter()
    mp = ensure_local_model_path(
        p1['model']['repo_id'], p1['model'].get('local_path'), auto_download=effective_auto,
        cache_dir=effective_cache, revision=str(cache_cfg.get('revision', 'main')),
    )
    cache_resolution_seconds=time.perf_counter()-cache_started
    model_load_started=time.perf_counter()
    tok = AutoTokenizer.from_pretrained(mp, local_files_only=True)
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(mp, local_files_only=True, device_map={'': device},
                                                quantization_config=q, dtype=torch.bfloat16)
    policy = PeftModel.from_pretrained(base, resolve(p1['output']['adapter_dir']), is_trainable=False)
    raw = torch.load(resolve(p2['output']['final_checkpoint']), map_location='cpu', weights_only=False)
    arch = raw['architecture']
    layer = int(raw.get('selected_layer', raw.get('layer_index', arch.get('layer_index', 0))))
    if raw.get('architecture_type') == 'layered_symbolic_result_mapper' or 'candidate_layers' in raw:
        layers = [int(x) for x in raw.get('candidate_layers', arch.get('candidate_layers', [layer]))]
        mapper = LayeredSymbolicResultMapper(hidden_size=int(policy.config.hidden_size), candidate_layers=layers,
            result_dim=int(arch['result_dim']), gate_dim=int(arch['gate_dim']), max_integer=int(arch['max_integer_result']),
            ascii_vocab_size=int(arch['ascii_vocab_size']), gate_bias_init=float(arch['gate_bias_init'])).to(policy.device)
    else:
        layers = [layer]
        mapper = SymbolicResultMapper(hidden_size=int(policy.config.hidden_size), result_dim=int(arch['result_dim']),
            gate_dim=int(arch['gate_dim']), max_integer=int(arch['max_integer_result']),
            ascii_vocab_size=int(arch['ascii_vocab_size']), gate_bias_init=float(arch['gate_bias_init'])).to(policy.device)
    mapper.load_state_dict(raw['mapper_state_dict'])
    mapper.eval(); policy.eval(); torch.cuda.synchronize()
    model_load_seconds=time.perf_counter()-model_load_started
    return tok, policy, mapper, layer, layers, arch, raw, mp, effective_auto, effective_cache, cache_resolution_seconds, model_load_seconds


def _group_summary(rows, key):
    groups = defaultdict(lambda: {'count': 0, 'phase1': 0, 'final': 0, 'system': 0, 'gates': [], 'latency': 0.0, 'input_tokens': 0, 'output_tokens': 0})
    for r in rows:
        g = groups[str(r.get(key, ''))]
        g['count'] += 1; g['phase1'] += int(r['phase1_exact']); g['final'] += int(r['final_exact']); g['system'] += int(r['system_success'])
        if r.get('phase2_gate') is not None: g['gates'].append(float(r['phase2_gate']))
        g['latency'] += float(r.get('latency_seconds', 0.0) or 0.0); g['input_tokens'] += int(r.get('input_tokens', 0) or 0); g['output_tokens'] += int(r.get('output_tokens', 0) or 0)
    return {k: {'count': v['count'], 'phase1_exact': v['phase1']/v['count'], 'final_exact': v['final']/v['count'],
                'system_success': v['system']/v['count'], 'mean_gate': sum(v['gates'])/len(v['gates']) if v['gates'] else None,
                'average_seconds_per_prompt': v['latency']/v['count'], 'average_input_tokens': v['input_tokens']/v['count'],
                'average_output_tokens': v['output_tokens']/v['count']}
            for k,v in sorted(groups.items())}


def summarize(rows, runtime=None):
    tasks=[r for r in rows if not r['is_control']]; controls=[r for r in rows if r['is_control']]
    scalar=[r for r in tasks if r['operation'] in SCALAR_RESULT_OPERATIONS]; transform=[r for r in tasks if r['operation'] in STRING_RESULT_OPERATIONS]
    correctly_called_scalar=[r for r in scalar if r['phase1_exact']]; gates=[float(r['phase2_gate']) for r in scalar if r.get('phase2_gate') is not None]
    out={
        'example_count':len(rows),'task_count':len(tasks),'control_count':len(controls),
        'phase1_execution_readiness':sum(r['phase1_exact'] for r in rows)/max(1,len(rows)),
        'control_no_call_accuracy':sum(r['no_call_correct'] for r in controls)/max(1,len(controls)),
        'control_false_positive_rate':1.0-(sum(r['no_call_correct'] for r in controls)/max(1,len(controls))),
        'scalar_phase2_final_exact':sum(r['final_exact'] for r in scalar)/max(1,len(scalar)),
        'scalar_phase2_given_correct_phase1':sum(r['final_exact'] for r in correctly_called_scalar)/max(1,len(correctly_called_scalar)),
        'transform_executor_direct_exact':sum(r['final_exact'] for r in transform)/max(1,len(transform)),
        'supported_task_system_exact':sum(r['final_exact'] for r in tasks)/max(1,len(tasks)),
        'overall_system_success':sum(r['system_success'] for r in rows)/max(1,len(rows)),
        'phase2_mean_gate':sum(gates)/len(gates) if gates else None,
        'by_operation':_group_summary(rows,'operation'),'by_category':_group_summary(rows,'category'),
        'by_source_style':_group_summary(rows,'source_style'),'by_generation_style':_group_summary(rows,'generation_style'),
        'by_length_regime':_group_summary(rows,'length_regime'),'by_result_kind':_group_summary(rows,'result_kind'),
        'per_example':rows,
    }
    if runtime is not None: out['runtime']=runtime
    return out


def evaluate(name, rows, policy, tok, mapper, layer, arch, p1, p2, completed, ledger):
    exr=CharacterExecutor(); out=[]
    for ex in tqdm(rows,desc=f'pipeline {name}',unit='ex',dynamic_ncols=True):
        key=f'{name}:{ex.example_id}'
        if key in completed:
            out.append(completed[key]); continue
        torch.cuda.synchronize(); example_started=time.perf_counter()
        raw,parsed,p1meta=policy_generate(policy,tok,ex,p1); p1ok=call_is_exact(parsed,ex)
        no=ex.is_control and parsed.get('decision')=='NO_CALL' and bool(parsed.get('valid'))
        generated=None; result=None; err=None; final=False; gate=None; executed_operation=None; path='no_call' if no else None
        executor_seconds=0.0; p2meta={'input_tokens':0,'output_tokens':0,'seconds':0.0}
        if parsed.get('decision')=='CALL' and bool(parsed.get('valid')):
            try:
                ex_started=time.perf_counter(); execution=execute_parsed_call(parsed,exr); executor_seconds=time.perf_counter()-ex_started
                result=execution.result; executed_operation=execution.operation
                if execution.operation in SCALAR_RESULT_OPERATIONS:
                    path='phase2_scalar'; generated,gate,p2meta=phase2_generate(policy,tok,mapper,layer,result,ex.prompt,int(p2['evaluation']['max_new_tokens']),arch)
                    final=generated.strip()==ex.expected.strip()
                else:
                    path='executor_direct_transform'; generated=str(result); final=generated==ex.expected
            except Exception as e: err=f'{type(e).__name__}: {e}'
        torch.cuda.synchronize(); latency=time.perf_counter()-example_started
        success=no if ex.is_control else final
        item={
            'example_id':key,'source_example_id':ex.example_id,'split':name,'operation':ex.operation,'category':ex.category,
            'source_style':ex.source_style,'generation_style':ex.generation_style,'length_regime':ex.length_regime,'result_kind':ex.result_kind,
            'is_control':ex.is_control,'policy_output':raw,'base_decision':parsed.get('decision'),'parsed_operation':parsed.get('operation'),
            'parsed_arguments':parsed.get('arguments',{}),'phase1_exact':bool(p1ok),'no_call_correct':bool(no),'executed_operation':executed_operation,
            'executor_result':result,'answer_path':path,'phase2_layer':int(layer) if path=='phase2_scalar' else None,'phase2_gate':gate,
            'generated_final':generated,'expected':None if ex.is_control else ex.expected,'final_exact':bool(final),'system_success':bool(success),'error':err,
            'phase1_seconds':p1meta['seconds'],'executor_seconds':executor_seconds,'phase2_seconds':p2meta['seconds'],'latency_seconds':latency,
            'phase1_input_tokens':p1meta['input_tokens'],'phase1_output_tokens':p1meta['output_tokens'],
            'phase2_input_tokens':p2meta['input_tokens'],'phase2_output_tokens':p2meta['output_tokens'],
            'input_tokens':p1meta['input_tokens']+p2meta['input_tokens'],'output_tokens':p1meta['output_tokens']+p2meta['output_tokens'],
            'total_tokens':p1meta['input_tokens']+p2meta['input_tokens']+p1meta['output_tokens']+p2meta['output_tokens'],
            'returned_answer_tokens':0 if generated is None else len(tok(generated,add_special_tokens=False)['input_ids']),
        }
        ledger.append(item); completed[key]=item; out.append(item)
    return out


def main():
    ap=argparse.ArgumentParser(description='Evaluate the complete trained pipeline. With no arguments, runs Qwen3-8B on IID and held-out splits.')
    ap.add_argument('--phase1-config',default=str(DEFAULT_P1)); ap.add_argument('--phase2-config',default=str(DEFAULT_P2))
    ap.add_argument('--split',choices=['test','heldout','both'],default=None)
    ap.add_argument('--auto-download',action=argparse.BooleanOptionalAction,default=None)
    ap.add_argument('--model-cache-dir',default=None)
    ap.add_argument('--resume',action=argparse.BooleanOptionalAction,default=None)
    add_shard_arguments(ap)
    args=ap.parse_args()
    p1=cfgload(Path(args.phase1_config).resolve()); p2=cfgload(Path(args.phase2_config).resolve())
    if args.local_model_path: p1['model']['local_path']=args.local_model_path
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    results_dir=resolve(args.results_dir or p1['output']['results_dir']); results_dir.mkdir(parents=True,exist_ok=True)
    progress_path=results_dir/'pipeline_progress.jsonl'; state_path=results_dir/'pipeline_progress.state.json'
    resume=bool(p1.get('evaluation',{}).get('resume',True)) if args.resume is None else bool(args.resume)
    ledger=ResumeLedger(state_path,progress_path,enabled=resume); completed=ledger.load_completed()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    tok,policy,mapper,layer,layers,arch,raw_phase2,model_path,effective_auto,effective_cache,cache_resolution_seconds,model_load_seconds=load_all(p1,p2,auto_download=args.auto_download,cache_dir=args.model_cache_dir)
    ledger.record_setup(model_load_seconds)
    test=load_jsonl(resolve(p1['data']['test_file'])); held=load_jsonl(resolve(p1['evaluation']['challenge_file']))
    split=args.split or str(p1.get('evaluation',{}).get('split','both'))
    requested=[]
    if split in {'test','both'}: requested.append(('test',test))
    if split in {'heldout','both'}: requested.append(('heldout_template_challenge',held))
    if args.report_path and len(requested) != 1: ap.error('--report-path requires a single --split')
    sharded = [shard_examples(rows,args.shard_index,args.shard_count) for _,rows in requested]
    requested = [(name,part[0]) for (name,_),part in zip(requested,sharded)]
    split_rows={}
    try:
        for name,rows in requested: split_rows[name]=evaluate(name,rows,policy,tok,mapper,layer,arch,p1,p2,completed,ledger)
    except KeyboardInterrupt:
        ledger.mark_interrupted(); print('Interrupted safely; rerun the same command to resume.'); raise
    ledger.mark_complete(); ledger_state=ledger.snapshot()
    all_rows=[r for name,_ in requested for r in split_rows[name]]
    global_runtime=telemetry_summary(all_rows,ledger_state)
    global_runtime.update({'cache_resolution_seconds_current_session':cache_resolution_seconds,'peak_allocated_vram_mib':torch.cuda.max_memory_allocated()/(1024**2),'peak_reserved_vram_mib':torch.cuda.max_memory_reserved()/(1024**2)})
    report={
        'experiment':'character_operation_pipeline','phase2_candidate_layers':layers,'phase2_selected_layer':layer,
        'shard':sharded[0][1] if len(sharded)==1 else {'splits':{name:part[1] for (name,_),part in zip(requested,sharded)}},
        'phase2_layer_selection':raw_phase2.get('selection_rule','checkpoint-selected layer'),
        'inference_setup':inference_setup_metadata(model_repo_id=p1['model']['repo_id'],model_path=model_path,
            quantization=str(p1['model'].get('quantization','4bit')),max_new_tokens=max(int(p1['evaluation']['max_new_tokens']),int(p2['evaluation']['max_new_tokens'])),
            extra={'phase1_max_new_tokens':int(p1['evaluation']['max_new_tokens']),'phase2_max_new_tokens':int(p2['evaluation']['max_new_tokens']),
                   'phase1_adapter_enabled_for_policy':True,'phase1_adapter_disabled_for_phase2':True,'phase2_layer':layer,
                   'auto_download':effective_auto,'model_cache_dir':effective_cache,'resume_enabled':resume}),
        'runtime':global_runtime,
        'note':'Runtime benchmark_total_seconds is interruption-neutral: first setup + completed-example latencies. Resume setup overhead is reported separately.',
    }
    for name,_ in requested:
        rows=split_rows[name]
        fake_state=dict(ledger_state); fake_state['canonical_setup_seconds']=0.0
        report[name]=summarize(rows,telemetry_summary(rows,fake_state))
        task_rows=[r for r in rows if not r['is_control']]; control_rows=[r for r in rows if r['is_control']]
        report[name]['runtime_supported_tasks']=telemetry_summary(task_rows,fake_state)
        report[name]['runtime_controls']=telemetry_summary(control_rows,fake_state)
    rp=resolve(args.report_path) if args.report_path else results_dir/'pipeline_results.json'
    rp.parent.mkdir(parents=True,exist_ok=True); rp.write_text(json.dumps(report,indent=2),encoding='utf-8')
    for name,_ in requested:
        row=report[name]; rt=row.get('runtime',{})
        label='IID' if name=='test' else 'Held-out'
        print(f"{label}: overall={row['overall_system_success']:.4f} | tasks={row['supported_task_system_exact']:.4f} | readiness={row['phase1_execution_readiness']:.4f} | control_no_call={row['control_no_call_accuracy']:.4f} | avg={rt.get('average_seconds_per_prompt') or 0.0:.4f}s | p95={rt.get('p95_seconds_per_prompt') or 0.0:.4f}s")
    print(f"Runtime: total={global_runtime.get('benchmark_total_seconds',0.0):.2f}s | avg={global_runtime.get('average_seconds_per_prompt') or 0.0:.4f}s | output_tok/s={global_runtime.get('output_tokens_per_second') or 0.0:.2f} | peak_vram={global_runtime.get('peak_allocated_vram_mib',0.0):.1f} MiB")
    print(f'Results: {rp}')


if __name__=='__main__': main()

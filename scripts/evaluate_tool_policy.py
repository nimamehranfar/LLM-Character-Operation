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

from src.data.character_dataset import load_jsonl, random_take
from src.data.tool_policy_dataset import call_is_exact, parse_policy_output, render_tool_policy_example
from src.model.local_model import ensure_local_model_path
from src.model.device import select_device


def resolve_path(value: str) -> Path:
    p = Path(value); return p if p.is_absolute() else REPO_ROOT / p


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as h: return tomllib.load(h)


def chat_prompt(tokenizer, system: str, user: str) -> str:
    messages=[{"role":"system","content":system},{"role":"user","content":user}]
    try: return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_policy(cfg, *, auto_download=None, cache_dir=None):
    device = select_device(announce=True)
    cache_cfg=cfg.get("cache", {})
    effective_auto=bool(cache_cfg.get("auto_download", False)) if auto_download is None else bool(auto_download)
    effective_cache=cache_cfg.get("cache_dir") if cache_dir is None else cache_dir
    model_path=ensure_local_model_path(cfg["model"]["repo_id"], cfg["model"].get("local_path"), auto_download=effective_auto, cache_dir=effective_cache, revision=str(cache_cfg.get("revision", "main")))
    tok=AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
    base=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,device_map={"":device},quantization_config=quant,dtype=torch.bfloat16)
    model=PeftModel.from_pretrained(base, resolve_path(cfg["output"]["adapter_dir"]), is_trainable=False)
    model.eval(); return tok,model


@torch.no_grad()
def generate(model, tokenizer, system, user, max_new):
    prompt=chat_prompt(tokenizer,system,user)
    toks=tokenizer(prompt,return_tensors="pt",add_special_tokens=False)
    input_tokens=int(toks["input_ids"].numel())
    toks={k:v.to(model.device) for k,v in toks.items()}
    torch.cuda.synchronize(); started=time.perf_counter()
    out=model.generate(**toks,max_new_tokens=max_new,do_sample=False,use_cache=True,pad_token_id=tokenizer.eos_token_id,eos_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize(); seconds=time.perf_counter()-started
    generated=out[0,toks["input_ids"].shape[1]:]
    return tokenizer.decode(generated,skip_special_tokens=True), {"seconds":seconds,"input_tokens":input_tokens,"output_tokens":int(generated.numel())}


def evaluate(model, tokenizer, rows, cfg, name):
    counts=defaultdict(int); runtime={"seconds":0.0,"input_tokens":0,"output_tokens":0}; by_op=defaultdict(lambda:defaultdict(int)); by_cat=defaultdict(lambda:defaultdict(int)); by_source=defaultdict(lambda:defaultdict(int)); by_generation=defaultdict(lambda:defaultdict(int)); by_length=defaultdict(lambda:defaultdict(int)); by_result=defaultdict(lambda:defaultdict(int)); details=[]
    seed=int(cfg["training"]["seed"]); mask=float(cfg["data"]["function_mask_probability"]); max_new=int(cfg["evaluation"]["max_new_tokens"])
    for ex in tqdm(rows,desc=f"tool policy {name}",unit="ex",dynamic_ncols=True):
        rendered=render_tool_policy_example(ex,seed=seed,epoch=0,mask_probability=mask)
        raw,meta=generate(model,tokenizer,rendered.system_prompt,rendered.user_prompt,max_new)
        runtime["seconds"]+=meta["seconds"]; runtime["input_tokens"]+=meta["input_tokens"]; runtime["output_tokens"]+=meta["output_tokens"]
        parsed=parse_policy_output(raw,rendered.tool_id_to_operation)
        exact=call_is_exact(parsed,ex); expected_decision="NO_CALL" if ex.is_control else "CALL"
        decision_ok=parsed.get("decision")==expected_decision
        arg_ok=ex.is_control or dict(parsed.get("arguments",{}))=={str(k):str(v) for k,v in ex.arguments.items()}
        op_ok=ex.is_control or parsed.get("operation")==ex.operation
        counts["total"]+=1; counts["valid"]+=int(bool(parsed.get("valid"))); counts["decision"]+=int(decision_ok); counts["full"]+=int(exact)
        if ex.is_control: counts["controls"]+=1; counts["control_no_call"]+=int(parsed.get("decision")=="NO_CALL")
        else: counts["tasks"]+=1; counts["operation"]+=int(op_ok); counts["arguments"]+=int(arg_ok)
        for table,key in ((by_op,ex.operation),(by_cat,ex.category),(by_source,ex.source_style),(by_generation,ex.generation_style),(by_length,ex.length_regime),(by_result,ex.result_kind)):
            table[key]["n"]+=1; table[key]["decision"]+=int(decision_ok); table[key]["full"]+=int(exact)
        details.append({"example_id":ex.example_id,"operation":ex.operation,"category":ex.category,"source_style":ex.source_style,"generation_style":ex.generation_style,"length_regime":ex.length_regime,"result_kind":ex.result_kind,"decision":parsed.get("decision"),"parsed_operation":parsed.get("operation"),"arguments":parsed.get("arguments",{}),"valid":bool(parsed.get("valid")),"decision_correct":bool(decision_ok),"operation_correct":bool(op_ok),"arguments_exact":bool(arg_ok),"full_call_exact":bool(exact)})
    def conv(table):
        return {k:{"count":v["n"],"decision_accuracy":v["decision"]/v["n"],"full_call_exact":v["full"]/v["n"]} for k,v in sorted(table.items())}
    return {"split":name,"example_count":counts["total"],"valid_output_accuracy":counts["valid"]/max(1,counts["total"]),"decision_accuracy":counts["decision"]/max(1,counts["total"]),"control_no_call_accuracy":counts["control_no_call"]/max(1,counts["controls"]),"operation_accuracy_on_tasks":counts["operation"]/max(1,counts["tasks"]),"argument_exact_on_tasks":counts["arguments"]/max(1,counts["tasks"]),"execution_readiness":counts["full"]/max(1,counts["total"]),"runtime":{"inference_seconds":runtime["seconds"],"average_seconds_per_prompt":runtime["seconds"]/max(1,counts["total"]),"total_input_tokens":runtime["input_tokens"],"total_output_tokens":runtime["output_tokens"]},"by_operation":conv(by_op),"by_category":conv(by_cat),"by_source_style":conv(by_source),"by_generation_style":conv(by_generation),"by_length_regime":conv(by_length),"by_result_kind":conv(by_result),"per_example":details}


def main():
    ap=argparse.ArgumentParser(description="Evaluate structured operation routing and argument extraction."); ap.add_argument("--config",default=str(REPO_ROOT/"configs/experiments/qwen3_8b/tool_policy.toml")); ap.add_argument("--test-examples",type=int,default=-1); ap.add_argument("--heldout-examples",type=int,default=-1); ap.add_argument("--auto-download",action=argparse.BooleanOptionalAction,default=None); ap.add_argument("--model-cache-dir",default=None); args=ap.parse_args()
    cfg=load_config(Path(args.config).resolve())
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    # Evaluation is not constrained by the laptop training-memory budget.
    tok,model=load_policy(cfg,auto_download=args.auto_download,cache_dir=args.model_cache_dir); seed=int(cfg["training"]["seed"])
    test=random_take(load_jsonl(resolve_path(cfg["data"]["test_file"])),args.test_examples,seed+2)
    held=random_take(load_jsonl(resolve_path(cfg["evaluation"]["challenge_file"])),args.heldout_examples,seed+3)
    report={"test":evaluate(model,tok,test,cfg,"test"),"heldout_template_challenge":evaluate(model,tok,held,cfg,"heldout_template_challenge"),"adapter_dir":str(resolve_path(cfg["output"]["adapter_dir"]))}
    out=resolve_path(cfg["output"]["results_dir"])/"tool_policy_eval.json"; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    for label,key in (("IID","test"),("Held-out","heldout_template_challenge")):
        row=report[key]
        print(f"{label}: readiness={row['execution_readiness']:.4f} | decision={row['decision_accuracy']:.4f} | operation={row['operation_accuracy_on_tasks']:.4f} | arguments={row['argument_exact_on_tasks']:.4f} | control_no_call={row['control_no_call_accuracy']:.4f} | avg={row['runtime']['average_seconds_per_prompt']:.4f}s")
    print(f"Results: {out}")

if __name__=="__main__": main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import sys
import time
import tomllib
from typing import Any

import torch
from tqdm import tqdm
from peft import get_peft_model_state_dict, set_peft_model_state_dict

REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))

from src.data.character_dataset import load_jsonl, random_take
from src.training.recovery import RecoveryManager,capture_rng_state,restore_rng_state
from scripts.train_tool_policy import load_model_and_tokenizer, trainable_parameters, configure_vram_limit, chat_prompt

SCRIPT_VERSION="direct_answer_sft"
SYSTEM="Answer the user's exact character/string task directly. Do not call external tools. Output only the final answer and no explanation."


def resolve(v:str)->Path:
    p=Path(v); return p if p.is_absolute() else REPO_ROOT/p

def cfgload(p:Path)->dict[str,Any]:
    with p.open('rb') as h:return tomllib.load(h)

def positives(path:Path): return [x for x in load_jsonl(path) if not x.is_control]

def encode(tokenizer,user,target,max_len):
    prompt=chat_prompt(tokenizer,SYSTEM,user); pids=tokenizer(prompt,add_special_tokens=False).input_ids; tids=tokenizer(target,add_special_tokens=False).input_ids
    if tokenizer.eos_token_id is not None:tids=tids+[tokenizer.eos_token_id]
    if len(pids)+len(tids)>max_len:
        overflow=len(pids)+len(tids)-max_len
        if overflow>=len(pids): raise ValueError('target too long')
        pids=pids[overflow:]
    ids=torch.tensor([pids+tids],dtype=torch.long); labels=torch.tensor([[-100]*len(pids)+tids],dtype=torch.long); mask=torch.ones_like(ids)
    return ids,mask,labels

@torch.no_grad()
def eval_em(model,tok,rows,max_new):
    model.eval(); correct=0
    for ex in tqdm(rows,desc='Direct-SFT dev',unit='ex',dynamic_ncols=True):
        prompt=chat_prompt(tok,SYSTEM,ex.prompt); inp=tok(prompt,return_tensors='pt',add_special_tokens=False); inp={k:v.to(model.device) for k,v in inp.items()}
        out=model.generate(**inp,max_new_tokens=max_new,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
        ans=tok.decode(out[0,inp['input_ids'].shape[1]:],skip_special_tokens=True).strip(); correct+=int(ans==ex.expected.strip())
    return {'example_count':len(rows),'exact_match_accuracy':correct/max(1,len(rows))}

def cpu_state(model):return {k:v.detach().cpu() for k,v in get_peft_model_state_dict(model).items()}

def main():
    ap=argparse.ArgumentParser(description='Train the direct-answer SFT comparison model.');ap.add_argument('--config',default=str(REPO_ROOT/'configs/experiments/qwen3_8b/direct_sft.toml'));ap.add_argument('--auto-download',action=argparse.BooleanOptionalAction,default=None);ap.add_argument('--model-cache-dir',default=None);ap.add_argument('--fresh',action='store_true');ap.add_argument('--run-dir',default='');ap.add_argument('--status',action='store_true');args=ap.parse_args()
    cp=Path(args.config).resolve();cfg=cfgload(cp);seed=int(cfg['training']['seed']);random.seed(seed);torch.manual_seed(seed)
    root=resolve(cfg['output']['checkpoint_dir'])
    if args.status:RecoveryManager.print_status(checkpoint_root=root,config=cfg,script_version=SCRIPT_VERSION);return
    mgr=RecoveryManager.create_or_resume(checkpoint_root=root,config=cfg,config_path=str(cp),script_version=SCRIPT_VERSION,heartbeat_seconds=float(cfg['recovery']['heartbeat_seconds']),fresh=bool(args.fresh),explicit_run_dir=Path(args.run_dir).resolve() if args.run_dir else None)
    try:
        configure_vram_limit(int(cfg['training']['max_vram_mib']));train=positives(resolve(cfg['data']['train_file']));dev=positives(resolve(cfg['data']['dev_file']));train=[x for x in random_take(load_jsonl(resolve(cfg['data']['train_file'])),int(cfg['data']['train_examples']),seed) if not x.is_control];dev=[x for x in random_take(load_jsonl(resolve(cfg['data']['dev_file'])),int(cfg['data']['dev_examples']),seed+1) if not x.is_control]
        latest=mgr.load_checkpoint('direct_latest.pt');model_path,tok,model=load_model_and_tokenizer(cfg,latest.get('adapter_state_dict') if latest else None,auto_download=args.auto_download,cache_dir=args.model_cache_dir);params=trainable_parameters(model)
        try:
            import bitsandbytes as bnb; opt=bnb.optim.PagedAdamW8bit(params,lr=float(cfg['training']['learning_rate']),weight_decay=float(cfg['training']['weight_decay']))
        except Exception:opt=torch.optim.AdamW(params,lr=float(cfg['training']['learning_rate']),weight_decay=float(cfg['training']['weight_decay']))
        if latest:
            opt.load_state_dict(latest['optimizer_state_dict']);restore_rng_state(latest.get('rng_state',{}));epoch=int(latest['epoch']);cursor=int(latest['cursor']);order=list(latest['order']);best=float(latest.get('best_metric',-1));stale=int(latest.get('stale',0));hist=list(latest.get('history',[]));loss_sum=float(latest.get('loss_sum',0));loss_count=int(latest.get('loss_count',0))
        else:epoch,cursor,order,best,stale,hist,loss_sum,loss_count=1,0,[],-1.,0,[],0.,0
        def save():mgr.checkpoint('direct_latest.pt',{'adapter_state_dict':cpu_state(model),'optimizer_state_dict':opt.state_dict(),'rng_state':capture_rng_state(),'epoch':epoch,'cursor':cursor,'order':order,'best_metric':best,'stale':stale,'history':hist,'loss_sum':loss_sum,'loss_count':loss_count})
        mgr.set_emergency_saver(save);epochs=int(cfg['training']['epochs']);accum=int(cfg['training']['gradient_accumulation_steps']);maxlen=int(cfg['training']['max_sequence_length']);save_every=int(cfg['recovery']['checkpoint_every_examples']);pat=int(cfg['training']['early_stopping_patience'])
        while epoch<=epochs:
            if not order:order=list(range(len(train)));random.Random(seed+epoch*1009).shuffle(order)
            model.train();opt.zero_grad(set_to_none=True);a=0;bar=tqdm(total=len(order),initial=cursor,desc=f'Direct SFT epoch {epoch}/{epochs}',unit='ex',dynamic_ncols=True)
            while cursor<len(order):
                ex=train[order[cursor]];ids,mask,labels=encode(tok,ex.prompt,ex.expected,maxlen);ids=ids.to(model.device);mask=mask.to(model.device);labels=labels.to(model.device)
                out=model(input_ids=ids,attention_mask=mask,labels=labels,use_cache=False);(out.loss/accum).backward();loss_sum+=float(out.loss.detach().item());loss_count+=1;a+=1;cursor+=1
                if a>=accum or cursor==len(order):torch.nn.utils.clip_grad_norm_(params,float(cfg['training']['gradient_clip_norm']));opt.step();opt.zero_grad(set_to_none=True);a=0
                bar.update(1);bar.set_postfix(loss=f'{loss_sum/max(1,loss_count):.4f}')
                if cursor%save_every==0:save();mgr.progress(phase='direct_sft',stage='training',epoch=epoch,cursor=cursor,total=len(order),latest_checkpoint=str(mgr.paths.checkpoints/'direct_latest.pt'));mgr.check_stop(saver=save)
            bar.close();metrics=eval_em(model,tok,dev,int(cfg['evaluation']['max_new_tokens']));metric=float(metrics['exact_match_accuracy']);improved=metric>best
            if improved:best=metric;stale=0;mgr.checkpoint('direct_best.pt',{'adapter_state_dict':cpu_state(model),'epoch':epoch,'dev_metrics':metrics})
            else:stale+=1
            hist.append({'epoch':epoch,'mean_loss':loss_sum/max(1,loss_count),'dev':metrics,'best':improved});print(f"epoch {epoch}: dev_EM={metric:.4f}");epoch+=1;cursor=0;order=[];loss_sum=0.;loss_count=0;save()
            if stale>=pat:break
        b=mgr.load_checkpoint('direct_best.pt');set_peft_model_state_dict(model,b['adapter_state_dict']);out=resolve(cfg['output']['adapter_dir']);out.mkdir(parents=True,exist_ok=True);model.save_pretrained(out,safe_serialization=True);tok.save_pretrained(out)
        report={'experiment':SCRIPT_VERSION,'status':'complete','model':cfg['model']['repo_id'],'model_path':str(model_path),'train_positive_examples':len(train),'dev_positive_examples':len(dev),'best_epoch':b['epoch'],'best_dev':b['dev_metrics'],'history':hist,'adapter_dir':str(out),'note':'Direct-answer SFT comparison: no deterministic executor or result injection.'}
        rp=resolve(cfg['output']['results_dir'])/'direct_sft_report.json';rp.parent.mkdir(parents=True,exist_ok=True);rp.write_text(json.dumps(report,indent=2),encoding='utf-8');mgr.mark_complete(report=str(rp),adapter_dir=str(out));print(f'Saved: {rp}')
    except BaseException as exc:mgr.handle_exception(exc);raise
    finally:mgr.close()
if __name__=='__main__':main()

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import tomllib
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.character_dataset import load_jsonl
from src.evaluation.parallel import add_shard_arguments, shard_examples
from src.evaluation.runtime import ResumeLedger, inference_setup_metadata, telemetry_summary
from src.model.local_model import ensure_local_model_path
from src.model.device import model_dtype, select_device, synchronize

ZERO_SHOT_SYSTEM = "Perform the requested character/string operation exactly. Output only the final answer, with no explanation."


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def cfgload(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def render_prompt(tokenizer, user: str) -> str:
    messages = [
        {"role": "system", "content": ZERO_SHOT_SYSTEM},
        {"role": "user", "content": user},
    ]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{ZERO_SHOT_SYSTEM}\n\nUser: {user}\nAssistant:"


def stratified_sample(rows, examples: int, examples_per_operation: int, seed: int):
    rows = list(rows)
    if examples_per_operation > 0:
        grouped = defaultdict(list)
        for row in rows:
            grouped[row.operation].append(row)
        sampled = []
        for operation in sorted(grouped):
            pool = grouped[operation]
            rng = random.Random(seed + sum(map(ord, operation)))
            sampled.extend(rng.sample(pool, min(examples_per_operation, len(pool))))
        return sampled
    if examples > 0 and examples < len(rows):
        return random.Random(seed).sample(rows, examples)
    return rows


def load_model(cfg: dict, mode: str, *, auto_download: bool | None, cache_dir: str | None):
    device = select_device()
    dtype = model_dtype(device)
    model_cfg = cfg["model"]
    cache_cfg = cfg.get("cache", {})
    effective_auto = bool(cache_cfg.get("auto_download", False)) if auto_download is None else bool(auto_download)
    effective_cache = cache_dir if cache_dir is not None else cache_cfg.get("cache_dir")
    revision = str(cache_cfg.get("revision", "main"))
    cache_started = time.perf_counter()
    model_path = ensure_local_model_path(
        model_cfg["repo_id"],
        model_cfg.get("local_path"),
        auto_download=effective_auto,
        cache_dir=effective_cache,
        revision=revision,
    )
    cache_resolution_seconds = time.perf_counter() - cache_started
    model_load_started = time.perf_counter()
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=trust_remote_code)
    quantization = model_cfg.get("quantization", "4bit")
    model_kwargs = {
        "local_files_only": True,
        "trust_remote_code": trust_remote_code,
        "device_map": {"": device},
        "dtype": dtype,
    }
    if quantization == "4bit":
        if device.type != "cuda":
            raise RuntimeError("4-bit evaluation requires CUDA; use --quantization none for CPU fallback")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    base = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    if mode == "direct_sft":
        adapter = cfg.get("output", {}).get("adapter_dir") or cfg.get("comparison", {}).get("direct_adapter_dir")
        if not adapter:
            raise ValueError("direct_sft mode requires output.adapter_dir in the config")
        model = PeftModel.from_pretrained(base, resolve(adapter), is_trainable=False)
    else:
        model = base
    model.eval()
    synchronize(device)
    model_load_seconds = time.perf_counter() - model_load_started
    return model_path, tokenizer, model, effective_auto, effective_cache, cache_resolution_seconds, model_load_seconds


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate zero-shot base generation or the direct-SFT comparator.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["zero_shot", "direct_sft"], default="zero_shot")
    parser.add_argument("--split", choices=["test", "heldout"], default=None)
    parser.add_argument("--examples", type=int, default=None)
    parser.add_argument("--examples-per-operation", type=int, default=None)
    parser.add_argument("--auto-download", action=argparse.BooleanOptionalAction, default=None,
                        help="Download only missing models; existing local snapshots are always reused.")
    parser.add_argument("--model-cache-dir", default=None,
                        help="Override [cache].cache_dir for Hugging Face model files.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--quantization", choices=["4bit", "none"], help="Override the model config")
    parser.add_argument("--max-new-tokens", type=int, help="Override the generation token limit")
    add_shard_arguments(parser)
    args = parser.parse_args()
    device = select_device(announce=True)

    cfg = cfgload(Path(args.config).resolve())
    if args.local_model_path:
        cfg["model"]["local_path"] = args.local_model_path
    if args.quantization:
        cfg["model"]["quantization"] = args.quantization
    if args.max_new_tokens is not None:
        if args.max_new_tokens < 1:
            parser.error("--max-new-tokens must be positive")
        cfg.setdefault("evaluation", {})["max_new_tokens"] = args.max_new_tokens
    if device.type == "cpu" and cfg["model"].get("quantization", "4bit") == "4bit":
        raise RuntimeError("4-bit evaluation requires CUDA; use --quantization none for CPU fallback")

    data_cfg = cfg["data"]
    eval_cfg = cfg.get("evaluation", {})
    split = args.split or str(eval_cfg.get("split", "test"))
    split_path = data_cfg["test_file"] if split == "test" else eval_cfg.get("challenge_file", data_cfg.get("heldout_file"))
    if split_path is None:
        raise KeyError("Config must provide data.test_file and evaluation.challenge_file or data.heldout_file")
    rows = [row for row in load_jsonl(resolve(split_path)) if not row.is_control]
    seed = int(cfg.get("training", {}).get("seed", 20260919)) + 91
    examples = int(eval_cfg.get("examples", -1)) if args.examples is None else int(args.examples)
    per_op = int(eval_cfg.get("examples_per_operation", -1)) if args.examples_per_operation is None else int(args.examples_per_operation)
    rows = stratified_sample(rows, examples, per_op, seed)
    rows, shard = shard_examples(rows, args.shard_index, args.shard_count)

    results_dir = resolve(args.results_dir or cfg.get("output", {}).get("results_dir", "results/baselines/local"))
    results_dir.mkdir(parents=True, exist_ok=True)
    model_slug = cfg["model"].get("slug", cfg["model"]["repo_id"].replace("/", "__"))
    output_path = resolve(args.report_path) if args.report_path else results_dir / f"{model_slug}_{args.mode}_{split}.json"
    progress_path = results_dir / f"{model_slug}_{args.mode}_{split}.progress.jsonl"
    state_path = results_dir / f"{model_slug}_{args.mode}_{split}.progress.state.json"
    resume = bool(eval_cfg.get("resume", True)) if args.resume is None else bool(args.resume)
    ledger = ResumeLedger(state_path, progress_path, enabled=resume)
    completed = ledger.load_completed()

    if all(row.example_id in completed for row in rows) and output_path.exists():
        ledger.mark_complete()
        print(f"Already complete: {output_path}")
        return

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model_path, tokenizer, model, effective_auto, effective_cache, cache_resolution_seconds, model_load_seconds = load_model(
        cfg, args.mode, auto_download=args.auto_download, cache_dir=args.model_cache_dir
    )
    ledger.record_setup(model_load_seconds)

    max_new_tokens = int(eval_cfg.get("max_new_tokens", 192))
    try:
        for ex in tqdm(rows, desc=f"{args.mode} {split}", unit="ex", dynamic_ncols=True):
            if ex.example_id in completed:
                continue
            example_started = time.perf_counter()
            prompt = render_prompt(tokenizer, ex.prompt)
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_tokens = int(inputs["input_ids"].numel())
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            synchronize(device)
            started = time.perf_counter()
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            synchronize(device)
            generation_seconds = time.perf_counter() - started
            generated_ids = output[0, inputs["input_ids"].shape[1]:]
            output_tokens = int(generated_ids.numel())
            raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
            answer = raw.strip()
            latency = time.perf_counter() - example_started
            ok = answer == ex.expected.strip()
            item = {
                "example_id": ex.example_id,
                "operation": ex.operation,
                "category": ex.category,
                "source_style": ex.source_style,
                "generation_style": ex.generation_style,
                "length_regime": ex.length_regime,
                "result_kind": ex.result_kind,
                "expected": ex.expected,
                "generated": raw,
                "normalized_answer": answer,
                "exact": ok,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "latency_seconds": latency,
                "generation_seconds": generation_seconds,
            }
            ledger.append(item)
            completed[ex.example_id] = item
    except KeyboardInterrupt:
        ledger.mark_interrupted()
        print("Interrupted safely; rerun the same command to resume completed-example progress.")
        raise

    details = [completed[row.example_id] for row in rows if row.example_id in completed]
    tables = {
        "by_operation": defaultdict(lambda: {"n": 0, "correct": 0, "latency": 0.0, "input_tokens": 0, "output_tokens": 0}),
        "by_category": defaultdict(lambda: {"n": 0, "correct": 0}),
        "by_source_style": defaultdict(lambda: {"n": 0, "correct": 0}),
        "by_generation_style": defaultdict(lambda: {"n": 0, "correct": 0}),
        "by_length_regime": defaultdict(lambda: {"n": 0, "correct": 0}),
        "by_result_kind": defaultdict(lambda: {"n": 0, "correct": 0}),
    }
    fields = {
        "by_operation": "operation", "by_category": "category", "by_source_style": "source_style",
        "by_generation_style": "generation_style", "by_length_regime": "length_regime", "by_result_kind": "result_kind",
    }
    for item in details:
        for table_name, field_name in fields.items():
            key = str(item[field_name])
            tables[table_name][key]["n"] += 1
            tables[table_name][key]["correct"] += int(item["exact"])
            if table_name == "by_operation":
                tables[table_name][key]["latency"] += float(item.get("latency_seconds", 0.0))
                tables[table_name][key]["input_tokens"] += int(item.get("input_tokens", 0))
                tables[table_name][key]["output_tokens"] += int(item.get("output_tokens", 0))

    def convert(table):
        out = {}
        for key, value in sorted(table.items()):
            row = {"count": value["n"], "exact_match_accuracy": value["correct"] / max(1, value["n"])}
            if "latency" in value:
                row.update({"average_seconds_per_prompt": value["latency"] / max(1, value["n"]),
                            "average_input_tokens": value["input_tokens"] / max(1, value["n"]),
                            "average_output_tokens": value["output_tokens"] / max(1, value["n"])})
            out[key] = row
        return out

    ledger.mark_complete()
    ledger_state = ledger.snapshot()
    correct = sum(int(x["exact"]) for x in details)
    runtime = telemetry_summary(details, ledger_state)
    runtime.update({
        "cache_resolution_seconds_current_session": cache_resolution_seconds,
        "peak_allocated_vram_mib": torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0,
        "peak_reserved_vram_mib": torch.cuda.max_memory_reserved(device) / (1024 ** 2) if device.type == "cuda" else 0.0,
    })
    report = {
        "mode": args.mode,
        "shard": shard,
        "model_repo_id": cfg["model"]["repo_id"],
        "model_path": str(model_path),
        "split": split,
        "example_count": len(details),
        "examples_per_operation": per_op,
        "exact_match_accuracy": correct / max(1, len(details)),
        "inference_setup": inference_setup_metadata(
            model_repo_id=cfg["model"]["repo_id"], model_path=model_path,
            quantization=str(cfg["model"].get("quantization", "4bit")), max_new_tokens=max_new_tokens,
            extra={"auto_download": effective_auto, "model_cache_dir": effective_cache, "resume_enabled": resume},
        ),
        "runtime": runtime,
        **{name: convert(table) for name, table in tables.items()},
        "per_example": details,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{args.mode} {split}: accuracy={report['exact_match_accuracy']:.4f} | n={report['example_count']} | avg={runtime.get('average_seconds_per_prompt') or 0.0:.4f}s | p95={runtime.get('p95_seconds_per_prompt') or 0.0:.4f}s | output_tok/s={runtime.get('output_tokens_per_second') or 0.0:.2f} | peak_vram={runtime.get('peak_allocated_vram_mib',0.0):.1f} MiB")
    print(f"Results: {output_path}")


if __name__ == "__main__":
    main()

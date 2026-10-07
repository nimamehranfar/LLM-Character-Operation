from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import time
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.character_dataset import SCALAR_RESULT_OPERATIONS, load_jsonl
from src.evaluation.parallel import add_shard_arguments, shard_examples
from src.data.tool_policy_dataset import call_is_exact, parse_policy_output, render_tool_policy_example
from src.evaluation.pipeline import execute_parsed_call
from src.evaluation.tool_feedback import (
    PROTOCOL_VERSION, build_tools, clean_response, feedback_messages, initial_messages,
    parse_native_call, render_native, select_examples, summarize,
)
from src.executor.operations import CharacterExecutor


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path):
    with resolve(path).open("rb") as handle:
        return tomllib.load(handle)


def digest_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def generate(model, tokenizer, prompt, max_new_tokens):
    import torch
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    torch.cuda.synchronize()
    started = time.perf_counter()
    # Phi uses <|end|> to end assistant turns, rather than its generic EOS alone.
    eos_ids = [tokenizer.eos_token_id]
    if "<|end|>" in tokenizer.get_vocab():
        eos_ids.append(tokenizer.convert_tokens_to_ids("<|end|>"))
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                use_cache=True, pad_token_id=tokenizer.eos_token_id,
                                eos_token_id=list(dict.fromkeys(eos_ids)))
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    generated = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=False), {
        "seconds": seconds, "input_tokens": int(inputs["input_ids"].numel()),
        "output_tokens": int(generated.numel()),
    }


def evaluate_example(ex, model, tokenizer, interface, seed, mask_probability,
                     call_tokens, final_tokens, policy_config=None):
    started = time.perf_counter()
    tools, mapping, masked = build_tools(ex.example_id, seed=seed, mask_probability=mask_probability)
    messages = initial_messages(ex.prompt, tools, interface)
    if policy_config is None:
        prompt = render_native(tokenizer, messages, tools, interface)
    else:
        rendered = render_tool_policy_example(ex, seed=seed, mask_probability=mask_probability)
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": rendered.system_prompt},
             {"role": "user", "content": ex.prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    raw, call_meta = generate(model, tokenizer, prompt, call_tokens)
    parsed = (parse_native_call(raw, mapping, interface) if policy_config is None
              else parse_policy_output(clean_response(raw), mapping))
    phase1_exact = call_is_exact(parsed, ex)
    no_call = parsed.get("decision") == "NO_CALL" and bool(parsed.get("valid"))
    attempted = parsed.get("decision") == "CALL" or "tool_call" in raw
    result = None
    executor_exact = False
    error = parsed.get("error")
    executor_seconds = 0.0
    feedback_elapsed = 0.0
    final_raw = None
    answer = clean_response(raw) if no_call and policy_config is None else None
    executed_operation = None
    final_meta = {"seconds": 0.0, "input_tokens": 0, "output_tokens": 0}
    if parsed.get("decision") == "CALL" and parsed.get("valid"):
        executor_started = time.perf_counter()
        try:
            execution = execute_parsed_call(parsed, CharacterExecutor())
            result = execution.result
            executed_operation = execution.operation
        except (ValueError, IndexError, TypeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        executor_seconds = time.perf_counter() - executor_started
        if executed_operation is not None:
            # Scoring only; never use expected arguments/results for execution.
            executor_exact = not ex.is_control and str(result) == ex.expected
            if policy_config is not None:
                tool_name = next(name for name, operation in mapping.items() if operation == executed_operation)
                arguments = {key: int(value) if key in {"index", "word_index"} else value
                             for key, value in parsed["arguments"].items()}
                parsed = {**parsed, "tool_name": tool_name,
                          "native_call": {"name": tool_name, "arguments": arguments}}
            feedback_started = time.perf_counter()
            followup = feedback_messages(messages, parsed, result, interface)
            followup_prompt = render_native(tokenizer, followup, tools, interface)
            # Like the mapper pipeline, the answer stage uses the original base
            # model with the tool-policy adapter disabled.
            context = model.disable_adapter() if policy_config is not None else nullcontext()
            with context:
                final_raw, final_meta = generate(model, tokenizer, followup_prompt, final_tokens)
            answer = clean_response(final_raw)
            feedback_elapsed = time.perf_counter() - feedback_started
    latency = time.perf_counter() - started
    final_exact = not ex.is_control and answer is not None and answer.strip() == ex.expected.strip()
    # Match the project's deployed routing: scalar feedback, string executor return.
    # Route using the predicted operation, never the ground-truth operation.
    matched_exact = (final_exact if executed_operation in SCALAR_RESULT_OPERATIONS
                     else executor_exact) if executed_operation is not None else False
    matched_feedback = executed_operation in SCALAR_RESULT_OPERATIONS
    row = {
        "example_id": ex.example_id, "source_example_id": ex.example_id, "split": ex.split,
        "operation": ex.operation, "category": ex.category, "source_style": ex.source_style,
        "generation_style": ex.generation_style, "length_regime": ex.length_regime,
        "result_kind": ex.result_kind, "is_control": ex.is_control, "masked_names": masked,
        "policy_output": raw, "base_decision": parsed.get("decision"),
        "parsed_operation": parsed.get("operation"), "parsed_arguments": parsed.get("arguments", {}),
        "phase1_exact": phase1_exact, "no_call_correct": ex.is_control and no_call,
        "tool_attempted": attempted, "invalid_response": not bool(parsed.get("valid")),
        "native_schema_valid": parsed.get("schema_valid"), "executed_operation": executed_operation,
        "executor_result": result, "executor_exact": executor_exact, "generated_final": answer,
        "final_output": final_raw, "expected": None if ex.is_control else ex.expected,
        "final_exact": final_exact, "matched_pipeline_exact": matched_exact,
        "system_success": no_call if ex.is_control else final_exact, "error": error,
        "phase1_seconds": call_meta["seconds"], "executor_seconds": executor_seconds,
        "feedback_seconds": final_meta["seconds"], "latency_seconds": latency,
        "phase1_input_tokens": call_meta["input_tokens"], "phase1_output_tokens": call_meta["output_tokens"],
        "feedback_input_tokens": final_meta["input_tokens"], "feedback_output_tokens": final_meta["output_tokens"],
        "input_tokens": call_meta["input_tokens"] + final_meta["input_tokens"],
        "output_tokens": call_meta["output_tokens"] + final_meta["output_tokens"],
        "executor_direct_latency_seconds": latency - feedback_elapsed,
        "matched_pipeline_latency_seconds": latency if matched_feedback else latency - feedback_elapsed,
        "matched_pipeline_input_tokens": call_meta["input_tokens"] + (final_meta["input_tokens"] if matched_feedback else 0),
        "matched_pipeline_output_tokens": call_meta["output_tokens"] + (final_meta["output_tokens"] if matched_feedback else 0),
    }
    row["total_tokens"] = row["input_tokens"] + row["output_tokens"]
    return row


def main():
    parser = argparse.ArgumentParser(description="Evaluate released function calling plus ordinary text tool feedback, without a mapper.")
    parser.add_argument("--config", default="configs/baselines/tools/qwen3_4b.toml")
    parser.add_argument("--split", choices=["test", "heldout", "both"], default="both")
    parser.add_argument("--examples-per-operation", type=int, default=-1)
    parser.add_argument("--controls-per-category", type=int, default=-1)
    parser.add_argument("--mask-probability", type=float, default=None)
    parser.add_argument("--policy-config", help="Optional trained Phase-1 adapter config; isolates text feedback versus the mapper.")
    parser.add_argument("--auto-download", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight", action="store_true", help="Validate data and cached native templates without loading model weights or generating answers.")
    add_shard_arguments(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.local_model_path:
        cfg["model"]["local_path"] = args.local_model_path
    policy_cfg = load_config(args.policy_config) if args.policy_config else None
    if policy_cfg and policy_cfg["model"]["repo_id"] != cfg["model"]["repo_id"]:
        parser.error("The native baseline and policy adapter must use the same model")
    interface = cfg["model"]["tool_interface"]
    if policy_cfg and interface != "qwen":
        parser.error("The project's trained policy is supported only with its Qwen backbone")
    seed = int(policy_cfg["training"]["seed"] if policy_cfg else cfg["training"]["seed"])
    default_mask = float(policy_cfg["data"]["function_mask_probability"] if policy_cfg else 0.0)
    mask_probability = default_mask if args.mask_probability is None else args.mask_probability
    if not 0 <= mask_probability <= 1:
        parser.error("mask-probability must be between 0 and 1")
    if args.examples_per_operation < -1 or args.controls_per_category < -1:
        parser.error("Sample sizes must be -1 (all), 0 (none), or positive")
    call_tokens = int(policy_cfg["evaluation"]["max_new_tokens"] if policy_cfg else cfg["evaluation"]["max_new_tokens"])
    final_tokens = int(cfg["evaluation"]["final_max_new_tokens"])
    splits = ["test", "heldout"] if args.split == "both" else [args.split]
    if args.report_path and len(splits) != 1:
        parser.error("--report-path requires a single --split")
    datasets = {}
    for split in splits:
        path = resolve(cfg["data"][f"{split}_file"])
        rows = select_examples(load_jsonl(path), args.examples_per_operation, args.controls_per_category, seed + 91)
        if not rows:
            parser.error("The selected evaluation contains no examples")
        datasets[split] = (path, rows)
    import torch
    import transformers
    from transformers import AutoTokenizer
    from src.model.local_model import ensure_local_model_path, load_local_causal_lm
    from src.evaluation.runtime import ResumeLedger, inference_setup_metadata, telemetry_summary
    if not args.preflight and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: no model evaluation was run. --preflight validates cached templates on CPU.")
    cache = cfg.get("cache", {})
    auto_download = bool(cache.get("auto_download", False)) if args.auto_download is None else args.auto_download
    cache_dir = args.model_cache_dir if args.model_cache_dir is not None else cache.get("cache_dir")
    revision = str(cache.get("revision", "main"))
    cache_started = time.perf_counter()
    model_path = ensure_local_model_path(cfg["model"]["repo_id"], cfg["model"].get("local_path"),
        auto_download=auto_download, cache_dir=cache_dir, revision=revision)
    cache_seconds = time.perf_counter() - cache_started
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    probe = next(iter(datasets.values()))[1][0]
    tools, mapping, _ = build_tools(probe.example_id, seed=seed, mask_probability=mask_probability)
    messages = initial_messages(probe.prompt, tools, interface)
    render_native(tokenizer, messages, tools, interface)
    # Verify the released template also accepts a tool result, before loading weights.
    name = tools[0]["function"]["name"]
    fake_call = {"tool_name": name, "native_call": {"name": name, "arguments": {}}}
    probe_rendered = render_native(tokenizer, feedback_messages(messages, fake_call, "TEMPLATE_RESULT_PROBE", interface), tools, interface)
    if "TEMPLATE_RESULT_PROBE" not in probe_rendered:
        raise ValueError("The released chat template omitted the tool result")
    if args.preflight:
        print(f"Native template validated: {cfg['model']['repo_id']} ({interface}); CUDA={torch.cuda.is_available()}")
        for split, (_, rows) in datasets.items():
            print(f"{split}: tasks={sum(not row.is_control for row in rows)}, controls={sum(row.is_control for row in rows)}")
        if policy_cfg:
            adapter = resolve(policy_cfg["output"]["adapter_dir"])
            print(f"Phase-1 adapter exists: {adapter.is_dir()} ({adapter})")
        return
    adapter = None
    if policy_cfg:
        adapter = resolve(policy_cfg["output"]["adapter_dir"])
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Trained Phase-1 adapter missing: {adapter}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    setup_started = time.perf_counter()
    _, tokenizer, model = load_local_causal_lm(cfg["model"]["repo_id"], explicit_path=model_path,
                                           quantization=cfg["model"]["quantization"], revision=revision)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
        model.eval()
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - setup_started
    mode = "lora_tool_text_feedback" if policy_cfg else "native_tool_text_feedback"
    root = resolve(args.results_dir or cfg["output"]["results_dir"])
    root.mkdir(parents=True, exist_ok=True)
    from tqdm import tqdm
    for split, (dataset_path, rows) in datasets.items():
        rows, shard = shard_examples(rows, args.shard_index, args.shard_count)
        identity = {
            "protocol_version": PROTOCOL_VERSION, "model_repo_id": cfg["model"]["repo_id"],
            "model_path": str(model_path), "requested_revision": revision,
            "model_config_sha256": digest_file(model_path / "config.json"),
            "tokenizer_config_sha256": digest_file(model_path / "tokenizer_config.json"),
            "dataset_sha256": digest_file(dataset_path), "selected_ids": [row.example_id for row in rows],
            "mode": mode, "interface": interface, "mask_probability": mask_probability,
            "seed": seed, "call_max_new_tokens": call_tokens, "final_max_new_tokens": final_tokens,
            "quantization": cfg["model"]["quantization"], "torch": torch.__version__,
            "transformers": transformers.__version__,
            "implementation_sha256": {str(file.relative_to(REPO_ROOT)): digest_file(file) for file in (
                Path(__file__), REPO_ROOT / "src/evaluation/tool_feedback.py",
                REPO_ROOT / "src/evaluation/pipeline.py", REPO_ROOT / "src/executor/operations.py",
                REPO_ROOT / "src/data/tool_policy_dataset.py")},
            "adapter_files": {file.name: digest_file(file) for file in adapter.iterdir() if file.is_file()} if adapter else None,
        }
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        stem = f"{mode}_{split}_{fingerprint}"
        output_path = resolve(args.report_path) if args.report_path else root / f"{stem}.json"
        progress_path = root / f"{stem}.progress.jsonl"
        state_path = root / f"{stem}.progress.state.json"
        if not args.resume and progress_path.exists():
            raise FileExistsError("Existing progress would be overwritten; use --resume or another results_dir")
        ledger = ResumeLedger(state_path, progress_path, enabled=True, heartbeat_seconds=0)
        try:
            ledger.record_setup(setup_seconds)
            completed = ledger.load_completed() if args.resume else {}
            for ex in tqdm(rows, desc=f"{cfg['model']['slug']} {split}", unit="ex", dynamic_ncols=True):
                if ex.example_id in completed:
                    continue
                item = evaluate_example(ex, model, tokenizer, interface, seed, mask_probability,
                                        call_tokens, final_tokens, policy_cfg)
                ledger.append(item)
                completed[ex.example_id] = item
            details = [completed[ex.example_id] for ex in rows]
            ledger.mark_complete()
            state = ledger.snapshot()
            task_rows = [row for row in details if not row["is_control"]]
            control_rows = [row for row in details if row["is_control"]]
            subset_state = {**state, "canonical_setup_seconds": 0.0}
            runtime = telemetry_summary(details, state)
            runtime.update({"peak_allocated_vram_mib": torch.cuda.max_memory_allocated() / 1024**2,
                            "peak_reserved_vram_mib": torch.cuda.max_memory_reserved() / 1024**2,
                            "cache_resolution_seconds_current_session": cache_seconds})
            report = {
                "mode": mode, "model_repo_id": cfg["model"]["repo_id"], "split": split,
                "shard": shard,
                "protocol": identity, "protocol_fingerprint": fingerprint, **summarize(details),
                "runtime": runtime,
                "runtime_supported_tasks": telemetry_summary(task_rows, subset_state),
                "runtime_controls": telemetry_summary(control_rows, subset_state),
                "runtime_executor_direct": telemetry_summary(details, subset_state,
                    latency_field="executor_direct_latency_seconds", input_field="phase1_input_tokens",
                    output_field="phase1_output_tokens"),
                "runtime_matched_pipeline": telemetry_summary(details, subset_state,
                    latency_field="matched_pipeline_latency_seconds", input_field="matched_pipeline_input_tokens",
                    output_field="matched_pipeline_output_tokens"),
                "inference_setup": inference_setup_metadata(model_repo_id=cfg["model"]["repo_id"],
                    model_path=model_path, quantization=cfg["model"]["quantization"], max_new_tokens=call_tokens,
                    extra={"tool_interface": interface, "final_max_new_tokens": final_tokens,
                           "phase1_adapter_enabled": bool(adapter), "phase2_adapter_enabled": False,
                           "mapper_enabled": False, "mask_probability": mask_probability}),
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"{split}: task_exact={report['supported_task_system_exact']} | executor_exact={report['executor_direct_task_exact']} | control_no_call={report['control_no_call_accuracy']}")
            print(f"Results: {output_path}")
        except BaseException:
            ledger.mark_interrupted()
            raise
        finally:
            ledger.close()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, FileNotFoundError, ValueError, FileExistsError) as exc:
        print(f"Tool-feedback evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

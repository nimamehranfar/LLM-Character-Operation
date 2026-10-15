from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
import tomllib
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.tool_policy_dataset import (
    call_is_exact,
    parse_policy_output,
    render_tool_policy_example,
)
from src.data.character_dataset import load_jsonl, random_take
from src.model.local_model import ensure_local_model_path
from src.model.device import select_device
from src.training.batching import pad_training_sequences
from src.training.engine import add_training_arguments, apply_training_arguments, train_single_gpu
from src.training.distributed import initialize, finish, create_manager
from src.training.reporting import export_training_artifacts, run_manifest
from src.training.low_vram_qlora import prepare_4bit_lora_base_low_vram
from src.training.recovery import (
    GracefulStop,
    RecoveryManager,
    capture_rng_state,
    restore_rng_state,
)

SCRIPT_VERSION = "tool_policy_sft_v2_batched"


def console(message: str) -> None:
    tqdm.write(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}")


def human_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return (f"{h}h {m}m {s}s" if h else f"{m}m {s}s")


def resolve_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_vram_limit(max_vram_mib: int) -> None:
    device = select_device(announce=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    total_mib = props.total_memory / 1024**2
    if max_vram_mib < 0:
        raise ValueError("max-vram-mib must be nonnegative; 0 uses the whole visible GPU")
    fraction = min(1.0, max_vram_mib / total_mib) if max_vram_mib else 1.0
    if max_vram_mib:
        torch.cuda.set_per_process_memory_fraction(fraction, device=device_index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    console(f"GPU: {props.name}; physical={total_mib:.1f} MiB; PyTorch ceiling={fraction * total_mib:.1f} MiB")


def chat_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def encode_training_example(tokenizer, rendered, max_length: int, *, fail_on_truncation: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_text = chat_prompt(tokenizer, rendered.system_prompt, rendered.user_prompt)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    target_ids = tokenizer(rendered.target, add_special_tokens=False).input_ids
    eos = tokenizer.eos_token_id
    if eos is not None:
        target_ids = target_ids + [eos]
    total = len(prompt_ids) + len(target_ids)
    if total > max_length:
        if fail_on_truncation:
            raise ValueError(f"Sequence has {total} tokens; max_sequence_length={max_length}. "
                             "Increase the configured maximum to retain the complete tool catalogue.")
        overflow = total - max_length
        # Keep every target token and the end of the prompt, which contains the user query.
        if overflow >= len(prompt_ids):
            raise ValueError(
                f"Target too long for max_length={max_length}: prompt={len(prompt_ids)} target={len(target_ids)}"
            )
        prompt_ids = prompt_ids[overflow:]
    ids = torch.tensor(prompt_ids + target_ids, dtype=torch.long)
    labels = torch.tensor([-100] * len(prompt_ids) + target_ids, dtype=torch.long)
    mask = torch.ones_like(ids)
    return ids.unsqueeze(0), mask.unsqueeze(0), labels.unsqueeze(0)


def load_model_and_tokenizer(
    cfg: dict[str, Any],
    adapter_state: dict[str, torch.Tensor] | None,
    *,
    auto_download: bool | None = None,
    cache_dir: str | None = None,
):
    model_cfg = cfg["model"]
    cache_cfg = cfg.get("cache", {})
    effective_auto = bool(cache_cfg.get("auto_download", False)) if auto_download is None else bool(auto_download)
    effective_cache = cache_cfg.get("cache_dir") if cache_dir is None else cache_dir
    resolve_started = time.perf_counter()
    model_path = ensure_local_model_path(
        model_cfg["repo_id"], model_cfg.get("local_path"),
        auto_download=effective_auto, cache_dir=effective_cache,
        revision=str(cache_cfg.get("revision", "main")),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    resolution_seconds = time.perf_counter() - resolve_started
    load_started = time.perf_counter()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        device_map={"": select_device()},
        quantization_config=quant,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_4bit_lora_base_low_vram(
        model,
        use_gradient_checkpointing=bool(cfg["lora"].get("gradient_checkpointing", True)),
    )
    lcfg = cfg["lora"]
    lora = LoraConfig(
        r=int(lcfg["r"]),
        lora_alpha=int(lcfg["alpha"]),
        lora_dropout=float(lcfg["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lcfg["target_modules"]),
    )
    model = get_peft_model(model, lora)
    if adapter_state is not None:
        set_peft_model_state_dict(model, adapter_state)
    model.train()
    model.training_setup_timing = {"cache_resolution_download_and_tokenizer_seconds_current_session": resolution_seconds,
                                   "model_and_adapter_loading_seconds_current_session": time.perf_counter() - load_started}
    return model_path, tokenizer, model


def trainable_parameters(model) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def adapter_state_cpu(model) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()}


@torch.no_grad()
def generate_policy(model, tokenizer, rendered, max_new_tokens: int) -> str:
    model.eval()
    prompt = chat_prompt(tokenizer, rendered.system_prompt, rendered.user_prompt)
    toks = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    toks = {k: v.to(model.device) for k, v in toks.items()}
    output = model.generate(
        **toks,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = output[0, toks["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    model.train()
    return text


@torch.no_grad()
def evaluate_policy(model, tokenizer, examples, cfg: dict[str, Any], *, seed: int, epoch: int) -> dict[str, Any]:
    from src.evaluation.training_metrics import grouped_accuracy, inference_runtime, token_count_to_eos, wilson_interval
    seed = int(cfg["evaluation"].get("seed", seed))
    selected = random_take(examples, int(cfg["evaluation"].get("dev_examples", len(examples))), seed + 97)
    batch_size = int(cfg["evaluation"].get("batch_size", 1))
    if batch_size < 1:
        raise ValueError("evaluation.batch_size must be positive")
    details, batches = [], []
    original_padding, was_training = tokenizer.padding_side, model.training
    tokenizer.padding_side = "left"
    model.eval()
    try:
        for start in tqdm(range(0, len(selected), batch_size), desc=f"policy evaluation e{epoch}", unit="batch"):
            examples_batch = selected[start:start + batch_size]
            rendered = [render_tool_policy_example(
                ex, seed=seed, epoch=0, mask_probability=float(cfg["data"]["function_mask_probability"]))
                for ex in examples_batch]
            prompts = [chat_prompt(tokenizer, row.system_prompt, row.user_prompt) for row in rendered]
            tensors = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
            tensors = {key: value.to(model.device) for key, value in tensors.items()}
            torch.cuda.synchronize(model.device)
            tick = time.perf_counter()
            output = model.generate(**tensors, max_new_tokens=int(cfg["evaluation"]["max_new_tokens"]),
                                    do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize(model.device)
            elapsed = time.perf_counter() - tick
            batch_id = len(batches)
            batches.append({"batch_id": batch_id, "example_count": len(examples_batch), "generation_seconds": elapsed})
            suffix = output[:, tensors["input_ids"].shape[1]:]
            texts = tokenizer.batch_decode(suffix, skip_special_tokens=True)
            for index, (ex, render, text) in enumerate(zip(examples_batch, rendered, texts)):
                parsed = parse_policy_output(text, render.tool_id_to_operation)
                expected_decision = "NO_CALL" if ex.is_control else "CALL"
                details.append({
                    "example_id": ex.example_id, "operation": ex.operation, "category": ex.category,
                    "source_style": ex.source_style, "length_regime": ex.length_regime,
                    "template_family": ex.template_family, "masked_names": render.masked_names,
                    "generated": text, "parsed": parsed, "expected": render.target,
                    "valid": bool(parsed["valid"]), "decision_correct": parsed["decision"] == expected_decision,
                    "operation_correct": parsed.get("operation") == ex.operation,
                    "arguments_correct": dict(parsed.get("arguments", {})) == {str(k): str(v) for k, v in ex.arguments.items()},
                    "correct": call_is_exact(parsed, ex), "batch_id": batch_id,
                    "input_tokens": int(tensors["attention_mask"][index].sum()),
                    "output_tokens": token_count_to_eos(suffix[index], tokenizer.eos_token_id),
                    "generation_seconds": elapsed / len(examples_batch),
                })
    finally:
        tokenizer.padding_side = original_padding
        model.train(was_training)
    tasks = [row for row in details if row["operation"] != "NONE"]
    masked = [row for row in details if row["masked_names"]]
    unmasked = [row for row in details if not row["masked_names"]]
    mean = lambda rows, key: sum(bool(row[key]) for row in rows) / max(1, len(rows))
    correct = sum(bool(row["correct"]) for row in details)
    return {
        "example_count": len(details), "correct_count": correct,
        "valid_output_accuracy": mean(details, "valid"), "decision_accuracy": mean(details, "decision_correct"),
        "operation_accuracy_on_tasks": mean(tasks, "operation_correct"),
        "argument_exact_on_tasks": mean(tasks, "arguments_correct"),
        "full_call_exact_accuracy": mean(details, "correct"),
        "full_call_exact_ci95_wilson": wilson_interval(correct, len(details)),
        "masked_full_exact": mean(masked, "correct"), "unmasked_full_exact": mean(unmasked, "correct"),
        **grouped_accuracy(details), "runtime": inference_runtime(details, batches),
        "per_example": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the structured character-operation tool policy on one or two GPUs.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/experiments/qwen3_8b/tool_policy.toml"))
    parser.add_argument("--auto-download", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--max-vram-mib", type=int)
    add_training_arguments(parser)
    args = parser.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    apply_training_arguments(cfg, args)
    if args.max_vram_mib is not None:
        cfg["training"]["max_vram_mib"] = args.max_vram_mib
    rank, world = initialize()
    seed = int(cfg["training"]["seed"])
    set_seed(seed)
    checkpoint_root = resolve_path(cfg["output"]["checkpoint_dir"])
    if args.status:
        RecoveryManager.print_status(checkpoint_root=checkpoint_root, config=cfg, script_version=SCRIPT_VERSION)
        return
    manager = create_manager(
        checkpoint_root=checkpoint_root, config=cfg, config_path=str(cfg_path), script_version=SCRIPT_VERSION,
        heartbeat_seconds=float(cfg["recovery"]["heartbeat_seconds"]), fresh=bool(args.fresh),
        explicit_run_dir=Path(args.run_dir).resolve() if args.run_dir else None,
    )
    try:
        configure_vram_limit(int(cfg["training"]["max_vram_mib"]))
        train_path = resolve_path(cfg["data"]["train_file"])
        dev_path = resolve_path(cfg["data"]["dev_file"])
        test_path = resolve_path(cfg["data"]["test_file"])
        heldout_path = resolve_path(cfg["evaluation"]["challenge_file"])
        train_rows = random_take(load_jsonl(train_path), int(cfg["data"]["train_examples"]), seed)
        dev_rows = random_take(load_jsonl(dev_path), int(cfg["data"]["dev_examples"]), seed + 1)
        latest = manager.load_checkpoint("policy_latest.pt")
        loading_started = time.perf_counter()
        model_path, tokenizer, model = load_model_and_tokenizer(
            cfg, None, auto_download=args.auto_download, cache_dir=args.model_cache_dir)
        loading_seconds = time.perf_counter() - loading_started
        params = trainable_parameters(model)
        tr = cfg["training"]
        # Explicit optimizer selection avoids silently changing the paper protocol.
        optimizer_name = str(tr.get("optimizer", "paged_adamw_8bit"))
        optimizer_kwargs = dict(lr=float(tr["learning_rate"]), weight_decay=float(tr["weight_decay"]),
                                betas=(float(tr.get("adam_beta1", .9)), float(tr.get("adam_beta2", .999))),
                                eps=float(tr.get("adam_epsilon", 1e-8)))
        if optimizer_name == "paged_adamw_8bit":
            import bitsandbytes as bnb
            optimizer = bnb.optim.PagedAdamW8bit(params, **optimizer_kwargs)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(params, **optimizer_kwargs)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        maximum = int(tr["max_sequence_length"])

        def encode(index, epoch):
            rendered = render_tool_policy_example(
                train_rows[index], seed=seed, epoch=epoch,
                mask_probability=float(cfg["data"]["function_mask_probability"]))
            return encode_training_example(tokenizer, rendered, maximum,
                                           fail_on_truncation=bool(tr.get("fail_on_truncation", False)))

        # Auto mode budgets for the longest actual rendered input across every
        # planned epoch, rather than wasting capacity on unused context length.
        length_epochs = range(1, int(tr["epochs"]) + 1) if tr.get("auto_batch_size") else (1,)
        if tr.get("auto_batch_size") and rank == 0:
            console("Auto batch: measuring sequence lengths across all planned epochs")
        lengths = [max(encode(i, epoch)[0].numel() for epoch in length_epochs)
                   for i in range(len(train_rows))]
        # Rendering varies deterministically by epoch. The engine's epoch is
        # available through the recovery state updated at optimizer boundaries.
        current_epoch = [int(latest.get("loop", {}).get("epoch", 1)) if latest else 1]

        def prepare(indices):
            sequences = [encode(i, current_epoch[0]) for i in indices]
            tensors, _, target_tokens = pad_training_sequences(
                sequences, pad_token_id=int(tokenizer.pad_token_id), device=torch.device("cpu"))
            return {"tensors": tensors, "target_tokens": target_tokens, "count": len(indices)}

        def forward(batch, micro_step):
            tensors = {key: value.to(model.device) for key, value in batch["tensors"].items()}
            out = model(**tensors, use_cache=False)
            return out.loss, batch["target_tokens"], int(tensors["attention_mask"].sum()), batch["count"], {}

        def validate():
            from src.training.engine import teacher_forced_loss
            metrics = evaluate_policy(model, tokenizer, dev_rows, cfg, seed=seed, epoch=current_epoch[0])
            was_training = model.training
            model.eval()
            def dev_batches():
                size = int(cfg["evaluation"].get("batch_size", 1))
                for start in range(0, len(dev_rows), size):
                    sequences = [encode_training_example(tokenizer, render_tool_policy_example(
                        ex, seed=int(cfg["evaluation"].get("seed", seed)), epoch=0,
                        mask_probability=float(cfg["data"]["function_mask_probability"])),
                        maximum, fail_on_truncation=bool(tr.get("fail_on_truncation", False)))
                        for ex in dev_rows[start:start + size]]
                    yield pad_training_sequences(sequences, pad_token_id=int(tokenizer.pad_token_id),
                                                 device=torch.device("cpu"))[0]
            metrics["loss"] = teacher_forced_loss(model, dev_batches(), model.device)
            model.train(was_training)
            return metrics

        trained = train_single_gpu(
            module=model, optimizer=optimizer, manager=manager, cfg=cfg, rows=train_rows, lengths=lengths,
            prepare=prepare, forward=forward, evaluate=validate, state_cpu=lambda: adapter_state_cpu(model),
            restore=lambda state: set_peft_model_state_dict(model, state),
            checkpoint_prefix="policy", latest=latest, metric_key="full_call_exact_accuracy",
            set_epoch=lambda epoch: current_epoch.__setitem__(0, epoch))
        finish()
        if rank != 0:
            return
        final_started = time.perf_counter()
        final_metrics = {}
        for split, path in (("test", test_path), ("heldout", heldout_path)):
            final_metrics[split] = evaluate_policy(model, tokenizer, load_jsonl(path), cfg, seed=seed, epoch=0)
        final_evaluation_seconds = time.perf_counter() - final_started
        # Each run owns immutable exported weights; compatibility path remains
        # available for the existing pipeline configuration.
        out_dir = manager.paths.artifacts / "final_adapter"
        model.save_pretrained(out_dir, safe_serialization=True)
        tokenizer.save_pretrained(out_dir)
        compatibility = resolve_path(cfg["output"]["adapter_dir"])
        model.save_pretrained(compatibility, safe_serialization=True)
        tokenizer.save_pretrained(compatibility)
        manifest = run_manifest(
            config_path=cfg_path, data_paths=[train_path, dev_path, test_path, heldout_path],
            training={**tr, "batch_plan": trained["batch_plan"], "lora": cfg["lora"],
                      "evaluation": cfg["evaluation"], "model": cfg["model"], "cache": cfg.get("cache", {})},
            trainable_parameters=sum(p.numel() for p in params),
            total_parameters=model.num_parameters())
        report = {
            "experiment": SCRIPT_VERSION, "status": "complete", "model": cfg["model"]["repo_id"],
            "model_path": str(model_path), "batch_plan": trained["batch_plan"],
            "train_count": len(train_rows), "dev_count": len(dev_rows),
            "best_epoch": trained["best"]["epoch"], "best_dev": trained["best"]["metrics"],
            **final_metrics, "history": trained["history"], "optimizer_steps": trained["optimizer_steps"],
            "runtime": {"resolve_download_and_model_loading_seconds": loading_seconds,
                        **model.training_setup_timing,
                        "batch_preparation_seconds": trained["batch_preparation_seconds"],
                        "train_loop_wall_seconds_current_session": trained["train_loop_wall_seconds_current_session"],
                        "training_seconds": trained["training_seconds"],
                        "validation_seconds": trained["validation_seconds"],
                        "checkpoint_seconds": trained["checkpoint_seconds"],
                        "final_evaluation_seconds": final_evaluation_seconds,
                        "active_total_seconds": manager.active_elapsed_seconds(),
                        "wall_total_seconds_including_interruptions": manager.wall_elapsed_seconds(),
                        "peak_allocated_vram_mib": max(trained["peak_allocated_vram_mib"], torch.cuda.max_memory_allocated() / 1024**2),
                        "peak_reserved_vram_mib": max(trained["peak_reserved_vram_mib"], torch.cuda.max_memory_reserved() / 1024**2)},
            "per_gpu_training_peak_vram_mib": trained["per_gpu_training_peak_vram_mib"],
            "adapter_dir": str(out_dir), "compatibility_adapter_dir": str(compatibility),
            "recovery_run_dir": str(manager.paths.run_dir), "phase2_trained": False,
        }
        export_training_artifacts(manager.paths.artifacts, step_rows=trained["step_rows"],
                                  epoch_rows=trained["history"], manifest=manifest, report=report)
        report_path = resolve_path(cfg["output"]["results_dir"]) / "tool_policy_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        manager.mark_complete(report=str(manager.paths.artifacts / "training_report.json"), adapter_dir=str(out_dir))
        console(f"COPY RESULTS DIRECTORY: {manager.paths.artifacts}")
    except BaseException as exc:
        if rank == 0:
            manager.handle_exception(exc)
        raise
    finally:
        manager.close()
        finish()


if __name__ == "__main__":
    main()

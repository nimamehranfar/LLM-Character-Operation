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
from src.training.low_vram_qlora import prepare_4bit_lora_base_low_vram
from src.training.recovery import (
    GracefulStop,
    RecoveryManager,
    capture_rng_state,
    restore_rng_state,
)

SCRIPT_VERSION = "tool_policy_sft"


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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    props = torch.cuda.get_device_properties(0)
    total_mib = props.total_memory / 1024**2
    fraction = min(1.0, max_vram_mib / total_mib)
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    console(f"GPU: {props.name}; physical={total_mib:.1f} MiB; PyTorch ceiling={max_vram_mib} MiB")


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


def encode_training_example(tokenizer, rendered, max_length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_text = chat_prompt(tokenizer, rendered.system_prompt, rendered.user_prompt)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    target_ids = tokenizer(rendered.target, add_special_tokens=False).input_ids
    eos = tokenizer.eos_token_id
    if eos is not None:
        target_ids = target_ids + [eos]
    total = len(prompt_ids) + len(target_ids)
    if total > max_length:
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
    model_path = ensure_local_model_path(
        model_cfg["repo_id"], model_cfg.get("local_path"),
        auto_download=effective_auto, cache_dir=effective_cache,
        revision=str(cache_cfg.get("revision", "main")),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
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
        device_map={"": "cuda:0"},
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
def evaluate_policy(model, tokenizer, examples, cfg: dict[str, Any], *, seed: int, epoch: int) -> dict[str, float]:
    max_examples = int(cfg["evaluation"].get("dev_examples", len(examples)))
    selected = random_take(examples, max_examples, seed + 97)
    total = controls = tasks = valid = decision_correct = op_correct = args_correct = full_exact = 0
    masked_total = masked_exact = 0
    unmasked_total = unmasked_exact = 0
    bar = tqdm(selected, desc=f"dev e{epoch}", unit="ex", dynamic_ncols=True, leave=False)
    for example in bar:
        rendered = render_tool_policy_example(
            example,
            seed=seed,
            epoch=0,
            mask_probability=float(cfg["data"]["function_mask_probability"]),
        )
        text = generate_policy(model, tokenizer, rendered, int(cfg["evaluation"]["max_new_tokens"]))
        parsed = parse_policy_output(text, rendered.tool_id_to_operation)
        total += 1
        valid += int(bool(parsed["valid"]))
        expected_decision = "NO_CALL" if example.operation == "NONE" else "CALL"
        decision_correct += int(parsed["decision"] == expected_decision)
        exact = call_is_exact(parsed, example)
        full_exact += int(exact)
        if rendered.masked_names:
            masked_total += 1
            masked_exact += int(exact)
        else:
            unmasked_total += 1
            unmasked_exact += int(exact)
        if example.operation == "NONE":
            controls += 1
        else:
            tasks += 1
            op_correct += int(parsed.get("operation") == example.operation)
            expected_args = {str(k): str(v) for k, v in example.arguments.items()}
            args_correct += int(dict(parsed.get("arguments", {})) == expected_args)
        bar.set_postfix(exact=f"{full_exact/max(1,total):.3f}", decision=f"{decision_correct/max(1,total):.3f}")
    return {
        "example_count": total,
        "valid_output_accuracy": valid / max(1, total),
        "decision_accuracy": decision_correct / max(1, total),
        "operation_accuracy_on_tasks": op_correct / max(1, tasks),
        "argument_exact_on_tasks": args_correct / max(1, tasks),
        "full_call_exact_accuracy": full_exact / max(1, total),
        "masked_full_exact": masked_exact / max(1, masked_total),
        "unmasked_full_exact": unmasked_exact / max(1, unmasked_total),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the structured character-operation tool policy.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/experiments/qwen3_8b/tool_policy.toml"))
    parser.add_argument("--auto-download", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    seed = int(cfg["training"]["seed"])
    set_seed(seed)
    checkpoint_root = resolve_path(cfg["output"]["checkpoint_dir"])
    if args.status:
        RecoveryManager.print_status(checkpoint_root=checkpoint_root, config=cfg, script_version=SCRIPT_VERSION)
        return
    manager = RecoveryManager.create_or_resume(
        checkpoint_root=checkpoint_root,
        config=cfg,
        config_path=str(cfg_path),
        script_version=SCRIPT_VERSION,
        heartbeat_seconds=float(cfg["recovery"]["heartbeat_seconds"]),
        fresh=bool(args.fresh),
        explicit_run_dir=Path(args.run_dir).resolve() if args.run_dir else None,
    )
    try:
        configure_vram_limit(int(cfg["training"]["max_vram_mib"]))
        train_rows = load_jsonl(resolve_path(cfg["data"]["train_file"]))
        dev_rows = load_jsonl(resolve_path(cfg["data"]["dev_file"]))
        train_rows = random_take(train_rows, int(cfg["data"]["train_examples"]), seed)
        dev_rows = random_take(dev_rows, int(cfg["data"]["dev_examples"]), seed + 1)
        latest = manager.load_checkpoint("policy_latest.pt")
        adapter_state = latest.get("adapter_state_dict") if latest else None
        model_path, tokenizer, model = load_model_and_tokenizer(cfg, adapter_state, auto_download=args.auto_download, cache_dir=args.model_cache_dir)
        params = trainable_parameters(model)
        trainable_count = sum(p.numel() for p in params)
        console(f"Qwen path: {model_path}")
        console(f"Trainable LoRA parameters: {trainable_count:,}")
        console(f"Train={len(train_rows):,}; dev={len(dev_rows):,}; result injection is trained separately.")

        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.PagedAdamW8bit(params, lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
        except Exception:
            optimizer = torch.optim.AdamW(params, lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))

        epochs = int(cfg["training"]["epochs"])
        grad_accum = int(cfg["training"]["gradient_accumulation_steps"])
        max_length = int(cfg["training"]["max_sequence_length"])
        save_every = int(cfg["recovery"]["checkpoint_every_examples"])
        patience = int(cfg["training"]["early_stopping_patience"])
        min_delta = float(cfg["training"]["min_delta"])

        if latest:
            optimizer.load_state_dict(latest["optimizer_state_dict"])
            for state in optimizer.state.values():
                for key, value in list(state.items()):
                    if torch.is_tensor(value):
                        state[key] = value.to("cuda:0")
            restore_rng_state(latest.get("rng_state", {}))
            epoch = int(latest["epoch"])
            cursor = int(latest["cursor"])
            order = list(latest["order"])
            loss_sum = float(latest.get("loss_sum", 0.0))
            loss_count = int(latest.get("loss_count", 0))
            epoch_active_seconds = float(latest.get("epoch_active_seconds", 0.0))
            best_metric = float(latest.get("best_metric", -1.0))
            best_epoch = int(latest.get("best_epoch", 0))
            stale = int(latest.get("stale", 0))
            history = list(latest.get("history", []))
            console(f"Resuming epoch {epoch} at example {cursor}/{len(order)}")
        else:
            epoch, cursor, order = 1, 0, []
            loss_sum, loss_count, epoch_active_seconds = 0.0, 0, 0.0
            best_metric, best_epoch, stale, history = -1.0, 0, 0, []

        segment_start = time.perf_counter()

        def current_epoch_active() -> float:
            return epoch_active_seconds + (time.perf_counter() - segment_start)

        def save_latest() -> None:
            nonlocal segment_start, epoch_active_seconds
            epoch_active_seconds = current_epoch_active()
            segment_start = time.perf_counter()
            manager.checkpoint("policy_latest.pt", {
                "adapter_state_dict": adapter_state_cpu(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "rng_state": capture_rng_state(),
                "epoch": epoch,
                "cursor": cursor,
                "order": order,
                "loss_sum": loss_sum,
                "loss_count": loss_count,
                "epoch_active_seconds": epoch_active_seconds,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "stale": stale,
                "history": history,
            })

        manager.set_emergency_saver(save_latest)
        while epoch <= epochs:
            if not order:
                order = list(range(len(train_rows)))
                random.Random(seed + epoch * 1009).shuffle(order)
            bar = tqdm(total=len(order), initial=cursor, desc=f"tool-policy SFT epoch {epoch}/{epochs}", unit="ex", dynamic_ncols=True)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            accum = 0
            while cursor < len(order):
                idx = order[cursor]
                example = train_rows[idx]
                rendered = render_tool_policy_example(
                    example,
                    seed=seed,
                    epoch=epoch,
                    mask_probability=float(cfg["data"]["function_mask_probability"]),
                )
                ids, mask, labels = encode_training_example(tokenizer, rendered, max_length)
                ids = ids.to(model.device)
                mask = mask.to(model.device)
                labels = labels.to(model.device)
                out = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
                loss = out.loss / grad_accum
                loss.backward()
                raw_loss = float(out.loss.detach().item())
                loss_sum += raw_loss
                loss_count += 1
                accum += 1
                cursor += 1
                if accum >= grad_accum or cursor == len(order):
                    torch.nn.utils.clip_grad_norm_(params, float(cfg["training"]["gradient_clip_norm"]))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accum = 0
                bar.update(1)
                bar.set_postfix(loss=f"{loss_sum/max(1,loss_count):.4f}")
                if cursor % save_every == 0:
                    save_latest()
                    manager.progress(phase="policy_sft", stage="training", epoch=epoch, cursor=cursor, total=len(order), latest_checkpoint=str(manager.paths.checkpoints / "policy_latest.pt"))
                    manager.check_stop(saver=save_latest)
            bar.close()
            epoch_train_seconds = current_epoch_active()
            epoch_active_seconds = 0.0
            segment_start = time.perf_counter()
            dev_metrics = evaluate_policy(model, tokenizer, dev_rows, cfg, seed=seed, epoch=epoch)
            metric = float(dev_metrics["full_call_exact_accuracy"])
            improved = metric > best_metric + min_delta
            if improved:
                best_metric = metric
                best_epoch = epoch
                stale = 0
                manager.checkpoint("policy_best.pt", {"adapter_state_dict": adapter_state_cpu(model), "epoch": epoch, "dev_metrics": dev_metrics})
            else:
                stale += 1
            history.append({
                "epoch": epoch,
                "mean_loss": loss_sum / max(1, loss_count),
                "epoch_train_active_seconds": epoch_train_seconds,
                "dev": dev_metrics,
                "best": improved,
            })
            epoch_times = [float(x["epoch_train_active_seconds"]) for x in history]
            typical = statistics.median(epoch_times[-5:])
            remaining = max(0, min(epochs - epoch, patience - stale)) * typical
            console(
                f"epoch {epoch}: loss={history[-1]['mean_loss']:.4f}; dev exact={metric:.4f}; "
                f"decision={dev_metrics['decision_accuracy']:.4f}; active epoch={human_duration(epoch_train_seconds)}; "
                f"active run={human_duration(manager.active_elapsed_seconds())}; estimated remaining={human_duration(remaining)}"
            )
            epoch += 1
            cursor = 0
            order = []
            loss_sum = 0.0
            loss_count = 0
            save_latest()
            if stale >= patience:
                console(f"Early stopping after {stale} stale epochs; best epoch={best_epoch}, exact={best_metric:.4f}")
                break

        best = manager.load_checkpoint("policy_best.pt")
        if best is None:
            raise RuntimeError("No best policy checkpoint")
        set_peft_model_state_dict(model, best["adapter_state_dict"])
        out_dir = resolve_path(cfg["output"]["adapter_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir, safe_serialization=True)
        tokenizer.save_pretrained(out_dir)
        report = {
            "experiment": SCRIPT_VERSION,
            "status": "complete",
            "model": cfg["model"]["repo_id"],
            "model_path": str(model_path),
            "trainable_lora_parameters": trainable_count,
            "best_epoch": int(best["epoch"]),
            "best_dev": best["dev_metrics"],
            "history": history,
            "adapter_dir": str(out_dir),
            "recovery_run_dir": str(manager.paths.run_dir),
            "phase2_trained": False,
            "phase2_note": "Result injection is trained separately after tool-policy evaluation.",
        }
        report_path = resolve_path(cfg["output"]["results_dir"]) / "tool_policy_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        manager.mark_complete(report=str(report_path), adapter_dir=str(out_dir))
        console(f"Saved adapter: {out_dir}")
        console(f"Saved report: {report_path}")
    except BaseException as exc:
        manager.handle_exception(exc)
        raise
    finally:
        manager.close()


if __name__ == "__main__":
    main()

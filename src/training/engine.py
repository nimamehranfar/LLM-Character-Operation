"""Single-GPU, token-weighted training with optimizer-boundary recovery."""
from __future__ import annotations

import math
import json
import random
import time
from typing import Any, Callable

import torch
from tqdm import tqdm
from transformers import get_scheduler

from src.training.batching import length_grouped_order, resolve_batch_plan
from src.training.recovery import capture_rng_state, optimizer_to_device, restore_rng_state
from src.training.reporting import MetricJournal


def add_training_arguments(parser) -> None:
    parser.add_argument("--batch-size", type=int, help="Sequences in one GPU forward/backward pass")
    parser.add_argument("--global-batch-size", type=int, help="Effective sequences per optimizer update")
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--seed", type=int)


def apply_training_arguments(cfg, args) -> None:
    for key, value in (("per_device_train_batch_size", args.batch_size),
                       ("global_batch_size", args.global_batch_size),
                       ("gradient_accumulation_steps", args.gradient_accumulation_steps), ("seed", args.seed)):
        if value is not None:
            cfg["training"][key] = value
    if args.gradient_accumulation_steps is not None and args.global_batch_size is None:
        cfg["training"].pop("global_batch_size", None)
    if args.eval_batch_size is not None:
        cfg["evaluation"]["batch_size"] = args.eval_batch_size


@torch.no_grad()
def teacher_forced_loss(model, batches, device) -> float:
    numerator, denominator = 0.0, 0
    for batch in batches:
        tokens = int((batch["labels"][:, 1:] != -100).sum())
        if not tokens:
            continue
        tensors = {key: value.to(device) for key, value in batch.items()}
        output = model(**tensors, use_cache=False)
        numerator += float(output.loss) * tokens
        denominator += tokens
    return numerator / denominator if denominator else float("nan")


def train_single_gpu(*, module, optimizer, manager, cfg, rows, lengths,
                     prepare: Callable, forward: Callable, evaluate: Callable,
                     state_cpu: Callable, restore: Callable, checkpoint_prefix: str,
                     latest: dict | None, metric_key: str) -> dict[str, Any]:
    """prepare(indices) returns CPU batch metadata; forward(batch, micro_step)
    returns (mean target-token loss, target_tokens, input_tokens, example_count).
    prepare must not put an accumulation window on GPU at once.
    """
    tr = cfg["training"]
    plan = resolve_batch_plan(per_device_batch_size=int(tr.get("per_device_train_batch_size", 1)),
                              gradient_accumulation_steps=int(tr.get("gradient_accumulation_steps", 1)),
                              world_size=1, global_batch_size=tr.get("global_batch_size"))
    print(f"Batch: micro={plan.per_device_batch_size}; accumulation={plan.gradient_accumulation_steps}; "
          f"effective={plan.effective_global_batch_size}; GPU count=1", flush=True)
    if not rows:
        raise ValueError("Training selection is empty")
    epochs = int(tr["epochs"])
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if int(tr["early_stopping_patience"]) < 1:
        raise ValueError("early_stopping_patience must be positive")
    if not 0 <= float(tr.get("warmup_ratio", 0)) < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    steps_per_epoch = math.ceil(math.ceil(len(rows) / plan.per_device_batch_size) / plan.gradient_accumulation_steps)
    scheduler = get_scheduler(str(tr.get("lr_scheduler_type", "constant")), optimizer=optimizer,
                              num_warmup_steps=int(steps_per_epoch * epochs * float(tr.get("warmup_ratio", 0))),
                              num_training_steps=steps_per_epoch * epochs)
    state = dict(latest.get("loop", {})) if latest else {}
    if latest:
        restore(latest["trainable_state_dict"])
        optimizer.load_state_dict(latest["optimizer_state_dict"])
        optimizer_to_device(optimizer, next(module.parameters()).device)
        scheduler.load_state_dict(latest["scheduler_state_dict"])
        restore_rng_state(latest.get("rng_state", {}))
    defaults = {"epoch": 1, "cursor": 0, "order": [], "optimizer_step": 0,
                "micro_step": 0, "best_metric": -1.0, "stale": 0, "history": [], "steps": [],
                "loss_numerator": 0.0, "target_tokens": 0, "input_tokens": 0,
                "examples": 0, "epoch_train_seconds": 0.0, "training_seconds": 0.0,
                "validation_seconds": 0.0, "checkpoint_seconds": 0.0, "batch_preparation_seconds": 0.0,
                "peak_allocated_vram_mib": 0.0, "peak_reserved_vram_mib": 0.0}
    for key, value in defaults.items():
        state.setdefault(key, value)
    journal = MetricJournal(manager.paths.artifacts / "optimizer_steps.jsonl")
    # The checkpoint is authoritative: remove journal rows after its last committed update.
    journal.path.write_text("".join(json.dumps(r) + "\n" for r in state["steps"]), encoding="utf-8")
    parameters = [p for p in module.parameters() if p.requires_grad]
    device = next(module.parameters()).device
    save_every = int(cfg["recovery"].get("checkpoint_every_optimizer_steps", 25))
    if save_every < 1:
        raise ValueError("checkpoint_every_optimizer_steps must be positive")

    def save_latest():
        tick = time.perf_counter()
        manager.checkpoint(checkpoint_prefix + "_latest.pt", {
            "trainable_state_dict": state_cpu(), "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), "rng_state": capture_rng_state(),
            "loop": state,
        })
        state["checkpoint_seconds"] += time.perf_counter() - tick

    # Never serialize parameters after a partial update on an exception. The last
    # optimizer-boundary checkpoint remains the valid continuation point.
    manager.set_emergency_saver(None)
    loop_started = time.perf_counter()
    while state["epoch"] <= epochs and state["stale"] < int(tr["early_stopping_patience"]):
        epoch = state["epoch"]
        if not state["order"]:
            indices = list(range(len(rows)))
            state["order"] = (length_grouped_order(indices, lengths, seed=int(tr["seed"]) + epoch * 1009,
                               batch_size=plan.effective_global_batch_size)
                               if tr.get("group_by_length", True) else
                               random.Random(int(tr["seed"]) + epoch * 1009).sample(indices, len(indices)))
        order = state["order"]
        bar = tqdm(total=len(order), initial=state["cursor"], desc=f"epoch {epoch}/{epochs}", unit="ex")
        module.train()
        while state["cursor"] < len(order):
            window_indices = order[state["cursor"]:state["cursor"] + plan.effective_global_batch_size]
            prepare_started = time.perf_counter()
            batches = [prepare(window_indices[i:i + plan.per_device_batch_size])
                       for i in range(0, len(window_indices), plan.per_device_batch_size)]
            state["batch_preparation_seconds"] += time.perf_counter() - prepare_started
            # Each CPU batch declares its number of supervised tokens. This gives
            # exactly the loss of the concatenated effective batch, even at epoch end.
            target_count = sum(int(batch["target_tokens"]) for batch in batches)
            if not target_count:
                raise ValueError("Batch has no supervised tokens")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            tick = time.perf_counter()
            numerator = input_count = seen = 0
            gates, layer_ids = [], []
            layer_metrics = {}
            for batch in batches:
                loss, target_tokens, input_tokens, example_count, extra = forward(batch, state["micro_step"])
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss at optimizer step {state['optimizer_step'] + 1}")
                (loss * (target_tokens / target_count)).backward()
                numerator += float(loss.detach()) * target_tokens
                input_count += input_tokens
                seen += example_count
                if extra.get("gate") is not None:
                    gates.append(float(extra["gate"]))
                if extra.get("layer") is not None:
                    layer = int(extra["layer"])
                    layer_ids.append(layer)
                    item = layer_metrics.setdefault(str(layer), {"examples": 0, "target_tokens": 0,
                                                                "loss_numerator": 0.0, "gate_sum": 0.0})
                    item["examples"] += example_count
                    item["target_tokens"] += target_tokens
                    item["loss_numerator"] += float(loss.detach()) * target_tokens
                    item["gate_sum"] += float(extra.get("gate") or 0) * example_count
                state["micro_step"] += 1
                del loss
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float(tr["gradient_clip_norm"]),
                                                      error_if_nonfinite=True)
            learning_rate = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - tick
            state["optimizer_step"] += 1
            state["cursor"] += len(window_indices)
            state["loss_numerator"] += numerator
            state["target_tokens"] += target_count
            state["input_tokens"] += input_count
            state["examples"] += seen
            state["training_seconds"] += elapsed
            state["epoch_train_seconds"] += elapsed
            state["peak_allocated_vram_mib"] = max(state["peak_allocated_vram_mib"], torch.cuda.max_memory_allocated(device) / 1024**2)
            state["peak_reserved_vram_mib"] = max(state["peak_reserved_vram_mib"], torch.cuda.max_memory_reserved(device) / 1024**2)
            row = {"optimizer_step": state["optimizer_step"], "epoch": epoch,
                   "loss": numerator / target_count, "learning_rate": learning_rate,
                   "gradient_norm": float(grad_norm), "examples": seen,
                   "target_tokens": target_count, "input_tokens": input_count, "step_seconds": elapsed,
                   "examples_per_second": seen / elapsed, "tokens_per_second": input_count / elapsed,
                   "allocated_vram_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                   "reserved_vram_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
                   "mean_gate": sum(gates) / len(gates) if gates else None,
                   "layers": ",".join(str(x) for x in sorted(set(layer_ids))),
                   "layer_metrics": layer_metrics}
            state["steps"].append(row)
            journal.append(row)
            bar.update(len(window_indices))
            bar.set_postfix(loss=f"{row['loss']:.4f}", step=state["optimizer_step"])
            if state["optimizer_step"] % save_every == 0 or state["cursor"] == len(order):
                save_latest()
                manager.progress(phase=checkpoint_prefix, stage="training", epoch=epoch,
                                 cursor=state["cursor"], total=len(order))
            manager.check_stop(saver=save_latest)
        bar.close()
        tick = time.perf_counter()
        module.eval()
        metrics = evaluate()
        validation_seconds = time.perf_counter() - tick
        state["validation_seconds"] += validation_seconds
        state["peak_allocated_vram_mib"] = max(state["peak_allocated_vram_mib"], torch.cuda.max_memory_allocated(device) / 1024**2)
        state["peak_reserved_vram_mib"] = max(state["peak_reserved_vram_mib"], torch.cuda.max_memory_reserved(device) / 1024**2)
        metric = float(metrics[metric_key])
        if not math.isfinite(metric):
            raise RuntimeError("Non-finite development selection metric")
        improved = metric > state["best_metric"] + float(tr.get("min_delta", 0))
        if improved:
            state["best_metric"], state["stale"] = metric, 0
            checkpoint_started = time.perf_counter()
            manager.checkpoint(checkpoint_prefix + "_best.pt", {
                "trainable_state_dict": state_cpu(), "epoch": epoch, "metrics": metrics,
            })
            state["checkpoint_seconds"] += time.perf_counter() - checkpoint_started
        else:
            state["stale"] += 1
        epoch_row = {"epoch": epoch, "mean_loss": state["loss_numerator"] / max(1, state["target_tokens"]),
                     "train_seconds": state["epoch_train_seconds"], "validation_seconds": validation_seconds,
                     "examples": state["examples"], "target_tokens": state["target_tokens"],
                     "input_tokens": state["input_tokens"], "best": improved, "dev": metrics,
                     **{f"dev_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}}
        state["history"].append(epoch_row)
        print(f"epoch {epoch}: loss={epoch_row['mean_loss']:.5f}; dev {metric_key}={metric:.4%}; "
              f"training={state['epoch_train_seconds']:.1f}s; validation={validation_seconds:.1f}s", flush=True)
        state["epoch"] += 1
        state["cursor"], state["order"] = 0, []
        for key in ("loss_numerator", "target_tokens", "input_tokens", "examples", "epoch_train_seconds"):
            state[key] = 0
        save_latest()
        if state["stale"] >= int(tr["early_stopping_patience"]):
            break
    best = manager.load_checkpoint(checkpoint_prefix + "_best.pt")
    if best is None:
        raise RuntimeError("Training did not produce a valid best checkpoint")
    restore(best["trainable_state_dict"])
    module.eval()
    return {"batch_plan": plan.__dict__, "best": best, "history": state["history"],
            "step_rows": state["steps"], "optimizer_steps": state["optimizer_step"],
            "training_seconds": state["training_seconds"], "validation_seconds": state["validation_seconds"],
            "checkpoint_seconds": state["checkpoint_seconds"],
            "batch_preparation_seconds": state["batch_preparation_seconds"],
            "peak_allocated_vram_mib": state["peak_allocated_vram_mib"],
            "peak_reserved_vram_mib": state["peak_reserved_vram_mib"],
            "train_loop_wall_seconds_current_session": time.perf_counter() - loop_started}

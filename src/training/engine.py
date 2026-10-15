"""Token-weighted training on one GPU or two torchrun workers."""
from __future__ import annotations

import math
import json
import random
import time
from typing import Any, Callable

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import get_scheduler

from src.training.batching import length_grouped_order, resolve_batch_plan
from src.training.recovery import capture_rng_state, optimizer_to_device, restore_rng_state
from src.training.reporting import MetricJournal


def add_training_arguments(parser) -> None:
    parser.add_argument("--batch-size", type=int, help="Simultaneous sequences PER GPU; accumulation defaults to 1 when supplied")
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
    if (args.batch_size is not None or args.gradient_accumulation_steps is not None) and args.global_batch_size is None:
        cfg["training"].pop("global_batch_size", None)
        if args.gradient_accumulation_steps is None:
            cfg["training"]["gradient_accumulation_steps"] = 1
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
                     latest: dict | None, metric_key: str,
                     set_epoch: Callable[[int], None] | None = None) -> dict[str, Any]:
    """prepare(indices) returns CPU batch metadata; forward(batch, micro_step)
    returns (mean target-token loss, target_tokens, input_tokens, example_count).
    prepare must not put an accumulation window on GPU at once.
    """
    tr = cfg["training"]
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    plan = resolve_batch_plan(per_device_batch_size=int(tr.get("per_device_train_batch_size", 1)),
                              gradient_accumulation_steps=int(tr.get("gradient_accumulation_steps", 1)),
                              world_size=world, global_batch_size=tr.get("global_batch_size"))
    if rank == 0:
        print(f"Batch: simultaneous_per_gpu={plan.per_device_batch_size}; "
              f"simultaneous_total={plan.per_device_batch_size * world}; "
              f"accumulation={plan.gradient_accumulation_steps}; "
              f"effective_per_update={plan.effective_global_batch_size}; GPU count={world}", flush=True)
        from src.training.recovery import atomic_write_json
        atomic_write_json(manager.paths.artifacts / "effective_training_plan.json",
                          {"batch_plan": plan.__dict__, "training": tr, "data": cfg.get("data", {})})
    if not rows:
        raise ValueError("Training selection is empty")
    epochs = int(tr["epochs"])
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if int(tr["early_stopping_patience"]) < 1:
        raise ValueError("early_stopping_patience must be positive")
    if not 0 <= float(tr.get("warmup_ratio", 0)) < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    steps_per_epoch = math.ceil(len(rows) / plan.effective_global_batch_size)
    scheduler = get_scheduler(str(tr.get("lr_scheduler_type", "constant")), optimizer=optimizer,
                              num_warmup_steps=int(steps_per_epoch * epochs * float(tr.get("warmup_ratio", 0))),
                              num_training_steps=steps_per_epoch * epochs)
    state = dict(latest.get("loop", {})) if latest else {}
    if world > 1 and not latest:
        for parameter in module.parameters():
            if parameter.requires_grad:
                dist.broadcast(parameter.data, src=0)
        torch.manual_seed(int(tr["seed"]) + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(int(tr["seed"]) + rank)
    if latest:
        restore(latest["trainable_state_dict"])
        optimizer.load_state_dict(latest["optimizer_state_dict"])
        optimizer_to_device(optimizer, next(module.parameters()).device)
        scheduler.load_state_dict(latest["scheduler_state_dict"])
        rng_states = latest.get("rng_states")
        restore_rng_state(rng_states[rank] if rng_states else latest.get("rng_state", {}))
    defaults = {"epoch": 1, "cursor": 0, "order": [], "optimizer_step": 0,
                "micro_step": 0, "best_metric": -1.0, "stale": 0, "history": [], "steps": [],
                "loss_numerator": 0.0, "target_tokens": 0, "input_tokens": 0,
                "examples": 0, "epoch_train_seconds": 0.0, "training_seconds": 0.0,
                "validation_seconds": 0.0, "checkpoint_seconds": 0.0, "batch_preparation_seconds": 0.0,
                "peak_allocated_vram_mib": 0.0, "peak_reserved_vram_mib": 0.0}
    for key, value in defaults.items():
        state.setdefault(key, value)
    journal = MetricJournal(manager.paths.artifacts / "optimizer_steps.jsonl") if rank == 0 else None
    # The checkpoint is authoritative: remove journal rows after its last committed update.
    if journal:
        journal.path.write_text("".join(json.dumps(r) + "\n" for r in state["steps"]), encoding="utf-8")
    parameters = [p for p in module.parameters() if p.requires_grad]
    device = next(module.parameters()).device
    def memory_mib(kind: str) -> float:
        if device.type != "cuda":
            return 0.0
        value = (torch.cuda.max_memory_allocated(device) if kind == "allocated"
                 else torch.cuda.max_memory_reserved(device))
        return value / 1024**2
    save_every = int(cfg["recovery"].get("checkpoint_every_optimizer_steps", 25))
    if save_every < 1:
        raise ValueError("checkpoint_every_optimizer_steps must be positive")

    def save_latest():
        tick = time.perf_counter()
        rng = capture_rng_state()
        rng_states = [None] * world if rank == 0 else None
        if world > 1:
            dist.gather_object(rng, rng_states, dst=0)
        else:
            rng_states = [rng]
        if rank == 0:
            manager.checkpoint(checkpoint_prefix + "_latest.pt", {
                "trainable_state_dict": state_cpu(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "rng_states": rng_states,
                "batch_plan": plan.__dict__,
                "loop": state,
            })
            state["checkpoint_seconds"] += time.perf_counter() - tick
        if world > 1:
            dist.barrier()

    # Never serialize parameters after a partial update on an exception. The last
    # optimizer-boundary checkpoint remains the valid continuation point.
    if rank == 0:
        manager.set_emergency_saver(None)
    loop_started = time.perf_counter()
    while state["epoch"] <= epochs and state["stale"] < int(tr["early_stopping_patience"]):
        epoch = state["epoch"]
        if set_epoch is not None:
            set_epoch(epoch)
        if not state["order"]:
            indices = list(range(len(rows)))
            state["order"] = (length_grouped_order(indices, lengths, seed=int(tr["seed"]) + epoch * 1009,
                               batch_size=plan.effective_global_batch_size)
                               if tr.get("group_by_length", True) else
                               random.Random(int(tr["seed"]) + epoch * 1009).sample(indices, len(indices)))
        order = state["order"]
        bar = tqdm(total=len(order), initial=state["cursor"], desc=f"epoch {epoch}/{epochs}",
                   unit="ex", disable=rank != 0)
        module.train()
        while state["cursor"] < len(order):
            window_indices = order[state["cursor"]:state["cursor"] + plan.effective_global_batch_size]
            prepare_started = time.perf_counter()
            local_indices = window_indices[rank::world]
            batches = [prepare(local_indices[i:i + plan.per_device_batch_size])
                       for i in range(0, len(local_indices), plan.per_device_batch_size)]
            state["batch_preparation_seconds"] += time.perf_counter() - prepare_started
            # Each CPU batch declares its number of supervised tokens. This gives
            # exactly the loss of the concatenated effective batch, even at epoch end.
            local_target_count = sum(int(batch["target_tokens"]) for batch in batches)
            target_count = local_target_count
            if world > 1:
                count_tensor = torch.tensor([target_count], dtype=torch.long, device=device)
                dist.all_reduce(count_tensor)
                target_count = int(count_tensor.item())
            if not target_count:
                raise ValueError("Batch has no supervised tokens")
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
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
            if world > 1:
                # The sum, rather than the DDP mean, matches the global token-weighted loss.
                presence = torch.tensor([int(p.grad is not None) for p in parameters],
                                        dtype=torch.int32, device=device)
                dist.all_reduce(presence, op=dist.ReduceOp.SUM)
                for parameter, used in zip(parameters, presence.tolist()):
                    if not used:
                        continue
                    if parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                totals = torch.tensor([numerator, input_count, seen], dtype=torch.float64, device=device)
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
                numerator, input_count, seen = totals.tolist()
                seen, input_count = int(seen), int(input_count)
                extras = [None] * world
                dist.all_gather_object(extras, {"gates": gates, "layer_ids": layer_ids,
                                                "layer_metrics": layer_metrics})
                gates = [gate for item in extras for gate in item["gates"]]
                layer_ids = [layer for item in extras for layer in item["layer_ids"]]
                layer_metrics = {}
                for item in extras:
                    for layer, values in item["layer_metrics"].items():
                        merged = layer_metrics.setdefault(layer, {key: 0 for key in values})
                        for key, value in values.items():
                            merged[key] += value
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float(tr["gradient_clip_norm"]),
                                                      error_if_nonfinite=True)
            learning_rate = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
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
            state["peak_allocated_vram_mib"] = max(state["peak_allocated_vram_mib"], memory_mib("allocated"))
            state["peak_reserved_vram_mib"] = max(state["peak_reserved_vram_mib"], memory_mib("reserved"))
            row = {"optimizer_step": state["optimizer_step"], "epoch": epoch,
                   "loss": numerator / target_count, "learning_rate": learning_rate,
                   "gradient_norm": float(grad_norm), "examples": seen,
                   "target_tokens": target_count, "input_tokens": input_count, "step_seconds": elapsed,
                   "examples_per_second": seen / elapsed, "tokens_per_second": input_count / elapsed,
                   "allocated_vram_gib": memory_mib("allocated") / 1024,
                   "reserved_vram_gib": memory_mib("reserved") / 1024,
                   "mean_gate": sum(gates) / len(gates) if gates else None,
                   "layers": ",".join(str(x) for x in sorted(set(layer_ids))),
                   "layer_metrics": layer_metrics}
            state["steps"].append(row)
            if journal:
                journal.append(row)
            bar.update(len(window_indices))
            bar.set_postfix(loss=f"{row['loss']:.4f}", step=state["optimizer_step"])
            if state["optimizer_step"] % save_every == 0 or state["cursor"] == len(order):
                save_latest()
                if rank == 0:
                    manager.progress(phase=checkpoint_prefix, stage="training", epoch=epoch,
                                     cursor=state["cursor"], total=len(order))
            if rank == 0:
                manager.check_stop(saver=None)
        bar.close()
        tick = time.perf_counter()
        module.eval()
        metrics = evaluate() if rank == 0 else None
        if world > 1:
            values = [float(metrics[metric_key]) if rank == 0 else None]
            dist.broadcast_object_list(values, src=0)
            if rank != 0:
                metrics = {metric_key: values[0]}
        validation_seconds = time.perf_counter() - tick
        state["validation_seconds"] += validation_seconds
        state["peak_allocated_vram_mib"] = max(state["peak_allocated_vram_mib"], memory_mib("allocated"))
        state["peak_reserved_vram_mib"] = max(state["peak_reserved_vram_mib"], memory_mib("reserved"))
        metric = float(metrics[metric_key])
        if not math.isfinite(metric):
            raise RuntimeError("Non-finite development selection metric")
        improved = metric > state["best_metric"] + float(tr.get("min_delta", 0))
        if improved:
            state["best_metric"], state["stale"] = metric, 0
            if rank == 0:
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
        if rank == 0:
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
    gpu_peaks = [{"rank": rank, "peak_allocated_vram_mib": state["peak_allocated_vram_mib"],
                  "peak_reserved_vram_mib": state["peak_reserved_vram_mib"]}]
    if world > 1:
        gpu_peaks = [None] * world
        dist.all_gather_object(gpu_peaks, {"rank": rank,
            "peak_allocated_vram_mib": state["peak_allocated_vram_mib"],
            "peak_reserved_vram_mib": state["peak_reserved_vram_mib"]})
        state["peak_allocated_vram_mib"] = max(item["peak_allocated_vram_mib"] for item in gpu_peaks)
        state["peak_reserved_vram_mib"] = max(item["peak_reserved_vram_mib"] for item in gpu_peaks)
    return {"batch_plan": plan.__dict__, "best": best, "history": state["history"],
            "step_rows": state["steps"], "optimizer_steps": state["optimizer_step"],
            "training_seconds": state["training_seconds"], "validation_seconds": state["validation_seconds"],
            "checkpoint_seconds": state["checkpoint_seconds"],
            "batch_preparation_seconds": state["batch_preparation_seconds"],
            "peak_allocated_vram_mib": state["peak_allocated_vram_mib"],
            "peak_reserved_vram_mib": state["peak_reserved_vram_mib"],
            "per_gpu_training_peak_vram_mib": gpu_peaks,
            "train_loop_wall_seconds_current_session": time.perf_counter() - loop_started}

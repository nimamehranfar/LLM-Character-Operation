"""Conservative CUDA forward/backward capacity calibration without weight updates."""
from __future__ import annotations

import gc
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.training.recovery import atomic_write_json, capture_rng_state, restore_rng_state


def largest_passing_batch(maximum, fits):
    """Exponential search followed by integer bisection; require batch one to fit."""
    if maximum < 1:
        raise ValueError("maximum must be positive")
    if not fits(1):
        raise RuntimeError("Even one training sequence does not fit the selected VRAM budget")
    low, high = 1, min(2, maximum)
    while high > low:
        if not fits(high):
            break
        low = high
        high = min(maximum, high * 2)
    if low == maximum:
        return low
    while high - low > 1:
        middle = (low + high) // 2
        if fits(middle):
            low = middle
        else:
            high = middle
    return low


def calibrate_batch_size(*, module, optimizer, prepare, forward, lengths, cfg, manager):
    device = next(module.parameters()).device
    if device.type != "cuda":
        raise ValueError("--auto-batch-size requires CUDA")
    if not lengths:
        raise ValueError("Cannot calibrate on an empty training set")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    tr = cfg["training"]
    target = float(tr["vram_target"])
    total = torch.cuda.get_device_properties(device).total_memory
    budget = min(total * target, float(tr.get("max_vram_mib") or total / 1024**2) * 1024**2)
    # Include space for lazy optimizer-state allocation and later collective buffers.
    optimizer_headroom = max(2 * 1024**3, sum(p.numel() * 16 for p in module.parameters() if p.requires_grad))
    maximum = int(tr["auto_batch_max_size"])
    sequence_length = min(int(tr["max_sequence_length"]), max(int(n) for n in lengths))
    longest = max(range(len(lengths)), key=lengths.__getitem__)
    probe_steps = max(1, len(cfg.get("architecture", {}).get("candidate_layers", [])))
    rng = capture_rng_state()
    was_training = module.training
    started = time.perf_counter()
    trials = []
    torch.cuda.set_per_process_memory_fraction(budget / total, device=device.index)

    def fits(size):
        batch = loss = output = None
        passed, peak = True, 0
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            batch = prepare([longest] * size)
            tensors = batch["tensors"]
            # Exercise the dataset's maximum training length, including padding.
            # Stage 1's length scan covers every planned epoch's rendered input.
            for key, value in list(tensors.items()):
                if value.ndim != 2:
                    continue
                gap = max(0, sequence_length - value.shape[1])
                tensors[key] = F.pad(value, (0, gap), value=-100 if key == "labels" else 0)
                if key == "attention_mask":
                    tensors[key].fill_(1)
            # Stage 2 cycles injection layers: budget for every candidate layer.
            for probe_step in range(probe_steps):
                output = forward(batch, probe_step)
                loss = output[0]
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Non-finite loss during auto-batch calibration")
                loss.backward()
                optimizer.zero_grad(set_to_none=True)
                loss = output = None
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
            passed = peak + optimizer_headroom <= budget
        except torch.OutOfMemoryError:
            passed = False
            peak = torch.cuda.max_memory_allocated(device)
        finally:
            loss = output = batch = None
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
        # Every rank attempts the same candidate independently; an OOM on either
        # device rejects it for both, without collectives inside the failed pass.
        result = torch.tensor([int(passed)], device=device, dtype=torch.int32)
        if world > 1:
            dist.all_reduce(result, op=dist.ReduceOp.MIN)
        passed = bool(result.item())
        readings = [None] * world
        reading = {"rank": rank, "peak_allocated_vram_mib": peak / 1024**2,
                   "gpu_total_vram_mib": total / 1024**2, "budget_mib": budget / 1024**2}
        if world > 1:
            dist.all_gather_object(readings, reading)
        else:
            readings = [reading]
        trials.append({"simultaneous_per_gpu": size, "fits": passed, "workers": readings})
        if rank == 0:
            peaks = ", ".join(f"GPU {r['rank']}: {r['peak_allocated_vram_mib'] / 1024:.2f} GiB" for r in readings)
            print(f"Auto batch: per_gpu={size}; fits={passed}; {peaks}", flush=True)
        return passed

    try:
        module.train()
        selected = largest_passing_batch(maximum, fits)
        if rank == 0:
            report = {"selected_simultaneous_per_gpu": selected, "simultaneous_total": selected * world,
                      "gradient_accumulation_steps": 1, "vram_target": target,
                      "probe_sequence_length": sequence_length,
                      "candidate_layer_passes_per_trial": probe_steps,
                      "optimizer_headroom_bytes": optimizer_headroom,
                      "search_ceiling_reached": selected == maximum,
                      "calibration_seconds": time.perf_counter() - started, "trials": trials,
                      "note": "Capacity search at maximum measured training sequence length; not a throughput optimum or a guarantee of constant VRAM usage."}
            atomic_write_json(manager.paths.artifacts / "batch_calibration.json", report)
            print(f"Auto batch selected: per_gpu={selected}; simultaneous_total={selected * world}; "
                  "accumulation=1. Calibration: " + str(manager.paths.artifacts / "batch_calibration.json"), flush=True)
        return selected
    finally:
        optimizer.zero_grad(set_to_none=True)
        module.train(was_training)
        restore_rng_state(rng)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

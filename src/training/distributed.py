"""Small single-node torchrun helpers for the custom two-GPU trainers."""
from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from src.training.recovery import RecoveryManager, RunPaths


def initialize() -> tuple[int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return 0, 1
    if world != 2 or not torch.cuda.is_available() or torch.cuda.device_count() < world:
        raise RuntimeError("This trainer supports torchrun with exactly two visible CUDA GPUs")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # Rank one waits while rank zero runs full development-set generation.
    dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(hours=4))
    return dist.get_rank(), world


def finish() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


class FollowerManager:
    """Read checkpoints without competing for the rank-zero recovery lock."""

    def __init__(self, run_dir: str):
        self.paths = RunPaths(Path(run_dir))

    def load_checkpoint(self, name: str):
        path = self.paths.checkpoints / name
        return torch.load(path, map_location="cpu", weights_only=False) if path.exists() else None

    def close(self):
        pass


def create_manager(*, checkpoint_root, config, config_path, script_version,
                   heartbeat_seconds, fresh, explicit_run_dir):
    rank = dist.get_rank() if dist.is_initialized() else 0
    manager = (RecoveryManager.create_or_resume(
        checkpoint_root=checkpoint_root, config=config, config_path=config_path,
        script_version=script_version, heartbeat_seconds=heartbeat_seconds,
        fresh=fresh, explicit_run_dir=explicit_run_dir) if rank == 0 else None)
    if dist.is_initialized():
        paths = [str(manager.paths.run_dir) if rank == 0 else ""]
        dist.broadcast_object_list(paths, src=0)
        dist.barrier()
        if rank != 0:
            manager = FollowerManager(paths[0])
    return manager

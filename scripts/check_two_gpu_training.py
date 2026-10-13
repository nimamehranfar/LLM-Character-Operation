"""Fast NCCL and device-mapping preflight for a two-GPU SLURM allocation."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist

from src.model.device import select_device
from src.training.distributed import finish, initialize


def main() -> None:
    rank, world = initialize()
    if world != 2:
        raise RuntimeError("Launch this check with torchrun --nproc_per_node=2")
    try:
        device = select_device(announce=True)
        weight = torch.nn.Parameter(torch.tensor([1.0], device=device))
        loss = ((weight * (rank + 1) - 3.0) ** 2).sum()
        loss.backward()
        dist.all_reduce(weight.grad, op=dist.ReduceOp.SUM)
        torch.optim.SGD([weight], lr=0.01).step()
        values = [None] * world
        dist.all_gather_object(values, (str(device), torch.cuda.get_device_name(device),
                                        float(weight.detach())))
        if rank == 0:
            assert values[0][0] != values[1][0], values
            assert abs(values[0][2] - values[1][2]) < 1e-6, values
            print(f"Two-GPU NCCL preflight passed: {values}", flush=True)
    finally:
        finish()


if __name__ == "__main__":
    main()

"""CPU two-process gradient and recovery smoke test; run this file directly."""
from __future__ import annotations

import copy
import random
import socket
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist

from src.training.distributed import create_manager
from src.training.engine import train_single_gpu


def worker(rank: int, port: int):
    print(f"rank {rank}: initializing", flush=True)
    store = dist.TCPStore("127.0.0.1", port, 2, is_master=rank == 0, use_libuv=False)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=2)
    print(f"rank {rank}: initialized", flush=True)
    rank = dist.get_rank()
    torch.manual_seed(42)
    model = torch.nn.Linear(1, 1, bias=False)
    initial = float(model.weight.detach())
    rows = [(1., 2., 1), (2., 3., 3), (3., 4., 2), (4., 1., 4), (5., 2., 1)]
    cfg = {"training": {"per_device_train_batch_size": 1, "global_batch_size": 4,
                        "gradient_accumulation_steps": 2, "epochs": 2, "seed": 42,
                        "gradient_clip_norm": 10., "early_stopping_patience": 3,
                        "lr_scheduler_type": "constant", "warmup_ratio": 0., "group_by_length": False},
           "recovery": {"checkpoint_every_optimizer_steps": 1}}
    manager = create_manager(checkpoint_root=ROOT / ".local" / "distributed-smoke",
                             config=cfg, config_path="smoke", script_version="smoke",
                             heartbeat_seconds=30, fresh=True, explicit_run_dir=None)
    print(f"rank {rank}: manager ready", flush=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)

    def prepare(indices):
        return {"rows": [rows[i] for i in indices],
                "target_tokens": sum(rows[i][2] for i in indices)}

    def forward(batch, micro_step):
        xs = torch.tensor([[x] for x, _, _ in batch["rows"]])
        ys = torch.tensor([[y] for _, y, _ in batch["rows"]])
        weights = torch.tensor([[w] for _, _, w in batch["rows"]])
        loss = ((model(xs) - ys).square() * weights).sum() / weights.sum()
        return loss, batch["target_tokens"], batch["target_tokens"], len(batch["rows"]), {}

    try:
        result = train_single_gpu(module=model, optimizer=optimizer, manager=manager,
                                  cfg=cfg, rows=rows, lengths=[1] * len(rows),
                                  prepare=prepare, forward=forward,
                                  evaluate=lambda: {"accuracy": float(model.weight.detach())},
                                  state_cpu=lambda: copy.deepcopy(model.state_dict()),
                                  restore=model.load_state_dict, checkpoint_prefix="smoke",
                                  latest=None, metric_key="accuracy")
        if rank == 0:
            reference = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                reference.weight.fill_(initial)
            ref_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
            best_weight = None
            best_metric = -1.0
            for epoch in (1, 2):
                order = random.Random(42 + epoch * 1009).sample(list(range(len(rows))), len(rows))
                for start in range(0, len(order), 4):
                    indices = order[start:start + 4]
                    ref_optimizer.zero_grad()
                    numerator = sum((reference(torch.tensor([[rows[i][0]]])) - rows[i][1]).square()
                                    * rows[i][2] for i in indices)
                    loss = numerator / sum(rows[i][2] for i in indices)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(reference.parameters(), 10.)
                    ref_optimizer.step()
                metric = float(reference.weight.detach())
                if metric > best_metric:
                    best_metric, best_weight = metric, reference.weight.detach().clone()
            torch.testing.assert_close(model.weight, best_weight, rtol=1e-5, atol=1e-6)
            assert result["batch_plan"]["effective_global_batch_size"] == 4
            assert [row["examples"] for row in result["step_rows"]] == [4, 1, 4, 1]
            assert len(result["history"]) == 2
            print("Two-worker gradient, short-window, checkpoint, and metric smoke test passed", flush=True)
    finally:
        manager.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    torch.multiprocessing.spawn(worker, args=(port,), nprocs=2)

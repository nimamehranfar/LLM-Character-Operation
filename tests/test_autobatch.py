import argparse
from types import SimpleNamespace

import pytest
import torch

from src.training.autobatch import largest_passing_batch
from src.training.engine import add_training_arguments, apply_training_arguments


@pytest.mark.parametrize("limit,maximum", [(1, 1024), (7, 1024), (32, 64), (1024, 1024), (1, 1)])
def test_capacity_search_finds_largest_fitting_integer(limit, maximum):
    attempts = []
    def fits(size):
        attempts.append(size)
        return size <= limit
    assert largest_passing_batch(maximum, fits) == min(limit, maximum)
    assert max(attempts) <= maximum
    assert len(attempts) <= 24


def test_capacity_search_fails_if_one_sequence_does_not_fit():
    with pytest.raises(RuntimeError, match="Even one"):
        largest_passing_batch(1024, lambda size: False)


def apply(*flags):
    parser = argparse.ArgumentParser()
    add_training_arguments(parser)
    cfg = {"training": {"global_batch_size": 2048, "gradient_accumulation_steps": 256,
                        "per_device_train_batch_size": 4}, "evaluation": {}}
    apply_training_arguments(cfg, parser.parse_args(flags))
    return cfg["training"]


def test_explicit_simultaneous_batch_removes_hidden_accumulation():
    tr = apply("--batch-size", "32")
    assert tr["per_device_train_batch_size"] == 32
    assert tr["gradient_accumulation_steps"] == 1
    assert "global_batch_size" not in tr


def test_accumulation_is_available_when_explicitly_requested():
    assert apply("--batch-size", "32", "--gradient-accumulation-steps", "2")["gradient_accumulation_steps"] == 2
    assert apply("--batch-size", "32", "--global-batch-size", "2048")["global_batch_size"] == 2048


def test_auto_batch_removes_implicit_accumulation_and_rejects_conflicts():
    tr = apply("--auto-batch-size")
    assert tr["auto_batch_size"] and tr["gradient_accumulation_steps"] == 1
    assert "global_batch_size" not in tr
    for flags in [("--batch-size", "16"), ("--global-batch-size", "2048"),
                  ("--gradient-accumulation-steps", "2")]:
        with pytest.raises(ValueError):
            apply("--auto-batch-size", *flags)


@pytest.mark.parametrize("world,gpu_gib,oom_limit,expected", [(1, 80, 7, 7), (2, 80, 7, 5), (1, 8, 100, 5)])
def test_calibration_handles_oom_without_changing_weights_or_rng(monkeypatch, tmp_path, world, gpu_gib, oom_limit, expected):
    from src.training import autobatch
    from src.training.recovery import load_json
    parameter = torch.nn.Parameter(torch.tensor(3.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    fake_parameter = SimpleNamespace(device=torch.device("cuda:0"), requires_grad=True, numel=lambda: 1)
    class Module:
        training = False
        def parameters(self):
            return iter([fake_parameter])
        def train(self, mode=True):
            self.training = mode
    module = Module()
    peak = [0]
    current_size = [0]
    rng_restored = []
    monkeypatch.setattr(autobatch, "capture_rng_state", lambda: {"sentinel": True})
    monkeypatch.setattr(autobatch, "restore_rng_state", lambda state: rng_restored.append(state))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(total_memory=gpu_gib * 1024**3))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: peak[0])
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(autobatch.dist, "is_initialized", lambda: world > 1)
    monkeypatch.setattr(autobatch.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(autobatch.dist, "get_world_size", lambda: world)
    def reduce(value, **kwargs):
        # Simulate a second GPU with lower capacity than rank zero.
        if current_size[0] > 5:
            value.fill_(0)
    monkeypatch.setattr(autobatch.dist, "all_reduce", reduce)
    monkeypatch.setattr(autobatch.dist, "all_gather_object",
                        lambda result, reading: result.__setitem__(slice(None), [reading, dict(reading, rank=1)]))
    real_tensor = torch.tensor
    monkeypatch.setattr(torch, "tensor", lambda *a, **k: real_tensor(*a, **{key: value for key, value in k.items() if key != "device"}))
    def prepare(indices):
        return {"tensors": {"input_ids": torch.ones(len(indices), 2, dtype=torch.long),
                            "attention_mask": torch.ones(len(indices), 2, dtype=torch.long),
                            "labels": torch.ones(len(indices), 2, dtype=torch.long)}, "count": len(indices)}
    def forward(batch, step):
        assert batch["tensors"]["input_ids"].shape[1] == 16
        assert batch["tensors"]["attention_mask"].all()
        assert (batch["tensors"]["labels"][:, 2:] == -100).all()
        size = batch["count"]
        current_size[0] = size
        peak[0] = size * 1024**3
        if size > oom_limit:
            raise torch.OutOfMemoryError("simulated")
        return (parameter.square(), 1, 1, size, {})
    cfg = {"training": {"vram_target": .95, "auto_batch_max_size": 32, "max_sequence_length": 16}}
    selected = autobatch.calibrate_batch_size(module=module, optimizer=optimizer, prepare=prepare,
        forward=forward, lengths=[16], cfg=cfg, manager=SimpleNamespace(paths=SimpleNamespace(artifacts=tmp_path)))
    assert selected == expected
    assert parameter.item() == 3.0 and parameter.grad is None
    assert rng_restored == [{"sentinel": True}] and module.training is False
    report = load_json(tmp_path / "batch_calibration.json", {})
    assert report["gradient_accumulation_steps"] == 1
    assert report["selected_simultaneous_per_gpu"] == selected
    assert report["simultaneous_total"] == selected * world
    assert any(not trial["fits"] for trial in report["trials"])

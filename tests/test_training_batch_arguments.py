import argparse

from src.training.engine import add_training_arguments, apply_training_arguments
from src.training.batching import resolve_batch_plan


def arguments(*flags):
    parser = argparse.ArgumentParser()
    add_training_arguments(parser)
    cfg = {"training": {"per_device_train_batch_size": 4,
                        "global_batch_size": 2048, "gradient_accumulation_steps": 256},
           "evaluation": {}}
    apply_training_arguments(cfg, parser.parse_args(flags))
    return cfg["training"]


def test_requested_batch_is_simultaneous_per_gpu_without_hidden_accumulation():
    tr = arguments("--batch-size", "1024")
    plan = resolve_batch_plan(per_device_batch_size=tr["per_device_train_batch_size"],
        gradient_accumulation_steps=tr["gradient_accumulation_steps"], world_size=2,
        global_batch_size=tr.get("global_batch_size"))
    assert plan.per_device_batch_size == 1024
    assert plan.gradient_accumulation_steps == 1
    assert plan.effective_global_batch_size == 2048


def test_explicit_accumulation_is_respected():
    tr = arguments("--batch-size", "32", "--gradient-accumulation-steps", "2")
    assert tr["per_device_train_batch_size"] == 32
    assert tr["gradient_accumulation_steps"] == 2
    assert "global_batch_size" not in tr


def test_explicit_effective_batch_and_legacy_config_remain_available():
    assert arguments("--batch-size", "32", "--global-batch-size", "2048")["global_batch_size"] == 2048
    assert arguments()["per_device_train_batch_size"] == 4
    assert arguments()["gradient_accumulation_steps"] == 256

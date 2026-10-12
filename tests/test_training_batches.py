from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.model.result_injector import LayeredSymbolicResultMapper, ResultSymbol, TrainableLayeredResultInjector
from src.training.batching import pad_training_sequences, resolve_batch_plan
from src.training.engine import train_single_gpu
from src.training.recovery import GracefulStop, RecoveryManager


def test_batch_plan_distinguishes_microbatch_and_effective_batch():
    plan = resolve_batch_plan(per_device_batch_size=8, gradient_accumulation_steps=1,
                              world_size=1, global_batch_size=1024)
    assert plan.gradient_accumulation_steps == 128
    assert plan.effective_global_batch_size == 1024
    with pytest.raises(ValueError, match="multiple"):
        resolve_batch_plan(per_device_batch_size=8, gradient_accumulation_steps=1,
                           world_size=1, global_batch_size=1023)


def test_training_padding_masks_targets_and_retains_positions():
    items = [(torch.tensor([1, 2, 3]), torch.ones(3), torch.tensor([-100, 2, 3])),
             (torch.tensor([1, 4]), torch.ones(2), torch.tensor([-100, 4]))]
    tensors, lengths, tokens = pad_training_sequences(items, pad_token_id=0, device=torch.device("cpu"))
    assert lengths == [3, 2] and tokens == 3
    assert tensors["labels"].tolist() == [[-100, 2, 3], [-100, 4, -100]]
    assert tensors["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]


def mapper():
    return LayeredSymbolicResultMapper(hidden_size=16, candidate_layers=[0, 1], result_dim=8,
                                      gate_dim=4, max_integer=128, ascii_vocab_size=128, gate_bias_init=-2)


def test_batched_mapper_matches_independent_mixed_symbols():
    torch.manual_seed(17)
    module = mapper()
    hidden = torch.randn(3, 16)
    symbols = [ResultSymbol("integer", 0), ResultSymbol("integer", 128), ResultSymbol("character", 65)]
    batched, gates = module(hidden, symbols, layer_index=0)
    individual = [module(hidden[i:i + 1], symbol, layer_index=0)[0] for i, symbol in enumerate(symbols)]
    torch.testing.assert_close(batched, torch.cat(individual))
    batched.square().sum().backward()
    assert module.integer_feature_projection.weight.grad is not None
    assert module.character_feature_projection.weight.grad is not None
    assert gates.shape == (3, 1)
    with pytest.raises(ValueError, match="symbols"):
        module(hidden, symbols[:1], layer_index=0)


def test_batched_injection_backpropagates_through_frozen_causal_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    base = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                                      head_dim=8, attention_dropout=0.0))
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    module = mapper()
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 6, 7, 0]])
    labels = torch.tensor([[-100, -100, -100, 4, 5], [-100, -100, 6, 7, -100]])
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
    symbols = [ResultSymbol("integer", 3), ResultSymbol("character", 65)]
    with TrainableLayeredResultInjector(base, module, layer_index=0, symbol=symbols,
                                       position_index=[2, 1], prefill_only=False) as injector:
        loss = base(input_ids=input_ids, attention_mask=mask, labels=labels, use_cache=False).loss
    loss.backward()
    assert len(injector.last_gates) == 2
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert all(p.grad is None for p in base.parameters())
    assert not base.model.layers[0]._forward_hooks


def _run_engine(tmp_path, monkeypatch, micro, global_batch, stop=False, resume=None):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *args: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *args: 0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(42)
    module = torch.nn.Linear(1, 1, bias=False)
    cfg = {"training": {"per_device_train_batch_size": micro, "global_batch_size": global_batch,
                         "gradient_accumulation_steps": 1, "epochs": 2, "seed": 42,
                         "gradient_clip_norm": 10., "early_stopping_patience": 3,
                         "lr_scheduler_type": "linear", "warmup_ratio": 0., "group_by_length": False},
           "recovery": {"checkpoint_every_optimizer_steps": 1}}
    manager = RecoveryManager.create_or_resume(checkpoint_root=tmp_path, config=cfg, config_path="fixture",
        script_version="fixture", heartbeat_seconds=30, fresh=resume is None, explicit_run_dir=resume)
    optimizer = torch.optim.AdamW(module.parameters(), lr=.03)
    rows = [(1., 2., 1), (2., 3., 3), (3., 4., 2), (4., 1., 4), (5., 2., 1)]
    def prepare(indices):
        return {"rows": [rows[i] for i in indices], "target_tokens": sum(rows[i][2] for i in indices)}
    def forward(batch, micro_step):
        values = batch["rows"]
        xs = torch.tensor([[x] for x, _, _ in values])
        ys = torch.tensor([[y] for _, y, _ in values])
        weights = torch.tensor([[w] for _, _, w in values])
        loss = ((module(xs) - ys).square() * weights).sum() / weights.sum()
        return loss, batch["target_tokens"], batch["target_tokens"], len(values), {}
    if stop:
        manager.stop_requested = True
    try:
        trained = train_single_gpu(module=module, optimizer=optimizer, manager=manager, cfg=cfg, rows=rows,
            lengths=[1]*len(rows), prepare=prepare, forward=forward, evaluate=lambda: {"accuracy": .5},
            state_cpu=lambda: copy.deepcopy(module.state_dict()), restore=module.load_state_dict,
            checkpoint_prefix="fixture", latest=manager.load_checkpoint("fixture_latest.pt"), metric_key="accuracy")
        return module.weight.detach().clone(), trained, manager.paths.run_dir
    except GracefulStop:
        return None, None, manager.paths.run_dir
    finally:
        manager.close()


def test_accumulation_equals_full_batch_including_short_epoch_end(tmp_path, monkeypatch):
    one, report_one, _ = _run_engine(tmp_path / "a", monkeypatch, 1, 4)
    four, report_four, _ = _run_engine(tmp_path / "b", monkeypatch, 4, 4)
    torch.testing.assert_close(one, four, rtol=1e-5, atol=1e-6)
    assert report_one["optimizer_steps"] == report_four["optimizer_steps"] == 4
    assert [row["examples"] for row in report_one["step_rows"]] == [4, 1, 4, 1]


def test_optimizer_boundary_resume_reproduces_uninterrupted_run(tmp_path, monkeypatch):
    uninterrupted, _, _ = _run_engine(tmp_path / "full", monkeypatch, 1, 4)
    _, _, run_dir = _run_engine(tmp_path / "resumed", monkeypatch, 1, 4, stop=True)
    resumed, report, _ = _run_engine(tmp_path / "resumed", monkeypatch, 1, 4, resume=run_dir)
    torch.testing.assert_close(uninterrupted, resumed)
    assert report["optimizer_steps"] == 4
    assert len(report["step_rows"]) == 4


def test_generation_accounting_includes_eos_without_padding():
    from src.evaluation.training_metrics import inference_runtime, token_count_to_eos
    assert token_count_to_eos(torch.tensor([4, 9, 0, 0]), 9) == 2
    runtime = inference_runtime([{"output_tokens": 2}, {"output_tokens": 3}],
                                [{"generation_seconds": 4., "example_count": 2}])
    assert runtime["generation_seconds"] == 4.
    assert runtime["examples_per_second"] == .5
    assert runtime["amortized_seconds_per_example"] == 2.


def test_paper_artifacts_export_real_figures_and_raw_metrics(tmp_path):
    from src.training.reporting import export_training_artifacts
    rows = [{"optimizer_step": i, "epoch": 1, "loss": 1 / i, "learning_rate": .001,
             "gradient_norm": .5, "examples_per_second": 10., "tokens_per_second": 1000.,
             "allocated_vram_gib": 4., "reserved_vram_gib": 5., "mean_gate": .2,
             "layer_metrics": {"1": {"examples": 2, "target_tokens": 3,
                                       "loss_numerator": 3 / i, "gate_sum": .4}}}
            for i in (1, 2)]
    metrics = {"example_count": 2, "exact_match_accuracy": .5,
               "by_operation": {"COUNT_CHAR": {"count": 2, "exact_match_accuracy": .5}},
               "per_example": [{"example_id": "a", "correct": True}, {"example_id": "b", "correct": False}],
               "runtime": {"generation_seconds": 2., "examples_per_second": 1.}}
    report = {"experiment": "fixture", "best_epoch": 1,
              "selected_test_oracle_phase1": metrics,
              "test_layerwise_oracle_phase1": [{"layer": 1, "metrics": {**metrics, "mean_gate": .2}}]}
    export_training_artifacts(tmp_path, step_rows=rows,
        epoch_rows=[{"epoch": 1, "mean_loss": .5, "dev_loss": .7,
                     "dev_best_accuracy": .5, "train_seconds": 10., "validation_seconds": 2.}],
        manifest={"seed": 42}, report=report)
    for name in ("training_loss", "epoch_training_loss", "validation_accuracy", "layerwise_accuracy", "operation_accuracy"):
        for extension in ("pdf", "svg", "png"):
            assert (tmp_path / f"{name}.{extension}").stat().st_size > 100
    assert (tmp_path / "training_loss.pdf").read_bytes().startswith(b"%PDF")
    assert json.loads((tmp_path / "training_report.json").read_text())["best_epoch"] == 1
    assert "50.00% (n=2)" in (tmp_path / "summary.txt").read_text()
    assert (tmp_path / "training_layers.csv").is_file()


def test_final_launcher_binds_bundle_weights_and_preserves_evaluation_seed(tmp_path, monkeypatch):
    import scripts.run_final_training as runner
    import tomllib
    repo = runner.ROOT
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        script = Path(command[1]).name
        if script in ("train_tool_policy.py", "train_result_injector.py"):
            stage = "1" if script == "train_tool_policy.py" else "2"
            cfg = tomllib.loads(Path(command[command.index("--config") + 1]).read_text())
            run_dir = tmp_path / ("recovery" + stage)
            artifacts = run_dir / "artifacts"
            artifacts.mkdir(parents=True)
            weights = artifacts / ("final_adapter" if stage == "1" else "result_injector.pt")
            if stage == "1":
                weights.mkdir()
                (weights / "adapter_config.json").write_text("{}")
            else:
                weights.write_bytes(b"fixture")
            report = {"status": "complete", "recovery_run_dir": str(run_dir),
                      "adapter_dir": str(weights), "final_checkpoint": str(weights)}
            (artifacts / "training_report.json").write_text(json.dumps(report))
            result_dir = tmp_path / cfg["output"]["results_dir"]
            result_dir.mkdir(parents=True)
            report_name = "tool_policy_report.json" if stage == "1" else "result_injection_report.json"
            (result_dir / report_name).write_text(json.dumps(report))
    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(runner.sys, "argv", ["runner", "--stage", "both", "--seed", "123",
        "--phase1-config", str(repo / "configs/experiments/qwen3_8b/tool_policy_final.toml"),
        "--phase2-config", str(repo / "configs/experiments/qwen3_8b/result_injection_final.toml")])
    runner.main()
    bundle = next((tmp_path / "results/final_training").glob("*/manifest.json")).parent
    for stage in ("1", "2"):
        cfg = tomllib.loads((bundle / f"phase{stage}.toml").read_text())
        assert cfg["training"]["seed"] == 123
        assert cfg["evaluation"]["seed"] == 20260919
        key = "adapter_dir" if stage == "1" else "final_checkpoint"
        assert Path(cfg["output"][key]).exists()
        assert str(bundle) in cfg["output"][key]
    assert json.loads((bundle / "manifest.json").read_text())["status"] == "complete"
    assert bundle.with_suffix(".zip").is_file()
    assert any("evaluate_pipeline.py" in command[1] for command in commands)

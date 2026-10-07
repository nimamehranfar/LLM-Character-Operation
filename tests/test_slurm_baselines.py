from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import torch

from src.evaluation.baseline_runs import cleanup_owned_cache, owned_cache, validate_report
from src.model.device import select_device


def test_single_slurm_gpu_uses_visible_cuda_and_preserves_assignment(monkeypatch, capsys):
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-assigned-by-slurm")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "H100" if index == 0 else pytest.fail("Wrong GPU"))
    device = select_device(announce=True)
    assert device == torch.device("cuda")
    output = capsys.readouterr().out
    assert "cuda:0 | GPU: H100" in output
    assert "SLURM_JOB_ID=42" in output and "GPU-assigned-by-slurm" in output


def test_cpu_fallback_loader_maps_model_to_cpu(monkeypatch, tmp_path):
    import src.model.local_model as local
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(local, "ensure_local_model_path", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(local.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    captured = {}
    model = SimpleNamespace(eval=lambda: None, parameters=lambda: [])
    def load(*args, **kwargs):
        captured.update(kwargs)
        return model
    monkeypatch.setattr(local.AutoModelForCausalLM, "from_pretrained", load)
    local.load_local_causal_lm("test/model", quantization="none")
    assert captured["device_map"] == {"": torch.device("cpu")}
    assert captured["dtype"] == torch.float32
    with pytest.raises(RuntimeError, match="requires CUDA"):
        local.load_local_causal_lm("test/model", quantization="4bit")


def test_cleanup_rejects_shared_cache_and_wrong_owner(tmp_path):
    root = tmp_path / "cache"
    shared = root / "shared"
    shared.mkdir(parents=True)
    (shared / "weights.bin").write_bytes(b"keep")
    with pytest.raises(ValueError):
        cleanup_owned_cache(shared, root, "run", "model")
    directory = owned_cache(root, "run", "model")
    (directory / "weights.bin").write_bytes(b"delete")
    with pytest.raises(ValueError):
        cleanup_owned_cache(directory, root, "other-run", "model")
    cleanup_owned_cache(directory, root, "run", "model")
    assert not directory.exists()
    assert (shared / "weights.bin").read_bytes() == b"keep"
    with pytest.raises(ValueError):
        owned_cache(root, "../escape", "model")


def test_report_validation_checks_identity_coverage_and_accuracy(tmp_path):
    path = tmp_path / "report.json"
    report = {"mode": "zero_shot", "model_repo_id": "test/model", "split": "test",
              "example_count": 2, "exact_match_accuracy": 0.5,
              "per_example": [{"example_id": "a", "exact": True}, {"example_id": "b", "exact": False}]}
    path.write_text(json.dumps(report))
    validate_report(path, "test/model", "test", ["a", "b"])
    with pytest.raises(ValueError):
        validate_report(path, "test/model", "test", ["a", "c"])
    with pytest.raises(ValueError):
        validate_report(path, "test/model", "heldout", ["a", "b"])
    report["exact_match_accuracy"] = 1.0
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="aggregate"):
        validate_report(path, "test/model", "test", ["a", "b"])


@pytest.mark.parametrize("failure", [None, "child", "incomplete", "download"])
def test_runner_exports_distinct_results_and_cleans_only_successful_models(monkeypatch, tmp_path, failure):
    import scripts.run_zero_shot_matrix as runner
    from src.data.character_dataset import load_jsonl
    downloads = []
    monkeypatch.setattr(runner, "select_device", lambda **kwargs: torch.device("cuda"))
    def download(repo, explicit, **kwargs):
        from huggingface_hub import constants
        assert constants.HF_HUB_DISABLE_XET is True
        downloads.append(kwargs)
        if failure == "download":
            raise RuntimeError("gated model access denied")
        assert explicit is None and kwargs["use_registry"] is False
        directory = Path(kwargs["cache_dir"]) / "snapshots" / "abc123"
        directory.mkdir(parents=True)
        (directory / "model.safetensors").write_bytes(b"fake model")
        return directory.resolve()
    monkeypatch.setattr(runner, "ensure_local_model_path", download)
    def evaluate(command, **kwargs):
        assert "--mode" in command and command[command.index("--mode") + 1] == "zero_shot"
        assert "--local-model-path" in command and kwargs["env"]["LLM_CHARACTER_SKIP_MODEL_REGISTRY_WRITE"] == "1"
        split = command[command.index("--split") + 1]
        if failure == "child" and split == "heldout":
            return SimpleNamespace(returncode=1, stdout="", stderr="RuntimeError: simulated OOM")
        cfg = runner.load(Path(command[command.index("--config") + 1]))
        path = runner.resolve(cfg["data"]["test_file"] if split == "test" else cfg["evaluation"]["challenge_file"])
        rows = [row for row in load_jsonl(path) if not row.is_control]
        rows = runner.stratified_sample(rows, -1, 1, cfg["training"]["seed"] + 91)
        if failure == "incomplete" and split == "heldout":
            rows = rows[:-1]
        report = {"mode": "zero_shot", "model_repo_id": cfg["model"]["repo_id"], "split": split,
                  "example_count": len(rows), "exact_match_accuracy": 1.0,
                  "runtime": {"average_seconds_per_prompt": 0.1},
                  "per_example": [{"example_id": row.example_id, "exact": True} for row in rows]}
        Path(command[command.index("--report-path") + 1]).write_text(json.dumps(report))
        return SimpleNamespace(returncode=0, stdout="evaluated", stderr="")
    monkeypatch.setattr(runner.subprocess, "run", evaluate)
    result_root, cache_root = tmp_path / "results", tmp_path / "cache"
    shared = cache_root / "existing-user-model"
    shared.mkdir(parents=True)
    (shared / "weights").write_bytes(b"preserve")
    arguments = ["run_zero_shot_matrix.py", "--models", "qwen3_8b", "--split", "both",
                 "--examples-per-operation", "1", "--cleanup-models", "--model-cache-dir", str(cache_root),
                 "--results-dir", str(result_root)]
    monkeypatch.setattr(runner.sys, "argv", arguments)
    if failure:
        with pytest.raises(SystemExit) as error:
            runner.main()
        assert error.value.code == 1
    else:
        runner.main()
    run_dir = next(path for path in result_root.iterdir() if path.is_dir())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cache_dir = cache_root / manifest["run_id"] / "qwen3_8b"
    assert cache_dir.exists() == bool(failure)
    assert manifest["status"] == ("partial_or_failed" if failure else "success")
    assert (shared / "weights").read_bytes() == b"preserve"
    assert (run_dir / "summary.csv").is_file()
    archive_path = run_dir.with_suffix(".zip")
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert any(name.endswith("summary.csv") for name in archive.namelist())
        assert not any(name.endswith(".safetensors") for name in archive.namelist())
    if not failure:
        assert manifest["models"]["qwen3_8b"]["splits"]["heldout"]["example_count"] == 12
        monkeypatch.setattr(runner.sys, "argv", [*arguments, "--resume-run", str(run_dir)])
        runner.main()
        assert len(downloads) == 1  # Successful cleaned runs do not redownload.
        monkeypatch.setattr(runner.sys, "argv", arguments)
        runner.main()
        assert len(list(result_root.glob("*/manifest.json"))) == 2


def test_download_preparation_can_resume_offline_and_cleanup(monkeypatch, tmp_path):
    import scripts.run_zero_shot_matrix as runner
    downloads = []
    monkeypatch.setattr(runner, "select_device", lambda **kwargs: torch.device("cpu"))
    def download(repo, explicit, **kwargs):
        downloads.append(kwargs)
        directory = Path(kwargs["cache_dir"]) / "snapshots" / "abc123"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model.safetensors").write_bytes(b"fixture")
        return directory.resolve()
    monkeypatch.setattr(runner, "ensure_local_model_path", download)
    arguments = ["runner", "--models", "qwen3_8b", "--split", "heldout", "--examples-per-operation", "1",
                 "--cleanup-models", "--model-cache-dir", str(tmp_path / "cache"),
                 "--results-dir", str(tmp_path / "results")]
    monkeypatch.setattr(runner.sys, "argv", [*arguments, "--download-only"])
    runner.main()
    run_dir = next(path for path in (tmp_path / "results").iterdir() if path.is_dir())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "downloaded"
    directory = tmp_path / "cache" / manifest["run_id"] / "qwen3_8b"
    assert directory.is_dir()
    monkeypatch.setattr(runner, "select_device", lambda **kwargs: torch.device("cuda"))
    from src.data.character_dataset import load_jsonl
    cfg = runner.load(runner.CONFIG_DIR / "qwen3_8b.toml")
    rows = [row for row in load_jsonl(runner.resolve(cfg["evaluation"]["challenge_file"])) if not row.is_control]
    rows = runner.stratified_sample(rows, -1, 1, cfg["training"]["seed"] + 91)
    def evaluate(command, **kwargs):
        report = {"mode": "zero_shot", "model_repo_id": cfg["model"]["repo_id"], "split": "heldout",
                  "example_count": len(rows), "exact_match_accuracy": 1.0,
                  "runtime": {}, "per_example": [{"example_id": row.example_id, "exact": True} for row in rows]}
        Path(command[command.index("--report-path") + 1]).write_text(json.dumps(report))
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(runner.subprocess, "run", evaluate)
    monkeypatch.setattr(runner.sys, "argv", [*arguments, "--no-auto-download", "--resume-run", str(run_dir)])
    runner.main()
    assert downloads[-1]["auto_download"] is False
    assert not directory.exists()
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "success"


def test_login_node_setup_defers_only_gpu_checks(monkeypatch, tmp_path):
    import scripts.setup_environment as setup
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="fixture==1\n")
    monkeypatch.setattr(setup.subprocess, "run", run)
    monkeypatch.setattr(setup.sys, "argv", ["setup", "--venv", str(tmp_path), "--cuda", "cu128", "--skip-gpu-check"])
    setup.main()
    assert "check_environment.py" in commands[-1][1]
    assert "--require-cuda" not in commands[-1] and "--gpu-smoke-test" not in commands[-1]
    assert any("cu128" in str(part) for command in commands for part in command)


def test_real_cpu_evaluator_exports_and_deletes_owned_fixture(monkeypatch, tmp_path):
    """Exercise actual tokenizer/model generation in a child with CUDA hidden.

    A tiny random model is a plumbing fixture, never a research baseline.
    """
    import shutil
    import scripts.run_zero_shot_matrix as runner
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    fixture = tmp_path / "fixture"
    tokenizer = Tokenizer(WordLevel({"[EOS]": 0, "[UNK]": 1, "x": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="[EOS]", pad_token="[EOS]")
    fast.save_pretrained(fixture)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=3, n_embd=8, n_layer=1, n_head=1, n_positions=512,
                                     bos_token_id=0, eos_token_id=0, pad_token_id=0))
    model.save_pretrained(fixture)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = config_dir / "tiny.toml"
    config.write_text('[model]\nrepo_id="fixture/tiny"\nquantization="none"\n'
                      '[data]\ntest_file="data/character_operations_50k/test.jsonl"\n'
                      'heldout_file="data/character_operations_50k/heldout.jsonl"\n'
                      '[evaluation]\nmax_new_tokens=2\n', encoding="utf-8")
    monkeypatch.setattr(runner, "CONFIG_DIR", config_dir)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setattr(runner, "select_device", lambda **kwargs: torch.device("cpu"))
    def download(repo, explicit, **kwargs):
        path = Path(kwargs["cache_dir"]) / "snapshots" / "fixture"
        shutil.copytree(fixture, path)
        return path.resolve()
    monkeypatch.setattr(runner, "ensure_local_model_path", download)
    monkeypatch.setattr(runner.sys, "argv", ["runner", "--models", "tiny", "--split", "test", "--examples", "1",
                         "--cleanup-models", "--model-cache-dir", str(tmp_path / "cache"),
                         "--results-dir", str(tmp_path / "results")])
    runner.main()
    run_dir = next(path for path in (tmp_path / "results").iterdir() if path.is_dir())
    report = json.loads((run_dir / "models/tiny/test.json").read_text())
    assert report["example_count"] == 1
    assert report["inference_setup"]["device"] == "cpu"
    assert report["inference_setup"]["dtype"] == "float32"
    assert report["runtime"]["peak_allocated_vram_mib"] == 0.0
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "success" and manifest["models"]["tiny"]["cache_deleted"] is True
    assert not (tmp_path / "cache" / manifest["run_id"] / "tiny").exists()

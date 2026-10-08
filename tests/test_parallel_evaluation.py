from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.evaluate_multi_gpu import run_workers, worker_commands
from scripts.evaluate_layerwise import scalar_oracle_report_to_compat
from src.evaluation.parallel import collect_shards, merge_reports, shard_examples, visible_gpu_tokens
from src.evaluation.result_analysis import summarize_best_any_layer


def test_partition_preserves_global_selection_without_duplicates():
    rows = [SimpleNamespace(example_id=str(i)) for i in range(11)]
    shards = [shard_examples(rows, i, 4)[0] for i in range(4)]
    assert sorted(int(row.example_id) for part in shards for row in part) == list(range(11))
    assert [len(part) for part in shards] == [3, 3, 3, 2]
    with pytest.raises(ValueError):
        shard_examples(rows, 4, 4)
    with pytest.raises(ValueError):
        shard_examples([rows[0], rows[0]], 0, 1)


def test_gpu_selection_respects_scheduler_visibility_and_mig():
    assert visible_gpu_tokens("auto", 2, {"CUDA_VISIBLE_DEVICES": "GPU-abc,MIG-def"}) == ["GPU-abc", "MIG-def"]
    assert visible_gpu_tokens("1,0", 2, {"CUDA_VISIBLE_DEVICES": "3,7"}) == ["7", "3"]
    for selection in ("2", "0,0", "-1"):
        with pytest.raises(ValueError):
            visible_gpu_tokens(selection, 2, {})
    with pytest.raises(ValueError):
        visible_gpu_tokens("auto", 0, {})


FAKE_WORKER = '''
import argparse, json, os, pathlib, sys, time
p=argparse.ArgumentParser()
p.add_argument('--shard-index', type=int); p.add_argument('--shard-count', type=int)
p.add_argument('--report-path'); p.add_argument('--results-dir'); p.add_argument('--split')
a=p.parse_args()
token=os.environ['CUDA_VISIBLE_DEVICES']
if token=='fail': sys.exit(3)
if token=='wait': time.sleep(60)
ids=[str(i) for i in range(7)]
rows=[dict(example_id=key, operation='COUNT_CHAR', category='task', source_style='audit',
 generation_style='audit', length_regime='short', result_kind='integer', exact=int(key)%2==0,
 input_tokens=2, output_tokens=1, latency_seconds=1) for key in ids[a.shard_index::a.shard_count]]
report=dict(mode='zero_shot', model_repo_id='audit/model', split=a.split,
 shard=dict(index=a.shard_index,count=a.shard_count,selected_ids=ids),per_example=rows,
 runtime=dict(canonical_model_setup_seconds=0,peak_allocated_vram_mib=1),
 inference_setup=dict(gpu_token=token))
pathlib.Path(a.report_path).write_text(json.dumps(report))
'''


def test_actual_subprocess_workers_are_isolated_and_merge_weighted_scores(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text(FAKE_WORKER, encoding="utf-8")
    original = os.environ.get("CUDA_VISIBLE_DEVICES")
    jobs = worker_commands(script, [], ["GPU-one", "GPU-two"], tmp_path, "test")
    run_workers(jobs)
    reports = [json.loads(path.read_text()) for _, _, path in jobs]
    assert [r["inference_setup"]["gpu_token"] for r in reports] == ["GPU-one", "GPU-two"]
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == original
    result = merge_reports("direct", reports)
    assert result["example_count"] == 7
    assert result["exact_match_accuracy"] == 4/7  # An average of worker accuracies would incorrectly give 1/2.
    assert result["runtime"]["inference_seconds"] == 7
    assert result["runtime"]["total_output_tokens"] == 7
    assert [row["example_id"] for row in result["per_example"]] == [str(i) for i in range(7)]
    reports[1]["per_example"].pop()
    with pytest.raises(ValueError, match="Missing"):
        merge_reports("direct", reports)


def test_worker_failure_stops_other_workers(tmp_path):
    import time
    script = tmp_path / "worker.py"
    script.write_text(FAKE_WORKER, encoding="utf-8")
    jobs = worker_commands(script, [], ["wait", "fail"], tmp_path, "test")
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="code 3"):
        run_workers(jobs)
    assert time.monotonic() - started < 15


def test_merge_rejects_duplicate_or_inconsistent_manifests():
    report = {"shard": {"index": 0, "count": 2, "selected_ids": ["a", "b"]}, "rows": [{"example_id": "a"}]}
    with pytest.raises(ValueError, match="manifests"):
        collect_shards([report, report], lambda r: r["rows"])


def test_layerwise_report_can_be_consumed_by_oracle_summary():
    common = dict(example_id="a", operation="COUNT_CHAR", category="task", result_kind="integer", phase1_exact=True, gate=0.7)
    reports = [scalar_oracle_report_to_compat(layer, {"per_example": [{**common, "final_exact": ok}]})
               for layer, ok in ((1, False), (3, True))]
    result = summarize_best_any_layer(reports)
    assert result["exact_match_accuracy"] == 1
    assert result["per_example"][0]["first_successful_layer"] == 3
    assert reports[1]["metrics"]["per_example"][0]["gate"] == 0.7


def test_vram_override_uses_whole_gpu_and_keeps_laptop_default(monkeypatch):
    from scripts.train_tool_policy import configure_vram_limit
    import torch
    fractions = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(name="audit", total_memory=80*1024**3))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda fraction, device: fractions.append(fraction))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    configure_vram_limit(7600)
    configure_vram_limit(0)
    assert fractions == [7600/(80*1024), 1.0]
    with pytest.raises(ValueError):
        configure_vram_limit(-1)


@pytest.mark.parametrize("kind", ["tool-feedback", "pipeline"])
def test_task_and_control_scores_survive_multi_worker_merge(kind):
    common = dict(phase1_exact=True, no_call_correct=False, tool_attempted=True, invalid_response=False,
                  executor_exact=True, matched_pipeline_exact=True, final_exact=True, system_success=True,
                  category="audit", source_style="audit", generation_style="audit", length_regime="short",
                  result_kind="integer", phase2_gate=None, latency_seconds=1, input_tokens=2, output_tokens=1)
    rows = [{**common, "example_id": "a", "source_example_id": "a", "operation": "COUNT_CHAR", "is_control": False},
            {**common, "example_id": "b", "source_example_id": "b", "operation": "NONE", "is_control": True,
             "final_exact": False, "no_call_correct": True}]
    reports = []
    for index in range(2):
        report = dict(shard={"index": index, "count": 2, "selected_ids": ["a", "b"]},
                      runtime={"canonical_model_setup_seconds": 1}, mode="audit", model_repo_id="audit/model", split="test")
        if kind == "pipeline":
            report["test"] = {"per_example": rows[index::2]}
        else:
            report["protocol"] = {"model_repo_id": "audit/model", "selected_ids": [rows[index]["example_id"]]}
            report["per_example"] = rows[index::2]
        reports.append(report)
    merged = merge_reports(kind, reports)
    block = merged["test"] if kind == "pipeline" else merged
    assert block["task_count"] == 1
    assert block["control_count"] == 1
    assert block["supported_task_system_exact"] == 1
    assert block["control_no_call_accuracy"] == 1
    assert len(block["per_example"]) == 2


def test_snapshot_validation_rejects_missing_shards_and_wrong_revision(tmp_path, monkeypatch):
    from src.model import local_model
    (tmp_path / "config.json").write_text("{}")
    assert not local_model._valid_snapshot(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "part-1.safetensors", "b": "part-2.safetensors"}}))
    (tmp_path / "part-1.safetensors").write_bytes(b"weight")
    assert not local_model._valid_snapshot(tmp_path)
    (tmp_path / "part-2.safetensors").write_bytes(b"weight")
    assert local_model._valid_snapshot(tmp_path)
    monkeypatch.setattr(local_model, "_load_registry", lambda: {"audit/model": {"snapshot_path": str(tmp_path), "requested_revision": "old", "resolved_revision": "old"}})
    def missing(**kwargs):
        raise FileNotFoundError("Requested revision absent")
    monkeypatch.setattr(local_model, "snapshot_download", missing)
    with pytest.raises(FileNotFoundError):
        local_model.ensure_local_model_path("audit/model", revision="new")


def test_portable_package_excludes_local_artifacts(tmp_path):
    from scripts.package_project import project_files
    names = ["README.md", "src/main.py", "src/__pycache__/main.pyc", ".env", "checkpoints/model.pt",
             ".venv/token.py", "results/report.json", "data/test.jsonl", "old.rar"]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("audit")
    assert {path.relative_to(tmp_path).as_posix() for path in project_files(tmp_path)} == {"README.md", "src/main.py", "data/test.jsonl"}


@pytest.mark.parametrize("capabilities,expected", [("8.9\n", "cu126"), ("8.0\n9.0\n", "cu126"), ("10.0\n", "cu128"), ("12.0\n", "cu128")])
def test_installer_selects_cuda_for_gpu_generation(capabilities, expected, monkeypatch):
    from scripts import setup_environment
    monkeypatch.setattr(setup_environment.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=capabilities))
    assert setup_environment.select_cuda() == expected

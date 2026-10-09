import json
import subprocess
import sys
from pathlib import Path

from scripts.summarize_baseline_run import combined, summarize


def test_parallel_summary_uses_wall_time_and_highest_gpu_peak():
    report = {
        "example_count": 2,
        "per_example": [{"exact": True, "generation_seconds": 3},
                        {"exact": False, "generation_seconds": 4}],
        "runtime": {"active_operational_seconds": 99},
        "parallel_runtime": {
            "wall_seconds_current_session": 5,
            "workers": [{"runtime": {"peak_allocated_vram_mib": 1024, "peak_reserved_vram_mib": 2048}},
                        {"runtime": {"peak_allocated_vram_mib": 3072, "peak_reserved_vram_mib": 4096}}],
        },
    }
    result = summarize(report)
    assert result["run_seconds"] == 5
    assert result["generation_seconds"] == 7
    assert result["allocated_gib"] == 3
    assert result["reserved_gib"] == 4
    assert result["correct"] == 1
    other = {**result, "examples": 1, "correct": 1, "run_seconds": 2, "allocated_gib": 1}
    total = combined([result, other])
    assert total["run_seconds"] == 7
    assert total["examples"] == 3 and total["correct"] == 2
    assert total["allocated_gib"] == 3


def test_shared_cli_discovers_all_models_and_does_not_invent_missing_timings(tmp_path):
    for model in ("qwen3_8b", "qwen2_5_7b"):
        directory = tmp_path / "models" / model
        directory.mkdir(parents=True)
        report = {"split": "test", "model_repo_id": model, "example_count": 1,
                  "per_example": [{"exact": True}], "runtime": {}}
        (directory / "test.json").write_text(json.dumps(report))
    script = Path(__file__).resolve().parents[1] / "scripts/summarize_baseline_run.py"
    proc = subprocess.run([sys.executable, str(script), str(tmp_path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "MODEL: qwen3_8b" in proc.stdout and "MODEL: qwen2_5_7b" in proc.stdout
    assert proc.stdout.count("Accuracy:       100.00% (1/1)") == 4
    assert "Run time:       not recorded" in proc.stdout
    assert "Peak allocated: not recorded" in proc.stdout

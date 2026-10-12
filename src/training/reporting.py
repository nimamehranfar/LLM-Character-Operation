"""Paper-oriented training telemetry and Matplotlib figures in PDF/SVG/PNG."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
from pathlib import Path
import socket
import subprocess
from importlib.metadata import version, PackageNotFoundError
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_manifest(*, config_path: Path, data_paths: list[Path], training: dict[str, Any],
                 trainable_parameters: int, total_parameters: int) -> dict[str, Any]:
    try:
        git_base = ["git", "-c", "safe.directory=" + str(REPO_ROOT), "-C", str(REPO_ROOT)]
        git_commit = subprocess.run([*git_base, "rev-parse", "HEAD"], capture_output=True, text=True,
                                    check=True).stdout.strip()
        git_dirty = bool(subprocess.run([*git_base, "status", "--porcelain"], capture_output=True,
                                        text=True, check=True).stdout.strip())
    except Exception:
        git_commit, git_dirty = None, None
    packages = {}
    for name in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "huggingface_hub", "matplotlib", "numpy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return {
        "host": socket.gethostname(), "platform": platform.platform(),
        "python": platform.python_version(), "torch": torch.__version__,
        "packages": packages,
        "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu_name": properties.name, "gpu_total_vram_mib": properties.total_memory / 1024**2,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "git_commit": git_commit, "git_dirty": git_dirty,
        "source_sha256": {str(path.relative_to(REPO_ROOT)): sha256(path)
                          for root in (REPO_ROOT / "src", REPO_ROOT / "scripts")
                          for path in sorted(root.rglob("*.py"))},
        "config_path": str(config_path), "config_sha256": sha256(config_path),
        "data_sha256": {str(path): sha256(path) for path in data_paths},
        "trainable_parameters": int(trainable_parameters),
        "total_parameters": int(total_parameters),
        "trainable_parameter_percentage": 100 * trainable_parameters / max(1, total_parameters),
        "training": training,
    }


class MetricJournal:
    def __init__(self, path: Path, enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled
        self.rows: list[dict[str, Any]] = []
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(dict(row))
        if self.enabled:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                          for key, value in row.items()} for row in rows)


def _plot(path: Path, title: str, x_label: str, y_label: str,
          series: list[tuple[str, list[tuple[float, float]]]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    series = [(name, points) for name, points in series if points]
    if not series:
        return
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
        for name, points in series:
            axis.plot([x for x, _ in points], [y for _, y in points], label=name, linewidth=1.3,
                      marker="o" if len(points) <= 20 else None, markersize=3)
        axis.set(title=title, xlabel=x_label, ylabel=y_label)
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
        for extension in (".pdf", ".svg", ".png"):
            fig.savefig(path.with_suffix(extension), dpi=300)
        plt.close(fig)


def _evaluation_reports(report):
    """Only actual evaluated splits; best-any-layer is an oracle diagnostic."""
    for key, value in report.items():
        if key in ("test", "heldout", "best_dev") and isinstance(value, dict):
            yield key, value
        elif key.startswith(("selected_", "baseline_no_injection_")) and isinstance(value, dict):
            yield key, value


def export_training_artifacts(directory: Path, *, step_rows: list[dict[str, Any]],
                              epoch_rows: list[dict[str, Any]], manifest: dict[str, Any],
                              report: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "environment_and_protocol.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (directory / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(directory / "optimizer_steps.csv", step_rows)
    training_layers = []
    for row in step_rows:
        for layer, metrics in row.get("layer_metrics", {}).items():
            training_layers.append({"optimizer_step": row["optimizer_step"], "epoch": row["epoch"],
                                    "layer": int(layer), "examples": metrics["examples"],
                                    "loss": metrics["loss_numerator"] / max(1, metrics["target_tokens"]),
                                    "gate": metrics["gate_sum"] / max(1, metrics["examples"])})
    _write_csv(directory / "training_layers.csv", training_layers)
    _plot(directory / "training_loss_by_layer", "Mapper training loss by layer", "Optimizer update", "Cross entropy",
          [(f"layer {layer}", [(float(row["optimizer_step"]), row["loss"]) for row in training_layers if row["layer"] == layer])
           for layer in sorted({row["layer"] for row in training_layers})])
    _write_csv(directory / "epochs.csv", [
        {key: value for key, value in row.items() if isinstance(value, (int, float, bool, str))}
        for row in epoch_rows])
    x = lambda row: float(row.get("optimizer_step", 0))
    points = lambda field: [(x(r), float(r[field])) for r in step_rows if r.get(field) is not None]
    _plot(directory / "training_loss", "Training loss", "Optimizer update", "Target-token cross entropy",
          [("training", points("loss"))])
    _plot(directory / "learning_rate", "Learning rate", "Optimizer update", "Learning rate",
          [("learning rate", points("learning_rate"))])
    _plot(directory / "gradient_norm", "Gradient norm before clipping", "Optimizer update", "L2 norm",
          [("gradient norm", points("gradient_norm"))])
    _plot(directory / "training_throughput", "Training throughput", "Optimizer update", "Examples / second",
          [("examples/s", points("examples_per_second"))])
    _plot(directory / "token_throughput", "Training token throughput", "Optimizer update", "Input tokens / second",
          [("tokens/s", points("tokens_per_second"))])
    _plot(directory / "training_memory", "Peak PyTorch GPU memory", "Optimizer update", "GiB",
          [("allocated", points("allocated_vram_gib")), ("reserved", points("reserved_vram_gib"))])
    _plot(directory / "training_gate", "Mapper gate during training", "Optimizer update", "Mean gate",
          [("gate", points("mean_gate"))])
    _plot(directory / "epoch_training_loss", "Epoch training loss", "Epoch", "Target-token cross entropy",
          [("training", [(float(r["epoch"]), float(r["mean_loss"])) for r in epoch_rows]),
           ("development", [(float(r["epoch"]), float(r["dev_loss"])) for r in epoch_rows if "dev_loss" in r])])
    metric_names = sorted({key for row in epoch_rows for key in row
                           if key.startswith("dev_") and ("accuracy" in key or "exact" in key)})
    _plot(directory / "validation_accuracy", "Development accuracy", "Epoch", "Percent",
          [(name.removeprefix("dev_"), [(float(r["epoch"]), 100 * float(r[name])) for r in epoch_rows if name in r])
           for name in metric_names])
    _plot(directory / "epoch_time", "Epoch runtime", "Epoch", "Seconds",
          [(field.removesuffix("_seconds"), [(float(r["epoch"]), float(r[field])) for r in epoch_rows if field in r])
           for field in ("train_seconds", "validation_seconds")])
    layer_rows = []
    for key, value in report.items():
        if key.endswith("layerwise_oracle_phase1") and isinstance(value, list):
            layer_rows.extend({"split": key, "layer": row["layer"],
                               "accuracy_percent": 100 * row["metrics"]["exact_match_accuracy"],
                               "mean_gate": row["metrics"].get("mean_gate"),
                               "count": row["metrics"]["example_count"]}
                              for row in value)
    _write_csv(directory / "layerwise.csv", layer_rows)
    _plot(directory / "layerwise_accuracy", "Oracle result injection by layer", "Decoder layer", "Exact match (%)",
          [(split.removesuffix("_layerwise_oracle_phase1"), [(float(r["layer"]), r["accuracy_percent"])
            for r in layer_rows if r["split"] == split]) for split in sorted({r["split"] for r in layer_rows})])
    _plot(directory / "layerwise_gate", "Oracle result injection gates", "Decoder layer", "Mean gate",
          [(split.removesuffix("_layerwise_oracle_phase1"), [(float(r["layer"]), float(r["mean_gate"]))
            for r in layer_rows if r["split"] == split and r["mean_gate"] is not None])
           for split in sorted({r["split"] for r in layer_rows})])
    evaluation_rows, operation_rows = [], []
    for split, metrics in _evaluation_reports(report):
        runtime = metrics.get("runtime", {})
        evaluation_rows.append({"split": split, "example_count": metrics["example_count"],
                                "accuracy_percent": 100 * metrics.get("exact_match_accuracy",
                                                                       metrics.get("full_call_exact_accuracy", 0)),
                                **{k: v for k, v in runtime.items() if isinstance(v, (int, float, str))}})
        for operation, values in metrics.get("by_operation", {}).items():
            operation_rows.append({"split": split, "operation": operation, "count": values["count"],
                                   "accuracy_percent": 100 * values.get("exact_match_accuracy", values.get("accuracy", 0))})
        rows = metrics.get("per_example") or []
        _write_csv(directory / (split + "_predictions.csv"), rows)
    _write_csv(directory / "evaluation_summary.csv", evaluation_rows)
    _write_csv(directory / "operation_accuracy.csv", operation_rows)
    # Categorical accuracy figures use bars rather than connecting unrelated operations.
    if operation_rows:
        import matplotlib.pyplot as plt
        splits = sorted({row["split"] for row in operation_rows})
        operations = sorted({row["operation"] for row in operation_rows})
        fig, axis = plt.subplots(figsize=(max(8, len(operations) * .7), 4.8), constrained_layout=True)
        width = .8 / len(splits)
        for index, split in enumerate(splits):
            lookup = {r["operation"]: r["accuracy_percent"] for r in operation_rows if r["split"] == split}
            axis.bar([i - .4 + width * (index + .5) for i in range(len(operations))],
                     [lookup.get(op, float("nan")) for op in operations], width=width, label=split)
        axis.set_xticks(range(len(operations)), operations, rotation=35, ha="right")
        axis.set(ylabel="Accuracy (%)", ylim=(0, 100), title="Accuracy by operation")
        axis.legend(fontsize=7)
        for extension in (".pdf", ".svg", ".png"):
            fig.savefig(directory / ("operation_accuracy" + extension), dpi=300)
        plt.close(fig)
    summary_lines = [
        f"Experiment: {report['experiment']}", f"Best epoch: {report.get('best_epoch')}",
        "Batch plan: " + json.dumps(report.get("batch_plan", {})),
        "Runtime: " + json.dumps(report.get("runtime", {})), "", "Evaluation:"]
    summary_lines.extend(f"  {row['split']}: {row['accuracy_percent']:.2f}% (n={row['example_count']})"
                         for row in evaluation_rows)
    (directory / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

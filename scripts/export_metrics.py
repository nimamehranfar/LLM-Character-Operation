from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def system_rows(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in results_root.rglob("*.json"):
        payload = load(path)
        if not payload:
            continue
        rel = str(path.relative_to(results_root))
        if "exact_match_accuracy" in payload and "split" in payload:
            rows.append({
                "source": rel,
                "system": payload.get("mode", path.stem),
                "model": payload.get("model_repo_id", payload.get("model")),
                "split": payload.get("split"),
                "accuracy": payload.get("exact_match_accuracy"),
                "example_count": payload.get("example_count"),
            })
        for key, label in (("test", "test"), ("heldout_template_challenge", "heldout")):
            block = payload.get(key)
            if not isinstance(block, dict):
                continue
            accuracy = block.get("overall_system_success")
            if accuracy is None:
                accuracy = block.get("execution_readiness", block.get("exact_match_accuracy"))
            if accuracy is not None:
                rows.append({
                    "source": rel,
                    "system": payload.get("experiment", path.stem),
                    "model": payload.get("model_repo_id", payload.get("model")),
                    "split": label,
                    "accuracy": accuracy,
                    "example_count": block.get("example_count"),
                })
    return rows


def operation_rows(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in results_root.rglob("*.json"):
        payload = load(path)
        if not payload:
            continue
        rel = str(path.relative_to(results_root))
        blocks: list[tuple[str, dict[str, Any]]] = []
        if isinstance(payload.get("by_operation"), dict):
            blocks.append((str(payload.get("split", "unknown")), payload))
        for key, label in (("test", "test"), ("heldout_template_challenge", "heldout")):
            block = payload.get(key)
            if isinstance(block, dict) and isinstance(block.get("by_operation"), dict):
                blocks.append((label, block))
        for split, block in blocks:
            for operation, metrics in sorted(block.get("by_operation", {}).items()):
                row = {
                    "source": rel,
                    "system": payload.get("mode", payload.get("experiment", path.stem)),
                    "split": split,
                    "operation": operation,
                }
                if isinstance(metrics, dict):
                    row.update(metrics)
                rows.append(row)
    return rows


def layer_rows(results_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    layer_rows_out: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    for path in results_root.rglob("layerwise_analysis.json"):
        payload = load(path)
        if not payload:
            continue
        rel = str(path.relative_to(results_root))
        selected = payload.get("phase2_selected_layer")
        for split_name, block in (payload.get("splits") or {}).items():
            split = "test" if split_name == "test" else "heldout"
            if not isinstance(block, dict):
                continue
            for mode_key, mode in (("actual_phase1_layerwise", "actual"), ("oracle_phase1_layerwise", "oracle")):
                for item in block.get(mode_key, []) or []:
                    metrics = item.get("metrics") or {}
                    layer_rows_out.append({
                        "source": rel,
                        "split": split,
                        "mode": mode,
                        "layer": item.get("layer"),
                        "selected_layer": selected,
                        "example_count": metrics.get("example_count"),
                        "phase1_execution_readiness": metrics.get("phase1_execution_readiness"),
                        "final_exact": metrics.get("final_exact"),
                        "mean_gate": metrics.get("mean_gate"),
                    })
            actual = block.get("selected_layer_actual") or {}
            oracle = block.get("selected_layer_oracle") or {}
            best = block.get("oracle_best_any_layer_scalar") or {}
            oracle_rows.append({
                "source": rel,
                "split": split,
                "selected_layer": selected,
                "selected_actual_exact": actual.get("final_exact"),
                "selected_oracle_exact": oracle.get("final_exact"),
                "oracle_best_any_layer_accuracy": best.get("oracle_best_any_layer_accuracy"),
                "oracle_example_count": best.get("example_count"),
            })
    return layer_rows_out, oracle_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export chart-ready CSV files from experiment result JSONs.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-dir", default="results/tables")
    args = parser.parse_args()
    results_root = (REPO_ROOT / args.results_root).resolve() if not Path(args.results_root).is_absolute() else Path(args.results_root)
    output = (REPO_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    layer, oracle = layer_rows(results_root)
    outputs = {
        "system_comparison.csv": system_rows(results_root),
        "character_operation_analysis.csv": operation_rows(results_root),
        "layer_analysis.csv": layer,
        "oracle_analysis.csv": oracle,
    }
    for name, rows in outputs.items():
        write_csv(output / name, rows)
        print(f"{name}: {len(rows)} rows")


if __name__ == "__main__":
    main()

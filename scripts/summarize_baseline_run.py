"""Print accuracy, recorded evaluator time and GPU memory for any baseline run.

Uses only Python's standard library; no GPU or model loading is required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def complete_sum(values):
    values = list(values)
    return sum(values) if values and all(value is not None for value in values) else None


def peak(values):
    values = list(values)
    return max(values) if values and all(value is not None for value in values) else None


def measurement(value, unit):
    return "not recorded" if value is None else f"{value:.2f} {unit}"


def summarize(report):
    rows = report.get("per_example")
    n = int(report["example_count"])
    if rows is None or len(rows) != n or any("exact" not in row for row in rows):
        raise ValueError("Missing or incomplete per-example accuracy records")
    runtime = report.get("runtime", {})
    parallel = report.get("parallel_runtime")
    if parallel:
        elapsed = parallel.get("wall_seconds_current_session")
        memories = [worker.get("runtime", {}) for worker in parallel.get("workers", [])]
        scope = "parallel wall time, including worker startup/loading"
        resumed = bool(parallel.get("resumed"))
    else:
        elapsed = runtime.get("active_operational_seconds")
        memories = [runtime]
        scope = "active evaluator time, including loading"
        resumed = bool(runtime.get("resume_count", 0))
    return {
        "examples": n,
        "correct": sum(bool(row["exact"]) for row in rows),
        "run_seconds": elapsed,
        # Batched rows contain apportioned times; summing counts each batch once.
        "generation_seconds": complete_sum(row.get("generation_seconds") for row in rows),
        "allocated_gib": divide(peak(m.get("peak_allocated_vram_mib") for m in memories), 1024),
        "reserved_gib": divide(peak(m.get("peak_reserved_vram_mib") for m in memories), 1024),
        "scope": scope,
        "resumed": resumed,
    }


def divide(value, denominator):
    return value / denominator if value is not None and denominator else None


def print_summary(label, values):
    n, correct = values["examples"], values["correct"]
    accuracy = f"{100 * correct / n:.2f}%" if n else "N/A"
    print(f"\n  {label}")
    print(f"    Accuracy:       {accuracy} ({correct}/{n})")
    elapsed = values["run_seconds"]
    print(f"    Run time:       {measurement(elapsed, 's')}"
          + (f" ({elapsed / 60:.2f} min)" if elapsed is not None else ""))
    print(f"    Generation:     {measurement(values['generation_seconds'], 's')} (summed worker time)")
    print(f"    Peak allocated: {measurement(values['allocated_gib'], 'GiB')}")
    print(f"    Peak reserved:  {measurement(values['reserved_gib'], 'GiB')}")


def combined(summaries):
    return {
        "examples": sum(s["examples"] for s in summaries),
        "correct": sum(s["correct"] for s in summaries),
        "run_seconds": complete_sum(s["run_seconds"] for s in summaries),
        "generation_seconds": complete_sum(s["generation_seconds"] for s in summaries),
        "allocated_gib": peak(s["allocated_gib"] for s in summaries),
        "reserved_gib": peak(s["reserved_gib"] for s in summaries),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Results run directory, model directory, or final report JSON")
    args = parser.parse_args()
    path = args.path.expanduser().resolve()
    if not path.exists():
        parser.error(f"Path does not exist: {path}")
    failures = []
    groups = {}
    if path.is_file():
        groups[path.parent.name] = [path]
    else:
        model_root = path / "models" if (path / "models").is_dir() else path
        direct = [model_root / f"{split}.json" for split in ("test", "heldout")]
        if any(p.is_file() for p in direct):
            groups[model_root.name] = [p for p in direct if p.is_file()]
        else:
            for directory in sorted(p for p in model_root.iterdir() if p.is_dir()):
                reports = [directory / f"{split}.json" for split in ("test", "heldout")]
                if any(p.is_file() for p in reports):
                    groups[directory.name] = [p for p in reports if p.is_file()]
    manifest_path = path / "manifest.json" if path.is_dir() else path.parent / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            print(f"Run status: {manifest.get('status', 'not recorded')}")
            for model, info in manifest.get("models", {}).items():
                for split in manifest.get("protocol", {}).get("splits", []):
                    entry = info.get("splits", {}).get(split, {})
                    if entry.get("status") != "success":
                        print(f"WARNING: {model}/{split}: {entry.get('status', 'missing')} "
                              f"{entry.get('error', '')}")
        except (ValueError, OSError) as exc:
            failures.append(f"Cannot read manifest: {exc}")
    if not groups:
        parser.error("No final reports found; expected models/<model>/test.json or heldout.json")
    print(f"Results: {path}")
    for model, files in groups.items():
        print(f"\nMODEL: {model}")
        summaries = []
        for file in files:
            try:
                report = json.loads(file.read_text(encoding="utf-8"))
                values = summarize(report)
                summaries.append(values)
                setup = report.get("inference_setup", {})
                print(f"  {report.get('model_repo_id', model)} | "
                      f"GPU: {setup.get('gpu_name', 'not recorded')} | "
                      f"batch size: {setup.get('batch_size', 'not recorded')}")
                print_summary(str(report.get("split", file.stem)).upper(), values)
                print(f"    Time scope:     {values['scope']}")
                if values["resumed"]:
                    print("    NOTE: Resumed run; parallel time covers only the current session,")
                    print("          whereas predictions/generation can include previous sessions.")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                failures.append(f"{file}: {exc}")
        if summaries:
            print_summary("COMBINED REPORTED SPLITS", combined(summaries))
    print("\nTimes exclude parent downloads, queue time and parent-runner overhead.")
    print("VRAM is the highest per-GPU PyTorch peak across workers/splits, not their sum.")
    print("Allocated and reserved overlap; CUDA/context memory outside PyTorch is excluded.")
    for error in failures:
        print(f"ERROR: {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

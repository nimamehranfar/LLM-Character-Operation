"""Disjoint evaluation shards and validated, example-weighted report merging."""
from __future__ import annotations

from collections import defaultdict
import os


def add_shard_arguments(parser):
    parser.add_argument("--local-model-path", help="Use this complete local base-model snapshot")
    parser.add_argument("--results-dir", help="Override the output directory (relative to the repository or absolute).")
    parser.add_argument("--report-path", help="Override the report filename; select a single split when using this option.")
    parser.add_argument("--shard-index", type=int, default=0, help="Worker index; normally set by evaluate_multi_gpu.py.")
    parser.add_argument("--shard-count", type=int, default=1, help="Partition the selected examples across this many workers.")


def shard_examples(rows, index, count):
    if count < 1 or not 0 <= index < count:
        raise ValueError("Require shard-count >= 1 and 0 <= shard-index < shard-count")
    rows = list(rows)
    ids = [row.example_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Selected dataset contains duplicate example IDs")
    return rows[index::count], {"index": index, "count": count, "selected_ids": ids}


def visible_gpu_tokens(selection, device_count, environ=None):
    """Select *visible* device indices, preserving scheduler GPU/MIG identifiers."""
    environ = os.environ if environ is None else environ
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    tokens = [item.strip() for item in visible.split(",")] if visible is not None else [str(i) for i in range(device_count)]
    if device_count < 1 or len(tokens) != device_count:
        raise ValueError("No usable GPUs, or CUDA_VISIBLE_DEVICES does not match PyTorch's visible devices")
    indices = list(range(device_count)) if selection == "auto" else [int(i.strip()) for i in selection.split(",")]
    if not indices or len(indices) != len(set(indices)) or any(i < 0 or i >= device_count for i in indices):
        raise ValueError("--gpus must be auto or distinct indices into the visible GPUs, e.g. 0,1")
    return [tokens[i] for i in indices]


def collect_shards(reports, row_getter):
    if not reports:
        raise ValueError("No worker reports")
    count = len(reports)
    selected = reports[0]["shard"]["selected_ids"]
    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate IDs in shard manifest")
    by_id = {}
    indices = set()
    for report in reports:
        shard = report["shard"]
        index = shard["index"]
        if shard["count"] != count or shard["selected_ids"] != selected or index in indices or not 0 <= index < count:
            raise ValueError("Worker shard manifests disagree")
        indices.add(index)
        rows = row_getter(report)
        expected = selected[index::count]
        actual = [row.get("source_example_id", row["example_id"]) for row in rows]
        if actual != expected:
            raise ValueError("Missing, duplicated, reordered or unexpected worker examples")
        for key, row in zip(actual, rows):
            if key in by_id:
                raise ValueError("Duplicate worker example")
            by_id[key] = row
    if set(by_id) != set(selected):
        raise ValueError("Merged report does not cover the selected dataset")
    return [by_id[key] for key in selected]


def direct_summary(rows):
    groups = {}
    for field in ("operation", "category", "source_style", "generation_style", "length_regime", "result_kind"):
        table = defaultdict(list)
        for row in rows:
            table[str(row[field])].append(row)
        groups[f"by_{field}"] = {
            key: {"count": len(pool), "exact_match_accuracy": sum(bool(row["exact"]) for row in pool) / len(pool)}
            for key, pool in sorted(table.items())
        }
        if field == "operation":
            for key, pool in table.items():
                groups["by_operation"][key].update({
                    "average_seconds_per_prompt": sum(row.get("latency_seconds", 0) for row in pool) / len(pool),
                    "average_input_tokens": sum(row.get("input_tokens", 0) for row in pool) / len(pool),
                    "average_output_tokens": sum(row.get("output_tokens", 0) for row in pool) / len(pool),
                })
    return {"example_count": len(rows), "exact_match_accuracy": sum(bool(row["exact"]) for row in rows) / max(1, len(rows)),
            **groups, "per_example": rows}


def merge_reports(kind, reports):
    from src.evaluation.runtime import telemetry_summary
    first = reports[0]
    for report in reports[1:]:
        for field in ("mode", "model_repo_id", "split", "experiment", "phase2_selected_layer", "phase2_candidate_layers"):
            if report.get(field) != first.get(field):
                raise ValueError(f"Worker reports disagree on {field}")
        if kind == "tool-feedback":
            # Only shard selection and physical placement may vary.
            a = {k: v for k, v in first["protocol"].items() if k != "selected_ids"}
            b = {k: v for k, v in report["protocol"].items() if k != "selected_ids"}
            if a != b:
                raise ValueError("Worker tool-feedback protocols disagree")
    split = "test" if first.get("test") is not None else "heldout_template_challenge"
    getter = (lambda report: report[split]["per_example"]) if kind == "pipeline" else (lambda report: report["per_example"])
    rows = collect_shards(reports, getter)
    runtime = telemetry_summary(rows, {})
    runtime["timing_scope"] = "Sum of worker setup and example latencies; parallel wall time is in parallel_runtime"
    runtime["aggregate_worker_setup_seconds"] = sum(r["runtime"].get("canonical_model_setup_seconds", 0) for r in reports)
    runtime["canonical_model_setup_seconds"] = runtime["aggregate_worker_setup_seconds"]
    runtime["benchmark_total_seconds"] += runtime["aggregate_worker_setup_seconds"]
    for key in ("peak_allocated_vram_mib", "peak_reserved_vram_mib"):
        runtime[key] = max(r["runtime"].get(key, 0) for r in reports)
    result = {k: v for k, v in first.items() if k != "shard"}
    result["runtime"] = runtime
    result["inference_setup"] = {**first.get("inference_setup", {}), "device": "multiple worker GPUs",
                                 "worker_count": len(reports)}
    if kind == "pipeline":
        from scripts.evaluate_pipeline import summarize
        block = summarize(rows, telemetry_summary(rows, {}))
        block["runtime_supported_tasks"] = telemetry_summary([r for r in rows if not r["is_control"]], {})
        block["runtime_controls"] = telemetry_summary([r for r in rows if r["is_control"]], {})
        result[split] = block
    elif kind == "direct":
        result.update(direct_summary(rows))
    else:
        from src.evaluation.tool_feedback import summarize
        import hashlib
        import json
        result.update(summarize(rows))
        result["protocol"] = {**first["protocol"], "selected_ids": first["shard"]["selected_ids"]}
        result["protocol_fingerprint"] = hashlib.sha256(json.dumps(result["protocol"], sort_keys=True).encode()).hexdigest()[:16]
        result["runtime_supported_tasks"] = telemetry_summary([r for r in rows if not r["is_control"]], {})
        result["runtime_controls"] = telemetry_summary([r for r in rows if r["is_control"]], {})
        result["runtime_executor_direct"] = telemetry_summary(rows, {}, latency_field="executor_direct_latency_seconds",
                                                             input_field="phase1_input_tokens", output_field="phase1_output_tokens")
        result["runtime_matched_pipeline"] = telemetry_summary(rows, {}, latency_field="matched_pipeline_latency_seconds",
            input_field="matched_pipeline_input_tokens", output_field="matched_pipeline_output_tokens")
    return result

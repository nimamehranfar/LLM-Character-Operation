"""Shared generation accounting for training validation and final evaluation."""
from __future__ import annotations

import math
import statistics
from collections import defaultdict


def token_count_to_eos(tokens, eos_token_id: int | None) -> int:
    values = tokens.tolist()
    if eos_token_id is not None and eos_token_id in values:
        return values.index(eos_token_id) + 1
    return len(values)


def wilson_interval(correct: int, count: int) -> list[float] | None:
    if not count:
        return None
    z = 1.959963984540054
    p, denominator = correct / count, 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0, center - half), min(1, center + half)]


def grouped_accuracy(rows, correct_key="correct"):
    result = {}
    for field in ("operation", "category", "source_style", "length_regime", "template_family"):
        groups = defaultdict(list)
        for row in rows:
            groups[str(row.get(field, ""))].append(row)
        result["by_" + field] = {
            key: {"count": len(items), "correct": sum(bool(x[correct_key]) for x in items),
                  "accuracy": sum(bool(x[correct_key]) for x in items) / len(items),
                  "accuracy_ci95_wilson": wilson_interval(sum(bool(x[correct_key]) for x in items), len(items))}
            for key, items in sorted(groups.items())}
    return result


def inference_runtime(rows, batches):
    elapsed = sum(float(batch["generation_seconds"]) for batch in batches)
    latencies = sorted(float(batch["generation_seconds"]) for batch in batches)
    percentile = lambda q: None if not latencies else latencies[min(len(latencies) - 1, round(q * (len(latencies) - 1)))]
    output_tokens = sum(int(row.get("output_tokens", 0)) for row in rows)
    return {"generation_seconds": elapsed, "example_count": len(rows), "batch_count": len(batches),
            "amortized_seconds_per_example": elapsed / len(rows) if rows else None,
            "examples_per_second": len(rows) / elapsed if elapsed else None,
            "output_tokens": output_tokens, "input_tokens": sum(int(row.get("input_tokens", 0)) for row in rows),
            "output_tokens_per_second": output_tokens / elapsed if elapsed else None,
            "mean_batch_seconds": statistics.fmean(latencies) if latencies else None,
            "p50_batch_seconds": percentile(.5), "p95_batch_seconds": percentile(.95),
            "timing_scope": "CUDA-synchronized generation only; batch latency is not individual request latency",
            "batches": batches}

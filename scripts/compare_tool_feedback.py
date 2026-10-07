"""Pair tool-feedback and mapper outcomes by example ID, keeping splits separate."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_variants(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("mode") in {"native_tool_text_feedback", "lora_tool_text_feedback"}:
        rows = payload["per_example"]
        split = payload["split"]
        label = payload["mode"]
        fingerprint = payload["protocol_fingerprint"]
        return [(split, f"{label}:{fingerprint}", field, rows) for field in (
            "final_exact", "executor_exact", "matched_pipeline_exact")]
    variants = []
    for block_name, split in (("test", "test"), ("heldout_template_challenge", "heldout")):
        block = payload.get(block_name)
        if isinstance(block, dict) and "per_example" in block:
            variants.append((split, "lora_mapper", "final_exact", block["per_example"]))
    if not variants:
        raise ValueError(f"Not a tool-feedback or pipeline report: {path}")
    return variants


def paired_comparison(reference, candidate, reference_field="final_exact", candidate_field="final_exact"):
    def index(rows):
        indexed = {}
        for row in rows:
            if row["is_control"]:
                continue
            key = row.get("source_example_id", row["example_id"])
            if key in indexed:
                raise ValueError(f"Duplicate source example ID: {key}")
            indexed[key] = row
        return indexed
    left, right = index(reference), index(candidate)
    ids = sorted(left.keys() & right.keys())
    if not ids:
        raise ValueError("Reports have no supported-task examples in common")
    for key in ids:
        if left[key].get("expected") != right[key].get("expected") or left[key]["operation"] != right[key]["operation"]:
            raise ValueError(f"Ground truth disagrees for paired example {key}")
    better = sum(bool(right[key][candidate_field]) and not bool(left[key][reference_field]) for key in ids)
    worse = sum(bool(left[key][reference_field]) and not bool(right[key][candidate_field]) for key in ids)
    return {
        "paired_task_count": len(ids), "reference_task_count": len(left), "candidate_task_count": len(right),
        "identical_task_sets": set(left) == set(right),
        "reference_task_exact": sum(bool(left[key][reference_field]) for key in ids) / len(ids),
        "candidate_task_exact": sum(bool(right[key][candidate_field]) for key in ids) / len(ids),
        "candidate_only_correct": better, "reference_only_correct": worse,
        "both_correct": sum(bool(left[key][reference_field]) and bool(right[key][candidate_field]) for key in ids),
        "both_wrong": sum(not bool(left[key][reference_field]) and not bool(right[key][candidate_field]) for key in ids),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare native tools/text-feedback ablations against a trained mapper report on shared example IDs.")
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--feedback", nargs="+", required=True)
    parser.add_argument("--output", default="results/tables/tool_feedback_comparison.csv")
    args = parser.parse_args()
    references = {split: (field, rows) for split, _, field, rows in load_variants(args.pipeline)}
    output = []
    for path in args.feedback:
        for split, label, field, candidate in load_variants(path):
            if split not in references:
                continue
            reference_field, reference = references[split]
            metrics = paired_comparison(reference, candidate, reference_field, field)
            output.append({"split": split, "candidate": label, "candidate_answer_path": field,
                           "feedback_file": str(path), **metrics})
    if not output:
        raise ValueError("No matching splits to compare")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"Saved {len(output)} paired comparisons: {path}")


if __name__ == "__main__":
    main()

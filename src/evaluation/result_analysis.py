from __future__ import annotations

from collections import defaultdict
import statistics
import time
from typing import Any, Iterable

import torch

from src.data.character_dataset import CharacterExample
from src.executor.operations import CharacterExecutor
from src.model.result_injector import (
    LayeredSymbolicResultMapper,
    TrainableLayeredResultInjector,
    result_to_symbol,
)
from src.training.result_injection_data import render_answer_prompt


def exact_match(text: str, expected: str) -> bool:
    return text.strip() == expected.strip()


def _summary(details: list[dict[str, Any]]) -> dict[str, Any]:
    by_operation: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "correct": 0, "gates": []})
    by_category: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "correct": 0, "gates": []})
    by_source_style: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "correct": 0, "gates": []})
    by_length: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "correct": 0, "gates": []})
    gates: list[float] = []
    for row in details:
        for table, key in (
            (by_operation, row["operation"]),
            (by_category, row["category"]),
            (by_source_style, row.get("source_style", "")),
            (by_length, row.get("length_regime", "")),
        ):
            item = table[key]
            item["count"] += 1
            item["correct"] += int(row["correct"])
            if row.get("gate") is not None:
                item["gates"].append(float(row["gate"]))
        if row.get("gate") is not None:
            gates.append(float(row["gate"]))

    def convert(table: dict[str, dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, item in sorted(table.items()):
            gs = item["gates"]
            out[key] = {
                "count": item["count"],
                "correct_count": item["correct"],
                "exact_match_accuracy": item["correct"] / item["count"] if item["count"] else 0.0,
                "mean_gate": statistics.fmean(gs) if gs else None,
            }
        return out

    return {
        "example_count": len(details),
        "exact_match_accuracy": sum(int(x["correct"]) for x in details) / len(details) if details else 0.0,
        "mean_gate": statistics.fmean(gates) if gates else None,
        "median_gate": statistics.median(gates) if gates else None,
        "min_gate": min(gates) if gates else None,
        "max_gate": max(gates) if gates else None,
        "runtime": _runtime_summary(details),
        "by_operation": convert(by_operation),
        "by_category": convert(by_category),
        "by_source_style": convert(by_source_style),
        "by_length_regime": convert(by_length),
        "per_example": details,
    }


def _runtime_summary(details: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["generation_seconds"]) for row in details if row.get("generation_seconds") is not None]
    if not latencies:
        return {}
    total = sum(latencies)
    output_tokens = sum(int(row.get("output_tokens", 0)) for row in details)
    ordered = sorted(latencies)
    percentile = lambda q: None if not ordered else ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]
    return {
        "generation_seconds": total,
        "mean_seconds_per_example": statistics.fmean(latencies) if latencies else None,
        "p50_seconds_per_example": percentile(0.50),
        "p95_seconds_per_example": percentile(0.95),
        "examples_per_second": len(latencies) / total if total else None,
        "output_tokens": output_tokens,
        "output_tokens_per_second": output_tokens / total if total else None,
    }


@torch.no_grad()
def evaluate_oracle_generation(
    model: Any, tokenizer: Any, mapper: LayeredSymbolicResultMapper | None,
    examples: Iterable[CharacterExample], executor: CharacterExecutor, *,
    layer_index: int | None, max_new_tokens: int, max_integer_result: int,
    ascii_vocab_size: int, save_per_example: bool = True, batch_size: int = 1,
) -> dict[str, Any]:
    """Oracle Phase-1 upper bound: isolate reintegration, never end-to-end accuracy."""
    from contextlib import nullcontext
    from src.evaluation.training_metrics import inference_runtime, token_count_to_eos, wilson_interval
    details, batches = [], []
    selected = [ex for ex in examples if not ex.is_control]
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = next(model.parameters()).device
    original_padding_side = tokenizer.padding_side
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(selected), batch_size):
            batch = selected[start:start + batch_size]
            results = [ex.execute(executor) for ex in batch]
            prompts = [render_answer_prompt(tokenizer, ex.prompt) for ex in batch]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            context = nullcontext(None)
            if mapper is not None:
                if layer_index is None:
                    raise ValueError("layer_index is required with mapper")
                symbols = [result_to_symbol(result, max_integer=max_integer_result,
                           ascii_vocab_size=ascii_vocab_size) for result in results]
                context = TrainableLayeredResultInjector(
                    model, mapper, layer_index=int(layer_index), symbol=symbols,
                    position_index=[-1] * len(batch), prefill_only=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with context as injector:
                generated = model.generate(
                    **encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            batch_id = len(batches)
            batches.append({"batch_id": batch_id, "example_count": len(batch), "generation_seconds": elapsed})
            suffix = generated[:, encoded["input_ids"].shape[1]:]
            answers = tokenizer.batch_decode(suffix, skip_special_tokens=True)
            for row, ex in enumerate(batch):
                answer = answers[row].strip()
                gate = None if injector is None else injector.last_gates[row]
                details.append({
                    "example_id": ex.example_id, "operation": ex.operation, "category": ex.category,
                    "source_style": ex.source_style, "generation_style": ex.generation_style,
                    "length_regime": ex.length_regime, "result_kind": ex.result_kind,
                    "template_family": ex.template_family, "expected": ex.expected,
                    "executor_result": str(results[row]), "generated": answer,
                    "correct": exact_match(answer, ex.expected), "layer": layer_index,
                    "gate": gate, "generation_seconds": elapsed / len(batch),
                    "output_tokens": token_count_to_eos(suffix[row], tokenizer.eos_token_id),
                    "input_tokens": int(encoded["attention_mask"][row].sum()),
                    "batch_size": len(batch), "batch_id": batch_id})
    finally:
        tokenizer.padding_side = original_padding_side
    summary = _summary(details)
    summary["runtime"] = inference_runtime(details, batches)
    summary["accuracy_ci95_wilson"] = wilson_interval(sum(bool(x["correct"]) for x in details), len(details))
    if not save_per_example:
        summary["per_example"] = None
    return summary


def summarize_best_any_layer(layer_reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Oracle upper bound: success if any candidate layer generated the exact answer."""
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in layer_reports:
        layer = int(report["layer"])
        for row in report["metrics"].get("per_example") or []:
            item = dict(row)
            item["layer"] = layer
            by_id[item["example_id"]].append(item)
    details: list[dict[str, Any]] = []
    for example_id, rows in sorted(by_id.items()):
        successes = [r for r in rows if r["correct"]]
        chosen = min(successes, key=lambda r: int(r["layer"])) if successes else min(rows, key=lambda r: int(r["layer"]))
        details.append({
            "example_id": example_id,
            "operation": chosen["operation"],
            "category": chosen["category"],
            "source_style": chosen.get("source_style", ""),
            "length_regime": chosen.get("length_regime", ""),
            "correct": bool(successes),
            "first_successful_layer": None if not successes else int(min(r["layer"] for r in successes)),
            "successful_layers": [int(r["layer"]) for r in successes],
        })
    return _summary(details)

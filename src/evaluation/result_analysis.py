from __future__ import annotations

from collections import defaultdict
import statistics
from typing import Any, Iterable

import torch

from src.data.character_dataset import CharacterExample
from src.executor.operations import CharacterExecutor
from src.model.result_injector import (
    LayeredSymbolicResultMapper,
    TrainableLayeredResultInjector,
    result_to_symbol,
)
from src.training.result_injection_data import prepare_prompt_tensors


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
        "by_operation": convert(by_operation),
        "by_category": convert(by_category),
        "by_source_style": convert(by_source_style),
        "by_length_regime": convert(by_length),
        "per_example": details,
    }


@torch.no_grad()
def evaluate_oracle_generation(
    model: Any,
    tokenizer: Any,
    mapper: LayeredSymbolicResultMapper | None,
    examples: Iterable[CharacterExample],
    executor: CharacterExecutor,
    *,
    layer_index: int | None,
    max_new_tokens: int,
    max_integer_result: int,
    ascii_vocab_size: int,
    save_per_example: bool = True,
) -> dict[str, Any]:
    """Evaluate final generation with ground-truth operation/arguments/result.

    This is explicitly an oracle Phase-1 upper bound. No oracle values are used
    by the normal end-to-end path.
    """
    details: list[dict[str, Any]] = []
    for ex in examples:
        if ex.is_control:
            continue
        result = ex.execute(executor)
        encoded = prepare_prompt_tensors(tokenizer, ex.prompt, next(model.parameters()).device)
        context = None
        injector = None
        if mapper is not None:
            if layer_index is None:
                raise ValueError("layer_index is required with mapper")
            symbol = result_to_symbol(
                result,
                max_integer=max_integer_result,
                ascii_vocab_size=ascii_vocab_size,
            )
            context = TrainableLayeredResultInjector(
                model,
                mapper,
                layer_index=int(layer_index),
                symbol=symbol,
                position_index=-1,
                prefill_only=True,
            )
        if context is None:
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            with context as injector:
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        answer = tokenizer.decode(
            generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        details.append({
            "example_id": ex.example_id,
            "operation": ex.operation,
            "category": ex.category,
            "source_style": ex.source_style,
            "generation_style": ex.generation_style,
            "length_regime": ex.length_regime,
            "result_kind": ex.result_kind,
            "expected": ex.expected,
            "executor_result": str(result),
            "generated": answer,
            "correct": exact_match(answer, ex.expected),
            "layer": None if layer_index is None else int(layer_index),
            "gate": None if injector is None else injector.last_gate,
        })
    summary = _summary(details)
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

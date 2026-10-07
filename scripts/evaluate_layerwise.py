from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_pipeline import (
    cfgload,
    load_all,
    phase2_generate,
    policy_generate,
    resolve,
)
from src.data.character_dataset import load_jsonl, SCALAR_RESULT_OPERATIONS, STRING_RESULT_OPERATIONS
from src.data.tool_policy_dataset import call_is_exact
from src.evaluation.pipeline import execute_parsed_call
from src.evaluation.result_analysis import evaluate_oracle_generation, summarize_best_any_layer
from src.executor.operations import CharacterExecutor

SCRIPT_VERSION = "layerwise_analysis"


def _load_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as h:
        for line in h:
            if not line.strip():
                continue
            row = json.loads(line)
            out[str(row["example_id"])] = row
    return out


def _append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
    with path.open("ab") as h:
        h.write(payload)
        h.flush()
        os.fsync(h.fileno())


def _mean(values):
    values = [float(x) for x in values if x is not None]
    return sum(values) / len(values) if values else None


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_op = defaultdict(lambda: {"n": 0, "p1": 0, "final": 0, "gates": []})
    by_cat = defaultdict(lambda: {"n": 0, "p1": 0, "final": 0, "gates": []})
    by_result = defaultdict(lambda: {"n": 0, "p1": 0, "final": 0, "gates": []})
    gates = []
    for row in rows:
        for table, key in ((by_op, row["operation"]), (by_cat, row["category"]), (by_result, row["result_kind"])):
            g = table[key]
            g["n"] += 1
            g["p1"] += int(row.get("phase1_exact", True))
            g["final"] += int(row["final_exact"])
            if row.get("gate") is not None:
                g["gates"].append(float(row["gate"]))
        if row.get("gate") is not None:
            gates.append(float(row["gate"]))

    def convert(table):
        return {
            k: {
                "count": v["n"],
                "phase1_exact": v["p1"] / v["n"],
                "final_exact": v["final"] / v["n"],
                "mean_gate": _mean(v["gates"]),
            }
            for k, v in sorted(table.items())
        }

    return {
        "example_count": len(rows),
        "phase1_execution_readiness": sum(int(r.get("phase1_exact", True)) for r in rows) / max(1, len(rows)),
        "final_exact": sum(int(r["final_exact"]) for r in rows) / max(1, len(rows)),
        "mean_gate": _mean(gates),
        "by_operation": convert(by_op),
        "by_category": convert(by_cat),
        "by_result_kind": convert(by_result),
        "per_example": rows,
    }


def collect_phase1(split_name, examples, policy, tok, p1, cache_dir: Path):
    cache = cache_dir / f"{split_name}_phase1.jsonl"
    existing = _load_rows(cache)
    exr = CharacterExecutor()
    for ex in tqdm(examples, desc=f"Phase1 cache {split_name}", unit="ex", dynamic_ncols=True):
        if ex.example_id in existing:
            continue
        raw, parsed, _ = policy_generate(policy, tok, ex, p1)
        row = {
            "example_id": ex.example_id,
            "policy_output": raw,
            "parsed": parsed,
            "phase1_exact": bool(call_is_exact(parsed, ex)),
            "executor_result": None,
            "executed_operation": None,
            "executor_error": None,
        }
        if parsed.get("decision") == "CALL" and bool(parsed.get("valid")):
            try:
                execution = execute_parsed_call(parsed, exr)
                row["executor_result"] = execution.result
                row["executed_operation"] = execution.operation
            except Exception as exc:
                row["executor_error"] = f"{type(exc).__name__}: {exc}"
        _append_row(cache, row)
        existing[ex.example_id] = row
    return existing


def actual_layer_eval(split_name, examples, phase1_cache, policy, tok, mapper, layer, arch, p2, cache_dir: Path):
    cache = cache_dir / f"{split_name}_actual_layer_{layer}.jsonl"
    existing = _load_rows(cache)
    for ex in tqdm(examples, desc=f"Actual P1 layer {layer} {split_name}", unit="ex", dynamic_ncols=True):
        if ex.example_id in existing:
            continue
        p = phase1_cache[ex.example_id]
        parsed = p["parsed"]
        generated = None
        gate = None
        final = False
        path = None
        if ex.is_control:
            final = parsed.get("decision") == "NO_CALL" and bool(parsed.get("valid"))
            path = "no_call"
        elif p.get("executor_error") is None and p.get("executed_operation") is not None:
            if p["executed_operation"] in SCALAR_RESULT_OPERATIONS:
                path = "phase2_scalar"
                try:
                    generated, gate, _ = phase2_generate(
                        policy,
                        tok,
                        mapper,
                        int(layer),
                        p["executor_result"],
                        ex.prompt,
                        int(p2["evaluation"]["max_new_tokens"]),
                        arch,
                    )
                    final = generated.strip() == ex.expected.strip()
                except Exception:
                    final = False
            elif p["executed_operation"] in STRING_RESULT_OPERATIONS:
                path = "executor_direct_transform"
                generated = str(p["executor_result"])
                final = generated == ex.expected
        row = {
            "example_id": ex.example_id,
            "operation": ex.operation,
            "category": ex.category,
            "source_style": ex.source_style,
            "generation_style": ex.generation_style,
            "length_regime": ex.length_regime,
            "result_kind": ex.result_kind,
            "is_control": ex.is_control,
            "phase1_exact": bool(p["phase1_exact"]),
            "layer": int(layer),
            "gate": gate,
            "answer_path": path,
            "generated": generated,
            "expected": None if ex.is_control else ex.expected,
            "final_exact": bool(final),
        }
        _append_row(cache, row)
        existing[ex.example_id] = row
    ordered = [existing[ex.example_id] for ex in examples]
    return summarize_rows(ordered)


def oracle_layer_eval(split_name, examples, policy, tok, mapper, layer, arch, p2, cache_dir: Path):
    cache = cache_dir / f"{split_name}_oracle_layer_{layer}.jsonl"
    existing = _load_rows(cache)
    exr = CharacterExecutor()
    for ex in tqdm(examples, desc=f"Oracle P1 layer {layer} {split_name}", unit="ex", dynamic_ncols=True):
        if ex.example_id in existing:
            continue
        generated = None
        gate = None
        if ex.is_control:
            final = True
            path = "oracle_no_call"
        elif ex.operation in STRING_RESULT_OPERATIONS:
            result = ex.execute(exr)
            generated = str(result)
            final = generated == ex.expected
            path = "oracle_executor_direct_transform"
        else:
            result = ex.execute(exr)
            generated, gate, _ = phase2_generate(
                policy,
                tok,
                mapper,
                int(layer),
                result,
                ex.prompt,
                int(p2["evaluation"]["max_new_tokens"]),
                arch,
            )
            final = generated.strip() == ex.expected.strip()
            path = "oracle_phase1_phase2_scalar"
        row = {
            "example_id": ex.example_id,
            "operation": ex.operation,
            "category": ex.category,
            "source_style": ex.source_style,
            "generation_style": ex.generation_style,
            "length_regime": ex.length_regime,
            "result_kind": ex.result_kind,
            "is_control": ex.is_control,
            "phase1_exact": True,
            "layer": int(layer),
            "gate": gate,
            "answer_path": path,
            "generated": generated,
            "expected": None if ex.is_control else ex.expected,
            "final_exact": bool(final),
        }
        _append_row(cache, row)
        existing[ex.example_id] = row
    ordered = [existing[ex.example_id] for ex in examples]
    return summarize_rows(ordered)


def scalar_oracle_report_to_compat(layer: int, summary: dict[str, Any]) -> dict[str, Any]:
    scalar_rows = [r for r in summary["per_example"] if r["result_kind"] in {"integer", "character"}]
    metrics = summarize_rows(scalar_rows)
    metrics["per_example"] = [{**row, "correct": row["final_exact"], "gate": row.get("gate", row.get("phase2_gate"))}
                              for row in scalar_rows]
    return {"layer": int(layer), "metrics": metrics}


def main() -> None:
    ap = argparse.ArgumentParser(description="Layer-wise actual-policy and oracle analysis for the trained pipeline.")
    ap.add_argument("--phase1-config", default=str(REPO_ROOT / "configs/experiments/qwen3_8b/tool_policy.toml"))
    ap.add_argument("--phase2-config", default=str(REPO_ROOT / "configs/experiments/qwen3_8b/result_injection.toml"))
    ap.add_argument("--auto-download", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--model-cache-dir", default=None)
    ap.add_argument("--fresh-analysis", action="store_true")
    ap.add_argument("--selected-only", action="store_true", help="Evaluate only the dev-selected injection layer.")
    args = ap.parse_args()

    p1 = cfgload(Path(args.phase1_config).resolve())
    p2 = cfgload(Path(args.phase2_config).resolve())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    tok, policy, mapper, selected_layer, layers, arch, raw_phase2, *_ = load_all(p1, p2, auto_download=args.auto_download, cache_dir=args.model_cache_dir)
    if args.selected_only:
        layers = [selected_layer]

    out_dir = resolve(p1["output"]["results_dir"])
    cache_dir = out_dir / "layerwise_cache"
    if args.fresh_analysis and cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "experiment": SCRIPT_VERSION,
        "phase2_candidate_layers": layers,
        "phase2_selected_layer": selected_layer,
        "phase2_selection_rule": raw_phase2.get("selection_rule", "checkpoint selected"),
        "splits": {},
    }
    for split_name, path in (
        ("test", resolve(p1["data"]["test_file"])),
        ("heldout_template_challenge", resolve(p1["evaluation"]["challenge_file"])),
    ):
        examples = load_jsonl(path)
        p1_cache = collect_phase1(split_name, examples, policy, tok, p1, cache_dir)
        actual = []
        oracle = []
        for layer in layers:
            actual.append({
                "layer": int(layer),
                "metrics": actual_layer_eval(
                    split_name, examples, p1_cache, policy, tok, mapper, layer, arch, p2, cache_dir
                ),
            })
            oracle_summary = oracle_layer_eval(
                split_name, examples, policy, tok, mapper, layer, arch, p2, cache_dir
            )
            oracle.append({"layer": int(layer), "metrics": oracle_summary})

        scalar_oracle_compat = [scalar_oracle_report_to_compat(x["layer"], x["metrics"]) for x in oracle]
        oracle_best_scalar = summarize_best_any_layer(scalar_oracle_compat)
        report["splits"][split_name] = {
            "actual_phase1_layerwise": actual,
            "oracle_phase1_layerwise": oracle,
            "oracle_best_any_layer_scalar": oracle_best_scalar,
            "selected_layer_actual": next((x["metrics"] for x in actual if x["layer"] == selected_layer), None),
            "selected_layer_oracle": next((x["metrics"] for x in oracle if x["layer"] == selected_layer), None),
        }

    report_path = out_dir / "layerwise_analysis.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for split_name, block in report["splits"].items():
        selected=block.get("selected_layer_actual") or {}
        oracle=block.get("selected_layer_oracle") or {}
        label="IID" if split_name=="test" else "Held-out"
        print(f"{label}: selected_layer={selected_layer} | actual_exact={selected.get('final_exact',0.0):.4f} | oracle_exact={oracle.get('final_exact',0.0):.4f}")
    print(f"Results: {report_path}")


if __name__ == "__main__":
    main()

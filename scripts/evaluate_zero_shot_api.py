from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import tomllib
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.character_dataset import load_jsonl
from src.evaluation.runtime import ResumeLedger, telemetry_summary

ZERO_SHOT_SYSTEM = "Perform the requested character/string operation exactly. Output only the final answer, with no explanation."


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def stratified_sample(rows, examples_per_operation: int, seed: int):
    if examples_per_operation <= 0:
        return list(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.operation].append(row)
    sampled = []
    for operation in sorted(grouped):
        pool = grouped[operation]
        rng = random.Random(seed + sum(map(ord, operation)))
        sampled.extend(rng.sample(pool, min(examples_per_operation, len(pool))))
    return sampled


def request_chat_completion(base_url: str, api_key: str | None, model: str, prompt: str,
                            max_tokens: int, temperature: float, timeout: float) -> dict:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": ZERO_SHOT_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    usage = body.get("usage") or {}
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
    return {
        "content": str(body["choices"][0]["message"]["content"]),
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "usage_available": bool(usage),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot evaluation through an OpenAI-compatible hosted API.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/baselines/api/openai_compatible.toml"))
    parser.add_argument("--split", choices=["test", "heldout"], default=None)
    parser.add_argument("--examples-per-operation", type=int, default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    cfg = load_config(Path(args.config).resolve())
    api = cfg["api"]
    data = cfg["data"]
    evaluation = cfg.get("evaluation", {})
    split = args.split or str(evaluation.get("split", "test"))
    split_path = data["test_file"] if split == "test" else data["heldout_file"]
    rows = [row for row in load_jsonl(resolve(split_path)) if not row.is_control]
    per_op = args.examples_per_operation if args.examples_per_operation is not None else int(evaluation.get("examples_per_operation", 100))
    seed = int(evaluation.get("seed", 20260919))
    rows = stratified_sample(rows, per_op, seed)

    base_url = str(args.base_url or api.get("base_url", "")).strip()
    model = str(args.model or api.get("model", "")).strip()
    if not base_url or not model:
        raise ValueError("Provide API base URL and model in config or via --base-url and --model")
    key_env = str(args.api_key_env or api.get("api_key_env", "BASELINE_API_KEY"))
    api_key = os.getenv(key_env)
    if bool(api.get("require_api_key", True)) and not api_key:
        raise RuntimeError(f"Missing API key environment variable: {key_env}")

    output_dir = resolve(cfg.get("output", {}).get("results_dir", "results/baselines/api"))
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = str(api.get("slug", model.replace("/", "__")))
    progress_path = output_dir / f"{slug}_zero_shot_{split}.progress.jsonl"
    state_path = output_dir / f"{slug}_zero_shot_{split}.progress.state.json"
    report_path = output_dir / f"{slug}_zero_shot_{split}.json"
    resume = bool(evaluation.get("resume", True)) if args.resume is None else bool(args.resume)
    ledger = ResumeLedger(state_path, progress_path, enabled=resume)
    completed = ledger.load_completed()
    if len(completed) >= len(rows) and report_path.exists():
        print(f"Already complete: {report_path}")
        return
    if ledger.state.get('canonical_setup_seconds') is None:
        ledger.record_setup(0.0)

    retries = int(api.get("retries", 5))
    retry_seconds = float(api.get("retry_seconds", 2.0))
    timeout = float(api.get("timeout_seconds", 120.0))
    max_tokens = int(evaluation.get("max_new_tokens", 192))
    temperature = float(api.get("temperature", 0.0))

    try:
        for index, ex in enumerate(rows, start=1):
            if ex.example_id in completed:
                continue
            error = None
            response = None
            request_active = 0.0
            attempt_count = 0
            for attempt in range(retries):
                attempt_count += 1
                started = time.perf_counter()
                try:
                    response = request_chat_completion(base_url, api_key, model, ex.prompt, max_tokens, temperature, timeout)
                    request_active += time.perf_counter() - started
                    error = None
                    break
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, ValueError) as exc:
                    request_active += time.perf_counter() - started
                    error = f"{type(exc).__name__}: {exc}"
                    if attempt + 1 < retries:
                        time.sleep(retry_seconds * (attempt + 1))
            raw = None if response is None else response["content"]
            answer = "" if raw is None else raw.strip()
            item = {
                "example_id": ex.example_id,
                "operation": ex.operation,
                "category": ex.category,
                "source_style": ex.source_style,
                "generation_style": ex.generation_style,
                "length_regime": ex.length_regime,
                "result_kind": ex.result_kind,
                "expected": ex.expected,
                "generated": raw,
                "normalized_answer": answer,
                "exact": answer == ex.expected.strip(),
                "error": error,
                "attempt_count": attempt_count,
                "latency_seconds": request_active,
                "input_tokens": 0 if response is None else response["input_tokens"],
                "output_tokens": 0 if response is None else response["output_tokens"],
                "total_tokens": 0 if response is None else response["input_tokens"] + response["output_tokens"],
                "provider_usage_available": False if response is None else response["usage_available"],
            }
            ledger.append(item)
            completed[ex.example_id] = item
    except KeyboardInterrupt:
        ledger.mark_interrupted()
        print("Interrupted safely; rerun the same command to resume.")
        raise

    details = [completed[row.example_id] for row in rows if row.example_id in completed]
    correct = sum(int(item["exact"]) for item in details)
    failures = sum(int(item["error"] is not None) for item in details)
    by_operation = defaultdict(lambda: {"count": 0, "correct": 0})
    for item in details:
        stat = by_operation[item["operation"]]
        stat["count"] += 1
        stat["correct"] += int(item["exact"])
    ledger.mark_complete()
    ledger_state = ledger.snapshot()
    report = {
        "mode": "zero_shot_api",
        "model": model,
        "base_url": base_url,
        "split": split,
        "example_count": len(details),
        "examples_per_operation": per_op,
        "request_failures": failures,
        "exact_match_accuracy": correct / max(1, len(details)),
        "inference_setup": {
            "provider": "openai_compatible",
            "model": model,
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "batch_size": 1,
            "resume_enabled": resume,
        },
        "runtime": telemetry_summary(details, ledger_state),
        "by_operation": {
            op: {"count": stat["count"], "exact_match_accuracy": stat["correct"] / max(1, stat["count"])}
            for op, stat in sorted(by_operation.items())
        },
        "per_example": details,
    }
    report["runtime"]["provider_usage_coverage"] = sum(int(x.get("provider_usage_available", False)) for x in details) / max(1, len(details))
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    runtime=report["runtime"]
    print(f"zero_shot_api {split}: accuracy={report['exact_match_accuracy']:.4f} | n={report['example_count']} | failures={report['request_failures']} | avg={runtime.get('average_seconds_per_prompt',0.0):.4f}s | p95={runtime.get('p95_seconds_per_prompt',0.0):.4f}s")
    print(f"Results: {report_path}")


if __name__ == "__main__":
    main()

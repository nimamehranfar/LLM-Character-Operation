"""Run independent evaluation shards on visible GPUs, then validate and merge them."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.parallel import merge_reports, visible_gpu_tokens

SCRIPTS = {"tool-feedback": "evaluate_tool_feedback.py", "direct": "evaluate_direct.py", "pipeline": "evaluate_pipeline.py"}


def resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def worker_commands(script, arguments, gpu_tokens, output_dir, split):
    jobs = []
    for index, token in enumerate(gpu_tokens):
        directory = output_dir / split / f"worker-{index:03d}"
        report = directory / "report.json"
        command = [sys.executable, str(script), *arguments, "--split", split,
                   "--shard-index", str(index), "--shard-count", str(len(gpu_tokens)),
                   "--results-dir", str(directory), "--report-path", str(report)]
        jobs.append((token, command, report))
    return jobs


def run_workers(jobs):
    processes = []
    logs = []
    try:
        for token, command, report in jobs:
            report.parent.mkdir(parents=True, exist_ok=True)
            log_path = report.parent / "worker.log"
            log = log_path.open("a", encoding="utf-8")
            logs.append(log)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": token, "PYTHONUNBUFFERED": "1",
                   "LLM_CHARACTER_SKIP_MODEL_REGISTRY_WRITE": "1"}
            processes.append((subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), log_path))
            print(f"Started GPU {token}; log: {log_path}", flush=True)
        while True:
            codes = [process.poll() for process, _ in processes]
            for code, (_, log_path) in zip(codes, processes):
                if code is not None and code != 0:
                    raise RuntimeError(f"Worker exited with code {code}; inspect {log_path}")
            if all(code is not None for code in codes):
                break
            time.sleep(0.2)
    finally:
        # No workers left running after Ctrl+C or another worker's failure.
        for process, _ in processes:
            if process.poll() is None:
                process.terminate()
        for process, _ in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=SCRIPTS, default="tool-feedback")
    parser.add_argument("--gpus", default="auto", help="auto = all visible GPUs; otherwise visible indices such as 0,1")
    parser.add_argument("--split", choices=["test", "heldout", "both"], default="both")
    parser.add_argument("--output-dir", default="results/parallel")
    parser.add_argument("--dry-run", action="store_true", help="Print worker commands without downloading or loading models")
    parser.add_argument("evaluation_arguments", nargs=argparse.REMAINDER, help="After --, pass normal evaluator arguments")
    args = parser.parse_args()
    extra = args.evaluation_arguments
    if extra[:1] == ["--"]:
        extra = extra[1:]
    reserved = {"--split", "--results-dir", "--report-path", "--shard-count", "--shard-index", "--preflight"}
    if any(item.split("=", 1)[0] in reserved for item in extra):
        parser.error("Set --split before --; worker output/shard options and --preflight cannot be forwarded")
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument("--config")
    options.add_argument("--phase1-config", default="configs/experiments/qwen3_8b/tool_policy.toml")
    options.add_argument("--phase2-config", default="configs/experiments/qwen3_8b/result_injection.toml")
    options.add_argument("--policy-config")
    options.add_argument("--auto-download", action=argparse.BooleanOptionalAction, default=None)
    options.add_argument("--model-cache-dir")
    options.add_argument("--local-model-path")
    config_args, _ = options.parse_known_args(extra)
    if args.kind == "pipeline":
        config_paths = [resolve(config_args.phase1_config), resolve(config_args.phase2_config)]
    else:
        default = "configs/baselines/tools/qwen3_8b.toml" if args.kind == "tool-feedback" else "configs/baselines/local/qwen3_8b.toml"
        config_paths = [resolve(config_args.config or default)]
        if config_args.config is None:
            extra += ["--config", str(config_paths[0])]
        if config_args.policy_config:
            config_paths.append(resolve(config_args.policy_config))
    configs = [tomllib.loads(path.read_text(encoding="utf-8")) for path in config_paths]
    cfg = configs[0]
    import torch
    count = torch.cuda.device_count()
    if args.dry_run and args.gpus != "auto":
        # Permit planning a server run from a developer machine with fewer GPUs.
        count = max(count, max(int(i) for i in args.gpus.split(",")) + 1)
        tokens = visible_gpu_tokens(args.gpus, count, {})
    else:
        tokens = visible_gpu_tokens(args.gpus, count)
    slug = cfg["model"].get("slug", cfg["model"]["repo_id"].replace("/", "__"))
    if args.dry_run:
        directory = resolve(args.output_dir) / slug / args.kind / "dry-run"
        for split in (["test", "heldout"] if args.split == "both" else [args.split]):
            for token, command, _ in worker_commands(ROOT / "scripts" / SCRIPTS[args.kind], extra, tokens, directory, split):
                print(json.dumps({"CUDA_VISIBLE_DEVICES": token, "command": command}))
        return
    # Resolve/download once before launching replicas; use the same snapshot in every worker.
    from src.model.local_model import ensure_local_model_path
    cache = cfg.get("cache", {})
    auto = bool(cache.get("auto_download", False)) if config_args.auto_download is None else config_args.auto_download
    cache_dir = config_args.model_cache_dir if config_args.model_cache_dir is not None else cache.get("cache_dir")
    model_path = ensure_local_model_path(cfg["model"]["repo_id"], config_args.local_model_path or cfg["model"].get("local_path"),
        auto_download=auto, cache_dir=cache_dir, revision=str(cache.get("revision", "main")))
    files = {str(path.relative_to(ROOT)): digest(path) for directory in ("src", "scripts") for path in (ROOT / directory).rglob("*.py")}
    for path, config in zip(config_paths, configs):
        files[str(path)] = digest(path)
        for section in ("data", "evaluation"):
            for key, value in config.get(section, {}).items():
                if (key.endswith("_file") or key == "challenge_file") and isinstance(value, str):
                    path = resolve(value)
                    if path.is_file():
                        files[str(path)] = digest(path)
        for key in ("adapter_dir", "final_checkpoint"):
            value = config.get("output", {}).get(key)
            if value:
                path = resolve(value)
                for file in (path.rglob("*") if path.is_dir() else [path]):
                    if file.is_file():
                        files[str(file)] = digest(file)
    hardware = [{"name": torch.cuda.get_device_properties(i).name,
                 "uuid": str(getattr(torch.cuda.get_device_properties(i), "uuid", "")),
                 "vram": torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]
    identity = {"kind": args.kind, "arguments": extra, "split": args.split, "gpus": tokens,
                "hardware": hardware, "hostname": platform.node(), "files": files, "model_path": str(model_path),
                "model_config": digest(model_path / "config.json"),
                "versions": {name: version(name) for name in ("torch", "transformers", "peft", "bitsandbytes")}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    directory = resolve(args.output_dir) / slug / args.kind / fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    for split in (["test", "heldout"] if args.split == "both" else [args.split]):
        jobs = worker_commands(ROOT / "scripts" / SCRIPTS[args.kind], [*extra, "--local-model-path", str(model_path)], tokens, directory, split)
        had_previous_progress = any(path.exists() or any(path.parent.glob("*.progress*")) for _, _, path in jobs)
        started = time.perf_counter()
        run_workers(jobs)
        wall_seconds = time.perf_counter() - started
        reports = [json.loads(report.read_text(encoding="utf-8")) for _, _, report in jobs]
        merged = merge_reports(args.kind, reports)
        resumed = had_previous_progress or any(r["runtime"].get("resume_count", 0) > 0 for r in reports)
        row_count = len(merged["test" if split == "test" else "heldout_template_challenge"]["per_example"]) if args.kind == "pipeline" else len(merged["per_example"])
        merged["parallel_runtime"] = {"gpu_tokens": tokens, "worker_count": len(tokens),
            "wall_seconds_current_session": wall_seconds, "resumed": resumed,
            "examples_per_wall_second": row_count / wall_seconds if not resumed else None,
            "output_tokens_per_wall_second": merged["runtime"]["total_output_tokens"] / wall_seconds if not resumed else None,
            "note": "Wall time includes worker startup/model loading. Throughput is omitted on resumed runs.",
            "workers": [{"gpu_token": token, "report_path": str(path), "runtime": report["runtime"],
                         "inference_setup": report.get("inference_setup")} for (token, _, path), report in zip(jobs, reports)]}
        path = directory / f"{split}.json"
        path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Merged {row_count} examples from {len(tokens)} GPUs: {path}", flush=True)


if __name__ == "__main__":
    main()

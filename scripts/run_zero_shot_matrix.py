"""Run unaided pretrained baselines with separate, exportable results per run."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tomllib
import uuid

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.character_dataset import load_jsonl
from src.evaluation.baseline_runs import cleanup_owned_cache, digest, export_results, owned_cache, save_json, validate_report
from src.model.device import select_device
from src.model.local_model import ensure_local_model_path, isolated_download_cache
from scripts.evaluate_direct import stratified_sample

CONFIG_DIR = REPO_ROOT / "configs/baselines/local"
DEFAULT_MATRIX_CONFIG = REPO_ROOT / "configs/baselines/matrix.toml"


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def load(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def concise_failure(stderr: str, stdout: str) -> str:
    lines = [line.strip() for line in (stderr + "\n" + stdout).splitlines() if line.strip()]
    useful = [line for line in lines if any(key in line for key in
              ("Error:", "Exception:", "CUDA out of memory", "OutOfMemoryError"))]
    return useful[-1] if useful else (lines[-1] if lines else "evaluation subprocess failed")


def main() -> None:
    configs_by_name = {path.stem: path for path in sorted(CONFIG_DIR.glob("*.toml"))}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-config", default=str(DEFAULT_MATRIX_CONFIG))
    parser.add_argument("--models", nargs="+", choices=sorted(configs_by_name), help="One or more config names; omitted = all 13")
    parser.add_argument("--list-models", action="store_true", help="List configured model names and Hub repositories, then exit")
    parser.add_argument("--split", choices=["test", "heldout", "both"], default=None)
    samples = parser.add_mutually_exclusive_group()
    samples.add_argument("--examples-per-operation", type=int, help="N per operation; -1 = full split")
    samples.add_argument("--examples", type=int, help="Total prompts per split; -1 = full split")
    parser.add_argument("--auto-download", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--download-only", action="store_true", help="Populate the cache on a login node; no evaluation or deletion, even with --cleanup-models")
    parser.add_argument("--model-cache-dir", help="Persistent HF cache, or parent of owned run caches with --cleanup-models")
    parser.add_argument("--cleanup-models", action="store_true", help="Use an isolated cache; delete each model's files only after all its requested splits succeed")
    parser.add_argument("--results-dir", default="results/baselines/zero_shot", help="Parent directory; each invocation creates a unique run folder")
    parser.add_argument("--resume-run", help="Existing run folder; rerun with identical model/evaluation options")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None, help="Enable per-example recovery inside --resume-run")
    parser.add_argument("--quantization", choices=["4bit", "none"], help="Override all selected configs; default = configured 4bit")
    parser.add_argument("--max-new-tokens", type=int, help="Override generation limit; config default is 192")
    parser.add_argument("--batch-size", type=int, help="Prompts per generation call; default config value or 1")
    parser.add_argument("--gpus", help="Optional evaluation replicas on allocated visible GPUs: auto or 0,1; omit for one GPU")
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    if args.list_models:
        for name, path in configs_by_name.items():
            print(f"{name:16s} {load(path)['model']['repo_id']}")
        return
    for option in ("examples", "examples_per_operation"):
        value = getattr(args, option)
        if value is not None and (value == 0 or value < -1):
            parser.error(f"--{option.replace('_', '-')} must be positive or -1")
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")

    device = select_device(announce=True)
    matrix_path = resolve(args.matrix_config)
    matrix = load(matrix_path)
    evaluation, cache = matrix.get("evaluation", {}), matrix.get("cache", {})
    split = args.split or str(evaluation.get("split", "both"))
    splits = ["test", "heldout"] if split == "both" else [split]
    per_op = int(evaluation.get("examples_per_operation", -1)) if args.examples_per_operation is None else args.examples_per_operation
    if args.examples is not None:
        per_op = -1
    total_examples = args.examples if args.examples is not None else -1
    auto_download = bool(cache.get("auto_download", True)) if args.auto_download is None else args.auto_download
    resume = bool(evaluation.get("resume", True)) if args.resume is None else args.resume
    continue_on_error = bool(evaluation.get("continue_on_error", True)) if args.continue_on_error is None else args.continue_on_error
    configs = [configs_by_name[name] for name in sorted(set(args.models or configs_by_name))]
    loaded = {path.stem: load(path) for path in configs}
    if not args.download_only and device.type == "cpu" and any(
            (args.quantization or cfg["model"].get("quantization", "4bit")) == "4bit" for cfg in loaded.values()):
        parser.error("4-bit baselines need a GPU allocation. Use --download-only on the login node, or --quantization none for CPU evaluation")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    data_paths = set()
    expected = {}
    for slug, cfg in loaded.items():
        for one_split in splits:
            value = cfg["data"]["test_file"] if one_split == "test" else cfg.get("evaluation", {}).get("challenge_file", cfg["data"].get("heldout_file"))
            path = resolve(value)
            data_paths.add(path)
            rows = [row for row in load_jsonl(path) if not row.is_control]
            seed = int(cfg.get("training", {}).get("seed", 20260919)) + 91
            expected[slug, one_split] = [row.example_id for row in stratified_sample(rows, total_examples, per_op, seed)]
    protocol = {
        "mode": "zero_shot", "models": [path.stem for path in configs], "splits": splits,
        "examples": total_examples, "examples_per_operation": per_op, "quantization_override": args.quantization,
        "max_new_tokens_override": args.max_new_tokens, "gpus": args.gpus,
        "cleanup_models": args.cleanup_models,
        "configs": {path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else str(path): digest(path) for path in configs},
        "matrix_config_sha256": digest(matrix_path),
        "data": {path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else str(path): digest(path) for path in sorted(data_paths)},
        "source": {path.relative_to(REPO_ROOT).as_posix(): digest(path) for folder in ("scripts", "src") for path in sorted((REPO_ROOT / folder).rglob("*.py"))},
    }
    if args.batch_size is not None:
        protocol["batch_size_override"] = args.batch_size
    if args.resume_run:
        run_dir = resolve(args.resume_run)
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["protocol"] != protocol:
            parser.error("--resume-run protocol/config/data/source differs; use the original options or start a new run")
        run_id = manifest["run_id"]
        cache_root = Path(manifest["cache_root"]) if manifest.get("cache_root") else None
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        job = os.environ.get("SLURM_JOB_ID", "local")
        job = "".join(char for char in job if char.isalnum() or char in "_-")
        run_id = f"{stamp}_{job}_{uuid.uuid4().hex[:8]}"
        run_dir = resolve(args.results_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        cache_parent = args.model_cache_dir or cache.get("cache_dir")
        if args.cleanup_models:
            cache_root = resolve(cache_parent or str(Path(os.environ.get("SLURM_TMPDIR", str(REPO_ROOT / ".local"))) / "baseline-model-cache"))
        else:
            cache_root = resolve(cache_parent) if cache_parent else None
        manifest = {"run_id": run_id, "protocol": protocol, "cache_root": str(cache_root) if cache_root else None,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "python": platform.python_version(), "versions": {name: version(name) for name in ("torch", "transformers", "huggingface_hub", "peft", "bitsandbytes")},
                    "models": {}}
    (run_dir / "configs").mkdir(exist_ok=True)
    for path in configs:
        (run_dir / "configs" / path.name).write_bytes(path.read_bytes())
    (run_dir / "configs" / "matrix.toml").write_bytes(matrix_path.read_bytes())
    manifest["status"] = "running"
    save_json(run_dir / "manifest.json", manifest)
    print(f"Run directory: {run_dir}", flush=True)
    stop = False
    interrupted = False
    fatal = False
    try:
        for config in configs:
            slug, cfg = config.stem, loaded[config.stem]
            repo_id = cfg["model"]["repo_id"]
            model_entry = manifest["models"].setdefault(slug, {"repo_id": repo_id, "splits": {}})
            # Revalidate finished reports before skipping a completed model.
            done = True
            if not args.download_only:
                for one_split in splits:
                    entry = model_entry["splits"].get(one_split, {})
                    try:
                        if entry.get("status") != "success":
                            done = False
                            continue
                        validate_report(run_dir / entry["report"], repo_id, one_split, expected[slug, one_split])
                    except (OSError, ValueError, KeyError):
                        done = False
                        entry["status"] = "invalid_report"
                if done:
                    print(f"Already complete: {slug}", flush=True)
                    # Cleanup may have been interrupted after saving the last report.
                    if args.cleanup_models and not model_entry.get("cache_deleted", False):
                        directory = cache_root / run_id / slug
                        if directory.exists():
                            cleanup_owned_cache(directory, cache_root, run_id, slug)
                        model_entry["cache_deleted"] = True
                    continue
            directory = owned_cache(cache_root, run_id, slug) if args.cleanup_models else None
            revision = str(cfg.get("cache", {}).get("revision", cache.get("revision", "main")))
            model_cache = directory / "hub" if directory else cache_root or cfg.get("cache", {}).get("cache_dir")
            model_entry["requested_revision"] = revision
            try:
                print(f"Resolve/download: {slug} ({repo_id})", flush=True)
                with isolated_download_cache(directory):
                    snapshot = ensure_local_model_path(repo_id, None if directory else cfg["model"].get("local_path"),
                        auto_download=auto_download, cache_dir=model_cache, revision=revision, use_registry=not args.cleanup_models)
                if directory and not snapshot.is_relative_to(directory.resolve()):
                    raise ValueError("Downloaded snapshot escaped the owned cache")
                model_entry.update({"snapshot_path": str(snapshot), "resolved_revision": snapshot.name})
            except Exception as exc:
                reason = str(exc).splitlines()[-1]
                print(f"SKIP {slug}: {reason}", flush=True)
                model_entry["download_status"] = "failed"
                for one_split in splits:
                    model_entry["splits"].setdefault(one_split, {}).update({"status": "download_failed", "error": reason})
                save_json(run_dir / "manifest.json", manifest)
                if not continue_on_error:
                    break
                continue
            model_entry["download_status"] = "success"
            save_json(run_dir / "manifest.json", manifest)
            if args.download_only:
                print(f"Cached: {snapshot}", flush=True)
                continue
            for one_split in splits:
                entry = model_entry["splits"].get(one_split, {})
                if entry.get("status") == "success":
                    continue
                model_dir = run_dir / "models" / slug
                model_dir.mkdir(parents=True, exist_ok=True)
                report_path = model_dir / f"{one_split}.json"
                command = [sys.executable, str(REPO_ROOT / "scripts/evaluate_direct.py"),
                           "--config", str(config), "--mode", "zero_shot", "--split", one_split,
                           "--local-model-path", str(snapshot), "--no-auto-download",
                           "--examples-per-operation", str(per_op), "--examples", str(total_examples),
                           "--resume" if resume else "--no-resume"]
                if args.quantization:
                    command += ["--quantization", args.quantization]
                if args.max_new_tokens:
                    command += ["--max-new-tokens", str(args.max_new_tokens)]
                if args.batch_size is not None:
                    command += ["--batch-size", str(args.batch_size)]
                if args.gpus:
                    split_index = command.index("--split")
                    del command[split_index:split_index + 2]
                    parallel_dir = model_dir / "parallel" / one_split
                    command = [sys.executable, str(REPO_ROOT / "scripts/evaluate_multi_gpu.py"),
                               "--kind", "direct", "--gpus", args.gpus, "--split", one_split,
                               "--output-dir", str(parallel_dir), "--", *command[2:]]
                else:
                    command += ["--results-dir", str(model_dir), "--report-path", str(report_path)]
                env = {**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1", "PYTHONUNBUFFERED": "1",
                       "PYTHONIOENCODING": "utf-8",
                       "LLM_CHARACTER_SKIP_MODEL_REGISTRY_WRITE": "1"}
                if directory:
                    env.update({"HF_HUB_CACHE": str(directory / "hub"), "HF_MODULES_CACHE": str(directory / "modules"),
                                "HF_HUB_DISABLE_XET": "1", "HF_XET_CACHE": str(directory / "xet"),
                                "XDG_CACHE_HOME": str(directory / "auxiliary")})
                print(f"Evaluate: {slug} {one_split} ({len(expected[slug, one_split])} prompts)", flush=True)
                entry = {"status": "running", "report": report_path.relative_to(run_dir).as_posix()}
                model_entry["splits"][one_split] = entry
                save_json(run_dir / "manifest.json", manifest)
                proc = subprocess.run(command, cwd=REPO_ROOT, env=env, text=True, encoding="utf-8", errors="replace", capture_output=True)
                (model_dir / f"{one_split}.log").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
                try:
                    if proc.returncode:
                        raise RuntimeError(concise_failure(proc.stderr, proc.stdout))
                    if args.gpus:
                        candidates = list(parallel_dir.rglob(f"{one_split}.json"))
                        if len(candidates) != 1:
                            raise ValueError("Expected exactly one merged parallel report")
                        report_path.write_bytes(candidates[0].read_bytes())
                    report = validate_report(report_path, repo_id, one_split, expected[slug, one_split])
                    entry.update({"status": "success", "example_count": report["example_count"],
                                  "exact_match_accuracy": report["exact_match_accuracy"],
                                  "average_seconds_per_prompt": report["runtime"].get("average_seconds_per_prompt")})
                    print(f"SUCCESS {slug} {one_split}: accuracy={report['exact_match_accuracy']:.4f} n={report['example_count']}", flush=True)
                except Exception as exc:
                    entry.update({"status": "failed", "error": str(exc)})
                    print(f"FAIL {config.stem} {one_split}: {exc}", flush=True)
                    stop = not continue_on_error
                save_json(run_dir / "manifest.json", manifest)
                if stop:
                    break
            success = all(model_entry["splits"].get(part, {}).get("status") == "success" for part in splits)
            if directory and success:
                cleanup_owned_cache(directory, cache_root, run_id, slug)
                model_entry["cache_deleted"] = True
                save_json(run_dir / "manifest.json", manifest)
                print(f"Deleted owned model cache: {directory}", flush=True)
            if stop:
                break
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted; model files retained for unfinished evaluations.", flush=True)
    except Exception as exc:
        fatal = True
        manifest["error"] = str(exc)
        print(f"Run failed: {exc}", flush=True)
    finally:
        success = not fatal and len(manifest["models"]) == len(configs) and all(
            model.get("download_status") == "success" if args.download_only else
            all(model["splits"].get(part, {}).get("status") == "success" for part in splits)
            and (not args.cleanup_models or model.get("cache_deleted", False))
            for model in manifest["models"].values())
        manifest["status"] = "interrupted" if interrupted else ("downloaded" if success and args.download_only else "success" if success else "partial_or_failed")
        manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        archive = export_results(run_dir, manifest)
        print(f"Run status: {manifest['status']}\nCOPY RESULTS DIRECTORY: {run_dir}\nCOPY RESULTS ZIP: {archive}", flush=True)
    if interrupted:
        raise SystemExit(130)
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

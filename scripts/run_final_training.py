"""Run either or both training stages and preserve a distinct paper-results bundle."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import tomllib
import uuid

ROOT = Path(__file__).resolve().parents[1]


def resolve(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["1", "2", "both"], default="both")
    parser.add_argument("--train-gpus", type=int, choices=[1, 2], default=1,
                        help="One GPU, or two torchrun workers on one SLURM node")
    parser.add_argument("--phase1-config", default="configs/experiments/qwen3_8b/tool_policy_final.toml")
    parser.add_argument("--phase2-config", default="configs/experiments/qwen3_8b/result_injection_final.toml")
    parser.add_argument("--batch-size", type=int, help="Override micro-batch for each selected stage")
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-vram-mib", type=int, default=0)
    parser.add_argument("--auto-download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-cache-dir")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--evaluate-pipeline", action=argparse.BooleanOptionalAction, default=True,
                        help="After both stages, evaluate actual predicted calls on test and heldout")
    parser.add_argument("--results-dir", default="results/final_training")
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + os.environ.get("SLURM_JOB_ID", "local") + "_" + uuid.uuid4().hex[:8]
    bundle = resolve(args.results_dir) / run_id
    bundle.mkdir(parents=True)
    started = time.perf_counter()
    print(f"RUN RESULTS DIRECTORY: {bundle}", flush=True)
    manifest = {"run_id": run_id, "arguments": vars(args), "status": "running", "stages": {}}
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    common = ["--max-vram-mib", str(args.max_vram_mib),
              "--auto-download" if args.auto_download else "--no-auto-download"]
    for option in ("batch_size", "global_batch_size", "gradient_accumulation_steps", "eval_batch_size", "seed", "model_cache_dir"):
        value = getattr(args, option)
        if value is not None:
            common.extend(["--" + option.replace("_", "-"), str(value)])
    if args.fresh:
        common.append("--fresh")
    stage_specs = {
        "1": (args.phase1_config, "train_tool_policy.py", "tool_policy_report.json", "adapter_dir"),
        "2": (args.phase2_config, "train_result_injector.py", "result_injection_report.json", "final_checkpoint"),
    }
    exported_configs = {}
    try:
        for stage in (["1", "2"] if args.stage == "both" else [args.stage]):
            config_name, script, report_name, weight_key = stage_specs[stage]
            config_path = resolve(config_name)
            cfg = tomllib.loads(config_path.read_text(encoding="utf-8"))
            launcher = ([sys.executable, "-m", "torch.distributed.run", "--standalone",
                         "--nnodes=1", "--nproc_per_node=2"] if args.train_gpus == 2 else [sys.executable])
            subprocess.run([*launcher, str(ROOT / "scripts" / script), "--config", str(config_path), *common],
                           cwd=ROOT, check=True)
            report = json.loads((resolve(cfg["output"]["results_dir"]) / report_name).read_text(encoding="utf-8"))
            if report.get("status") != "complete":
                raise RuntimeError(f"Stage {stage} has no complete report")
            source = Path(report["recovery_run_dir"]) / "artifacts" if stage == "1" else Path(report["final_checkpoint"]).parent
            destination = bundle / ("stage" + stage)
            shutil.copytree(source, destination)
            weights = destination / ("final_adapter" if stage == "1" else "result_injector.pt")
            # Bind evaluation to this bundle's immutable weights instead of the
            # compatibility output that another seed/run could overwrite.
            text = config_path.read_text(encoding="utf-8")
            text = re.sub(r"(?m)^" + weight_key + r"\s*=.*$",
                          lambda match: weight_key + " = " + json.dumps(str(weights)), text)
            text = re.sub(r'(?m)^results_dir\s*=.*$',
                          lambda match: "results_dir = " + json.dumps(str(bundle / "pipeline")), text)
            if args.seed is not None:
                text = re.sub(r'(?m)^seed\s*=.*$', lambda match: f"seed = {args.seed}", text, count=1)
            exported_config = bundle / ("phase" + stage + ".toml")
            exported_config.write_text(text, encoding="utf-8")
            exported_configs[stage] = exported_config
            manifest["stages"][stage] = {"artifacts": str(destination), "checkpoint": str(weights)}
            (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if args.stage == "both" and args.evaluate_pipeline:
            pipeline_args = [sys.executable, str(ROOT / "scripts/evaluate_pipeline.py"),
                             "--phase1-config", str(exported_configs["1"]), "--phase2-config", str(exported_configs["2"]),
                             "--split", "both", "--no-resume", "--auto-download" if args.auto_download else "--no-auto-download"]
            if args.model_cache_dir:
                pipeline_args.extend(["--model-cache-dir", args.model_cache_dir])
            subprocess.run(pipeline_args, cwd=ROOT, check=True)
            subprocess.run([sys.executable, str(ROOT / "scripts/export_metrics.py"),
                            "--results-root", str(bundle), "--output-dir", str(bundle / "paper_tables")], cwd=ROOT, check=True)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "partial_or_failed"
        manifest["error"] = str(exc)
        raise
    finally:
        manifest["current_session_wall_seconds_before_zip"] = time.perf_counter() - started
        (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if manifest["status"] == "complete":
            archive = shutil.make_archive(str(bundle), "zip", root_dir=bundle.parent, base_dir=bundle.name)
            print(f"COPY RESULTS DIRECTORY: {bundle}", flush=True)
            print(f"COPY RESULTS ZIP: {archive}", flush=True)
            print(f"TOTAL CURRENT SESSION TIME (ZIP creation excluded): {manifest['current_session_wall_seconds_before_zip']:.2f} seconds", flush=True)
        else:
            print(f"PARTIAL RESULTS DIRECTORY: {bundle}; training checkpoints can be resumed.", flush=True)


if __name__ == "__main__":
    main()

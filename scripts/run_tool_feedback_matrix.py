from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("qwen3_4b", "qwen3_8b", "phi4_mini", "hermes3_8b")


def main():
    parser = argparse.ArgumentParser(description="Run the off-the-shelf function-calling comparison matrix.")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--split", choices=["test", "heldout", "both"], default="both")
    parser.add_argument("--examples-per-operation", type=int, default=-1)
    parser.add_argument("--controls-per-category", type=int, default=-1)
    parser.add_argument("--mask-probability", type=float, default=0.0)
    parser.add_argument("--auto-download", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--with-lora-text", action="store_true", help="Also evaluate each Qwen's trained selector with text feedback; requires its adapter.")
    parser.add_argument("--gpus", help="Use evaluate_multi_gpu.py: auto or visible indices such as 0,1")
    args = parser.parse_args()
    failures = 0
    runs = [(model, False) for model in args.models]
    if args.with_lora_text:
        runs += [(model, True) for model in args.models if model.startswith("qwen3_")]
    for model, lora in runs:
        command = [sys.executable, str(ROOT / "scripts/evaluate_tool_feedback.py"),
                   "--config", str(ROOT / f"configs/baselines/tools/{model}.toml"),
                   "--split", args.split, "--examples-per-operation", str(args.examples_per_operation),
                   "--controls-per-category", str(args.controls_per_category),
                   "--auto-download" if args.auto_download else "--no-auto-download"]
        if lora:
            command += ["--policy-config", str(ROOT / f"configs/experiments/{model}/tool_policy.toml")]
        else:
            command += ["--mask-probability", str(args.mask_probability)]
        if args.model_cache_dir:
            command += ["--model-cache-dir", args.model_cache_dir]
        if args.preflight:
            command += ["--preflight"]
        if args.gpus and not args.preflight:
            # The launcher owns --split and partitions after the normal sampling.
            split_index = command.index("--split")
            del command[split_index:split_index + 2]
            command = [sys.executable, str(ROOT / "scripts/evaluate_multi_gpu.py"),
                       "--kind", "tool-feedback", "--gpus", args.gpus, "--split", args.split, "--", *command[2:]]
        print(f"Running {model}: {'trained selector + text' if lora else 'off-the-shelf tools'}", flush=True)
        proc = subprocess.run(command, cwd=ROOT)
        if proc.returncode:
            failures += 1
            print(f"FAILED {model} ({'lora_text' if lora else 'native'}), exit={proc.returncode}", flush=True)
    print(f"Finished: {len(runs) - failures}/{len(runs)} {'preflights' if args.preflight else 'evaluations'} succeeded.")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

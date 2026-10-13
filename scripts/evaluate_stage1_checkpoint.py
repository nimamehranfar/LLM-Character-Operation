"""Evaluate a saved policy checkpoint; print only final accuracy and result path."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/qwen3_8b/tool_policy_final.toml")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open("rb") as handle:
        cfg = tomllib.load(handle)
    cfg["evaluation"]["batch_size"] = args.batch_size
    cfg["evaluation"]["dev_examples"] = -1
    checkpoint = args.checkpoint
    if checkpoint is None:
        candidates = list((ROOT / cfg["output"]["checkpoint_dir"]).rglob("policy_best.pt"))
        if not candidates:
            raise FileNotFoundError("No policy_best.pt checkpoint found")
        checkpoint = max(candidates, key=lambda p: p.stat().st_mtime)
    elif not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = checkpoint.parent.parent / "evaluation" / f"stage1_{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    output = directory / "report.json"
    diagnostics = directory / "diagnostics.log"
    try:
        with diagnostics.open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
            import torch
            from scripts.train_tool_policy import load_model_and_tokenizer, evaluate_policy
            from src.data.character_dataset import load_jsonl

            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            _, tokenizer, model = load_model_and_tokenizer(
                cfg, saved["trainable_state_dict"], auto_download=True)
            report = {"checkpoint": str(checkpoint), "epoch": saved["epoch"],
                      "batch_size": args.batch_size, "splits": {}}
            for split, filename in (("test", cfg["data"]["test_file"]),
                                    ("heldout", cfg["evaluation"]["challenge_file"])):
                report["splits"][split] = evaluate_policy(
                    model, tokenizer, load_jsonl(ROOT / filename), cfg,
                    seed=int(cfg["training"]["seed"]), epoch=0)
                output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except Exception:
        print(f"Evaluation failed. Diagnostics: {diagnostics}", file=sys.stderr)
        raise
    for split, metrics in report["splits"].items():
        print(f"{split}: {100 * metrics['full_call_exact_accuracy']:.2f}%")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()

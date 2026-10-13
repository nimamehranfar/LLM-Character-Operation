"""Evaluate a saved stage-2 mapper with oracle tool results; print final percentages."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def checkpoint_weights_and_layer(saved: dict) -> tuple[dict, int]:
    """Read either a recovery best checkpoint or the final exported mapper."""
    if "trainable_state_dict" in saved:
        return saved["trainable_state_dict"], int(saved["metrics"]["selected_layer"])
    if "mapper_state_dict" in saved:
        return saved["mapper_state_dict"], int(saved["selected_layer"])
    raise ValueError("Not a stage-2 best checkpoint or final mapper export")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/qwen3_8b/result_injection_final.toml")
    parser.add_argument("--checkpoint", type=Path,
                        help="Stage-2 best checkpoint or final result_injector.pt; default: latest best")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open("rb") as handle:
        cfg = tomllib.load(handle)
    checkpoint = args.checkpoint
    if checkpoint is None:
        candidates = list((ROOT / cfg["output"]["checkpoint_dir"]).rglob("phase2_layered_best.pt"))
        if not candidates:
            raise FileNotFoundError("No phase2_layered_best.pt checkpoint found")
        checkpoint = max(candidates, key=lambda p: p.stat().st_mtime)
    elif not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = checkpoint.parent.parent / "evaluation" / f"stage2_{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    output = directory / "report.json"
    diagnostics = directory / "diagnostics.log"
    try:
        with diagnostics.open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
            import torch
            from scripts.train_result_injector import scalar_rows
            from src.executor.operations import CharacterExecutor
            from src.model.local_model import load_local_causal_lm
            from src.model.result_injector import LayeredSymbolicResultMapper
            from src.evaluation.result_analysis import evaluate_oracle_generation

            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            weights, layer = checkpoint_weights_and_layer(saved)
            arch = cfg["architecture"]
            layers = [int(x) for x in arch["candidate_layers"]]
            if layer not in layers:
                raise ValueError(f"Saved layer {layer} is absent from config candidate layers {layers}")
            cache = cfg.get("cache", {})
            _, tokenizer, model = load_local_causal_lm(
                cfg["model"]["repo_id"], cfg["model"].get("local_path"),
                quantization=cfg["model"]["quantization"], auto_download=True,
                cache_dir=cache.get("cache_dir"), revision=str(cache.get("revision", "main")))
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            model.eval()
            mapper = LayeredSymbolicResultMapper(
                hidden_size=int(model.config.hidden_size), candidate_layers=layers,
                result_dim=int(arch["result_dim"]), gate_dim=int(arch["gate_dim"]),
                max_integer=int(arch["max_integer_result"]),
                ascii_vocab_size=int(arch["ascii_vocab_size"]),
                gate_bias_init=float(arch["gate_bias_init"])).to(next(model.parameters()).device)
            mapper.load_state_dict(weights, strict=True)
            mapper.eval()
            executor = CharacterExecutor()
            report = {
                "checkpoint": str(checkpoint), "epoch": saved.get("epoch"),
                "selected_layer_from_dev": layer, "batch_size": args.batch_size,
                "evaluation_type": "oracle_executor_result; stage-2 reintegration only; not end-to-end",
                "splits": {},
            }
            for split, filename in (("test", cfg["data"]["test_file"]),
                                    ("heldout", cfg["evaluation"]["challenge_file"])):
                rows = scalar_rows(ROOT / filename)
                common = dict(
                    model=model, tokenizer=tokenizer, examples=rows, executor=executor,
                    max_new_tokens=int(cfg["evaluation"]["max_new_tokens"]),
                    max_integer_result=int(arch["max_integer_result"]),
                    ascii_vocab_size=int(arch["ascii_vocab_size"]),
                    save_per_example=True, batch_size=args.batch_size,
                )
                report["splits"][split] = {
                    "without_mapper": evaluate_oracle_generation(mapper=None, layer_index=None, **common),
                    "with_mapper": evaluate_oracle_generation(mapper=mapper, layer_index=layer, **common),
                }
                output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except Exception:
        print(f"Evaluation failed. Diagnostics: {diagnostics}", file=sys.stderr)
        raise
    for split, metrics in report["splits"].items():
        print(f"{split}: mapper {100 * metrics['with_mapper']['exact_match_accuracy']:.2f}% | "
              f"without mapper {100 * metrics['without_mapper']['exact_match_accuracy']:.2f}%")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()

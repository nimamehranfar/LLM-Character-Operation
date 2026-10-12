from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
import tomllib
from typing import Any

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.character_dataset import SCALAR_RESULT_OPERATIONS, load_jsonl
from src.evaluation.result_analysis import evaluate_oracle_generation, summarize_best_any_layer
from src.executor.operations import CharacterExecutor
from src.model.local_model import load_local_causal_lm
from src.model.result_injector import (
    LayeredSymbolicResultMapper,
    TrainableLayeredResultInjector,
    result_to_symbol,
)
from src.training.recovery import RecoveryManager, capture_rng_state, restore_rng_state
from src.training.result_injection_data import prepare_training_tensors

SCRIPT_VERSION = "layered_result_injection_v2_batched"


def resolve_path(v: str | Path) -> Path:
    p = Path(v)
    return p if p.is_absolute() else REPO_ROOT / p


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as h:
        return tomllib.load(h)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scalar_rows(path: Path):
    return [x for x in load_jsonl(path) if x.operation in SCALAR_RESULT_OPERATIONS]


def take(rows, n: int, seed: int):
    rows = list(rows)
    if n < 0 or n >= len(rows):
        return rows
    return random.Random(seed).sample(rows, n)


def state_cpu(module):
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def candidate_layers(arch: dict[str, Any]) -> list[int]:
    values = arch.get("candidate_layers")
    if values is None:
        values = [int(arch["layer_index"])]
    layers = [int(x) for x in values]
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("architecture.candidate_layers must be a non-empty unique list")
    return layers


def layerwise_eval(model, tokenizer, mapper, rows, executor, layers, cfg, *, save_per_example: bool):
    arch = cfg["architecture"]
    out = []
    for layer in layers:
        metrics = evaluate_oracle_generation(
            model,
            tokenizer,
            mapper,
            rows,
            executor,
            layer_index=layer,
            max_new_tokens=int(cfg["evaluation"]["max_new_tokens"]),
            max_integer_result=int(arch["max_integer_result"]),
            ascii_vocab_size=int(arch["ascii_vocab_size"]),
            save_per_example=save_per_example,
            batch_size=int(cfg["evaluation"].get("batch_size", 1)),
        )
        out.append({"layer": int(layer), "metrics": metrics})
    return out


def select_layer(layer_reports: list[dict[str, Any]]) -> int:
    best = max(
        layer_reports,
        key=lambda row: (float(row["metrics"]["exact_match_accuracy"]), -int(row["layer"])),
    )
    return int(best["layer"])


def main() -> None:
    from src.training.engine import add_training_arguments, apply_training_arguments, train_single_gpu
    from src.training.reporting import export_training_artifacts, run_manifest
    from src.training.result_injection_data import prepare_training_batch
    from src.training.recovery import atomic_torch_save

    ap = argparse.ArgumentParser(description="Train the multi-layer symbolic result injector on one GPU.")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/experiments/qwen3_8b/result_injection.toml"))
    ap.add_argument("--auto-download", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--model-cache-dir", default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--max-vram-mib", type=int)
    add_training_arguments(ap)
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    apply_training_arguments(cfg, args)
    if args.max_vram_mib is not None:
        cfg["training"]["max_vram_mib"] = args.max_vram_mib
    seed = int(cfg["training"]["seed"])
    set_seed(seed)
    root = resolve_path(cfg["output"]["checkpoint_dir"])
    if args.status:
        RecoveryManager.print_status(checkpoint_root=root, config=cfg, script_version=SCRIPT_VERSION)
        return
    manager = RecoveryManager.create_or_resume(
        checkpoint_root=root, config=cfg, config_path=str(cfg_path), script_version=SCRIPT_VERSION,
        heartbeat_seconds=float(cfg["recovery"]["heartbeat_seconds"]), fresh=bool(args.fresh),
        explicit_run_dir=Path(args.run_dir).resolve() if args.run_dir else None,
    )
    try:
        from scripts.train_tool_policy import configure_vram_limit
        configure_vram_limit(int(cfg["training"].get("max_vram_mib", 7600)))
        paths = {split: resolve_path(cfg["data"][f"{split}_file"]) for split in ("train", "dev", "test")}
        paths["heldout"] = resolve_path(cfg["evaluation"].get("challenge_file", "data/character_operations_50k/heldout.jsonl"))
        train = take(scalar_rows(paths["train"]), int(cfg["data"]["train_examples"]), seed + 1)
        dev = take(scalar_rows(paths["dev"]), int(cfg["data"]["dev_examples"]), seed + 2)
        test = take(scalar_rows(paths["test"]), int(cfg["data"]["test_examples"]), seed + 3)
        heldout = scalar_rows(paths["heldout"])
        loading_started = time.perf_counter()
        cache = cfg.get("cache", {})
        model_path, tokenizer, model = load_local_causal_lm(
            cfg["model"]["repo_id"], cfg["model"].get("local_path"), quantization=cfg["model"]["quantization"],
            auto_download=bool(cache.get("auto_download", False)) if args.auto_download is None else args.auto_download,
            cache_dir=cache.get("cache_dir") if args.model_cache_dir is None else args.model_cache_dir,
            revision=str(cache.get("revision", "main")))
        loading_seconds = time.perf_counter() - loading_started
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        model.config.use_cache = False
        # Frozen layers before the injection do not require backward activations;
        # layers after it propagate the loss to the trainable mapper.
        device = next(model.parameters()).device
        executor = CharacterExecutor()
        arch, tr = cfg["architecture"], cfg["training"]
        layers = candidate_layers(arch)
        if any(not 0 <= layer < int(model.config.num_hidden_layers) for layer in layers):
            raise ValueError("Candidate injection layer is outside the model")
        mapper = LayeredSymbolicResultMapper(
            hidden_size=int(model.config.hidden_size), candidate_layers=layers,
            result_dim=int(arch["result_dim"]), gate_dim=int(arch["gate_dim"]),
            max_integer=int(arch["max_integer_result"]), ascii_vocab_size=int(arch["ascii_vocab_size"]),
            gate_bias_init=float(arch["gate_bias_init"])).to(device)
        optimizer = torch.optim.AdamW(
            mapper.parameters(), lr=float(tr["learning_rate"]), weight_decay=float(tr["weight_decay"]),
            betas=(float(tr.get("adam_beta1", .9)), float(tr.get("adam_beta2", .999))),
            eps=float(tr.get("adam_epsilon", 1e-8)))
        max_length = int(tr["max_sequence_length"])
        # Compute lengths once, and validate every answer before starting a long job.
        lengths = [prepare_training_tensors(tokenizer, ex.prompt, ex.expected,
                   max_sequence_length=max_length, device=torch.device("cpu"))[0]["input_ids"].numel()
                   for ex in train]

        def prepare(indices):
            examples = [train[i] for i in indices]
            tensors, positions, target_tokens = prepare_training_batch(
                tokenizer, examples, max_sequence_length=max_length, device=torch.device("cpu"))
            symbols = [result_to_symbol(ex.execute(executor), max_integer=int(arch["max_integer_result"]),
                       ascii_vocab_size=int(arch["ascii_vocab_size"])) for ex in examples]
            return {"tensors": tensors, "positions": positions, "symbols": symbols,
                    "target_tokens": target_tokens, "count": len(examples)}

        def forward(batch, micro_step):
            layer = layers[micro_step % len(layers)]
            tensors = {key: value.to(device) for key, value in batch["tensors"].items()}
            with TrainableLayeredResultInjector(model, mapper, layer_index=layer, symbol=batch["symbols"],
                                               position_index=batch["positions"], prefill_only=False) as injector:
                output = model(**tensors, use_cache=False)
            return (output.loss, batch["target_tokens"], int(tensors["attention_mask"].sum()),
                    batch["count"], {"layer": layer, "gate": injector.last_gate})

        dev_eval = take(dev, int(tr.get("layer_eval_examples", len(dev))), seed + 8001)
        def validate():
            reports = layerwise_eval(model, tokenizer, mapper, dev_eval, executor, layers, cfg, save_per_example=False)
            selected = select_layer(reports)
            loss_sum, token_count = 0.0, 0
            size = int(cfg["evaluation"].get("batch_size", 1))
            with torch.no_grad():
                for start in range(0, len(dev_eval), size):
                    examples = dev_eval[start:start + size]
                    tensors, positions, tokens = prepare_training_batch(
                        tokenizer, examples, max_sequence_length=max_length, device=device)
                    symbols = [result_to_symbol(ex.execute(executor), max_integer=int(arch["max_integer_result"]),
                               ascii_vocab_size=int(arch["ascii_vocab_size"])) for ex in examples]
                    with TrainableLayeredResultInjector(model, mapper, layer_index=selected, symbol=symbols,
                                                       position_index=positions, prefill_only=False):
                        output = model(**tensors, use_cache=False)
                    loss_sum += float(output.loss) * tokens
                    token_count += tokens
            return {"best_accuracy": max(row["metrics"]["exact_match_accuracy"] for row in reports),
                    "selected_layer": selected, "layerwise": reports,
                    "loss": loss_sum / max(1, token_count)}

        latest = manager.load_checkpoint("phase2_layered_latest.pt")
        trained = train_single_gpu(
            module=mapper, optimizer=optimizer, manager=manager, cfg=cfg, rows=train, lengths=lengths,
            prepare=prepare, forward=forward, evaluate=validate, state_cpu=lambda: state_cpu(mapper),
            restore=mapper.load_state_dict, checkpoint_prefix="phase2_layered", latest=latest, metric_key="best_accuracy")
        mapper.eval()
        final_started = time.perf_counter()
        dev_layerwise = layerwise_eval(model, tokenizer, mapper, dev, executor, layers, cfg, save_per_example=True)
        selected_layer = select_layer(dev_layerwise)
        test_layerwise = layerwise_eval(model, tokenizer, mapper, test, executor, layers, cfg, save_per_example=True)
        heldout_layerwise = layerwise_eval(model, tokenizer, mapper, heldout, executor, layers, cfg, save_per_example=True)

        def baseline(rows):
            return evaluate_oracle_generation(
                model, tokenizer, None, rows, executor, layer_index=None,
                max_new_tokens=int(cfg["evaluation"]["max_new_tokens"]),
                max_integer_result=int(arch["max_integer_result"]), ascii_vocab_size=int(arch["ascii_vocab_size"]),
                batch_size=int(cfg["evaluation"].get("batch_size", 1)), save_per_example=True)

        baseline_test, baseline_heldout = baseline(test), baseline(heldout)
        final_evaluation_seconds = time.perf_counter() - final_started
        payload = {
            "architecture_type": "layered_symbolic_result_mapper", "candidate_layers": layers,
            "selected_layer": selected_layer, "layer_index": selected_layer, "mapper_state_dict": state_cpu(mapper),
            "architecture": dict(arch), "model_repo_id": cfg["model"]["repo_id"],
            "supported_result_kinds": ["integer", "character"], "supported_operations": sorted(SCALAR_RESULT_OPERATIONS),
            "selection_rule": "highest exact match on dev only; tie -> earliest layer",
        }
        immutable_checkpoint = manager.paths.artifacts / "result_injector.pt"
        final_path = resolve_path(cfg["output"]["final_checkpoint"])
        atomic_torch_save(immutable_checkpoint, payload)
        atomic_torch_save(final_path, payload)
        manifest = run_manifest(
            config_path=cfg_path, data_paths=list(paths.values()),
            training={**tr, "batch_plan": trained["batch_plan"], "architecture": arch,
                      "evaluation": cfg["evaluation"], "model": cfg["model"], "cache": cache},
            trainable_parameters=sum(p.numel() for p in mapper.parameters()),
            total_parameters=model.num_parameters() + sum(p.numel() for p in mapper.parameters()))
        selected = lambda reports: next(x["metrics"] for x in reports if x["layer"] == selected_layer)
        report = {
            "experiment": SCRIPT_VERSION, "status": "complete", "model": cfg["model"]["repo_id"],
            "model_path": str(model_path), "candidate_layers": layers, "selected_layer": selected_layer,
            "layer_selection_rule": "dev-only oracle-result exact match; test and heldout never select",
            "train_count": len(train), "dev_count": len(dev), "test_count": len(test), "heldout_count": len(heldout),
            "batch_plan": trained["batch_plan"], "optimizer_steps": trained["optimizer_steps"],
            "best_epoch": trained["best"]["epoch"], "history": trained["history"],
            "baseline_no_injection_test": baseline_test, "baseline_no_injection_heldout": baseline_heldout,
            "dev_layerwise_oracle_phase1": dev_layerwise, "test_layerwise_oracle_phase1": test_layerwise,
            "heldout_layerwise_oracle_phase1": heldout_layerwise,
            "selected_dev_oracle_phase1": selected(dev_layerwise), "selected_test_oracle_phase1": selected(test_layerwise),
            "selected_heldout_oracle_phase1": selected(heldout_layerwise),
            "oracle_best_any_layer_test": summarize_best_any_layer(test_layerwise),
            "oracle_best_any_layer_heldout": summarize_best_any_layer(heldout_layerwise),
            "runtime": {"resolve_download_and_model_loading_seconds": loading_seconds,
                        **model.training_setup_timing,
                        "batch_preparation_seconds": trained["batch_preparation_seconds"],
                        "train_loop_wall_seconds_current_session": trained["train_loop_wall_seconds_current_session"],
                        "training_seconds": trained["training_seconds"], "validation_seconds": trained["validation_seconds"],
                        "checkpoint_seconds": trained["checkpoint_seconds"], "final_evaluation_seconds": final_evaluation_seconds,
                        "active_total_seconds": manager.active_elapsed_seconds(),
                        "wall_total_seconds_including_interruptions": manager.wall_elapsed_seconds(),
                        "peak_allocated_vram_mib": max(trained["peak_allocated_vram_mib"], torch.cuda.max_memory_allocated() / 1024**2),
                        "peak_reserved_vram_mib": max(trained["peak_reserved_vram_mib"], torch.cuda.max_memory_reserved() / 1024**2)},
            "final_checkpoint": str(immutable_checkpoint), "compatibility_checkpoint": str(final_path),
            "note": "All mapper evaluations here use oracle executor results to isolate reintegration. "
                    "End-to-end evaluation must separately use the trained policy's predicted calls.",
        }
        export_training_artifacts(manager.paths.artifacts, step_rows=trained["step_rows"],
                                  epoch_rows=trained["history"], manifest=manifest, report=report)
        report_path = resolve_path(cfg["output"]["results_dir"]) / "result_injection_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        manager.mark_complete(report=str(manager.paths.artifacts / "training_report.json"), checkpoint=str(immutable_checkpoint))
        print(f"Selected layer from dev only: {selected_layer}")
        print(f"COPY RESULTS DIRECTORY: {manager.paths.artifacts}")
    except BaseException as exc:
        manager.handle_exception(exc)
        raise
    finally:
        manager.close()


if __name__ == "__main__":
    main()

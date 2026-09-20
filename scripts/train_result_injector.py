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

SCRIPT_VERSION = "layered_result_injection"


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
    ap = argparse.ArgumentParser(description="Train the multi-layer symbolic result injector.")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/experiments/qwen3_8b/result_injection.toml"))
    ap.add_argument("--auto-download", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--model-cache-dir", default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    seed = int(cfg["training"]["seed"])
    set_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    props = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, int(cfg["training"].get("max_vram_mib", 7600)) / (props.total_memory / 1024**2)), 0
    )
    torch.cuda.reset_peak_memory_stats()

    root = resolve_path(cfg["output"]["checkpoint_dir"])
    if args.status:
        RecoveryManager.print_status(checkpoint_root=root, config=cfg, script_version=SCRIPT_VERSION)
        return
    manager = RecoveryManager.create_or_resume(
        checkpoint_root=root,
        config=cfg,
        config_path=str(cfg_path),
        script_version=SCRIPT_VERSION,
        heartbeat_seconds=float(cfg["recovery"]["heartbeat_seconds"]),
        fresh=bool(args.fresh),
        explicit_run_dir=Path(args.run_dir).resolve() if args.run_dir else None,
    )

    try:
        train = take(scalar_rows(resolve_path(cfg["data"]["train_file"])), int(cfg["data"]["train_examples"]), seed + 1)
        dev = take(scalar_rows(resolve_path(cfg["data"]["dev_file"])), int(cfg["data"]["dev_examples"]), seed + 2)
        test = take(scalar_rows(resolve_path(cfg["data"]["test_file"])), int(cfg["data"]["test_examples"]), seed + 3)
        cache_cfg = cfg.get("cache", {})
        effective_auto = bool(cache_cfg.get("auto_download", False)) if args.auto_download is None else bool(args.auto_download)
        effective_cache = cache_cfg.get("cache_dir") if args.model_cache_dir is None else args.model_cache_dir
        model_path, tokenizer, model = load_local_causal_lm(
            cfg["model"]["repo_id"], cfg["model"].get("local_path"),
            quantization=cfg["model"]["quantization"], auto_download=effective_auto,
            cache_dir=effective_cache, revision=str(cache_cfg.get("revision", "main")),
        )
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        device = next(model.parameters()).device
        executor = CharacterExecutor()
        hidden = int(model.config.hidden_size)
        arch = cfg["architecture"]
        tr = cfg["training"]
        layers = candidate_layers(arch)
        num_hidden_layers = int(model.config.num_hidden_layers)
        for layer in layers:
            if not 0 <= layer < num_hidden_layers:
                raise ValueError(f"Candidate layer {layer} is outside model range 0..{num_hidden_layers-1}")

        mapper = LayeredSymbolicResultMapper(
            hidden_size=hidden,
            candidate_layers=layers,
            result_dim=int(arch["result_dim"]),
            gate_dim=int(arch["gate_dim"]),
            max_integer=int(arch["max_integer_result"]),
            ascii_vocab_size=int(arch["ascii_vocab_size"]),
            gate_bias_init=float(arch["gate_bias_init"]),
        ).to(device)
        opt = torch.optim.AdamW(
            mapper.parameters(), lr=float(tr["learning_rate"]), weight_decay=float(tr["weight_decay"])
        )

        latest = manager.load_checkpoint("phase2_layered_latest.pt")
        if latest:
            mapper.load_state_dict(latest["mapper_state_dict"])
            opt.load_state_dict(latest["optimizer_state_dict"])
            restore_rng_state(latest.get("rng_state", {}))
            epoch = int(latest["epoch"])
            cursor = int(latest["cursor"])
            order = list(latest["order"])
            loss_sum = float(latest.get("loss_sum", 0.0))
            loss_count = int(latest.get("loss_count", 0))
            best = float(latest.get("best_metric", -1.0))
            stale = int(latest.get("stale", 0))
            history = list(latest.get("history", []))
            epoch_active = float(latest.get("epoch_active_seconds", 0.0))
            layer_stats = dict(latest.get("layer_stats", {}))
        else:
            epoch, cursor, order = 1, 0, []
            loss_sum, loss_count, best, stale = 0.0, 0, -1.0, 0
            history, epoch_active, layer_stats = [], 0.0, {}

        segment = time.perf_counter()
        save_every = int(cfg["recovery"]["checkpoint_every_examples"])

        def active() -> float:
            return epoch_active + (time.perf_counter() - segment)

        def save_latest() -> None:
            manager.checkpoint(
                "phase2_layered_latest.pt",
                {
                    "mapper_state_dict": state_cpu(mapper),
                    "optimizer_state_dict": opt.state_dict(),
                    "rng_state": capture_rng_state(),
                    "epoch": epoch,
                    "cursor": cursor,
                    "order": order,
                    "loss_sum": loss_sum,
                    "loss_count": loss_count,
                    "best_metric": best,
                    "stale": stale,
                    "history": history,
                    "epoch_active_seconds": active(),
                    "layer_stats": layer_stats,
                    "candidate_layers": layers,
                },
            )

        manager.set_emergency_saver(save_latest)
        max_len = int(tr["max_sequence_length"])
        clip = float(tr["gradient_clip_norm"])
        epochs = int(tr["epochs"])
        patience = int(tr["early_stopping_patience"])
        layer_eval_n = int(tr.get("layer_eval_examples", min(400, len(dev))))
        dev_eval = take(dev, layer_eval_n, seed + 8001)

        while epoch <= epochs:
            if not order:
                order = list(range(len(train)))
                random.Random(seed + epoch * 1013).shuffle(order)
                layer_stats = {
                    str(layer): {"count": 0, "loss_sum": 0.0, "gate_sum": 0.0, "gate_count": 0}
                    for layer in layers
                }
            mapper.train()
            bar = tqdm(total=len(order), initial=cursor, desc=f"result injection epoch {epoch}/{epochs}", unit="ex", dynamic_ncols=True)
            while cursor < len(order):
                ex = train[order[cursor]]
                # Deterministic balanced assignment: one training pass per example,
                # rotating across every accepted candidate layer rather than hard-coding one.
                layer = layers[(cursor + epoch - 1) % len(layers)]
                result = ex.execute(executor)
                symbol = result_to_symbol(
                    result,
                    max_integer=int(arch["max_integer_result"]),
                    ascii_vocab_size=int(arch["ascii_vocab_size"]),
                )
                tensors, pos = prepare_training_tensors(
                    tokenizer, ex.prompt, ex.expected, max_sequence_length=max_len, device=device
                )
                opt.zero_grad(set_to_none=True)
                with TrainableLayeredResultInjector(
                    model,
                    mapper,
                    layer_index=layer,
                    symbol=symbol,
                    position_index=pos,
                    prefill_only=False,
                ) as inj:
                    out = model(**tensors, use_cache=False)
                    loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mapper.parameters(), clip)
                opt.step()

                value = float(loss.detach().item())
                loss_sum += value
                loss_count += 1
                stat = layer_stats[str(layer)]
                stat["count"] += 1
                stat["loss_sum"] += value
                if inj.last_gate is not None:
                    stat["gate_sum"] += float(inj.last_gate)
                    stat["gate_count"] += 1
                cursor += 1
                bar.update(1)
                bar.set_postfix(layer=layer, loss=f"{loss_sum/max(1,loss_count):.4f}")
                if cursor % save_every == 0:
                    save_latest()
                    manager.progress(
                        phase="result_injection",
                        stage="training",
                        epoch=epoch,
                        cursor=cursor,
                        total=len(order),
                        latest_checkpoint=str(manager.paths.checkpoints / "phase2_layered_latest.pt"),
                    )
                    manager.check_stop(saver=save_latest)
            bar.close()
            mapper.eval()

            dev_layerwise = layerwise_eval(
                model, tokenizer, mapper, dev_eval, executor, layers, cfg, save_per_example=False
            )
            metric = max(float(row["metrics"]["exact_match_accuracy"]) for row in dev_layerwise)
            epoch_selected = select_layer(dev_layerwise)
            improved = metric > best
            if improved:
                best = metric
                stale = 0
                manager.checkpoint(
                    "phase2_layered_best.pt",
                    {
                        "mapper_state_dict": state_cpu(mapper),
                        "epoch": epoch,
                        "dev_layerwise": dev_layerwise,
                        "selected_layer": epoch_selected,
                        "best_metric": best,
                    },
                )
            else:
                stale += 1

            train_by_layer = {}
            for layer in layers:
                stat = layer_stats[str(layer)]
                train_by_layer[str(layer)] = {
                    "count": int(stat["count"]),
                    "mean_loss": stat["loss_sum"] / max(1, stat["count"]),
                    "mean_gate": stat["gate_sum"] / max(1, stat["gate_count"]) if stat["gate_count"] else None,
                }
            history.append({
                "epoch": epoch,
                "mean_loss": loss_sum / max(1, loss_count),
                "dev_layerwise": dev_layerwise,
                "dev_best_accuracy": metric,
                "dev_selected_layer": epoch_selected,
                "train_by_layer": train_by_layer,
                "active_seconds": active(),
                "best": improved,
            })
            print(
                f"epoch {epoch}: loss={history[-1]['mean_loss']:.4f} "
                f"best_dev_EM={metric:.4f} selected_layer={epoch_selected}"
            )
            epoch += 1
            cursor, order, loss_sum, loss_count, epoch_active = 0, [], 0.0, 0, 0.0
            segment = time.perf_counter()
            save_latest()
            if stale >= patience:
                break

        best_ck = manager.load_checkpoint("phase2_layered_best.pt")
        if best_ck is None:
            raise RuntimeError("No result-injection best checkpoint was produced")
        mapper.load_state_dict(best_ck["mapper_state_dict"])
        mapper.eval()

        # Full dev sweep selects the deployment layer. Test/heldout never choose it.
        dev_layerwise = layerwise_eval(
            model, tokenizer, mapper, dev, executor, layers, cfg, save_per_example=True
        )
        selected_layer = select_layer(dev_layerwise)
        test_layerwise = layerwise_eval(
            model, tokenizer, mapper, test, executor, layers, cfg, save_per_example=True
        )
        baseline_test = evaluate_oracle_generation(
            model,
            tokenizer,
            None,
            test,
            executor,
            layer_index=None,
            max_new_tokens=int(cfg["evaluation"]["max_new_tokens"]),
            max_integer_result=int(arch["max_integer_result"]),
            ascii_vocab_size=int(arch["ascii_vocab_size"]),
            save_per_example=True,
        )
        oracle_best_any = summarize_best_any_layer(test_layerwise)
        selected_test = next(x["metrics"] for x in test_layerwise if int(x["layer"]) == selected_layer)
        selected_dev = next(x["metrics"] for x in dev_layerwise if int(x["layer"]) == selected_layer)

        final_path = resolve_path(cfg["output"]["final_checkpoint"])
        final_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "architecture_type": "layered_symbolic_result_mapper",
                "candidate_layers": layers,
                "selected_layer": selected_layer,
                "layer_index": selected_layer,  # compatibility alias
                "mapper_state_dict": state_cpu(mapper),
                "architecture": dict(arch),
                "model_repo_id": cfg["model"]["repo_id"],
                "supported_result_kinds": ["integer", "character"],
                "supported_operations": sorted(SCALAR_RESULT_OPERATIONS),
                "selection_rule": "highest oracle-result final exact match on dev only; tie -> earliest layer",
            },
            final_path,
        )

        report = {
            "experiment": SCRIPT_VERSION,
            "status": "complete",
            "model": cfg["model"]["repo_id"],
            "model_path": str(model_path),
            "candidate_layers": layers,
            "selected_layer": selected_layer,
            "layer_selection_rule": "dev-only oracle-result final exact match; test never selects layer",
            "train_count": len(train),
            "dev_count": len(dev),
            "test_count": len(test),
            "baseline_no_injection_test": baseline_test,
            "dev_layerwise_oracle_phase1": dev_layerwise,
            "test_layerwise_oracle_phase1": test_layerwise,
            "selected_dev_oracle_phase1": selected_dev,
            "selected_test_oracle_phase1": selected_test,
            "oracle_best_any_layer_test": oracle_best_any,
            "history": history,
            "peak_allocated_vram_mib": torch.cuda.max_memory_allocated() / (1024**2),
            "peak_reserved_vram_mib": torch.cuda.max_memory_reserved() / (1024**2),
            "final_checkpoint": str(final_path),
            "note": (
                "Result injection is trained across all configured candidate layers. Oracle tool-policy metrics use the "
                "ground-truth operation/arguments/result only to isolate result reintegration. The normal end-to-end "
                "evaluator never receives oracle operation or arguments. Arbitrary-string transform results remain "
                "deterministic executor outputs and are reported separately."
            ),
        }
        rp = resolve_path(cfg["output"]["results_dir"]) / "result_injection_report.json"
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
        manager.mark_complete(report=str(rp), checkpoint=str(final_path))
        print(f"Selected injection layer from dev only: {selected_layer}")
        print(f"Saved: {rp}")
    except BaseException as exc:
        manager.handle_exception(exc)
        raise
    finally:
        manager.close()


if __name__ == "__main__":
    main()

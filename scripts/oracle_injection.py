from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Allow `python scripts/oracle_injection.py` from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.executor.operations import (  # noqa: E402
    CharacterExecutor,
    CharacterOperation,
    ExecutionRequest,
)
from src.model.oracle_injector import ResidualResultInjector  # noqa: E402


def mib(value: int) -> float:
    return value / (1024**2)


def parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def build_inputs(tokenizer, prompt: str):
    messages = [
        {
            "role": "user",
            "content": prompt + "\nAnswer with only the final value.",
        }
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(rendered, return_tensors="pt")
    return encoded, rendered


def get_next_token_stats(model, encoded, target_token_id: int) -> dict[str, object]:
    with torch.inference_mode():
        outputs = model(**encoded, use_cache=False)
        logits = outputs.logits[:, -1, :].float()
        probs = torch.softmax(logits, dim=-1)

    target_prob = probs[0, target_token_id].item()
    target_logit = logits[0, target_token_id].item()
    rank = int((logits[0] > target_logit).sum().item()) + 1

    top_probs, top_ids = torch.topk(probs[0], k=5)
    return {
        "target_probability": target_prob,
        "target_logit": target_logit,
        "target_rank": rank,
        "top_token_ids": [int(x) for x in top_ids.tolist()],
        "top_probabilities": [float(x) for x in top_probs.tolist()],
    }


def decode_token_list(tokenizer, token_ids: list[int]) -> list[str]:
    return [
        tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        for token_id in token_ids
    ]


def generate_answer(model, tokenizer, encoded, max_new_tokens: int) -> str:
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    continuation = generated[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(continuation, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Oracle deterministic execution + hidden-state residual injection diagnostic."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--text", default="strawberry")
    parser.add_argument("--character", default="r")
    parser.add_argument(
        "--layers",
        default="8,17,26,35",
        help="Comma-separated zero-based decoder layer indices.",
    )
    parser.add_argument(
        "--scales",
        default="0.25,0.5,1,2,4,8",
        help="Comma-separated residual injection strengths.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "oracle_injection",
        help="Directory where the JSON experiment report is saved.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. Run this on the RTX 4070 laptop.")

    if len(args.character) != 1:
        raise ValueError("--character must be exactly one character for COUNT_CHAR.")

    executor = CharacterExecutor()
    request = ExecutionRequest(
        operation=CharacterOperation.COUNT_CHAR,
        text=args.text,
        character=args.character,
    )
    exact_result = executor.execute(request)
    prompt = f'How many "{args.character}" characters are in "{args.text}"?'

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": "cuda:0"},
        dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    encoded_cpu, rendered_prompt = build_inputs(tokenizer, prompt)
    encoded = {name: tensor.to("cuda") for name, tensor in encoded_cpu.items()}

    target_ids = tokenizer.encode(str(exact_result), add_special_tokens=False)
    if not target_ids:
        raise RuntimeError("Exact result encoded to zero target tokens.")
    target_token_id = int(target_ids[0])

    baseline_stats = get_next_token_stats(model, encoded, target_token_id)
    baseline_stats["top_tokens"] = decode_token_list(
        tokenizer, baseline_stats["top_token_ids"]
    )
    baseline_answer = generate_answer(
        model, tokenizer, encoded, max_new_tokens=args.max_new_tokens
    )

    layers = parse_csv_ints(args.layers)
    scales = parse_csv_floats(args.scales)
    num_layers = int(model.config.num_hidden_layers)
    for layer_index in layers:
        if not 0 <= layer_index < num_layers:
            raise ValueError(
                f"Requested layer {layer_index}, but model has layers 0..{num_layers - 1}."
            )

    sweep: list[dict[str, object]] = []
    for layer_index in layers:
        for scale in scales:
            with ResidualResultInjector(
                model,
                tokenizer,
                layer_index=layer_index,
                result=exact_result,
                scale=scale,
            ):
                stats = get_next_token_stats(model, encoded, target_token_id)
            sweep.append(
                {
                    "layer_index": layer_index,
                    "scale": scale,
                    "target_probability": stats["target_probability"],
                    "target_logit": stats["target_logit"],
                    "target_rank": stats["target_rank"],
                    "top_tokens": decode_token_list(tokenizer, stats["top_token_ids"]),
                    "top_probabilities": stats["top_probabilities"],
                }
            )

    best = max(sweep, key=lambda row: float(row["target_probability"]))

    with ResidualResultInjector(
        model,
        tokenizer,
        layer_index=int(best["layer_index"]),
        result=exact_result,
        scale=float(best["scale"]),
    ):
        injected_answer = generate_answer(
            model, tokenizer, encoded, max_new_tokens=args.max_new_tokens
        )

    completed_at = datetime.now(timezone.utc)
    report = {
        "experiment": "oracle_residual_result_injection_v1",
        "completed_at_utc": completed_at.isoformat(),
        "model": args.model,
        "quantization": "4bit_nf4",
        "prompt": prompt,
        "rendered_prompt": rendered_prompt,
        "oracle_request": {
            "operation": request.operation.value,
            "text": request.text,
            "character": request.character,
        },
        "deterministic_result": exact_result,
        "target_token_id": target_token_id,
        "target_token_decoded": tokenizer.decode(
            [target_token_id], clean_up_tokenization_spaces=False
        ),
        "target_result_token_ids": [int(x) for x in target_ids],
        "baseline": {
            "greedy_answer": baseline_answer,
            **baseline_stats,
        },
        "best_injection": best,
        "injected_greedy_answer": injected_answer,
        "sweep": sweep,
        "cuda_peak_allocated_mib": round(
            mib(torch.cuda.max_memory_allocated()), 2
        ),
        "interpretation_note": (
            "This is an oracle interface diagnostic. The operation and arguments are "
            "provided correctly, and the sweep selects the strongest layer/scale on this "
            "example. It is not an end-to-end accuracy result and not the final injector."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = completed_at.strftime("%Y%m%dT%H%M%SZ")
    model_slug = args.model.replace("/", "--")
    output_path = args.output_dir / (
        f"oracle_residual_result_injection_v1_{model_slug}_{timestamp}.json"
    )
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Saved JSON report: {output_path}")
    print(
        "Baseline -> injected: "
        f"{baseline_answer!r} -> {injected_answer!r}; "
        f"best layer={best['layer_index']}, scale={best['scale']}, "
        f"target rank={baseline_stats['target_rank']} -> {best['target_rank']}"
    )


if __name__ == "__main__":
    main()

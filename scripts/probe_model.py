from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def mib(value: int) -> float:
    return value / (1024 ** 2)


def build_inputs(tokenizer, prompt: str):
    messages = [{"role": "user", "content": prompt + "\nAnswer with only the final value."}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(text, return_tensors="pt"), text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--prompt",
        default='How many "r" characters are in "strawberry"?',
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--quantization",
        choices=("4bit", "none"),
        default="4bit",
        help="Use NF4 4-bit quantization by default so Qwen3-4B fits comfortably in 8 GB VRAM.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. Run this in the RTX 4070 environment.")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model_kwargs = {
        "device_map": {"": "cuda:0"},
        "dtype": torch.bfloat16,
    }

    if args.quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()

    # The base transformer stays frozen. Only our future auxiliary modules train.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    encoded, rendered_prompt = build_inputs(tokenizer, args.prompt)
    encoded = {name: tensor.to("cuda") for name, tensor in encoded.items()}

    with torch.inference_mode():
        output = model(**encoded, output_hidden_states=True, use_cache=False)

    hidden_states = output.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden states.")

    expected_hidden_size = model.config.hidden_size
    last_shape = tuple(hidden_states[-1].shape)
    if last_shape[-1] != expected_hidden_size:
        raise RuntimeError(
            f"Hidden size mismatch: tensor={last_shape[-1]}, config={expected_hidden_size}"
        )

    # This is only a sanity-check generation, not yet an accuracy benchmark.
    # Greedy decoding makes this probe reproducible across runs.
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    continuation = generated[0, encoded["input_ids"].shape[1] :]
    answer = tokenizer.decode(continuation, skip_special_tokens=True)

    report = {
        "model": args.model,
        "quantization": args.quantization,
        "compute_dtype": "torch.bfloat16",
        "stored_parameter_elements_after_quantization": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "num_hidden_state_tensors": len(hidden_states),
        "last_hidden_state_shape": list(last_shape),
        "config_hidden_size": expected_hidden_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "cuda_allocated_mib": round(mib(torch.cuda.memory_allocated()), 2),
        "cuda_reserved_mib": round(mib(torch.cuda.memory_reserved()), 2),
        "cuda_peak_allocated_mib": round(mib(torch.cuda.max_memory_allocated()), 2),
        "prompt": args.prompt,
        "evaluation_instruction": "Answer with only the final value.",
        "rendered_prompt": rendered_prompt,
        "greedy_answer": answer,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

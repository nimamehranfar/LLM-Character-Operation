from __future__ import annotations

from typing import Any
import torch


def render_answer_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt + "\nAnswer with only the final value."}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def prepare_training_tensors(
    tokenizer: Any,
    prompt: str,
    expected: str,
    *,
    max_sequence_length: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int]:
    rendered = render_answer_prompt(tokenizer, prompt)
    prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
    answer_ids = tokenizer.encode(expected, add_special_tokens=False)
    if not answer_ids:
        raise ValueError(f"Expected answer {expected!r} tokenized to zero tokens.")
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id.")
    full_ids = prompt_ids + answer_ids + [int(eos_id)]
    if len(full_ids) > max_sequence_length:
        raise ValueError(
            f"Sequence length {len(full_ids)} exceeds configured maximum {max_sequence_length}."
        )
    labels = [-100] * len(prompt_ids) + answer_ids + [int(eos_id)]
    return (
        {
            "input_ids": torch.tensor([full_ids], device=device, dtype=torch.long),
            "attention_mask": torch.ones((1, len(full_ids)), device=device, dtype=torch.long),
            "labels": torch.tensor([labels], device=device, dtype=torch.long),
        },
        len(prompt_ids) - 1,
    )


def prepare_prompt_tensors(tokenizer: Any, prompt: str, device: torch.device) -> dict[str, torch.Tensor]:
    rendered = render_answer_prompt(tokenizer, prompt)
    encoded = tokenizer(rendered, return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}

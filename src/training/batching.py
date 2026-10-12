"""Padding, length grouping and global-batch planning for custom trainers."""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence

import torch


@dataclass(frozen=True)
class BatchPlan:
    per_device_batch_size: int
    gradient_accumulation_steps: int
    world_size: int
    effective_global_batch_size: int


def resolve_batch_plan(*, per_device_batch_size: int, gradient_accumulation_steps: int,
                       world_size: int, global_batch_size: int | None = None) -> BatchPlan:
    for name, value in (("per_device_batch_size", per_device_batch_size),
                        ("gradient_accumulation_steps", gradient_accumulation_steps),
                        ("world_size", world_size)):
        if int(value) < 1:
            raise ValueError(f"{name} must be positive")
    accumulation = int(gradient_accumulation_steps)
    base = int(per_device_batch_size) * int(world_size)
    if global_batch_size is not None:
        if int(global_batch_size) < base or int(global_batch_size) % base:
            raise ValueError(
                f"global_batch_size must be a positive multiple of per-device batch x world size ({base})"
            )
        accumulation = int(global_batch_size) // base
    return BatchPlan(int(per_device_batch_size), accumulation, int(world_size), base * accumulation)


def pad_training_sequences(sequences: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                           *, pad_token_id: int, device: torch.device):
    if not sequences:
        raise ValueError("Cannot collate an empty training batch")
    maximum = max(int(ids.numel()) for ids, _, _ in sequences)
    batch = len(sequences)
    input_ids = torch.full((batch, maximum), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((batch, maximum), dtype=torch.long)
    labels = torch.full((batch, maximum), -100, dtype=torch.long)
    lengths = []
    target_tokens = 0
    for row, (ids, mask, one_labels) in enumerate(sequences):
        ids, mask, one_labels = ids.flatten(), mask.flatten(), one_labels.flatten()
        length = int(ids.numel())
        input_ids[row, :length] = ids
        attention_mask[row, :length] = mask
        labels[row, :length] = one_labels
        lengths.append(length)
        target_tokens += int((one_labels != -100).sum().item())
    return ({"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device),
             "labels": labels.to(device)}, lengths, target_tokens)


def length_grouped_order(indices: list[int], lengths: Sequence[int], *, seed: int,
                         batch_size: int, mega_batch_multiplier: int = 50) -> list[int]:
    if len(indices) != len(lengths):
        raise ValueError("indices and lengths must have identical length")
    rng = random.Random(seed)
    pairs = list(zip(indices, lengths))
    rng.shuffle(pairs)
    size = max(int(batch_size), int(batch_size) * int(mega_batch_multiplier))
    grouped = []
    for start in range(0, len(pairs), size):
        bucket = sorted(pairs[start:start + size], key=lambda item: item[1], reverse=True)
        grouped.extend(index for index, _ in bucket)
    return grouped


def optimizer_steps(example_count: int, plan: BatchPlan) -> int:
    local_examples = math.ceil(example_count / plan.world_size)
    micro_batches = math.ceil(local_examples / plan.per_device_batch_size)
    return math.ceil(micro_batches / plan.gradient_accumulation_steps)

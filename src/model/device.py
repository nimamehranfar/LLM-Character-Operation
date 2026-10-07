"""Use CUDA's visible devices, including the GPU allocation supplied by SLURM."""
from __future__ import annotations

import os

import torch


def select_device(*, announce: bool = False) -> torch.device:
    # Do not interpret physical GPU IDs or change CUDA_VISIBLE_DEVICES. A single
    # GPU allocated by SLURM is CUDA's sole visible device, at ordinal zero.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if announce:
        if device.type == "cuda":
            index = torch.cuda.current_device()
            description = f"cuda:{index} | GPU: {torch.cuda.get_device_name(index)}"
        else:
            description = "cpu | GPU: none"
        print(f"Device: {description} | SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', 'local')} "
              f"| CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}", flush=True)
    return device


def model_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)

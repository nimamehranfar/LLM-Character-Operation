"""Use CUDA's visible devices, including the GPU allocation supplied by SLURM."""
from __future__ import annotations

import os

import torch


def select_device(*, announce: bool = False) -> torch.device:
    # Do not interpret physical GPU IDs or change CUDA_VISIBLE_DEVICES. A single
    # GPU allocated by SLURM is CUDA's sole visible device, at ordinal zero.
    if torch.cuda.is_available():
        index = int(os.environ.get("LOCAL_RANK", torch.cuda.current_device()))
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"LOCAL_RANK={index} exceeds the visible CUDA device count")
        torch.cuda.set_device(index)
        device = torch.device("cuda", index)
    else:
        device = torch.device("cpu")
    if announce:
        if device.type == "cuda":
            index = device.index
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

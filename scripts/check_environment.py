"""Check runtime dependencies, frozen data, checkpoints, and every visible GPU."""
from __future__ import annotations

import argparse
import hashlib
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--gpu-smoke-test", action="store_true", help="Run a tiny 4-bit layer on every visible GPU")
    parser.add_argument("--require-checkpoints", action="store_true", help="Require the default Qwen3-8B trained pipeline weights")
    args = parser.parse_args()
    errors = []
    print(f"Python {platform.python_version()}; {platform.system()} {platform.machine()}")
    if sys.version_info < (3, 11):
        errors.append("Python 3.11+ required")
    for package in ("torch", "transformers", "accelerate", "peft", "bitsandbytes", "huggingface_hub",
                    "safetensors", "tqdm", "sentencepiece", "google.protobuf", "tiktoken", "einops"):
        try:
            import_module(package)
        except Exception as exc:
            errors.append(f"{package}: {type(exc).__name__}: {exc}")
    try:
        import_module("transformers_stream_generator")
    except Exception as exc:
        # Only the legacy Qwen-7B remote-code baseline needs this package.
        errors.append(f"Legacy Qwen-7B generator unavailable: {exc}; install requirements.txt (Transformers 4.56.2)")
    dataset = ROOT / "data/character_operations_50k"
    for line in (dataset / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        path = dataset / name.strip().lstrip("*")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f"Dataset checksum mismatch: {path}")
    for name in ("checkpoints/qwen3_8b/tool_policy/final_adapter/adapter_config.json",
                 "checkpoints/qwen3_8b/result_injection/result_injector.pt"):
        present = (ROOT / name).is_file()
        print(f"{'Present' if present else 'Missing'} trained artifact: {name}")
        if args.require_checkpoints and not present:
            errors.append(f"Missing {name}; transfer compatible weights or train them")
    adapter = ROOT / "checkpoints/qwen3_8b/tool_policy/final_adapter"
    if args.require_checkpoints and not any((adapter / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        errors.append("Missing trained selector adapter weights")
    try:
        import torch
        print(f"PyTorch {version('torch')}; CUDA runtime {torch.version.cuda}; visible GPUs {torch.cuda.device_count()}")
        if (args.require_cuda or args.gpu_smoke_test) and not torch.cuda.is_available():
            errors.append("CUDA unavailable; check NVIDIA driver and installed PyTorch wheel")
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            print(f"GPU {index}: {props.name}; {props.total_memory / 1024**3:.1f} GiB; capability {props.major}.{props.minor}")
            if args.gpu_smoke_test:
                import bitsandbytes as bnb
                with torch.cuda.device(index), torch.no_grad():
                    layer = bnb.nn.Linear4bit(16, 16, bias=False, compute_dtype=torch.bfloat16,
                                             quant_type="nf4", compress_statistics=True).to(f"cuda:{index}")
                    output = layer(torch.ones(1, 16, device=f"cuda:{index}", dtype=torch.bfloat16))
                    torch.cuda.synchronize(index)
                    if not torch.isfinite(output).all():
                        raise RuntimeError("Non-finite output in 4-bit smoke test")
                    del layer, output
                print(f"GPU {index}: 4-bit smoke test passed")
    except Exception as exc:
        errors.append(f"GPU/runtime check: {type(exc).__name__}: {exc}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("Environment and dataset checks passed.")


if __name__ == "__main__":
    main()

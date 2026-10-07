"""Create a Windows/Linux virtual environment and install a CUDA-compatible runtime."""
from __future__ import annotations

import argparse
import platform
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def select_cuda():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                                capture_output=True, text=True, check=True)
        capabilities = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        if capabilities and max(capabilities) >= 10:
            return "cu128"
    except (OSError, ValueError, subprocess.CalledProcessError):
        print("GPU generation could not be detected; using cu126. Override with --cuda if needed.")
    return "cu126"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", default=".venv")
    parser.add_argument("--cuda", choices=["auto", "cu126", "cu128", "cu130"], default="auto")
    parser.add_argument("--torch-index-url", help="Override the PyTorch wheel index for your platform")
    parser.add_argument("--dev", action="store_true", help="Also install the test dependencies")
    parser.add_argument("--skip-gpu-check", action="store_true", help="Install on a login node without a GPU; check CUDA later inside the job")
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.error("Python 3.11 or newer is required")
    if platform.machine().lower() in {"aarch64", "arm64"} and not args.torch_index_url:
        parser.error("ARM64 servers require a platform-specific PyTorch index; supply --torch-index-url")
    cuda = select_cuda() if args.cuda == "auto" else args.cuda
    index = args.torch_index_url or f"https://download.pytorch.org/whl/{cuda}"
    directory = Path(args.venv).expanduser()
    if not directory.is_absolute():
        directory = ROOT / directory
    python = directory / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.is_file():
        subprocess.run([sys.executable, "-m", "venv", str(directory)], check=True)
    print(f"Installing runtime into {directory}; CUDA wheel family: {cuda}", flush=True)
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "torch>=2.5", "--index-url", index], check=True)
    requirement = ROOT / ("requirements-dev.txt" if args.dev else "requirements.txt")
    subprocess.run([str(python), "-m", "pip", "install", "-r", str(requirement)], check=True)
    # Capture the actual package versions for reproducibility, without assuming
    # a Windows wheel lock can be installed on a different Linux architecture.
    installed = subprocess.run([str(python), "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (directory / "installed-requirements.txt").write_text(installed.stdout, encoding="utf-8")
    check = [str(python), str(ROOT / "scripts/check_environment.py")]
    if not args.skip_gpu_check:
        check += ["--require-cuda", "--gpu-smoke-test"]
    subprocess.run(check, check=True)
    if args.skip_gpu_check:
        print("Dependencies/data verified; GPU check deferred. Run scripts/check_environment.py --require-cuda --gpu-smoke-test inside your allocation.")
    print("Exact installed versions are saved inside the virtual environment.")


if __name__ == "__main__":
    main()

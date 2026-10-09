# Windows and Linux deployment

Use Python 3.11+ and an NVIDIA GPU with a compatible driver. On SLURM, verify GPU access through `scripts/check_environment.py` inside an allocation; `nvidia-smi` access is not required for default H100 installation. Install the driver through the server administrator or operating-system instructions if it is absent. The launcher respects the scheduler's `CUDA_VISIBLE_DEVICES`.

## Get the complete source

Git only transfers committed and pushed files. Local untracked evaluator files and uncommitted changes must be included before cloning/pulling elsewhere. Model weights, `.venv/`, `.local/`, `results/` and trained `checkpoints/` are excluded from Git.

To transfer the current source and frozen dataset without publishing Git changes:

```text
python scripts/package_project.py
```

This produces `outputs/LLM-Character-Operation-portable.zip`. Extract it into a new directory on the target machine. The archive includes a SHA-256 manifest and excludes environments, secrets, model caches, checkpoints and old results. It is a source snapshot, not a Git checkout.

## Install

Use Python 3.11 for the validated dependency stack. The default `torch==2.8.0+cu126` wheel supports Python 3.11–3.13 but not Python 3.14. A Conda `base` environment running Python 3.14 needs a separate Python 3.11 environment before bootstrap:

```bash
conda create -n llm-character python=3.11 pip -y
conda activate llm-character
python scripts/setup_environment.py --dev --skip-gpu-check
source .venv/bin/activate
```

If an existing project `.venv` was created with Python 3.14, first move it to an unused backup path and let setup create a new one with Python 3.11.

For SLURM jobs and clean baseline exports, see [SLURM_BASELINES.md](SLURM_BASELINES.md). Installation on a login node without a GPU supports `--skip-gpu-check`; verify CUDA later inside an allocation.

Windows PowerShell, from the project directory:

```powershell
python scripts/setup_environment.py --dev
.\.venv\Scripts\Activate.ps1
```

Linux Bash, from the project directory:

```bash
python3 scripts/setup_environment.py --dev
source .venv/bin/activate
```

The bootstrap creates or updates `.venv`, installs PyTorch first from the CUDA wheel index, installs project/test dependencies, records installed versions and checks every visible GPU with a tiny NF4 computation. Its default is `torch==2.8.0+cu126`, matching `requirements.txt` for the H100 server and RTX 4070 on Linux/Windows x86_64. Default installation does not call `nvidia-smi`. Direct `python -m pip install -r requirements.txt` also selects that exact GPU wheel; the CUDA suffix excludes CPU-only builds.

Explicit `--cuda cu128` uses PyTorch 2.8.0 with CUDA 12.8; `--cuda cu130` uses PyTorch 2.9.1 with CUDA 13.0. Optional `--cuda auto` probes GPU generation, selecting `cu128` for compute capability 10+ and `cu126` otherwise, with `cu126` as its detection-failure fallback. Remaining dependencies install from `.venv/runtime-requirements.txt`, excluding the default PyTorch pin so an explicit CUDA/platform override is preserved. Example overrides:

```text
python scripts/setup_environment.py --cuda cu128 --dev
python scripts/setup_environment.py --torch-index-url https://download.pytorch.org/whl/cu128 --dev
```

Choose a wheel family supported by the GPU and installed driver. A newer GPU can require a newer driver. ARM64 servers require their platform-specific PyTorch index supplied through `--torch-index-url`; the bootstrap does not guess it. Direct requirements installation targets Linux/Windows x86_64 with the CUDA 12.6 pin; use the bootstrap's platform/CUDA override when a different build is needed.

Fresh installs pin `transformers==4.56.2` because the legacy Qwen-7B generator depends on exports removed in later releases, including `BeamSearchScorer` in 4.57. Existing research reports retain their recorded versions and are not rewritten. Package lists inside `.venv/installed-requirements.txt` document that machine's installation; hardware-specific wheels should not be copied blindly between Windows and Linux.

The other direct runtime dependencies are pinned to locally tested versions. PyTorch is selected separately for the machine's CUDA wheel family. The installed package record also captures transitive dependencies, so retain it with the experiment reports.

Check an existing environment and run CPU/unit checks:

```text
python scripts/check_environment.py --require-cuda --gpu-smoke-test
python -m pytest -q
```

## One-GPU native tool pilot

This downloads the pinned Qwen3-8B snapshot if needed and tests five examples per operation plus two controls per category on each split:

```text
python scripts/evaluate_tool_feedback.py --config configs/baselines/tools/qwen3_8b.toml --split both --examples-per-operation 5 --controls-per-category 2 --auto-download
```

Leave out the sample options for full evaluation. Native tools require no trained selector or mapper checkpoints. The default cache is Hugging Face's cache; override with `--model-cache-dir /your/model/cache` on Linux or an appropriate Windows directory. On Windows, commands can use `.\.venv\Scripts\python.exe` without activating the environment; on Linux use `.venv/bin/python`.

## Multi-GPU evaluation

Each GPU runs a complete model copy on a disjoint set of examples. Sampling happens before partitioning, so the chosen test examples are the same as on one GPU. Each model copy must fit on its individual GPU. The model precision, prompts, mapper, decoding and scoring rules remain unchanged.

Native tool pilot on all visible GPUs:

```text
python scripts/evaluate_multi_gpu.py --kind tool-feedback --gpus auto --split both -- --config configs/baselines/tools/qwen3_8b.toml --examples-per-operation 5 --controls-per-category 2 --auto-download
```

Both complete baseline matrices:

```text
python scripts/run_tool_feedback_matrix.py --gpus auto --auto-download
python scripts/run_zero_shot_matrix.py --gpus auto --auto-download
```

One unaided base model:

```text
python scripts/evaluate_multi_gpu.py --kind direct --gpus auto --split both -- --config configs/baselines/local/qwen3_8b.toml --mode zero_shot --auto-download
```

The trained pipeline:

```text
python scripts/evaluate_multi_gpu.py --kind pipeline --gpus auto --split both -- --auto-download
```

The same trained selector with ordinary text feedback:

```text
python scripts/evaluate_multi_gpu.py --kind tool-feedback --gpus auto --split both -- --config configs/baselines/tools/qwen3_8b.toml --policy-config configs/experiments/qwen3_8b/tool_policy.toml --auto-download
```

Select visible GPUs with `--gpus 0,1` or inspect commands with `--dry-run`. Put launcher arguments before `--` and normal evaluator arguments after it. GPU indices refer to the current allocation: when a scheduler exposes physical GPUs 3 and 7, visible indices 0 and 1 select those GPUs. GPU UUIDs/MIG identifiers in `CUDA_VISIBLE_DEVICES` are preserved.

Combined reports are written under `results/parallel/<model>/<kind>/<run-fingerprint>/test.json` and `heldout.json`. Worker reports, logs and resumable progress remain in separate split/worker directories. The fingerprint includes configs, source, dataset, adapter/mapper hashes, package versions and hardware to prevent mixing different experiments during resume. Re-run the same command to continue; use a different `--output-dir` for a separate run. Do not run the identical command concurrently against the same output directory.

Merging requires every selected example exactly once and recomputes accuracy from all examples. `parallel_runtime.wall_seconds_current_session` records elapsed time including worker startup/loading. Aggregate prompt latency is a separate measure and must not be interpreted as parallel wall time. Throughput is omitted for resumed runs, where old examples were generated in earlier sessions. GPU speedup is workload-dependent; more GPUs do not guarantee a linear speedup. Small pilots may spend most of their time loading replicas.

## Trained artifacts and server training

Transfer compatible artifacts into:

```text
checkpoints/qwen3_8b/tool_policy/final_adapter/
checkpoints/qwen3_8b/result_injection/result_injector.pt
```

The whole adapter directory is required, including its weights. Check readiness with `python scripts/check_environment.py --require-cuda --require-checkpoints`. Copy corresponding Qwen3-4B artifacts when using its experiment configs. Historical checkpoints for a different protocol are not automatically compatible.

Alternatively train on the target machine:

```text
python scripts/train_tool_policy.py --max-vram-mib 0
python scripts/train_result_injector.py --max-vram-mib 0
python scripts/train_direct_sft.py --max-vram-mib 0
```

`--max-vram-mib 0` removes the laptop memory ceiling on the selected GPU. Without this option, the existing 7600 MiB training budget remains. Training remains single-GPU, with the original batch size and optimization settings; this change does not implement distributed training or automatically enlarge batches. Independent training runs can be assigned different GPUs with `CUDA_VISIBLE_DEVICES` and distinct configs/output paths. The new launcher accelerates evaluation, not a single training run.

On Linux, for example:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_tool_policy.py --max-vram-mib 0
```

Do not transfer `.venv` or the Windows `.local/model_cache.json`; recreate them on the server. Download pinned base weights there or transfer a complete Hugging Face cache. Start new performance measurements in a fresh results directory instead of resuming timings from the laptop.

## Validation limits

CPU tests exercise real worker subprocesses, GPU visibility isolation, example coverage, failure cleanup, report merging and mapper summaries. A local GPU smoke/pilot check does not prove Linux or physical multi-GPU performance. Run the environment check and a small pilot on the target allocation before the full benchmark.

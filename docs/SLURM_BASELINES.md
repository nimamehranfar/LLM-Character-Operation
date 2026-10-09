# SLURM: clean pretrained baselines

Run commands from the project root. These evaluate the original configured checkpoints with direct generation: no trained LoRA adapter, executor, tool feedback or result mapper. The configs include both base and chat models; the runner does not replace them with instruction-tuned variants. Defaults are NF4 4-bit weights, greedy generation, a 192-token output limit and the frozen character-task splits. Qwen3 thinking is disabled when its released template supports it. Reports record the actual model, GPU, quantization and predictions.

## Install and authenticate

If the server's Conda `base` environment uses Python 3.14, create a Python 3.11 environment first. The pinned PyTorch 2.8.0 CUDA wheel does not support Python 3.14. Python 3.11 matches the locally tested dependency stack:

```bash
conda create -n llm-character python=3.11 pip -y
conda activate llm-character
python scripts/setup_environment.py --dev --skip-gpu-check
source .venv/bin/activate
```

The bootstrap creates the project's `.venv` using this Python interpreter, as required by the included SLURM template. If `.venv` was previously created with Python 3.14, move it to an unused backup name before setup, for example `mv .venv .venv-python314-backup`; do not reuse that environment for this pin.

Transfer the updated source or extract `outputs/LLM-Character-Operation-portable.zip` first. Uncommitted local changes do not transfer through `git pull`.

```bash
python3 scripts/setup_environment.py --dev --skip-gpu-check
source .venv/bin/activate
hf auth login
hf auth whoami
mkdir -p logs
python scripts/run_zero_shot_matrix.py --list-models
```

Use a Hugging Face read token. Public models do not require login. Gated models require accepting their model-page terms and receiving access on the same account; login alone does not grant access. See [HF authentication](https://huggingface.co/docs/huggingface_hub/v0.36.0/guides/cli) and [gated model access](https://huggingface.co/docs/hub/models-gated).

`--skip-gpu-check` allows installation on a login node without a GPU. Run `python scripts/check_environment.py --require-cuda --gpu-smoke-test` inside a GPU allocation. Load your site's Python/CUDA modules first if required. For wheel selection and driver requirements see [DEPLOYMENT.md](DEPLOYMENT.md); if the login node cannot identify the target GPU, select its wheel family explicitly using `--cuda`.

## Submit jobs

The requirements and setup script default to `torch==2.8.0+cu126` for the H100 cluster (Linux x86_64). `python -m pip install -r requirements.txt` selects this GPU build directly. Default setup does not require `nvidia-smi`; its Python GPU check runs inside your allocation, or can be deferred with `--skip-gpu-check`.

The template requests one GPU, four CPU cores, 64 GB RAM and 24 hours. Adjust to your site's rules; add `--partition YOUR_PARTITION` and/or `--account YOUR_ACCOUNT` to `sbatch` if required. The full matrix may need more time; saved per-example progress allows recovery.

```bash
# Quick pilot: one prompt for each of 12 operations, on both splits (24 total).
sbatch scripts/slurm_zero_shot.sbatch --models qwen3_8b --split both --examples-per-operation 1

# ALL 13 models, full test and heldout splits, deleting owned downloads on success.
sbatch scripts/slurm_zero_shot.sbatch --split both --examples-per-operation -1 --cleanup-models

# One model, full data.
sbatch scripts/slurm_zero_shot.sbatch --models qwen3_4b --split both --cleanup-models

# Batched H100 pilot: 5 prompts per operation = 60 per split, 120 total.
sbatch scripts/slurm_zero_shot.sbatch --models qwen3_8b --split both --examples-per-operation 5 --batch-size 64 --auto-download

# Several models with a persistent scratch cache parent.
sbatch scripts/slurm_zero_shot.sbatch --models qwen3_8b qwen3_4b qwen2_5_7b --split heldout --cleanup-models --model-cache-dir /path/to/your/shared/scratch/hf-baselines
```

To let SLURM schedule separate one-GPU jobs for all models concurrently when resources are available:

```bash
for model in $(python scripts/run_zero_shot_matrix.py --list-models | awk '{print $1}'); do
  sbatch scripts/slurm_zero_shot.sbatch --models "$model" --split both --cleanup-models
done
```

This submits 13 jobs. Each has its own assigned visible GPU 0 and its own result folder/ZIP; the scheduler controls actual concurrency. Use the single all-model job above when you prefer sequential downloads/evaluations.

For an interactive job:

```bash
srun --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=01:00:00 \
  .venv/bin/python scripts/run_zero_shot_matrix.py \
  --models qwen3_8b --split heldout --examples-per-operation 1 --auto-download
```

The template enables automatic downloading. Device startup output includes `cuda:0`, the actual GPU model, SLURM job ID and `CUDA_VISIBLE_DEVICES`. The code leaves the scheduler's GPU visibility unchanged: visible device 0 is the GPU assigned to this job and need not be physical GPU 0. [SLURM GPU allocation](https://slurm.schedmd.com/gres.html#GPU_Management).

Models load on the dynamic device; inputs and restored optimizer tensors follow their model's device. Direct baselines support CPU using `--quantization none`. Existing 4-bit training/pipeline commands require CUDA and now dynamically select it too. Training remains single GPU; use its `--max-vram-mib 0` option on the server to remove the laptop memory ceiling.

## Compute nodes without internet

Prepare owned caches on an internet-connected login node, choosing shared storage reachable from the compute node:

```bash
python scripts/run_zero_shot_matrix.py --models qwen3_8b --split both \
  --auto-download --download-only --cleanup-models \
  --model-cache-dir /path/to/your/shared/scratch/hf-baselines
```

Preparation retains downloads and prints a unique run folder. Evaluate using that exact folder and the same model/split/sampling/quantization settings:

```bash
sbatch scripts/slurm_zero_shot.sbatch --models qwen3_8b --split both \
  --resume-run /absolute/path/to/printed/run-folder \
  --no-auto-download --cleanup-models
```

Omit `--models` from both commands for all models. The manifest retains the shared cache location. Do not prepare on login-node-local scratch if compute nodes cannot read it. Remote-code models need their dependencies installed too; missing/offline dependencies remain recorded failures.

## Command options

```bash
python scripts/run_zero_shot_matrix.py --help
python scripts/run_zero_shot_matrix.py --list-models
```

| Argument | Meaning |
|---|---|
| `--models NAME [NAME ...]` | One or several config names; omitted = all 13. |
| `--list-models` | Print names and exact Hub repositories, then exit. |
| `--split test`, `heldout`, `both` | Default both. Heldout uses withheld prompt templates. |
| `--examples-per-operation N` | N per operation per split; `1` = 12 per split; `-1` = full split. |
| `--examples N` | Total random prompts per split; mutually exclusive with per-operation sampling. |
| `--auto-download` / `--no-auto-download` | Fetch missing weights / require existing weights; default enables download. |
| `--download-only` | Populate cache without evaluation; no deletion, even when cleanup is selected. |
| `--model-cache-dir PATH` | Persistent HF cache, or parent of isolated run caches with cleanup selected. |
| `--cleanup-models` | Isolate model downloads; delete each model's cache after all its requested splits succeed. |
| `--results-dir PATH` | Parent for unique run folders; default `results/baselines/zero_shot`. |
| `--resume-run PATH` | Recover a specific run with matching model/settings/config/data/source. |
| `--resume` / `--no-resume` | Enable/disable per-example recovery; default enabled. A new invocation still creates a new folder. |
| `--quantization 4bit` / `none` | Override quantization for all selected models. `none` needs more memory. |
| `--max-new-tokens N` | Override generation limit; default config value is 192. |
| `--batch-size N` | Positive maximum prompts per generation call; default config value or 1. With 60 prompts per split and size 64, actual batch size is 60. |
| `--continue-on-error` / `--no-continue-on-error` | Continue after a model/split failure (default) / stop on first failure. |
| `--matrix-config PATH` | Override matrix defaults. |
| `--gpus auto` or visible indices | Optional replicas across allocated GPUs; omit for single-GPU SLURM jobs. |

For batch sizes above 1, per-example timing is the batch wall time divided by its actual prompt count. The runtime totals therefore count each batch once; the average describes throughput cost per example, not individual response latency. Each row also records the actual batch size, batch ID, full batch latency and full batch generation time. Input-token counts exclude left padding, and output-token counts exclude padding after an answer's EOS. A memory failure is reported rather than silently reducing the requested batch size.

Available model names:

```text
aya23_8b       baichuan2_7b   chatglm3_6b   gemma_7b
llama2_7b      llama3_1_8b    llama3_8b     mistral_7b
qwen2_5_7b     qwen3_4b       qwen3_8b      qwen_7b
yi_6b
```

Some legacy repositories use their original remote Python implementations. Requirements cover their tokenizers, but each model's end-to-end compatibility still needs evaluation on the target machine. Missing access and runtime failures are reported instead of being counted as completed baselines.

## Copy results and recover runs

Each invocation creates `results/baselines/zero_shot/<UTC-time>_<job-id>_<unique-id>/` and a ZIP beside it. Both are distinct from previous runs. Contents:

- `manifest.json`: model revisions, config/data/source hashes, software versions, statuses and cleanup records.
- `summary.csv`: separate model/split rows with counts, accuracy, average time and errors.
- `models/<name>/test.json` and `heldout.json`: full predictions, operation breakdowns, timing, tokens and GPU memory.
- Per-split logs and progress files, plus copies of the configs.

The final terminal output prints `Run status`, `COPY RESULTS DIRECTORY` and `COPY RESULTS ZIP` with absolute paths. The ZIP is integrity-checked. Complete successful matrices exit 0; partial/failed matrices exit 1; Ctrl+C exports progress and exits 130. A skipped gated model makes the matrix incomplete. Pilots report their actual prompt counts and do not count as full benchmarks.

Copy the printed ZIP from your workstation, replacing the account, hostname and path:

```bash
scp your_user@your_cluster:/absolute/path/to/printed-run.zip .
```

Print accuracy percentages, completed counts, recorded run/generation time and peak VRAM for every model in a results directory:

```bash
.venv/bin/python scripts/summarize_baseline_run.py /absolute/path/to/printed-run-folder
```

The same standard-library script handles all configured model names and single/multi-GPU reports. It combines splits separately for each model, uses parallel wall time for multi-GPU runs, and shows missing measurements as `not recorded`. Times exclude parent downloads and queue time; VRAM is the highest per-GPU PyTorch peak, not summed GPU capacity.

To recover, repeat the original command with `--resume-run /absolute/path/to/run-folder`. Validated completed splits are skipped; unfinished splits resume from saved progress. After a SLURM timeout, submit recovery as a new job. Use shared persistent scratch if the cache must survive allocation teardown. Changed code/data/config/settings require a new run folder.

## Cleanup behavior

Cleanup-selected downloads go to `<cache-parent>/<run-id>/<model>/hub`; dynamic Python modules and auxiliary caches are isolated under the same owned model folder. The runner disables Xet for owned downloads so its separate shared chunk cache is not populated. Without a cache parent, SLURM jobs use `SLURM_TMPDIR/baseline-model-cache` if available, otherwise project `.local/baseline-model-cache`. Cleanup checks ownership and complete, matching results before deleting a model folder. Shared HF caches and existing local models are never deletion targets.

Failed/unfinished model caches are retained. Successfully completed models may be deleted even if another model later fails. Without cleanup, existing cached/local weights are reused. Empty cache-parent folders may remain. Cleanup removes owned downloaded model/module files; results, login credentials, installed environments and scheduler logs remain. To remove a saved HF login separately after all jobs finish, use `hf auth logout`; unset `HF_TOKEN` separately if supplied through the environment. [HF authentication](https://huggingface.co/docs/huggingface_hub/v0.36.0/guides/cli).

## Local Windows pilot

```powershell
.\.venv\Scripts\python.exe scripts/run_zero_shot_matrix.py --models qwen3_8b --split heldout --examples-per-operation 1 --no-auto-download
```

Use `--auto-download` when the configured checkpoint is absent. Model selection and result export options also work locally.

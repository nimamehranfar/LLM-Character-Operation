# LLM Character Operations

This repository implements a structured LLM interface to deterministic character/string operations. The language model decides whether an exact operation is needed and emits the operation plus arguments; deterministic code executes it; scalar results can be reintegrated into the language model through learned residual-stream injection.

The source and frozen dataset are self-contained for the current experiment protocol. Model weights, checkpoints, caches and generated results are local artifacts excluded from Git.

## Supported operations

`COUNT_CHAR`, `STRING_LENGTH`, `FIND_CHAR`, `WORD_COUNT`, `WORD_LENGTH_AT`, `INSERT_TEXT`, `INSERT_TEXT_IN_WORD`, `INSERT_WORDS`, `CHAR_AT`, `CHAR_AT_IN_WORD`, `REVERSE`, `REVERSE_WORD_AT`.

Integer and single-character results use the learned result-injection path. Arbitrary string-transform results are returned directly from the deterministic executor and are reported separately.

## Frozen dataset

`data/character_operations_50k/` contains the exact 50,000-example dataset used by the current protocol:

| Split | Examples |
|---|---:|
| Train | 35,000 |
| Dev | 5,000 |
| IID test | 5,000 |
| Held-out wording/style | 5,000 |

The dataset contains 63% supported tasks and 37% controls. Positive strings use the preserved 30% real-word / 50% augmented-real-word / 20% random-string mix. Held-out template families are disjoint from training.

The committed dataset can be regenerated deterministically:

```powershell
python scripts/generate_dataset.py
```

## Installation

Python **3.11 or newer** and a compatible NVIDIA driver are required for local 4-bit model evaluation/training. The same source runs on Windows and Linux.

Windows:

```powershell
python scripts/setup_environment.py --dev
.\.venv\Scripts\Activate.ps1
```

Linux:

```bash
python3 scripts/setup_environment.py --dev
source .venv/bin/activate
```

Setup chooses CUDA 12.6 for older GPU generations or CUDA 12.8 for Blackwell, then checks each visible GPU. Override with `--cuda cu126`, `--cuda cu128` or `--torch-index-url` to match the server. It does not install the NVIDIA driver. Runtime requirements pin Transformers to 4.56.2 for the legacy comparison models. Exact installed versions are recorded in `.venv/installed-requirements.txt`.

See [Windows/Linux deployment and multi-GPU commands](docs/DEPLOYMENT.md) for model downloads, checkpoint transfer, server evaluation and portable packaging.

For an existing environment:

```text
python scripts/check_environment.py --require-cuda --gpu-smoke-test
```

## Model cache behavior

Every primary experiment config has:

```toml
[cache]
auto_download = true
cache_dir = ""
revision = "<pinned commit>"
```

Existing local snapshots are always reused first. `cache_dir = ""` uses the normal Hugging Face cache. To keep weights somewhere else, either edit the config or pass:

```powershell
--model-cache-dir "D:\hf-models"
```

You can force local-only behavior with `--no-auto-download`.

The zero-shot matrix's download behavior is configured in `configs/baselines/matrix.toml`; `--no-auto-download` forces local-only use. To fetch missing ungated models into a chosen directory:

```powershell
python scripts/run_zero_shot_matrix.py --auto-download --model-cache-dir "D:\hf-models"
```

Some comparison repositories require accepting provider/model terms before download.

## Zero-shot base-model baselines

Run all configured local baseline models on both IID and held-out splits:

```powershell
python scripts/run_zero_shot_matrix.py
```

The default is the full positive-task set. Configuration is in `configs/baselines/matrix.toml` and `configs/baselines/local/`.

For one model:

```powershell
python scripts/evaluate_direct.py `
  --config configs/baselines/local/qwen3_8b.toml `
  --mode zero_shot `
  --split test
```

For models that do not fit locally, use an OpenAI-compatible hosted endpoint:

```powershell
$env:BASELINE_API_KEY="..."
python scripts/evaluate_zero_shot_api.py `
  --base-url <provider-v1-url> `
  --model <provider-model-id> `
  --split test
```

The hosted config defaults to a deterministic 100-example-per-operation subset to control cost.

## Off-the-shelf tool-calling comparisons

Compare the released Qwen3-4B/8B, Phi-4-mini and Hermes-3-8B models using the same
character executor with ordinary text tool feedback:

```powershell
python scripts/run_tool_feedback_matrix.py --preflight
python scripts/run_tool_feedback_matrix.py
```

On a multi-GPU server, distribute each model's selected examples across all visible GPUs:

```text
python scripts/run_tool_feedback_matrix.py --gpus auto --auto-download
```

The zero-shot matrix also accepts `--gpus auto`. Direct evaluation commands retain their single-GPU defaults. Use `scripts/evaluate_multi_gpu.py` for a single model or the trained pipeline; see the deployment guide.

These baselines use existing function-calling interfaces, with no additional
training or mapper. A trained-selector/text-feedback comparison is also available
to isolate the mapper's contribution. See [the protocol and commands](docs/TOOL_FEEDBACK_BASELINES.md)
for pilot runs, name masking, paired comparisons and separate control/task scores.

## Training sequence

The primary backbone is Qwen3-8B. Qwen3-4B configs are included for smaller-backbone replication.

### Direct-answer SFT comparison

```powershell
python scripts/train_direct_sft.py
```

Evaluate it with:

```powershell
python scripts/evaluate_direct.py `
  --config configs/experiments/qwen3_8b/direct_sft.toml `
  --mode direct_sft `
  --split test
```

### Structured tool policy

```powershell
python scripts/train_tool_policy.py
python scripts/evaluate_tool_policy.py
```

### Multi-layer result injection

```powershell
python scripts/train_result_injector.py
```

Qwen3-8B candidate layers are `1, 3, 5, 7, 9, 11, 13, 15`. The deployment layer is selected using dev data only. Test/held-out data never select the layer.

### Complete pipeline

After the tool-policy adapter and result-injection checkpoint exist, the complete evaluation requires no arguments:

```powershell
python scripts/evaluate_pipeline.py
```

It runs IID and held-out evaluation by default.

### Layer-wise and oracle analysis

```powershell
python scripts/evaluate_layerwise.py
```

This records actual-policy and oracle-result behavior at every candidate layer, per-operation accuracy, gates, selected-layer accuracy, and an oracle upper bound. Oracle information is diagnostic only and is never used by the normal pipeline.

## Recovery and interruption handling

Training scripts use durable recovery checkpoints and heartbeat state. Evaluation scripts persist completed examples. Re-run the same command after interruption to continue.

Recorded inference time excludes downtime between processes. The first model setup time is kept as the canonical setup cost; repeated model loading after resume is reported separately as resume overhead rather than silently inflating prompt latency.

Use `--fresh` only when intentionally starting a new training run. Evaluation resume can be controlled with `--resume` / `--no-resume` where applicable.

## Runtime telemetry

Zero-shot/direct-SFT evaluation and the complete trained pipeline save:

- total interruption-neutral benchmark time;
- model setup time;
- average, p50, p95 and p99 prompt latency;
- input/output/total token counts;
- output tokens per second;
- peak allocated/reserved VRAM;
- resume count and resume setup overhead.

The trained pipeline additionally records tool-policy generation time/tokens, deterministic executor time, result-injection generation time/tokens, selected layer, and gate values.

Terminal output is intentionally concise. Complete per-example records and detailed breakdowns are written to JSON/JSONL files under `results/`.

## Chart-ready exports

```powershell
python scripts/export_metrics.py
python scripts/export_runtime.py
```

`export_metrics.py` creates:

- `system_comparison.csv`
- `character_operation_analysis.csv`
- `layer_analysis.csv`
- `oracle_analysis.csv`

`export_runtime.py` creates `results/runtime_comparison.csv` for latency/token/VRAM comparisons.

## Tests

```powershell
pip install -r requirements-dev.txt
pytest -q
```

The test suite validates deterministic executor semantics, all 50,000 ground-truth rows, held-out template isolation, structured-call round trips, candidate-layer configs, recovery behavior, low-VRAM preparation, and runtime accounting.

## Repository layout

```text
configs/
  baselines/             zero-shot local/API model configs
  experiments/           Qwen3-8B and Qwen3-4B train/eval configs
data/
  character_operations_50k/
  resources/
scripts/                  generation, training, evaluation, export commands
src/
  data/
  evaluation/
  executor/
  model/
  training/
tests/
```

Primary Qwen snapshot revisions are pinned in config so a fresh cache resolves the same model revision used by this experiment protocol.

## Local baseline matrix notes

See [SLURM and clean baseline commands](docs/SLURM_BASELINES.md) for one-GPU jobs, all 13 configured models, model selection, automatic downloads and optional cleanup. Each invocation creates a unique result folder, CSV summary and ZIP. Incomplete matrices exit with code 1. Use `--resume-run <folder>` with matching evaluation options to recover a specific run.

Install the full runtime requirements before evaluating the heterogeneous comparison models:

```powershell
pip install -r requirements.txt
```

Some Hugging Face repositories are gated. If your account has access, authenticate once with `hf auth login`; otherwise the matrix records the model as `SKIP` and continues. Individual model/split failures are also reported concisely and do not abort the remaining matrix by default. Set `evaluation.continue_on_error = false` in `configs/baselines/matrix.toml` or pass `--no-continue-on-error` for fail-fast behavior.

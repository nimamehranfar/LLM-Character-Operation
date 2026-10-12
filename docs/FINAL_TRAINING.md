# Final training on one GPU

Both stages now support actual padded batches, gradient accumulation, and an explicit effective batch. The SLURM launcher requests one GPU and preserves the scheduler's `CUDA_VISIBLE_DEVICES` allocation. The script prints the device and GPU model. These commands do not pool multiple GPUs.

`--batch-size` means sequences processed simultaneously on the GPU. `--global-batch-size` means sequences per optimizer update. On one GPU:

```text
effective batch = batch-size × gradient-accumulation-steps
```

For example, `--batch-size 8 --global-batch-size 1024` accumulates 128 micro-batches before each update. This does not allocate 1,024 sequences simultaneously. The final, short update at an epoch boundary uses the actual remaining examples; losses are weighted by supervised token counts. Increasing the effective batch reduces optimizer updates and can change accuracy. It is a hyperparameter experiment, not a quality-neutral speed setting.

## Pull the pushed changes on the server

Run from the server login terminal after pushing your local changes:

```bash
cd ~/Projects/LLM-Character-Operation
git pull --ff-only
source .venv/bin/activate
python -m pip install 'matplotlib>=3.9,<4'
mkdir -p logs
python scripts/run_final_training.py --help
```

Matplotlib is the only newly added package. A fresh installation still uses `scripts/setup_environment.py` and the existing Python 3.11/CUDA requirements. Do not replace a working GPU Torch installation just to add plotting support.

## Run both stages and the actual pipeline evaluation

```bash
sbatch \
  --account=ric_g.ai.a._430 \
  --partition=aiq \
  --qos=ric_g.ai.a._430_aiq_qos \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=128G \
  --time=24:00:00 \
  scripts/slurm_train.sbatch \
  --stage both \
  --auto-download
```

The default uses the new `tool_policy_final.toml` and `result_injection_final.toml` presets. After both trainers complete, the launcher evaluates the actual pipeline with predicted policy calls on test and heldout. Stage 2's oracle accuracy is reported separately and must never be presented as actual end-to-end accuracy. The serial actual-pipeline evaluator provides individual request timings; the trainers' batched evaluations provide batch latency and throughput.

Monitor the submitted job, replacing `JOBID` with the number printed by `sbatch`:

```bash
tail -F logs/character-training-JOBID.out
squeue -j JOBID
sacct -j JOBID --format=JobID,State,Elapsed,AllocTRES,MaxRSS
```

`sacct` elapsed is scheduler job time; the JSON runtime fields are the actual component measurements. `MaxRSS` is host RAM, not GPU VRAM.

## Train just one stage

Use the same `sbatch` resource options and change the final arguments:

```bash
scripts/slurm_train.sbatch --stage 1 --auto-download
scripts/slurm_train.sbatch --stage 2 --auto-download
```

On an already allocated interactive GPU, the equivalent commands are:

```bash
.venv/bin/python scripts/train_tool_policy.py \
  --config configs/experiments/qwen3_8b/tool_policy_final.toml \
  --batch-size 4 --global-batch-size 16 --auto-download

.venv/bin/python scripts/train_result_injector.py \
  --config configs/experiments/qwen3_8b/result_injection_final.toml \
  --batch-size 8 --global-batch-size 32 --auto-download
```

Stage 1 uses all training examples, including no-call controls. Stage 2 uses only the seven scalar-result operations, because the mapper supports integer and single-character outputs. It does not train on all 50,000 examples or on arbitrary string-result operations. Train, development, test, and heldout remain separate.

To explicitly try an effective batch of 1,024 for stage 1:

```bash
sbatch \
  --account=ric_g.ai.a._430 \
  --partition=aiq \
  --qos=ric_g.ai.a._430_aiq_qos \
  --gres=gpu:1 --cpus-per-task=8 --mem=128G --time=24:00:00 \
  scripts/slurm_train.sbatch \
  --stage 1 --batch-size 4 --global-batch-size 1024 --auto-download
```

This is 256 accumulation steps. For stage 2, `--stage 2 --batch-size 8 --global-batch-size 1024` is 128 accumulation steps. A literal `--batch-size 1024` is permitted by the CLI but has not been shown to fit an H100 for training. The earlier evaluation OOMs do not establish a safe training batch.

Supported controls on both trainers: `--batch-size`, `--global-batch-size`, `--gradient-accumulation-steps`, `--eval-batch-size`, `--seed`, `--max-vram-mib`, `--auto-download` / `--no-auto-download`, `--model-cache-dir`, `--fresh`, `--run-dir`, and `--status`. `--global-batch-size` derives the accumulation count and must divide exactly by the micro-batch. Giving only an accumulation override removes the config's effective-batch override. `--max-vram-mib 0` allows the whole allocated GPU.

## Saved artifacts

The launcher prints a distinct `COPY RESULTS DIRECTORY` and `COPY RESULTS ZIP` under `results/final_training/<timestamp>_<job>_<suffix>`. Each bundle contains immutable adapters/mapper weights, the run manifest, stage artifacts, and, for `--stage both`, actual pipeline results and comparison CSVs. Existing compatibility checkpoint paths are also written, but final evaluation is bound to the bundle's immutable weights.

Each trainer's artifacts include:

| Output | Contents |
|---|---|
| `training_report.json` | Best epoch, denominators, test/heldout scores, per-example predictions, batch plan, runtime and VRAM |
| `environment_and_protocol.json` | Effective settings, optimizer, quantization, trainable parameter count/percentage, GPU, packages, git revision, source/config/data hashes |
| `optimizer_steps.jsonl`, `optimizer_steps.csv` | Every update: token-weighted loss, learning rate, gradient norm, examples/tokens, duration, throughput, memory, mapper gate/layers |
| `epochs.csv` | Epoch loss, development loss/accuracy, train and validation time |
| `evaluation_summary.csv` | Accuracy percentages, completed counts, generation throughput and latency scope |
| `*_predictions.csv` | Raw predictions, correctness, IDs, task metadata, input/output tokens, batch accounting |
| `operation_accuracy.csv` | Per-operation percentages and denominators |
| `layerwise.csv` | Stage 2 development/test/heldout oracle accuracy and gates by layer |
| `summary.txt` | Copyable important scores and measured runtime |
| PDF, SVG, PNG figures | Training/development loss, development accuracy, learning rate, gradients, examples/s, tokens/s, memory, epoch time, per-operation accuracy; stage 2 gates and layer accuracy |

Figures use Matplotlib, vector PDF/SVG and 300 DPI PNG. Metrics retain fractions in JSON and percentages in paper tables/accuracy figures. Wilson 95% intervals quantify finite-example accuracy uncertainty; they do not replace variation across independent training seeds.

Timing separates cache resolution/download plus tokenizer initialization, model/adapter loading, batch preparation, CUDA-synchronized forward/backward/optimizer time, validation, checkpoint writing, final evaluation, and total active/session time. Model setup fields are explicitly current-session measurements. Completed-update training time is cumulative across recovery; total active time also includes operational overhead. It is not a summed generation estimate from the full SLURM duration. Peak allocated/reserved fields measure PyTorch-managed GPU memory, not all device consumers or energy. Energy, FLOPs, and GPU utilization are not invented when no measurement is available.

Checkpoint resume restores optimizer, scheduler, RNG, shuffle order, partial epoch aggregates, and committed update metrics. Signals checkpoint at an optimizer boundary. A failed/OOM update is discarded and the last atomic checkpoint remains authoritative. New script versions do not resume the old single-example trainer's checkpoint format. Rerunning with unchanged settings resumes the most recent incomplete training; `--fresh` starts a distinct run. Keep the same effective configuration on resume.

## Why these parameters

There is no universal standard setting for these two different training objectives. The final presets are evidence-informed starting points; they are not claimed to be optimal without development validation.

* [QLoRA, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/1feb87871436031bdc0f2beaa62a049b-Paper-Conference.pdf) reports an effective batch of 16 and learning rate 2e-4 for its 7B model, NF4/double quantization/BF16, rank 64, alpha 16, all linear LoRA targets, dropout 0.1 for smaller models, clipping 0.3, and constant learning rate. Stage 1 adopts these optimizer/adapter choices. Its four-epoch cap, task prompts, development stopping, and sequence maximum are project choices. This is not a reproduction of the QLoRA dataset or results.
* [ReFT, NeurIPS 2024](https://proceedings.nips.cc/paper_files/paper/2024/file/75008a0fba53bf13b0bb3b7bff986e0e-Paper-Conference.pdf) uses development-selected settings, AdamW/linear scheduling, batch searches including 32, epoch searches including 6, warmup searches including 0.06, and multiple final seeds. Stage 2 uses settings within those ranges; this mapper is not the ReFT method.
* [Toolformer](https://arxiv.org/abs/2302.04761) reports batch 128 and 10% warmup for a different full-model tool-use objective. Those values are not automatically transplanted into this QLoRA/mapper experiment.

Keep old configurations for reproducing previous pilot runs. Final presets have distinct checkpoint/result directories and explicitly change LoRA rank/targets and optimizer settings; compare them as a new protocol.

Before reporting final research claims, freeze hyperparameters using development only and run at least three independent seeds, for example `--seed 20260919`, `--seed 20260920`, and `--seed 20260921`. Report mean and standard deviation across those runs. Include matched clean baseline, native-tool baseline, trained-policy text feedback, and actual mapper pipeline results; oracle mapper results alone cannot demonstrate the method's improvement. These runs can be submitted sequentially on one GPU.

The final presets keep `evaluation.seed` fixed while `--seed` changes the training seed, so comparisons across seeds use the same masked tool catalogue. Final bundles retain separate training and evaluation seeds.

## Local verification

From PowerShell in the project:

```powershell
.venv\Scripts\python.exe -m pip install 'matplotlib>=3.9,<4'
.venv\Scripts\python.exe -m pytest tests/test_training_batches.py -q --basetemp=.local/training-check-temp
.venv\Scripts\python.exe scripts/run_final_training.py --help
```

The CPU fixtures verify batch weighting, mapper gradients, recovery, timing contracts and artifact generation. They are not model-accuracy experiments. The full Qwen/H100 training, chosen micro-batch fit, and final model quality require the actual server run.

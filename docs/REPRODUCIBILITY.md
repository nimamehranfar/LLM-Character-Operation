# Reproducibility

The frozen dataset is committed under `data/character_operations_50k/`. `SHA256SUMS.txt` records the exact hashes of the four JSONL splits and manifest.

The dataset generator uses seed `20260919` and the bundled lexical seed resource. Regenerating the dataset reproduces the committed JSONL bytes.

Primary model revisions are pinned in the experiment configs:

- Qwen3-8B: `b968826d9c46dd6066d109eabc6255188de91218`
- Qwen3-4B: `1cfa9a7208912126459214e8b04321603b3df60c`

Training seeds, LoRA settings, candidate injection layers, VRAM limits, early-stopping settings, and recovery intervals are all stored in TOML configs.

GPU kernels and library implementations can still introduce small cross-hardware numerical differences. For the closest reproduction, keep the same CUDA/PyTorch/Transformers/PEFT/bitsandbytes stack and GPU family in addition to using the pinned configs and model revisions.

No model checkpoints are bundled. Reproducing trained results therefore starts from the pinned base model and runs the training commands documented in `README.md`.

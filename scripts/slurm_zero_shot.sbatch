#!/bin/bash
#SBATCH --job-name=character-baselines
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
if [[ ! -x .venv/bin/python ]]; then
    echo "Missing environment. Run: python3 scripts/setup_environment.py"
    exit 1
fi
export PYTHONUNBUFFERED=1
# SLURM supplies CUDA_VISIBLE_DEVICES. Do not replace it with physical IDs.
srun .venv/bin/python scripts/run_zero_shot_matrix.py --auto-download "$@"

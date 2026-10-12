#!/bin/bash
#SBATCH --job-name=character-training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --signal=TERM@120
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
if [[ ! -x .venv/bin/python ]]; then
    echo "Missing environment. Run scripts/setup_environment.py first."
    exit 1
fi
export PYTHONUNBUFFERED=1
# Keep SLURM's CUDA_VISIBLE_DEVICES assignment intact. Each trainer uses cuda:0.
srun .venv/bin/python scripts/run_final_training.py "$@"

#!/usr/bin/env bash
#SBATCH --job-name=grain-smoke
#SBATCH --partition=gpu-highvram
#SBATCH --gpus=1
#SBATCH --constraint=a40|a100|l40s|h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke-%j.out
#SBATCH --error=logs/smoke-%j.err

set -euo pipefail
: "${PROJECT_ROOT:?Export PROJECT_ROOT before submitting}"
cd "$PROJECT_ROOT"
PYTHON="${ENV_PREFIX:-$PROJECT_ROOT/.hpc-env}/bin/python"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
nvidia-smi
"$PYTHON" -m voron_matsam.baseline doctor
"$PYTHON" -m voron_matsam.baseline run \
    --manifest "$PROJECT_ROOT/data/processed/paper-bright/manifest.json" \
    --checkpoint "$PROJECT_ROOT/data/models/cpsam" \
    --split 05 --seed 0 --epochs 2 --aug-multiplier 1 --limit-test-images 1 \
    --output "${SMOKE_OUTPUT:-$PROJECT_ROOT/outputs/smoke-${SLURM_JOB_ID}}"

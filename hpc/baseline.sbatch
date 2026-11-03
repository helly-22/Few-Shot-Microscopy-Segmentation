#!/usr/bin/env bash
#SBATCH --job-name=grain-baseline
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=a40|a100|l40s|h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-5%1
#SBATCH --output=logs/baseline-%A_%a.out
#SBATCH --error=logs/baseline-%A_%a.err

set -euo pipefail
: "${PROJECT_ROOT:?Submit through bash hpc/submit_baselines.sh}"
cd "$PROJECT_ROOT"
PYTHON="${ENV_PREFIX:-$PROJECT_ROOT/.hpc-env}/bin/python"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="${SEED:-0}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/grain-mpl-${SLURM_JOB_ID}"
splits=(zero 05 10 25 50 75)
split="${splits[$SLURM_ARRAY_TASK_ID]}"
output="${RESULTS_ROOT:-$PROJECT_ROOT/outputs/baselines}/${split}_seed${SEED:-0}"
nvidia-smi
"$PYTHON" -m voron_matsam.baseline doctor
"$PYTHON" -m voron_matsam.baseline run \
    --manifest "$PROJECT_ROOT/data/processed/paper-bright/manifest.json" \
    --checkpoint "$PROJECT_ROOT/data/models/cpsam" \
    --split "$split" --seed "${SEED:-0}" --output "$output"

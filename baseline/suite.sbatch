#!/usr/bin/env bash
#SBATCH --job-name=grain-suite
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=a40|a100|l40s|h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-9%1
#SBATCH --output=logs/suite-%A_%a.out
#SBATCH --error=logs/suite-%A_%a.err

set -euo pipefail
: "${PROJECT_ROOT:?Submit through baseline/submit.sh}"
: "${METHOD:?Choose a method with baseline/submit.sh}"
cd "$PROJECT_ROOT"
PYTHON="${BASELINE_ENV_ROOT:-$PROJECT_ROOT/.baseline-envs}/$METHOD/bin/python"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMBA_NUM_THREADS="$OMP_NUM_THREADS"
export TF_NUM_INTRAOP_THREADS="$OMP_NUM_THREADS"
export TF_NUM_INTEROP_THREADS=1
export CELLPOSE_LOCAL_MODELS_PATH="$PROJECT_ROOT/data/models/cellpose-cache/$METHOD"
index="${SLURM_ARRAY_TASK_ID:-0}"
seed="$((index / 2))"
shots=2
if (( index % 2 == 1 )); then shots=5; fi
export PYTHONHASHSEED="$seed"
arguments=()
suffix=""
if [[ "${SMOKE:-0}" == 1 ]]; then arguments+=(--smoke); suffix="_smoke${SLURM_JOB_ID}"; fi
if [[ "$METHOD" == watershed ]]; then arguments+=(--cpu); fi
if [[ "$METHOD" != watershed ]]; then nvidia-smi; fi
name="${shots}shot_seed${seed}"
if [[ "$METHOD" == watershed || "$METHOD" == sam2 ]]; then name="zero_shot"; fi
doctor=()
if [[ "$METHOD" == watershed ]]; then doctor+=(--cpu); fi
"$PYTHON" -m baseline doctor --method "$METHOD" "${doctor[@]}"
"$PYTHON" -m baseline run --method "$METHOD" --shots "$shots" --seed "$seed" \
    --manifest "${MANIFEST:-data/processed/paper-bright/manifest.json}" \
    --plan "${PLAN:-outputs/baseline_plan.json}" \
    --output "${RESULTS_ROOT:-outputs/baseline_suite}/$METHOD/${name}${suffix}" \
    "${arguments[@]}"

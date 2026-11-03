#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
device="${SMALL_DEVICE:-cuda}"
case "$device" in cpu|cuda) ;; *) echo "SMALL_DEVICE must be cpu or cuda" >&2; exit 1;; esac
PYTHON="${SMALL_PYTHON:-${BASELINE_ENV_ROOT:-$PROJECT_ROOT/.baseline-envs}/unet_small_$device/bin/python}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
extra=()
if [[ "$device" == cpu ]]; then extra+=(--cpu); else nvidia-smi; fi
"$PYTHON" -m baseline doctor --method unet_watershed "${extra[@]}"
"$PYTHON" -m baseline run --method unet_watershed \
    --config baseline/configs/unet_small.json --shots 2 --seed 0 \
    --manifest "${MANIFEST:-data/processed/paper-bright/manifest.json}" \
    --plan "${PLAN:-outputs/baseline_plan.json}" --evaluate-split validation \
    --output "${SMALL_OUTPUT:-outputs/small_unet/validation-${SLURM_JOB_ID:-local}}" \
    "${extra[@]}"

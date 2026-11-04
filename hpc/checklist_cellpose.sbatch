#!/usr/bin/env bash
#SBATCH --job-name=checklist-cpsam
#SBATCH --partition=gpu-highvram
#SBATCH --gpus=1
#SBATCH --constraint=a40|a100|l40s|h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/checklist-%j.out
#SBATCH --error=logs/checklist-%j.err
set -euo pipefail
: "${PROJECT_ROOT:?}"
: "${CHECKLIST_RUN_ROOT:?}"
cd "$PROJECT_ROOT"
PYTHON="${ENV_PREFIX:-$PROJECT_ROOT/.hpc-env}/bin/python"
export PYTHONPATH="${CHECKLIST_SOURCE_ROOT:?}/src:$CHECKLIST_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/checklist-mpl-$SLURM_JOB_ID"
manifest="$PROJECT_ROOT/data/processed/connected-v2/manifest.json"
"$PYTHON" -m pytest -q "$CHECKLIST_SOURCE_ROOT/tests"
"$PYTHON" -m voron_matsam.baseline check-data --manifest "$manifest"
nvidia-smi
for split in zero 05; do
  "$PYTHON" -m voron_matsam.baseline run --manifest "$manifest" \
    --checkpoint "$PROJECT_ROOT/data/models/cpsam" --split "$split" \
    --output "$CHECKLIST_RUN_ROOT/${split}_seed0" --save-raw
done
checkpoint="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["trained_checkpoint"])' "$CHECKLIST_RUN_ROOT/05_seed0/run.json")"
"$PYTHON" "$CHECKLIST_SOURCE_ROOT/scripts/cellpose_validation_diagnostic.py" \
  --manifest "$manifest" --checkpoint "$checkpoint" \
  --output "$CHECKLIST_RUN_ROOT/05_validation"

#!/usr/bin/env bash
#SBATCH --job-name=checklist-data
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/checklist-data-%j.out
#SBATCH --error=logs/checklist-data-%j.err
set -euo pipefail
: "${PROJECT_ROOT:?}"
cd "$PROJECT_ROOT"
PYTHON="${ENV_PREFIX:-$PROJECT_ROOT/.hpc-env}/bin/python"
export PYTHONPATH="${CHECKLIST_SOURCE_ROOT:?}/src:$CHECKLIST_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
destination="$PROJECT_ROOT/data/processed/connected-v2"
if [[ -d "$destination" ]]; then
  "$PYTHON" -m voron_matsam.baseline check-data --manifest "$destination/manifest.json"
else
  "$PYTHON" -m voron_matsam.baseline prepare --output "$destination" --label-policy connected_uint16
fi

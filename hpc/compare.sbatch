#!/usr/bin/env bash
#SBATCH --job-name=grain-compare
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/compare-%j.out
#SBATCH --error=logs/compare-%j.err

set -euo pipefail
: "${PROJECT_ROOT:?Submit through hpc/submit_reproduction.sh}"
: "${RUN_ROOT:?Submit through hpc/submit_reproduction.sh}"
: "${RESULTS_ROOT:?Submit through hpc/submit_reproduction.sh}"
cd "$PROJECT_ROOT"
"${ENV_PREFIX:-$PROJECT_ROOT/.hpc-env}/bin/python" -m voron_matsam.baseline compare \
    --results "$RESULTS_ROOT" --output "$RUN_ROOT/paper_comparison.json"

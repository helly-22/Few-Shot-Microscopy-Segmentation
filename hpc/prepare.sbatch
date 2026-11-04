#!/usr/bin/env bash
#SBATCH --job-name=grain-prepare
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/prepare-%j.out
#SBATCH --error=logs/prepare-%j.err

set -euo pipefail
: "${PROJECT_ROOT:?Submit through hpc/submit_reproduction.sh}"
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
bash hpc/setup_unity.sh
bash hpc/fetch_assets.sh
"${ENV_PREFIX:-$PROJECT_ROOT/.hpc-env}/bin/python" scripts/audit_prepared_data.py \
    --output "${RUN_ROOT:-$PROJECT_ROOT/outputs}/prepared_data_audit.json"

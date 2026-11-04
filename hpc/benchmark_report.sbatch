#!/usr/bin/env bash
#SBATCH --job-name=other-model-report
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/other-report-%j.out
#SBATCH --error=logs/other-report-%j.err
set -euo pipefail
: "${PROJECT_ROOT:?}"
: "${BENCHMARK_ROOT:?}"
: "${BENCHMARK_SOURCE:?}"
cd "$BENCHMARK_SOURCE"
export PYTHONPATH="$BENCHMARK_SOURCE/src:$BENCHMARK_SOURCE"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/other-report-mpl-$SLURM_JOB_ID"
export OMP_NUM_THREADS=4
args=(--root "$BENCHMARK_ROOT" --manifest "$PROJECT_ROOT/data/processed/connected-v2/manifest.json"
      --cellpose-reference "$PROJECT_ROOT/outputs/checklist-20261104T1715Z/report/results.json")
if [[ "${PARTIAL_REPORT:-0}" == 1 ]]; then args+=(--partial); fi
"$PROJECT_ROOT/.hpc-env/bin/python" "$BENCHMARK_ROOT/report_other_models.py" "${args[@]}"

#!/usr/bin/env bash
#SBATCH --job-name=checklist-report
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/checklist-report-%j.out
#SBATCH --error=logs/checklist-report-%j.err
set -euo pipefail
: "${PROJECT_ROOT:?}"
: "${CHECKLIST_RUN_ROOT:?}"
cd "$PROJECT_ROOT"
PYTHON="${ENV_PREFIX:-$PROJECT_ROOT/.hpc-env}/bin/python"
export PYTHONPATH="${CHECKLIST_SOURCE_ROOT:?}/src:$CHECKLIST_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/checklist-report-mpl-$SLURM_JOB_ID"
"$PYTHON" -c 'import matplotlib; print("Report Matplotlib:", matplotlib.__version__)'
"$PYTHON" "$CHECKLIST_RUN_ROOT/report_checklist_runs.py" \
  --run-root "$CHECKLIST_RUN_ROOT" \
  --manifest "$PROJECT_ROOT/data/processed/connected-v2/manifest.json" \
  --source-root "$CHECKLIST_SOURCE_ROOT" \
  --targets "$CHECKLIST_SOURCE_ROOT/configs/paper_targets.csv"
"$PYTHON" -c 'import shutil,sys; shutil.make_archive(sys.argv[1]+"/checklist_results", "zip", sys.argv[1], "report")' "$CHECKLIST_RUN_ROOT"

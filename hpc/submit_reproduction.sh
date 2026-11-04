#!/usr/bin/env bash
# Submit preparation, smoke, six baselines, and comparison with success dependencies.
# Extra arguments apply to every job, e.g. --account=pi_yourgroup.
set -euo pipefail
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export ENV_PREFIX="${ENV_PREFIX:-$PROJECT_ROOT/.hpc-env}"
export RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/paper-$(date -u +%Y%m%dT%H%M%SZ)}"
export RESULTS_ROOT="$RUN_ROOT/baselines"
export SMOKE_OUTPUT="$RUN_ROOT/smoke"
mkdir -p logs "$(dirname "$RUN_ROOT")"
# Never combine a new submission with previous outputs or launch records.
mkdir "$RUN_ROOT"
mkdir "$RESULTS_ROOT"
record="$RUN_ROOT/jobs.tsv"
printf 'stage\tjob_id\n' > "$record"
submit() {
    local stage="$1"
    shift
    local submitted job
    if ! submitted="$(sbatch --parsable --export=ALL "$@")"; then
        echo "Submission failed for $stage; already submitted jobs are recorded in $record" >&2
        return 1
    fi
    job="${submitted%%;*}"
    [[ "$job" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $submitted" >&2; return 1; }
    printf '%s\t%s\n' "$stage" "$job" >> "$record"
    printf '%s\n' "$job"
}
prepare="$(submit prepare "$@" hpc/prepare.sbatch)"
smoke="$(submit smoke "$@" --dependency="afterok:$prepare" --kill-on-invalid-dep=yes hpc/smoke.sbatch)"
baselines="$(submit baselines "$@" --dependency="afterok:$smoke" --kill-on-invalid-dep=yes hpc/baseline.sbatch)"
submit compare "$@" --dependency="afterok:$baselines" --kill-on-invalid-dep=yes hpc/compare.sbatch > /dev/null
cat "$record"
printf 'Run directory: %s\n' "$RUN_ROOT"

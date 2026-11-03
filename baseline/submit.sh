#!/usr/bin/env bash
# Remaining arguments are passed to sbatch (account, resources, dependency, etc.).
set -euo pipefail
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export METHOD="${1:?Usage: bash baseline/submit.sh METHOD [sbatch options]}"
shift
case "$METHOD" in cellpose_sam|cellpose2|cellpose3|unet_watershed|sam2|stardist|watershed) ;; *) echo "Unknown method" >&2; exit 1;; esac
cd "$PROJECT_ROOT"
mkdir -p logs
[[ -x "${BASELINE_ENV_ROOT:-$PROJECT_ROOT/.baseline-envs}/$METHOD/bin/python" ]] || { echo "Set up the method environment first" >&2; exit 1; }
[[ -f "${PLAN:-outputs/baseline_plan.json}" ]] || { echo "Create the shared split plan first" >&2; exit 1; }
extra=()
script="$PROJECT_ROOT/baseline/suite.sbatch"
if [[ "$METHOD" == watershed ]]; then script="$PROJECT_ROOT/baseline/watershed.sbatch"; fi
if [[ "$METHOD" == watershed || "$METHOD" == sam2 ]]; then extra+=(--array=0); fi
if [[ "${SMOKE:-0}" == 1 ]]; then extra+=(--array=0 --time=00:30:00); fi
sbatch "${extra[@]}" "$@" --export=ALL "$script"

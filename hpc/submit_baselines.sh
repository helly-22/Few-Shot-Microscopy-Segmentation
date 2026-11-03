#!/usr/bin/env bash
# All extra arguments go to sbatch, e.g. --account=... --constraint=a5500.
set -euo pipefail
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs outputs/baselines
PYTHON="${ENV_PREFIX:-$PROJECT_ROOT/.hpc-env}/bin/python"
"$PYTHON" -m voron_matsam.baseline check-data --manifest data/processed/paper-bright/manifest.json
[[ -f data/models/cpsam ]] || { echo "Run bash hpc/fetch_assets.sh first" >&2; exit 1; }
sbatch "$@" --export=ALL "$PROJECT_ROOT/hpc/baseline.sbatch"

#!/usr/bin/env bash
# Download before GPU allocation. No network access is needed by training jobs.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ENV_PREFIX:-$PROJECT_ROOT/.hpc-env}/bin/python"
cd "$PROJECT_ROOT"
if [[ ! -f data/raw/grains-v4.zip ]]; then
    "$PYTHON" -m voron_matsam.baseline download-data
fi
"$PYTHON" -m voron_matsam.baseline download-weights
if [[ ! -f data/processed/paper-bright/manifest.json ]]; then
    "$PYTHON" -m voron_matsam.baseline prepare
fi
"$PYTHON" -m voron_matsam.baseline check-data --manifest data/processed/paper-bright/manifest.json

#!/usr/bin/env bash
# Minimal small-U-Net environment; no foundation-model weights are downloaded.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
device="${1:-cuda}"
case "$device" in cpu|cuda) ;; *) echo "Usage: bash baseline/setup_small_unity.sh [cpu|cuda]" >&2; exit 1;; esac
prefix="${BASELINE_ENV_ROOT:-$PROJECT_ROOT/.baseline-envs}/unet_small_$device"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PROJECT_ROOT/.cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$PROJECT_ROOT/.cache/conda}"
module load conda/latest
eval "$(conda shell.bash hook)"
if [[ ! -x "$prefix/bin/python" ]]; then
    conda create --prefix "$prefix" python=3.11 pip -y
fi
conda activate "$prefix"
index=cu118
if [[ "$device" == cpu ]]; then index=cpu; fi
python -m pip install torch==2.6.0 --index-url "https://download.pytorch.org/whl/$index"
python -m pip install -r "$PROJECT_ROOT/baseline/requirements/unet_watershed.txt"
python -m pip install --no-deps -e "$PROJECT_ROOT"
python -m pip check
cd "$PROJECT_ROOT"
python -m baseline doctor --method unet_watershed --cpu
mkdir -p outputs/runtime
python -m pip freeze > "outputs/runtime/unet-small-$device-pip-freeze.txt"

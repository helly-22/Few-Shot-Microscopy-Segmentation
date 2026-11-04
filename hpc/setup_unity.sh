#!/usr/bin/env bash
# Run in a Unity interactive CPU allocation after copying the project to /work.
set -eo pipefail
if ! type module >/dev/null 2>&1; then
    source /etc/profile
fi
set -u
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ENV_PREFIX="${ENV_PREFIX:-$PROJECT_ROOT/.hpc-env}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PROJECT_ROOT/.cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$PROJECT_ROOT/.cache/conda}"
module load conda/latest
eval "$(conda shell.bash hook)"
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    conda create --prefix "$ENV_PREFIX" python=3.11 pip -y
fi
conda activate "$ENV_PREFIX"
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r "$PROJECT_ROOT/requirements/baseline.txt"
python -m pip install --no-deps -e "$PROJECT_ROOT"
python -m pip check
python -m pytest "$PROJECT_ROOT/tests" -q
mkdir -p "$PROJECT_ROOT/outputs/runtime"
python -m pip freeze > "$PROJECT_ROOT/outputs/runtime/unity-pip-freeze.txt"
conda list --explicit > "$PROJECT_ROOT/outputs/runtime/unity-conda-explicit.txt"
echo "Environment ready at $ENV_PREFIX"

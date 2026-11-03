#!/usr/bin/env bash
# Run inside a CPU allocation. One mutually isolated environment per method.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
method="${1:?Usage: bash baseline/setup_unity.sh METHOD}"
case "$method" in cellpose_sam|cellpose2|cellpose3|unet_watershed|sam2|stardist|watershed) ;; *) echo "Unknown method: $method" >&2; exit 1;; esac
prefix="${BASELINE_ENV_ROOT:-$PROJECT_ROOT/.baseline-envs}/$method"
export CELLPOSE_LOCAL_MODELS_PATH="$PROJECT_ROOT/data/models/cellpose-cache/$method"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PROJECT_ROOT/.cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$PROJECT_ROOT/.cache/conda}"
module load conda/latest
eval "$(conda shell.bash hook)"
if [[ ! -x "$prefix/bin/python" ]]; then
    conda create --prefix "$prefix" python=3.11 pip -y
fi
conda activate "$prefix"
case "$method" in
    cellpose2|cellpose3)
        python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121 ;;
    stardist|watershed)
        python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu ;;
    *)
        python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 ;;
esac
export SAM2_BUILD_CUDA=0
python -m pip install -r "$PROJECT_ROOT/baseline/requirements/$method.txt"
python -m pip install --no-deps -e "$PROJECT_ROOT"
python -m pip check
cd "$PROJECT_ROOT"
python -m baseline doctor --method "$method" --cpu
# Fetch serially during setup. Training jobs never request implicit downloads.
python -m baseline assets --method "$method"
mkdir -p outputs/runtime
python -m pip freeze > "outputs/runtime/$method-pip-freeze.txt"
echo "Ready: $prefix"

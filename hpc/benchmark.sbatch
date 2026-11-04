#!/usr/bin/env bash
#SBATCH --job-name=grain-other-model
#SBATCH --partition=gpu-highvram
#SBATCH --gpus=1
#SBATCH --constraint=a40|a100|l40s|h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/other-model-%j.out
#SBATCH --error=logs/other-model-%j.err
set -euo pipefail
: "${PROJECT_ROOT:?}"
: "${BENCHMARK_ROOT:?}"
: "${BENCHMARK_SOURCE:?}"
: "${BENCHMARK_METHOD:?}"
cd "$BENCHMARK_SOURCE"
export PYTHONPATH="$BENCHMARK_SOURCE/src:$BENCHMARK_SOURCE"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/other-model-mpl-$SLURM_JOB_ID"
PYTHON="${ENV_PREFIX:-$PROJECT_ROOT/.hpc-env}/bin/python"
args=(--method "$BENCHMARK_METHOD" --manifest "$PROJECT_ROOT/data/processed/connected-v2/manifest.json"
      --output "$BENCHMARK_ROOT/$BENCHMARK_METHOD" --seed 0 --steps 100)
case "$BENCHMARK_METHOD" in
  unet) args+=(--epochs 100) ;;
  stardist)
    args+=(--epochs 400)
    libraries="$("$PYTHON" -c 'import glob,site; print(":".join(p for root in site.getsitepackages() for p in glob.glob(root+"/nvidia/*/lib")))')"
    export LD_LIBRARY_PATH="$libraries${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ;;
  matsam) args+=(--checkpoint "$PROJECT_ROOT/data/models/matsam_sam_vit_l.pth"
                 --upstream "$PROJECT_ROOT/external/matsam-benchmark") ;;
  *) echo "Unknown GPU benchmark: $BENCHMARK_METHOD" >&2; exit 1 ;;
esac
nvidia-smi
"$PYTHON" -m baseline.benchmark "${args[@]}"

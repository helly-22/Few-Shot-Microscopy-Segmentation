#!/usr/bin/env bash
#SBATCH --job-name=grain-small-unet
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/small-unet-%j.out
#SBATCH --error=logs/small-unet-%j.err

set -euo pipefail
: "${PROJECT_ROOT:?Export PROJECT_ROOT to the repository root before sbatch}"
export SMALL_DEVICE=cuda
bash "$PROJECT_ROOT/baseline/run_small.sh"

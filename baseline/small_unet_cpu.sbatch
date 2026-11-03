#!/usr/bin/env bash
#SBATCH --job-name=grain-small-unet-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/small-unet-cpu-%j.out
#SBATCH --error=logs/small-unet-cpu-%j.err

set -euo pipefail
: "${PROJECT_ROOT:?Export PROJECT_ROOT to the repository root before sbatch}"
export SMALL_DEVICE=cpu
bash "$PROJECT_ROOT/baseline/run_small.sh"

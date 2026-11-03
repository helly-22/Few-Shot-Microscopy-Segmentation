#!/usr/bin/env bash
#SBATCH --job-name=grain-watershed
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-9%1
#SBATCH --output=logs/watershed-%A_%a.out
#SBATCH --error=logs/watershed-%A_%a.err

set -euo pipefail
bash "$PROJECT_ROOT/baseline/suite.sbatch"

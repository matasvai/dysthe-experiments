#!/usr/bin/env bash
#SBATCH --job-name=dysthe-breathing-preview
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export RUN_DIR="${RUN_DIR:-runs/breathing-preview-12288}"
export DT="${DT:-0.5}"
export GRID="${GRID:-32}"
export WALL_SECONDS="${WALL_SECONDS:-1500}"
# Execute the shared body; its SBATCH comments do not change this allocation.
bash cluster/breathing.slurm

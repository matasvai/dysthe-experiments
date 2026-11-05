#!/usr/bin/env bash
#SBATCH --job-name=dysthe-long-cpu
#SBATCH --partition=med
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
set -euo pipefail
: "${DYSTHE_VISUAL_ROOT:?Set absolute path to the visualization package root}"
: "${DYSTHE_PYTHON:?Set absolute path to a prepared Python environment}"
: "${DYSTHE_RUN_DIR:?Set a NEW output directory visible to compute nodes}"
DYSTHE_LONG_CONFIG="${DYSTHE_LONG_CONFIG:-$DYSTHE_VISUAL_ROOT/long_horizon/configs/periodic-targets.json}"
cd "$DYSTHE_VISUAL_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
"$DYSTHE_PYTHON" long_horizon/run_long_horizon.py --config "$DYSTHE_LONG_CONFIG" --output "$DYSTHE_RUN_DIR"

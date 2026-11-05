#!/usr/bin/env bash
#SBATCH --job-name=dysthe-long-reference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
"$PYTHON_BIN" scripts/long_time_reference.py \
  --max-time "${MAX_TIME:-2}" \
  --output "${REFERENCE_DIR:-runs/long-reference-${SLURM_JOB_ID:-local}}"

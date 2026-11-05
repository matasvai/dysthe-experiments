#!/usr/bin/env bash
#SBATCH --job-name=dysthe-breathing
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
RUN_DIR="${RUN_DIR:-runs/breathing-12288}"
"$PYTHON_BIN" -c "import numpy, scipy, matplotlib"
"$PYTHON_BIN" scripts/late_window_reference.py --config configs/benchmark.json \
  --output "$RUN_DIR" --integration-end "${TOTAL_T:-12288}" \
  --start 1500 --end 2500 --snapshots 501 --dt "${DT:-0.04}" \
  --shape "${GRID:-64}" --breathing --showcases --wall-seconds "${WALL_SECONDS:-82000}"
"$PYTHON_BIN" scripts/plot_breathing.py --run "$RUN_DIR" --output "$RUN_DIR/report"

#!/usr/bin/env bash
#SBATCH --job-name=dysthe-late-window
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
WINDOW_START="${WINDOW_START:-1024}"
WINDOW_END="${WINDOW_END:-2048}"
# Fixed folder allows a later submission to resume; a lock prevents concurrent writers.
"$PYTHON_BIN" scripts/late_window_reference.py \
  --config "${CONFIG:-configs/benchmark.json}" \
  --output "${REFERENCE_DIR:-runs/late-window-${WINDOW_START}-${WINDOW_END}}" \
  --start "$WINDOW_START" --end "$WINDOW_END" --dt "${DT:-0.01}" --shape "${GRID:-64}" \
  --snapshots "${SNAPSHOTS:-257}" \
  --wall-seconds "${WALL_SECONDS:-82000}"

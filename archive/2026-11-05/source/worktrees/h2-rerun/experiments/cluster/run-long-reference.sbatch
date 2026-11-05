#!/bin/bash -l
#SBATCH --job-name=dysthe-reference
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%A_%a.out

# Slurm executes a spool copy: anchor to the submitted repository, not $0.
set -euo pipefail
die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
stage="${1:-reference}"
[[ $# -le 1 ]] || die 'Usage: sbatch run-long-reference.sbatch [reference|sensitivity|domain|summarize]'
case "$stage" in reference|sensitivity|domain|summarize) ;; *) die "Unknown CPU stage: $stage" ;; esac
[[ -n "${SLURM_JOB_ID:-}" ]] || die 'Submit through sbatch; compute work must not run on a login node.'
[[ "${SLURM_NTASKS:-1}" == 1 && "${SLURM_JOB_NUM_NODES:-1}" == 1 ]] || die 'Exactly one task on one node is required.'
[[ "${DYSTHE_PYTHON:-}" == /* && -x "${DYSTHE_PYTHON:-}" ]] || die 'Set DYSTHE_PYTHON to an executable absolute path in the prepared environment.'
repo_dir="${DYSTHE_EXPERIMENTS_DIR:-${SLURM_SUBMIT_DIR:-}}"
[[ -n "$repo_dir" ]] || die 'Set DYSTHE_EXPERIMENTS_DIR or submit from the repository.'
cd -- "$repo_dir"
repo_dir="$PWD"
[[ -f src/dysthe_experiments/long_distance.py ]] || die 'Repository anchor is not dysthe-experiments with the long-distance campaign.'
long_config="${LONG_CONFIG:-$repo_dir/configs/long-distance-a100.json}"
run_dir="${RUN_DIR:-$repo_dir/runs/long-distance-a100}"
[[ "$long_config" == /* ]] || long_config="$repo_dir/$long_config"
[[ "$run_dir" == /* ]] || run_dir="$repo_dir/$run_dir"
[[ -f "$long_config" ]] || die "Configuration not found: $long_config"
if [[ "$stage" == summarize ]]; then
    [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || die 'Summarize is one job, not an array.'
    task_index=0
else
    task_index="${SLURM_ARRAY_TASK_ID:-}"
    [[ "$task_index" =~ ^[0-9]+$ ]] || die 'Reference and diagnostic stages require a numeric Slurm array task ID.'
fi
export PYTHONPATH="$repo_dir/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="$run_dir/launcher/$stage-$task_index/matplotlib"
mkdir -p -- "$MPLCONFIGDIR"
printf 'Dysthe long-distance: job=%s stage=%s task=%s run=%s\n' "$SLURM_JOB_ID" "$stage" "$task_index" "$run_dir"
srun --ntasks=1 "$DYSTHE_PYTHON" - "$long_config" "${SLURM_CPUS_PER_TASK:-1}" <<'PY'
import json, sys
threads = json.load(open(sys.argv[1]))['threads']
if type(threads) is not int or threads < 1 or threads > int(sys.argv[2]):
    raise SystemExit('Configuration requests invalid or more CPU threads than allocated')
PY
srun --ntasks=1 "$DYSTHE_PYTHON" -m dysthe_experiments.long_distance \
    --config "$long_config" --run "$run_dir" --stage "$stage" \
    --task-index "$task_index" --device cpu

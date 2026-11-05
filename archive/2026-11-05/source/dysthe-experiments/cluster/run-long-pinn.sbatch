#!/bin/bash -l
#SBATCH --job-name=dysthe-long-pinn
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:A100-PCI-40GB:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%A_%a.out

# One array member owns one case/seed. There is no installation or CPU fallback.
set -euo pipefail
die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
[[ $# == 0 ]] || die 'Usage: sbatch --array=... cluster/run-long-pinn.sbatch'
[[ -n "${SLURM_JOB_ID:-}" ]] || die 'Submit through sbatch; compute work must not run on a login node.'
[[ "${SLURM_NTASKS:-1}" == 1 && "${SLURM_JOB_NUM_NODES:-1}" == 1 ]] || die 'Exactly one task on one node is required.'
task_index="${SLURM_ARRAY_TASK_ID:-}"
[[ "$task_index" =~ ^[0-9]+$ ]] || die 'A numeric Slurm array task ID is required.'
[[ "${DYSTHE_PYTHON:-}" == /* && -x "${DYSTHE_PYTHON:-}" ]] || die 'Set DYSTHE_PYTHON to an executable absolute path in the prepared environment.'
command -v flock >/dev/null || die 'The per-task launcher guard requires flock.'
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
launcher_dir="$run_dir/launcher/pinn-$task_index"
mkdir -p -- "$launcher_dir"
exec 9>"$launcher_dir/.lock"
flock -n 9 || die 'Another launcher owns this case/seed; wait or choose a different RUN_DIR.'
export PYTHONPATH="$repo_dir/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="$launcher_dir/matplotlib"
mkdir -p -- "$MPLCONFIGDIR"
printf 'Dysthe long-distance PINN: job=%s task=%s run=%s\n' "$SLURM_JOB_ID" "$task_index" "$run_dir"
srun --ntasks=1 "$DYSTHE_PYTHON" "$repo_dir/cluster/long-preflight.py" \
    "$long_config" "$launcher_dir" "$repo_dir" "$task_index"
srun --ntasks=1 "$DYSTHE_PYTHON" -m dysthe_experiments.long_distance \
    --config "$long_config" --run "$run_dir" --stage pinn \
    --task-index "$task_index" --device cuda

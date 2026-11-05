#!/bin/bash -l
#SBATCH --job-name=dysthe-slabs
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:A100-PCI-40GB:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

# Submit from dysthe-experiments; Slurm runs a copy of this script from its spool.
# No dependency installation, network access, or job submission occurs here.
set -euo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
stage="${1:-all}"
[[ $# -le 1 ]] || die 'Usage: sbatch cluster/run-a100-pilot.sbatch [all|prepare|pinns|diagnostics]'
case "$stage" in all|prepare|pinns|diagnostics) ;; *) die "Unknown stage: $stage" ;; esac
[[ -n "${SLURM_JOB_ID:-}" ]] || die 'Submit through sbatch; do not run this workload on a login node.'
[[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || die 'Job arrays are unsupported: one process owns each run directory.'
[[ "${SLURM_NTASKS:-1}" == 1 ]] || die 'This pilot requires exactly one Slurm task.'
[[ "${SLURM_JOB_NUM_NODES:-1}" == 1 ]] || die 'This pilot requires exactly one node.'
[[ -n "${DYSTHE_PYTHON:-}" ]] || die 'Set DYSTHE_PYTHON to the absolute path of the prepared Python environment.'
[[ "$DYSTHE_PYTHON" == /* && -x "$DYSTHE_PYTHON" ]] || die 'DYSTHE_PYTHON must be an executable absolute path.'
command -v flock >/dev/null || die 'The run-directory guard requires flock.'

repo_dir="${DYSTHE_EXPERIMENTS_DIR:-${SLURM_SUBMIT_DIR:-}}"
[[ -n "$repo_dir" ]] || die 'Submit from dysthe-experiments, or set DYSTHE_EXPERIMENTS_DIR.'
cd -- "$repo_dir"
repo_dir="$PWD"
[[ -f src/dysthe_experiments/cpu_pilot.py ]] || die 'Repository anchor is not dysthe-experiments.'

pilot_config="${PILOT_CONFIG:-$repo_dir/configs/a100-pilot.json}"
run_dir="${RUN_DIR:-$repo_dir/runs/a100-pilot}"
[[ "$pilot_config" == /* ]] || pilot_config="$repo_dir/$pilot_config"
[[ "$run_dir" == /* ]] || run_dir="$repo_dir/$run_dir"
[[ -f "$pilot_config" ]] || die "Configuration not found: $pilot_config"
mkdir -p -- "$run_dir"
exec 9>"$run_dir/.pilot.lock"
flock -n 9 || die 'Another launcher owns this run directory; choose a separate RUN_DIR.'

# Core and learning resolve from the prepared environment's pinned packages.
export PYTHONPATH="$repo_dir/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="$run_dir/.cache/matplotlib"
mkdir -p -- "$MPLCONFIGDIR"

printf 'Dysthe slab PINN: job=%s stage=%s config=%s run=%s\n' "$SLURM_JOB_ID" "$stage" "$pilot_config" "$run_dir"

# Check the allocated device and the complete local Python source before any
# expensive stage. A restart must use the same source, config and runtime.
srun --ntasks=1 "$DYSTHE_PYTHON" - "$pilot_config" "$run_dir" "$repo_dir" "$stage" <<'PY'
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
import torch
import dysthe_core
import dysthe_learning
import dysthe_experiments
from dysthe_core.torch_spatial import rhs
from dysthe_experiments.cpu_pilot import read_config
from dysthe_learning.pilot_models import SpectralPINN

config_path, run_path, repo_path = map(Path, sys.argv[1:4])
c = read_config(config_path)
if c['threads'] > int(os.environ.get('SLURM_CPUS_PER_TASK', '1')):
    raise RuntimeError('Configuration requests more CPU threads than allocated')
if torch.__version__.split('+')[0].split('.')[:2] != ['2', '10']:
    raise RuntimeError('This pilot was locally tested with PyTorch 2.10; prepare a CUDA-enabled 2.10.x environment')
if not torch.cuda.is_available() or torch.version.cuda is None:
    raise RuntimeError('A CUDA-enabled PyTorch build and an allocated GPU are required; no CPU fallback')
if torch.cuda.device_count() != 1:
    raise RuntimeError('Exactly one GPU must be visible through the Slurm allocation')
gpu = torch.cuda.get_device_name(0)
if 'A100' not in gpu:
    raise RuntimeError(f'This launcher requires an A100, received {gpu}')
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# Compare the complete RHS in double precision on a smooth nonconstant field.
# This is an arithmetic/runtime gate, not a numerical-accuracy or learning result.
tau = torch.arange(64, dtype=torch.float64) * (2 * np.pi / 64) - np.pi
u = 0.5 + 0.1 * torch.exp(2j * tau) - 0.03j * torch.exp(-3j * tau)
cpu_rhs = rhs(u, 0.05, 2 * np.pi)
gpu_rhs = rhs(u.cuda(), 0.05, 2 * np.pi).cpu()
torch.testing.assert_close(gpu_rhs, cpu_rhs, rtol=1e-10, atol=1e-11)
rhs_relative_error = float(torch.linalg.vector_norm(gpu_rhs - cpu_rhs)
                           / torch.linalg.vector_norm(cpu_rhs))
# Exercise the actual model, full residual, JVP and parameter backpropagation.
torch.manual_seed(20261026)
model = SpectralPINN(u.to(device='cuda', dtype=torch.complex64),
                     2 * np.pi, max_mode=8, width=8, depth=1)
s = torch.tensor([0.0, 0.02, 0.05], device='cuda')
y, dy = torch.func.jvp(model, (s,), (torch.ones_like(s),))
loss = (dy - rhs(y, 0.05, 2 * np.pi)).abs().square().mean()
loss.backward()
torch.cuda.synchronize()
gradients = [parameter.grad for parameter in model.parameters()]
if (not torch.isfinite(loss) or not torch.isfinite(y).all()
        or not torch.isfinite(dy).all()
        or any(g is None or not torch.isfinite(g).all() for g in gradients)
        or not any(torch.count_nonzero(g) > 0 for g in gradients)):
    raise RuntimeError('CUDA spectral PINN JVP/backward check failed')

sources = {}
for module in (dysthe_core, dysthe_learning, dysthe_experiments):
    package = Path(module.__file__).resolve().parent
    for path in sorted(package.rglob('*.py')):
        name = f'{module.__name__}/{path.relative_to(package)}'
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
for relative in ('scope.json', 'pyproject.toml', 'cluster/run-a100-pilot.sbatch'):
    sources[f'dysthe-experiments/{relative}'] = hashlib.sha256((repo_path / relative).read_bytes()).hexdigest()
identity = {'config': c, 'source_sha256': sources, 'runtime': {
    'python': platform.python_version(), 'torch': torch.__version__,
    'cuda_build': torch.version.cuda, 'numpy': np.__version__, 'scipy': scipy.__version__,
    'gpu': gpu, 'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
    'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG'],
    'nvidia_tf32_override': os.environ['NVIDIA_TF32_OVERRIDE'],
    'package_versions': {name: importlib.metadata.version(name)
                         for name in ('dysthe-core', 'dysthe-learning', 'dysthe-experiments')},
}}
lock_path = run_path / 'a100-runtime-lock.json'
if lock_path.exists():
    if json.loads(lock_path.read_text()) != identity:
        raise RuntimeError('Source, configuration or runtime changed; use a new RUN_DIR instead of mixing results')
else:
    if any(run_path.glob('**/last*.pt')):
        raise RuntimeError('Untracked existing checkpoints: start a fresh GPU run, or copy only verified reference data')
    lock_path.write_text(json.dumps(identity, indent=2) + '\n')
job = {'job_id': os.environ['SLURM_JOB_ID'], 'stage': sys.argv[4],
       'utc': datetime.now(timezone.utc).isoformat(), 'identity': identity,
       'cuda_rhs_relative_error': rhs_relative_error,
       'cuda_pinn_smoke_loss': float(loss.detach())}
(run_path / f'a100-job-{os.environ["SLURM_JOB_ID"]}.json').write_text(json.dumps(job, indent=2) + '\n')
print(json.dumps({'gpu': gpu, 'torch': torch.__version__, 'cuda_build': torch.version.cuda,
                  'source_files': len(sources), 'cuda_rhs_relative_error': rhs_relative_error,
                  'runtime_check': 'passed'}), flush=True)
PY

run_stage() {
    local selected_stage="$1" selected_device="$2"
    srun --ntasks=1 "$DYSTHE_PYTHON" -m dysthe_experiments.cpu_pilot \
        --config "$pilot_config" --run "$run_dir" --device "$selected_device" \
        --stage "$selected_stage" --pinn-variants slabs
}

if [[ "$stage" == all || "$stage" == prepare ]]; then run_stage prepare cpu; fi
if [[ "$stage" == all || "$stage" == pinns ]]; then run_stage pinns cuda; fi
if [[ "$stage" == all || "$stage" == diagnostics ]]; then run_stage diagnostics cpu; fi
printf 'Completed Dysthe stage %s; results remain in %s\n' "$stage" "$run_dir"

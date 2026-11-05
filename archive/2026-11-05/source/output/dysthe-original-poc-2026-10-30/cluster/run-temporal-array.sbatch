#!/usr/bin/env bash
# CPU reference calculations only. This file does not submit other jobs.
# med is the supplied CPU-partition example; override if the site differs.
#SBATCH --job-name=dysthe-temporal
#SBATCH --partition=med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --output=dysthe-temporal-%A_%a.out
set -euo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
: "${DYSTHE_POC_DIR:?Set DYSTHE_POC_DIR to the absolute standalone package directory}"
: "${DYSTHE_PYTHON:?Set DYSTHE_PYTHON to the absolute prepared Python executable}"
: "${DYSTHE_RESULTS_ROOT:?Set DYSTHE_RESULTS_ROOT to an absolute result-storage directory}"
: "${SLURM_ARRAY_JOB_ID:?Run as a Slurm array job}"
: "${SLURM_ARRAY_TASK_ID:?Run as a Slurm array job}"
[[ "$DYSTHE_POC_DIR" == /* && -f "$DYSTHE_POC_DIR/run.py" ]] || die 'Invalid package directory'
[[ "$DYSTHE_PYTHON" == /* && -x "$DYSTHE_PYTHON" ]] || die 'Invalid prepared Python executable'
[[ "$DYSTHE_RESULTS_ROOT" == /* ]] || die 'Results root must be absolute'
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ ]] || die 'Array task ID must be a nonnegative integer'

cd -- "$DYSTHE_POC_DIR"
config_path="${DYSTHE_CONFIG:-config-refined.json}"
[[ "$config_path" == /* ]] || config_path="$DYSTHE_POC_DIR/$config_path"
[[ -f "$config_path" ]] || die "Missing config: $config_path"
task_root="$DYSTHE_RESULTS_ROOT/temporal-$SLURM_ARRAY_JOB_ID/task-$SLURM_ARRAY_TASK_ID"
mkdir -p -- "$task_root"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"

# One epsilon per task preserves all within-epsilon model comparisons and
# all three profile families. Each task owns a different output directory.
"$DYSTHE_PYTHON" - "$config_path" "$task_root" "$SLURM_ARRAY_TASK_ID" <<'PY'
import hashlib
import json
import os
import platform
import socket
import sys
from pathlib import Path
import numpy
import scipy

source, root, index = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
config = json.loads(source.read_text())
if len(config['epsilons']) != 2 or index not in (0, 1):
    raise SystemExit('This array requires exactly two configured epsilon values and task IDs 0,1')
config['epsilons'] = [config['epsilons'][index]]
(root / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
files = [source, Path('run.py'), Path('test_run.py'), Path('requirements.txt')]
runtime = {
    'status': 'runtime_record_only_not_a_scientific_pass',
    'hostname': socket.gethostname(), 'platform': platform.platform(),
    'python': platform.python_version(), 'python_executable': sys.executable,
    'numpy': numpy.__version__, 'scipy': scipy.__version__,
    'slurm': {k: os.environ.get(k) for k in (
        'SLURM_JOB_ID', 'SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID',
        'SLURM_JOB_PARTITION', 'SLURM_CPUS_PER_TASK')},
    'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    'model_ids': config['model_ids'], 'epsilon': config['epsilons'][0],
}
(root / 'cluster-runtime.json').write_text(json.dumps(runtime, indent=2) + '\n')
PY

"$DYSTHE_PYTHON" -m unittest -v test_run.py
"$DYSTHE_PYTHON" -u run.py --config "$task_root/config.json" --output "$task_root/results"
"$DYSTHE_PYTHON" supplement.py --results "$task_root/results" --export "$task_root/portable"
printf 'Completed task outputs: %s\n' "$task_root"

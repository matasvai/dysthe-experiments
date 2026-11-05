#!/usr/bin/env bash
# Local source-tree launch; no package downloads or cluster submission.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)
project_dir=$(cd -- "$repo_dir/.." && pwd)
export PYTHONPATH="$project_dir/dysthe-core/src:$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
exec python3 -m dysthe_experiments.simulator_server \
  --data "$repo_dir/runs/small-simulator-20261016" --port "${SIMULATOR_PORT:-8765}" "$@"

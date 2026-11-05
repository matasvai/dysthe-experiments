#!/bin/bash
# Print a dependency plan by default. Only --submit calls sbatch.
set -euo pipefail
die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
usage() {
    cat <<'TEXT'
Usage: cluster/submit-long-distance.sh [--submit] [--gpu-partition gpu1|gpu2]
       [--account NAME] [--config PATH] [--run PATH] [--max-cpu N] [--max-gpu N]

The default prints five sbatch commands without submitting or creating results.
--submit requires an absolute DYSTHE_PYTHON in a prepared environment.
Reference arrays run first; PINNs, sensitivity, and domain checks depend on them.
The final summary runs after all jobs finish, including failures/cancellations.
CPU and GPU array concurrency defaults to 4 and 2 respectively.
TEXT
}
submit=false
gpu_partition=gpu2
account=''
max_cpu=4
max_gpu=2
repo_dir="${DYSTHE_EXPERIMENTS_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
long_config="${LONG_CONFIG:-configs/long-distance-a100.json}"
run_dir="${RUN_DIR:-runs/long-distance-a100}"
while (($#)); do
    case "$1" in
        --submit) submit=true; shift ;;
        --help|-h) usage; exit 0 ;;
        --gpu-partition|--account|--config|--run|--max-cpu|--max-gpu)
            [[ $# -ge 2 && -n "$2" ]] || die "Missing value for $1"
            case "$1" in
                --gpu-partition) gpu_partition="$2" ;;
                --account) account="$2" ;;
                --config) long_config="$2" ;;
                --run) run_dir="$2" ;;
                --max-cpu) max_cpu="$2" ;;
                --max-gpu) max_gpu="$2" ;;
            esac
            shift 2 ;;
        *) die "Unknown option: $1" ;;
    esac
done
case "$gpu_partition" in
    gpu2) gpu_resource=gpu:A100-PCI-40GB:1 ;;
    gpu1) gpu_resource=gpu:A100-SXM4-40GB:1 ;;
    *) die 'Choose gpu1 or gpu2; this campaign requires an A100.' ;;
esac
[[ "$max_cpu" =~ ^[1-9][0-9]*$ && "$max_gpu" =~ ^[1-9][0-9]*$ ]] || die 'Array concurrency must be positive integers.'
cd -- "$repo_dir"
repo_dir="$PWD"
[[ -f src/dysthe_experiments/long_distance.py ]] || die 'Repository anchor does not contain the long-distance campaign.'
[[ "$long_config" == /* ]] || long_config="$repo_dir/$long_config"
[[ "$run_dir" == /* ]] || run_dir="$repo_dir/$run_dir"
[[ -f "$long_config" ]] || die "Configuration not found: $long_config"
if "$submit"; then
    [[ "${DYSTHE_PYTHON:-}" == /* && -x "${DYSTHE_PYTHON:-}" ]] || die 'Set DYSTHE_PYTHON to an executable absolute path in the prepared environment.'
    command -v sbatch >/dev/null || die 'sbatch is unavailable; use the cluster submission host.'
fi
count_python="${DYSTHE_PYTHON:-python3}"
counts=$("$count_python" - "$long_config" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
if c['model_id'] != 'water-wave-dysthe-spatial-v1':
    raise SystemExit('Only the approved spatial water-wave Dysthe model is supported')
g, t, v = (c[k] for k in ('groups_per_family', 'train_per_family', 'validation_per_family'))
if any(type(x) is not int for x in (g, t, v)) or not (0 <= t < g and 0 < v < g - t):
    raise SystemExit('Configuration must provide validation and test fields in each family')
seeds = c['seeds']
if not seeds or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
    raise SystemExit('Seed list must contain distinct integers')
cases = 3 * (g - t)
print(cases, cases * len(seeds))
PY
)
read -r case_count pinn_count <<< "$counts"
[[ "$case_count" =~ ^[1-9][0-9]*$ && "$pinn_count" =~ ^[1-9][0-9]*$ ]] || die 'Could not derive array sizes from configuration.'
export DYSTHE_EXPERIMENTS_DIR="$repo_dir" LONG_CONFIG="$long_config" RUN_DIR="$run_dir"
common=(--parsable --export=ALL "--chdir=$repo_dir" "--output=$run_dir/logs/%x-%A_%a.out")
[[ -z "$account" ]] || common+=("--account=$account")
if "$submit"; then
    mkdir -p -- "$run_dir/logs" "$run_dir/launcher"
    submission_log="$run_dir/launcher/submission-$(date -u +%Y%m%dT%H%M%SZ)-$$.tsv"
    printf 'stage\tjob_id\n' > "$submission_log"
fi
submit_job() {
    local name="$1" placeholder="$2" response
    shift 2
    printf '%q ' sbatch "${common[@]}" "$@" >&2
    printf '\n' >&2
    if "$submit"; then
        response=$(sbatch "${common[@]}" "$@")
        response="${response%%;*}"
        [[ "$response" =~ ^[0-9]+$ ]] || die "Unexpected sbatch job ID for $name: $response"
        printf '%s\t%s\n' "$name" "$response" >> "$submission_log"
        printf '%s\n' "$response"
    else
        printf '%s\n' "$placeholder"
    fi
}
reference_id=$(submit_job reference REFERENCE_JOB_ID \
    "--array=0-$((case_count - 1))%$max_cpu" "$repo_dir/cluster/run-long-reference.sbatch" reference)
pinn_id=$(submit_job pinn PINN_JOB_ID \
    "--array=0-$((pinn_count - 1))%$max_gpu" "--partition=$gpu_partition" "--gres=$gpu_resource" \
    "--dependency=afterok:$reference_id" --kill-on-invalid-dep=yes "$repo_dir/cluster/run-long-pinn.sbatch")
sensitivity_id=$(submit_job sensitivity SENSITIVITY_JOB_ID \
    "--array=0-$((case_count - 1))%$max_cpu" "--dependency=afterok:$reference_id" \
    --kill-on-invalid-dep=yes "$repo_dir/cluster/run-long-reference.sbatch" sensitivity)
domain_id=$(submit_job domain DOMAIN_JOB_ID \
    "--array=0-$((case_count - 1))%$max_cpu" "--dependency=afterok:$reference_id" \
    --kill-on-invalid-dep=yes "$repo_dir/cluster/run-long-reference.sbatch" domain)
summary_id=$(submit_job summarize SUMMARY_JOB_ID \
    "--dependency=afterany:$reference_id:$pinn_id:$sensitivity_id:$domain_id" \
    "$repo_dir/cluster/run-long-reference.sbatch" summarize)
if "$submit"; then
    printf 'Submitted reference=%s pinn=%s sensitivity=%s domain=%s summary=%s\n' \
        "$reference_id" "$pinn_id" "$sensitivity_id" "$domain_id" "$summary_id"
    printf 'Submission record: %s\n' "$submission_log"
else
    printf 'Plan only: %s reference cases, %s case/seed PINN tasks. Nothing submitted.\n' "$case_count" "$pinn_count"
    printf 'Use --submit with a prepared DYSTHE_PYTHON on the cluster to execute this plan.\n'
fi

# CPU and cluster handoff

**Not run on cluster in this package.** The commands below are templates for a future operator. No jobs were submitted while preparing this handoff. The temporal reference calculations and the spatial PINN repair are different model problems, and must retain their own names, outputs and conclusions.

## Temporal reference: CPU array

`run-temporal-array.sbatch` uses the standalone `run.py`, with two array members: epsilon 0.05 and 0.1. Each task runs all three initial spatial profiles and all four model variants, retaining comparisons and failed gates. It uses `config-refined.json` by default; the included `config-audit.json` instead selects the independent higher-resolution T=16 audit. Each task records source hashes, runtime versions and Slurm identifiers, then exports compact CSV/JSON/plot data.

The template requests the supplied CPU partition `med`, two CPUs, 8 GiB and four hours per task. These are resource requests, not measured cluster costs. Confirm the site's partition, allocation rules and storage paths before submission. No hostname, username or account is assumed. Override the partition at submission if required, and add `--account="$DYSTHE_ACCOUNT"` only after setting that variable to an actual allocation.

Copy the entire standalone package to storage visible from the compute nodes. On the preparation/login host, create a Python >=3.11 environment and install `requirements.txt`; do not install packages inside compute jobs. Set these variables to real absolute paths:

```bash
export DYSTHE_POC_DIR=/absolute/path/to/dysthe-original-poc-2026-10-30
export DYSTHE_PYTHON=/absolute/path/to/prepared/venv/bin/python
export DYSTHE_RESULTS_ROOT=/absolute/path/to/new/temporal-results
```

After replacing those path examples, the following checks the shell syntax and creates only the log directory. Inspect the commands before the final submission line.

```bash
cd "$DYSTHE_POC_DIR"
bash -n cluster/run-temporal-array.sbatch
"$DYSTHE_PYTHON" -m unittest -v test_run.py
mkdir -p "$DYSTHE_RESULTS_ROOT/logs"

# Future submission example; not executed while making this package.
sbatch --export=ALL --chdir="$DYSTHE_POC_DIR" \
  --output="$DYSTHE_RESULTS_ROOT/logs/%x-%A_%a.out" \
  cluster/run-temporal-array.sbatch
```

For the independent T=16 high-resolution audit, use another results root and set the configuration before the same submission command:

```bash
export DYSTHE_CONFIG="$DYSTHE_POC_DIR/config-audit.json"
```

Unset `DYSTHE_CONFIG` to return to `config-refined.json`. A task writes beneath `temporal-JOBID/task-INDEX/`; there is no concurrent shared summary. Read both task summaries. The epsilon split preserves within-epsilon comparisons, but does not create one merged JSON. A failed numerical gate is retained and blocks the next configured horizon; successful process exit alone is not numerical qualification. The array does not run the optional domain-size check automatically.

Return both task directories' `config.json`, `cluster-runtime.json`, all result JSON, `portable/` and the Slurm logs, including failures. Keep the complete `results/*.npz` arrays for full-grid verification. Portable complex64 arrays are visualization products, not numerical reference data. If the cluster regenerates outputs with another Python or library version, use new output directories and compare numerical convergence, not only array hashes.

## Separate spatial PINN repair: pinned reproducible checkout

This section concerns **`water-wave-dysthe-spatial-v1`**, the spatial free-surface-envelope model in `dysthe-experiments`. It does not train a PINN for the temporal equation in this directory. The recorded repair uses persisted/checksummed canonical inputs and invertible `1/(1+k^2)` spectral coefficient scaling while retaining the full residual and 48 output modes. Keep the unscaled matched control and the L2/H1/H2/peak gates.

The following pins were checked against local repair checkouts during handoff preparation; remote availability was not refreshed here:

| Component | Revision |
|---|---|
| `dysthe-experiments`, branch `codex/pinn-h2-rerun` | `f53afd9e853883e29a9111c14ec115189bc7394e` |
| `dysthe-learning`, branch `codex/pinn-h2-rerun` | `bdfe60e571e9c1e676bd87b8a0097a7f4b795951` |
| `dysthe-core`, pinned dependency | `a47a42321475feb6bc55031f4b752419f4b7fda9` |

Use a new clone and a separately prepared CUDA PyTorch 2.10.x environment; the NumPy/SciPy-only temporal environment is not a GPU training environment. The pinned project and its runtime preflight are the executable authority for dependency compatibility.

```bash
git clone --branch codex/pinn-h2-rerun \
  https://github.com/matasvai/dysthe-experiments.git dysthe-h2-rerun
cd dysthe-h2-rerun
git checkout --detach f53afd9e853883e29a9111c14ec115189bc7394e

export DYSTHE_PYTHON=/absolute/path/to/prepared/cuda-venv/bin/python
"$DYSTHE_PYTHON" -m pip install -e '.[pilot]'
"$DYSTHE_PYTHON" -m pip install --no-deps --force-reinstall \
  'dysthe-learning @ git+https://github.com/matasvai/dysthe-learning.git@bdfe60e571e9c1e676bd87b8a0097a7f4b795951'
"$DYSTHE_PYTHON" scripts/check.py

# Plan only: deliberately omit --submit.
bash cluster/submit-long-distance.sh --gpu-partition gpu2 \
  --config configs/long-distance-h2-smoke.json --run runs/h2-smoke-new
```

The existing spatial launcher's GPU selectors encode this **site-supplied configuration**, not universal Slurm names:

| Example partition | Example GPU resource string |
|---|---|
| `gpu1` | `gpu:A100-SXM4-40GB:1` |
| `gpu2` | `gpu:A100-PCI-40GB:1` |

Do not assume these strings work on another cluster. That launcher also contains the previous site's CPU partition defaults; inspect its generated plan and `cluster/run-long-reference.sbatch`. Do not replace an A100 with another device while retaining A100 evidence labels. The runtime checks must pass on the allocated compute node. The printed plan does not establish a valid allocation, successful execution or GPU performance.

When the site configuration is confirmed, adding `--submit` to the smoke command submits it. Inspect its runtime checks and complete summary before requesting the matched xi=4 runs, each with a fresh run directory:

```bash
# Plan candidate, then matched control. Add --submit only when ready to execute.
bash cluster/submit-long-distance.sh --gpu-partition gpu2 \
  --config configs/long-distance-h2-check.json --run runs/h2-check-new
bash cluster/submit-long-distance.sh --gpu-partition gpu2 \
  --config configs/long-distance-h2-control.json --run runs/h2-control-new
```

The tiny smoke optimization budget is an execution check. The xi=4 candidate/control accuracy check uses L2 <=1% and H1/H2/peak <=5% at every sampled station. Only after inspecting those gates use `configs/long-distance-h2-extended.json` with another fresh run directory. A fit through xi=8/16 is not an unseen-distance forecast. These already inspected inputs are a diagnostic rerun, not a new untouched generalization test.

After actual runs finish, preserve all task failures and GPU runtime records with the existing metadata exporter, in a CPU allocation. Set `RUN_DIR` to that completed run and `EVIDENCE_DIR` to a new directory outside it:

```bash
"$DYSTHE_PYTHON" -m dysthe_experiments.export_evidence \
  --run "$RUN_DIR" --destination "$EVIDENCE_DIR"
"$DYSTHE_PYTHON" -m dysthe_experiments.export_evidence --verify "$EVIDENCE_DIR"
```

Return the whole verified export, not only the top-level summary; retain the raw arrays and checkpoints on cluster storage. The detailed pinned runbook is [docs/h2-rerun.md](https://github.com/matasvai/dysthe-experiments/blob/f53afd9e853883e29a9111c14ec115189bc7394e/docs/h2-rerun.md). This handoff makes no new repaired-A100 accuracy, timing or completion claim.

## Separate smaller-step audit

The primary high-steepness wavetrain required a further smaller-step check. To regenerate it in a CPU allocation, using a new directory and the prepared environment:

```bash
cd "$DYSTHE_POC_DIR"
srun --partition=med --cpus-per-task=2 --mem=8G --time=00:30:00 \
  "$DYSTHE_PYTHON" rerun_time_audit.py \
  --output "$DYSTHE_RESULTS_ROOT/fresh-time-audit"
```

This is a submission template, not an executed cluster run. The default is the same c=3/2 sideband, epsilon0.1,T16, N4096h0.0003125 versus N2048 and h0.000625. `--help` lists bounded alternatives. The archived `time_audit.py` is preserved for historical source identity; use the new rerun CLI for a portable fresh run.

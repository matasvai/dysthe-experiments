# AI.Panther A100 pilot

`run-a100-pilot.sbatch` runs the selected **slab spectral PINN** for
`water-wave-dysthe-spatial-v1`. The completed CPU comparison motivated this
candidate; it does not establish a GPU speedup or solution existence. Each
initial field receives a separate optimization. This is a numerical solver task,
distinct from an operator's prediction on unseen initial fields without fitting.

The default sequence prepares and qualifies numerical references on the allocated
node, trains only `--pinn-variants slabs` on its GPU, then runs CPU diagnostics.
It does not run the FNO/PINO comparison or the other PINN variants. These remain
available through the campaign CLI for separate, explicitly configured studies.

## Resources and prerequisites

The [official AI.Panther submission guide](https://help.fit.edu/TDClient/39/Portal/KB/PrintArticle?ID=2107)
lists four nodes in each of `gpu1` and `gpu2`, with four 40 GB A100s per node and
a seven-day partition limit. The verified resource strings are:

| Partition | One-GPU request |
| --- | --- |
| `gpu2` | `--gres=gpu:A100-PCI-40GB:1` |
| `gpu1` | `--gres=gpu:A100-SXM4-40GB:1` |

The script requests one node, one task, one `gpu2` A100, four CPU cores, 16 GB
host memory, and two hours. The latter three are initial allocation choices,
not measured GPU resource requirements. No account is assumed. If your access
requires one, supply `sbatch --account=YOUR_ALLOCATED_ACCOUNT` at submission.
Changing to `gpu1` requires overriding **both** partition and GPU type.

Set `DYSTHE_PYTHON` to an absolute Python executable in a prepared environment.
Core and learning are installed dependencies; sibling checkouts are not required.
The launcher requires CUDA-enabled PyTorch 2.10.x and exactly one visible A100.
Before work starts, it compares the complete CUDA and CPU RHS in complex128 on
a nonconstant field (`rtol=1e-10`, `atol=1e-11`), then tests the actual spectral
PINN's distance JVP and full-residual backward pass for finite, nonzero gradients.
The local CPU run used Python 3.14.7,
PyTorch 2.10.0+cu128, NumPy 2.4.2, and SciPy 1.17.0. These observations are not
a tested cluster environment or a portable lock file. Choose a supported Python
and CUDA wheel combination, retain its exact dependency lock, and validate it on
an allocated node. A CPU-only PyTorch wheel cannot run this job.

The script uses a login shell because the cluster's A100 hosts use the AMD
software stack. Consult the [site container guidance](https://help.fit.edu/TDClient/39/Portal/KB/Article/23937/Using-NGC-Containers-on-AI-Panther)
when choosing modules or a container; this launcher does not assume an installed
module version or silently select a container. Follow the site's rule that
computational work belongs on allocated compute nodes, including reference
generation and environment validation. The job does not install packages or
access the network.

## Install the reviewed source and optional dependencies

The package extras are `dysthe-core[reference,learning]`,
`dysthe-learning[pilot]`, and `dysthe-experiments[pilot]`. The extras express
supported dependency ranges, not a reproducible environment lock. Install the
reviewed experiments revision whose `pyproject.toml` pins the matching core and
learning commits. Their Git dependency URLs use full commit hashes; these must
include the new physics and model modules. Do not substitute older scaffold
revisions or rely on mutable branch names.

For a connected installation, prepare a virtual environment, choose the official
CUDA PyTorch wheel index compatible with the cluster driver, and install:

```bash
python3 -m venv "$DYSTHE_VENV"
export DYSTHE_PYTHON="$DYSTHE_VENV/bin/python"
"$DYSTHE_PYTHON" -m pip install 'torch==2.10.0' --index-url "$DYSTHE_TORCH_INDEX"
cd "$DYSTHE_EXPERIMENTS_DIR"
"$DYSTHE_PYTHON" -m pip install -e '.[pilot]'
```

Set the three path/index variables first. `DYSTHE_TORCH_INDEX` must refer to the
chosen CUDA wheel repository, not the CPU index used in CI. This online command
resolves the full-SHA core/learning dependencies without sibling repositories.
Run it during environment preparation, not inside the batch job.

For an offline environment, first prepare an approved wheelhouse and a complete
requirements lock containing exact versions and SHA256 hashes, including the
chosen CUDA PyTorch distribution and its dependencies, NumPy, SciPy, matplotlib,
setuptools, and wheel. Build core and learning wheels from the exact commits
pinned by experiments and retain their source SHAs and wheel hashes. Build the
environment on a compatible machine or compute allocation. Set
`DYSTHE_EXPERIMENTS_DIR`, `DYSTHE_VENV`, `DYSTHE_WHEELHOUSE`,
`DYSTHE_RUNTIME_LOCK`, `DYSTHE_CORE_WHEEL`, and `DYSTHE_LEARNING_WHEEL` to
absolute paths, with the last two referring to those verified wheels:

```bash
python3 -m venv "$DYSTHE_VENV"
export DYSTHE_PYTHON="$DYSTHE_VENV/bin/python"
"$DYSTHE_PYTHON" -m pip install --no-index --find-links "$DYSTHE_WHEELHOUSE" \
  --require-hashes -r "$DYSTHE_RUNTIME_LOCK"
"$DYSTHE_PYTHON" -m pip install --no-index --no-deps \
  "$DYSTHE_CORE_WHEEL" "$DYSTHE_LEARNING_WHEEL"
"$DYSTHE_PYTHON" -m pip install --no-deps --no-build-isolation \
  -e "$DYSTHE_EXPERIMENTS_DIR"
"$DYSTHE_PYTHON" -m pip freeze > "$DYSTHE_VENV/installed-packages.txt"
```

This repository does not ship the wheelhouse or a GPU-tested hash lock. Do not
substitute the CPU wheel index from CI for a CUDA environment. `pip freeze`
records installed versions but does not replace the wheel hashes or source
manifest. Record each repository's full 40-character reviewed commit and verify
the wheel hashes before installing. An unpublished local change needs a source
snapshot and content hashes as well; a Git commit alone cannot identify it.
The launcher records hashes of the Python sources actually imported, experiments
package metadata, its scope contract, and itself, as well as package versions.

Run the experiments checks in the prepared environment before submitting:

```bash
cd "$DYSTHE_EXPERIMENTS_DIR"
"$DYSTHE_PYTHON" scripts/check_scope.py
"$DYSTHE_PYTHON" scripts/check.py
```

Core and learning use the same `scripts/check_scope.py` and `scripts/check.py`
commands in their source checkouts; their CI must pass at the pinned revisions.
The CPU reproducibility tests check split isolation, model rejection, complex
phase error, plane-wave refinement, and exact next-step optimizer/RNG recovery.
Passing these checks is not GPU validation. The allocation preflight is also
only a runtime check; inspect accuracy, residual, timing, and memory results
before increasing the experiment.

## Submit and resume

Submit from `dysthe-experiments`; Slurm copies the script to its spool, so the
launcher anchors paths to `SLURM_SUBMIT_DIR`, not to that copied script. An
explicit `DYSTHE_EXPERIMENTS_DIR` can override the anchor. `PILOT_CONFIG` and
`RUN_DIR` may be absolute or relative to the repository. Defaults are
`configs/a100-pilot.json` and `runs/a100-pilot`.

```bash
cd "$DYSTHE_EXPERIMENTS_DIR"
export PILOT_CONFIG=configs/a100-pilot.json
export RUN_DIR=runs/a100-pilot-01
sbatch cluster/run-a100-pilot.sbatch
```

An optional positional stage is `all`, `prepare`, `pinns`, or `diagnostics`.
Here `all` means the three selected stages above; it does not pass the CLI's
broader `--stage all`. For a prepared run, submit only training with:

```bash
sbatch cluster/run-a100-pilot.sbatch pinns
```

For an independent `gpu1` run, use a new `RUN_DIR` and both overrides:

```bash
export RUN_DIR=runs/a100-gpu1-pilot-01
sbatch --partition=gpu1 --gres=gpu:A100-SXM4-40GB:1 \
  cluster/run-a100-pilot.sbatch
```

Resubmit the same command with the same source, runtime, configuration, and run
directory after an interruption. The campaign restores model weights, Adam
state, epoch, history, and CPU/CUDA random states from each slab's `last-*.pt`.
Checkpoint writes use a temporary file followed by atomic replacement; they
occur at validation intervals and slab completion. Work since the last saved
checkpoint may be lost. There is no automatic requeue, signal-triggered save,
or archive operation. `best-*.pt` is selected by full Dysthe residual, without
using the numerical solution as a training label or checkpoint criterion.

The launcher's runtime/source lock rejects changed inputs on restart, and a file
lock rejects concurrent writers. Job arrays are intentionally rejected because
the CLI already iterates cases and seeds inside one run. Start a fresh directory
for a CPU-to-GPU comparison, a different GPU type, or an altered configuration;
do not mix their checkpoints. Verified `dataset.npz`, `manifest.json`, and the
matching `config.json` can be copied together to a fresh run to reuse references.
The campaign rechecks configuration, scope, split isolation, and dataset hash.
Only load your own trusted checkpoint files.

Each PINN summary includes worst errors over every saved station, station count,
and a provisional gate of L2 error at most 1%, H1 at most 5%, and peak-amplitude
error at most 5%. These sampled checks do not bound errors between stations.
`invocation_elapsed_seconds` includes per-case setup, checkpoint loading,
prediction and audits; `training_seconds` accumulates optimization/validation
time across resumes. Dataset preparation and loading precede both per-case
timers. `gpu_peak_allocated_bytes` records the process's peak allocated tensor
memory during the case, with the peak counter reset before each case; it is
null on CPU. A resume flag distinguishes replay from a fresh fit.

For reevaluation of a completed comparison with selected variants, use a copied
run: the CLI rewrites the aggregate PINN summary for the selected variants and
the environment record for the current invocation. Retain original training
provenance. The A100 launcher also keeps a separate record for each job.

`CUBLAS_WORKSPACE_CONFIG=:4096:8` is set before Python, TF32 is disabled, and the
campaign requests deterministic PyTorch operations. See [PyTorch reproducibility](https://docs.pytorch.org/docs/2.10/notes/randomness.html)
and [NVIDIA cuBLAS reproducibility](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility).
These settings do not guarantee bitwise agreement between CPU and GPU or across
different software and hardware. GPU restart equivalence still needs a measured
check on the cluster.

Logs remain as `slurm-dysthe-slabs-JOBID.out` in the submission directory.
Provenance, arrays, checkpoints, and summaries remain in `RUN_DIR`; no results
are deleted or archived. Keep raw arrays and checkpoints out of Git. Preserve
the configuration, dependency lock, source hashes, dataset checksum, allocation
details, and total optimization time when reporting accuracy or cost.

## Export long-distance evidence, including failures

For the long-distance campaign, retain the complete run on cluster storage and
export its small evidence files after the jobs have finished. Exporting only
`summary.json` loses the per-task exceptions, per-horizon H2 errors, sampled
station errors, and GPU provenance needed to diagnose a failed gate.

Set `EVIDENCE_DIR` to a **new directory outside `RUN_DIR`**, then run in a CPU
allocation because the exporter reads and hashes every numerical artifact:

```bash
"$DYSTHE_PYTHON" -m dysthe_experiments.export_evidence \
  --run "$RUN_DIR" --destination "$EVIDENCE_DIR"
"$DYSTHE_PYTHON" -m dysthe_experiments.export_evidence \
  --verify "$EVIDENCE_DIR"
```

The export preserves every run JSON file, including task errors and tracebacks,
all per-horizon `result.json` files, canonical `inlets.json` and `cases.json`,
runtime locks, job records, source manifests, and small diagnostic logs. Their
original bytes live under `metadata/`. `artifact-inventory.json` records the
path, size, and SHA256 of arrays and checkpoints; `.npz`, `.pt`, and other raw
research files remain on the cluster and are never deserialized by the exporter.
`source-manifests.json` indexes the recorded run-time source identities.
`evidence-manifest.json` lists checksums, missing expected evidence, malformed
JSON retained verbatim, and declared artifact checksum mismatches. Such findings
remain visible rather than preventing a diagnostic export or being repaired.

The exporter rejects existing destinations, symlinks, and changes detected while
reading the run. It fails loudly if a metadata file exceeds 4 MiB or the total
exceeds 256 MiB; inspect the file and explicitly adjust `--max-file-bytes` or
`--max-total-bytes` if needed. It never silently drops oversized task results.
Those overrides do not change the repository's 5 MB per-file review limit.
Runtime cache directories are excluded and listed. Other unrecognized file
types are inventoried as not copied. Files outside `RUN_DIR` are not collected:
the dependency submission helper places Slurm output in `RUN_DIR/logs`; for
manual `sbatch` calls, choose an output path there before submission.

Transfer the complete exported directory, then run `--verify` on the received
copy. Verification detects altered, missing, or unexpected exported files; it
does not prove scientific accuracy or authenticate the producer. Keep the raw
cluster run and its prepared-environment lock for deeper investigation. A
failed or incomplete run remains failed or incomplete in this evidence package.

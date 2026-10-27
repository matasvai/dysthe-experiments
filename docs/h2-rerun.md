# H2 and random-input repair: cluster rerun

Workstream: `codex/pinn-h2-rerun`. This runbook accompanies the executable
regression checks and CPU diagnostic evidence. Historical results under
`results/long-distance-gpu2-2026-10-26` remain unchanged.

## What changed

- Every run now saves its selected complex128 inlets once and verifies their
  checksums in every stage. Random inputs are no longer independently rebuilt
  by CPU and GPU hosts before an exact reference-identity comparison. The
  original random-input exception was identified by matching all 12 error
  records' bytes and SHA-256 hashes against the published manifest. A local
  BLAS-dispatch experiment reproduced the cause; the original cluster's exact
  CPU dispatch remains unknown.
- The candidate spectral PINN multiplies its learned correction coefficient
  at physical wavenumber `k=2*pi*m/L` by `1/(1+k*k)`. This invertible
  parameterization retains all 48 output modes, the initial condition,
  full Dysthe residual, and checkpoint selection by residual alone. The
  unscaled method remains available as a matched control. No reference labels
  are used in optimization or checkpoint selection.
- Both trainer and campaign pass flags include complex L2, H1, H2, and peak
  errors. H2 uses the Fourier weight `(1+k*k)^2` in its squared norm.
  No phase alignment, filtering, or relaxed accuracy threshold is applied.
- Results include per-station H2 error contributions by frequency band,
  reference projection floors, gradient norms before clipping, complete
  exception traces, and a metadata exporter. These distinguish representation
  limits, optimization errors, and execution failures.

## Prepare a clean checkout and environment

Keep old run directories intact. Changed source/configuration deliberately
cannot resume an old run. Use the same prepared CUDA PyTorch **2.10.x**
Python environment as the previous campaign; the runtime preflight verifies
an allocated A100, full-physics arithmetic, and PINN differentiation.

```bash
git clone --branch codex/pinn-h2-rerun https://github.com/matasvai/dysthe-experiments.git dysthe-h2-rerun
cd dysthe-h2-rerun
export DYSTHE_PYTHON=/absolute/path/to/prepared/venv/bin/python
"$DYSTHE_PYTHON" -m pip install -e '.[pilot]'
"$DYSTHE_PYTHON" scripts/check.py
```

Installing this checkout updates the learning dependency to its exact repair
commit; installing only experiment files over the old learning package is
insufficient. The core equation/solver dependency is unchanged. Preparation
requires network/package access, but compute jobs never install packages.

## Tests and their goals

| Configuration | Goal | Interpretation |
|---|---|---|
| `long-distance-h2-smoke.json` | Exercise canonical inlets, all input families, CUDA JVP/backprop, checkpointing, and result export with a tiny budget. | Require no execution failures and successful runtime checks. Its deliberately tiny optimization budget is not an accuracy benchmark. |
| `long-distance-h2-check.json` | Test the conditioned 48-mode PINN through xi=4 on all 12 existing evaluation inlets and 3 seeds. | Require every sampled L2 <=1%, H1/H2/peak <=5%; retain every failure. |
| `long-distance-h2-control.json` | Repeat the same xi=4 test with the legacy unscaled coefficients and repaired inlet handling. | Compare matched case/seed errors, full residuals, training/invocation time, and gradient norms. The random-family comparison is now executable. |
| `long-distance-h2-extended.json` | After the xi=4 check, test PINN fits through xi=8,16; reference diagnostics through xi=64. | Each task stops after a failed accuracy gate. A fitted interval is not an unseen-distance forecast or an existence proof. |

The xi=4 control and candidate differ only in the named campaign and coefficient
scaling. They use the same initial data, architecture, 400 Adam updates per
0.5-distance slab, residual, seeds, reference discretization, and thresholds.
These inlets have already been inspected; this is a diagnostic rerun, not a
new untouched test set. A future generalization claim needs new locked inlets.

Run the short workflow first, then the xi=4 check (separate directories):

```bash
bash cluster/submit-long-distance.sh --submit --gpu-partition gpu2 \
  --config configs/long-distance-h2-smoke.json --run runs/h2-smoke-v2
bash cluster/submit-long-distance.sh --submit --gpu-partition gpu2 \
  --config configs/long-distance-h2-check.json --run runs/h2-check-v2
```

Run the second command only after inspecting the smoke summary/runtime records.
Use `--gpu-partition gpu1` for the SXM4 A100 partition; the launcher switches
both partition and GPU resource type. Add `--account YOUR_ACCOUNT` if required
by your allocation. Omit `--submit` to print the dependency plan only.

For the matched control, substitute `long-distance-h2-control.json` and
`runs/h2-control-v2`. After the xi=4 accuracy check succeeds, substitute
`long-distance-h2-extended.json` and `runs/h2-extended-v2` for the longer study.
The existing CPU reference, sensitivity, and domain qualification gates remain
active. Packet-domain disagreement is a separate interpretation limit even
when a periodic-box neural prediction is accurate.

## Return complete evidence

After jobs finish, use the metadata exporter described in
[the cluster guide](../cluster/README.md). Include task JSON (especially
exceptions), each horizon's `result.json`, `pinn-summary.json`, inlet metadata,
and A100 runtime records. Keep raw inlet arrays, trajectories, and checkpoints
on cluster storage; the exported manifest retains their locations/checksums.
Do not copy only the top-level aggregate summary: it cannot locate H2 error
or explain missing random results.

CPU diagnostics and software checks accompany this branch. No A100 execution
of the repaired code is claimed before your rerun. High prediction H2 error
measures disagreement with the reference, not singularity of the actual wave.

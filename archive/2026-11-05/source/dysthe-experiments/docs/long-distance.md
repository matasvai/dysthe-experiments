# Long-distance water-wave Dysthe campaign

Workstream: `codex/long-distance-a100`. This is an executable **proposed**
experiment, not an A100 result. The prior CPU pilot qualified a per-inlet slab
PINN through xi=2; it did not establish long-distance neural accuracy.

The [editable protocol](../protocols/long-distance/protocol.tex) and
[PDF](../protocols/long-distance/protocol.pdf) specify the scientific question,
equation, diagnostics and interpretation limits. The physical model remains
`water-wave-dysthe-spatial-v1`, the spatial free-surface envelope with complete
periodic retarded-time profiles and the nonlocal mean-flow term.

## Tests and purposes

| Stage | Distances | Purpose and acceptance |
|---|---|---|
| Numerical reference | 4, 8, 16, 32, 64 | Independent grid and step comparisons, including complex H2 error, spectral tails and wave action. Failed qualification stops this inlet's longer reference stages. |
| Slab PINN | 4, 8, 16 | Full Dysthe residual training, no numerical labels. Worst saved-station errors must be at most 1% complex L2, 5% H1, 5% H2 and 5% peak amplitude. Failure stops this case/seed's longer fits. |
| Sensitivity | 4, 8, 16 | Centered response to one declared H1-normalized perturbation, with delta halving and independent grid/step refinement. All signed-response H1 differences must be at most 1%. |
| Packet domain | 4, 8, 16, 32, 64 | Reconstruct the same analytic localized packet in a doubled period at the same spacing and physical initial cutoff; refine both boxes independently. Compare on the original interval, differentiating before restriction. H1 difference at most 1%, with initial discrepancy at most 0.1%. |
| NLS control | Each numerical reference horizon | Independently qualify epsilon=0 at the same scaled inlet and distance; report its difference from Dysthe as a model comparison, not a numerical error. |

Reference thresholds are L2 1e-5, H1 1e-4 and H2 1e-3 for both independent
refinements. Fine-grid action drift must be at most 1e-6 and action in the upper
20% of its retained Fourier band at most 1e-8. These are chosen numerical gates,
not mathematical existence or physical-validity thresholds. The additional 5%
neural H2 gate was not part of the earlier CPU pilot.

The production [configuration](../configs/long-distance-a100.json) generates
36 fixed initial fields: 24 training-reserved, six validation and six test.
The per-inlet PINN and diagnostic stages use the 12 validation/test fields from
packet, sideband and random families. Each neural inlet is fitted separately
with seeds 11, 29 and 47; this is not a pretrained operator generalizing to new
inlets. The unused training fields preserve the existing family/split recipe.

The numerical reference uses N=1024 and h=0.00125, with N=512 at the same step
and h=0.0025 at the same grid. Saved stations are spaced by 0.05. Each neural
fit uses N=256, Fourier band K=48, a width-64/depth-three spectral PINN, 64
collocation slices per update and 400 Adam updates per slab of length 0.5.
For each horizon, the fit starts from the same original inlet. Early slabs are
retrained for the 4/8/16 fits; these are independent horizon experiments, not a
single extended checkpoint or an unseen-distance forecast. Within every fit,
slab endpoints are inherited predictions and are never reset to numerical truth.
Checkpoints are selected only by the full Dysthe residual.

## Prepare the cluster environment

Clone this workstream, or fetch and switch an existing clean checkout:

```bash
git clone --branch codex/long-distance-a100 https://github.com/matasvai/dysthe-experiments.git
cd dysthe-experiments
```

Use the [existing environment instructions](../cluster/README.md#install-the-reviewed-source-and-optional-dependencies)
to prepare CUDA-enabled PyTorch 2.10.x, NumPy and SciPy and the core/learning
commits pinned in `pyproject.toml`. Set `DYSTHE_PYTHON` to that environment's
absolute executable path. Use the same prepared runtime for the CPU reference
and GPU tasks, because package/source identities are checked across stages.
The launcher installs nothing and uses no network access on compute nodes.
Install this checkout in the prepared environment if needed:

```bash
"$DYSTHE_PYTHON" -m pip install --no-deps --no-build-isolation -e .
"$DYSTHE_PYTHON" scripts/check_scope.py
"$DYSTHE_PYTHON" scripts/check.py
```

Use a persistent allocated project directory on `/shared` for `RUN_DIR`; exact
project path and account depend on your allocation. The default remains
`runs/long-distance-a100` inside this checkout. Large arrays and checkpoints are
ignored by Git and are never automatically removed.

## Print or submit the job plan

From the repository, with `DYSTHE_PYTHON` exported:

```bash
# Prints commands; submits nothing.
bash cluster/submit-long-distance.sh --run "$RUN_DIR"

# Submit when the environment and project directory are ready.
bash cluster/submit-long-distance.sh --submit --run "$RUN_DIR"
```

Add `--account ACCOUNT` only if your site allocation requires it. `gpu2` uses
one `A100-PCI-40GB` per task. For `gpu1`, use
`--gpu-partition gpu1`; the helper switches both partition and GRES to
`A100-SXM4-40GB`. CPU stages use `long`. Each array member requests one task,
four CPU cores and 16 GB host RAM for at most 12 hours; GPU members also request
one A100. These are allocation requests, not measured resource requirements.
Concurrency defaults to four CPU array members and two GPU members; lower it
with `--max-cpu` and `--max-gpu` to match your allocation.

The helper submits five jobs: a 12-member numerical-reference array, a
36-member PINN array, a 12-member sensitivity array, a 12-member domain array,
and one final summary. PINNs/sensitivity/domain depend on successful completion
of the reference array; summary runs after every array finishes, including
failed or cancelled jobs. A scientific gate failure is recorded normally and
does not make its job an execution failure: other cases remain usable. An
execution/configuration failure fails the job, cancels dependent arrays and
leaves missing/failed entries visible in the final summary.

Domain tests apply only to packet inlets. Sideband and random tasks are marked
not applicable; padding a periodic wavetrain would change the problem.

For a first cluster runtime test, choose the tiny configuration and a different
run directory (still submit to compute nodes):

```bash
bash cluster/submit-long-distance.sh --submit \
  --config configs/long-distance-smoke.json --run "$RUN_DIR-smoke"
```

This exercises six cases, one seed, and distances 0.1/0.2. Its scientific gates
can legitimately fail, particularly the coarse packet-domain check. It is a
workflow check, not a substitute for the proposed long-distance study.

The local [workflow verification](../results/long-distance-workflow-2026-10-26/verification.json)
ran all 24 case/stage tasks in the smoke configuration. Reference and sensitivity
checks completed for all six cases. One PINN exceeded the field-error gate at
xi=0.2; both coarse packet-domain checks failed the declared domain criteria;
four nonpacket domain tasks were not applicable. Those outcomes remain in the
[summary](../results/long-distance-workflow-2026-10-26/summary.json). Replaying
all six completed PINN tasks preserved all 180 array/checkpoint hashes. This
verifies workflow and completed-run reuse, not A100 performance or GPU restart
equivalence. Unit tests also exercise optimizer/RNG recovery after interruption.

## Resume and inspect

Keep the exact configuration, code, runtime, GPU type and run directory when
resuming. To resubmit one failed PINN array member after correcting an external
interruption, for example task 7:

```bash
export LONG_CONFIG=configs/long-distance-a100.json
sbatch --array=7 cluster/run-long-pinn.sbatch
```

`RUN_DIR` and `DYSTHE_PYTHON` must also be exported. Task indices are case-major,
then seed: tasks 0/1/2 are the first inlet with seeds 11/29/47. Reference and
diagnostic indices run over the 12 inlet records in `cases.json`. Duplicate
writers are rejected. Existing numerical trajectories are checksum-verified and
reused; interrupted PINNs restore model/Adam/RNG state. Completed predictions
and best checkpoints are hash-checked and reused. There is no automatic requeue
or signal-triggered save; work since the most recent checkpoint can be lost.
Changed code, thresholds or inputs require a new run directory.

Key outputs inside the run directory:

- `identity.json`, `config.json`, `cases.json`: source/runtime identity and inlet records.
- `tasks/*.json`: completion, gate failure, execution failure and blocked horizon per task.
- `reference/CASE/xi-DISTANCE/result.json`: refinement, NLS control, peaks, recurrence candidates, H2 growth, spectrum and modified invariants.
- `pinn/CASE/seed-SEED/xi-DISTANCE/result.json`: all-station errors, first failed station, residual audits, checkpoint hashes and timings.
- `sensitivity/` and `domain/`: signed-response and domain comparisons with independent refinement evidence.
- `summary.json`: missing/failing tasks, last passed horizons, validation/test worst and median neural errors, and coverage counts. An aggregate with missing cases is explicitly incomplete.
- `launcher/`, `invocations/`, `logs/`: submission IDs, per-task runtime checks and Slurm output.

For each reference station the recorded nonlinear distance is
`initial_peak**2 * xi`, an initial-peak comparison convention. Also inspect
`epsilon_peak` and `epsilon_spectral_rms_wavenumber`; neither has a universal
acceptance threshold here. Fourier outputs are smooth by construction, and a
small loss cannot establish smoothness of the continuum solution. Neither
finite-distance convergence nor an H2 bound from these calculations proves
global existence or validity of the model over arbitrarily long physical distances.

Primary background: [Fedele and Dutykh](https://arxiv.org/abs/1110.3605) for the
spatial equation, invariants and Fourier integration; [Armaroli et al.](https://doi.org/10.1103/PhysRevE.96.012222)
for Dysthe recurrence and spectral asymmetry. The numerical tolerances and
horizon ladder above are campaign choices, not claims from those papers.

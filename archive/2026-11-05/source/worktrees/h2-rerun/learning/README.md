# dysthe-learning

**Scope: water-wave Dysthe only.** See [SCOPE.md](SCOPE.md) for the fixed
physical model, exclusions and enforced boundaries.

Comparable learning methods consuming the same water-wave Dysthe fields and contracts.

**Status: executable CPU pilot models.** `ContinuousFNO` and `SpectralPINN` are
implemented in `dysthe_learning.pilot_models`, with hard projected initial data,
complex fields and continuous-distance derivatives. PINO uses the FNO backbone
with the full Dysthe residual in the experiment trainer. The general method
registry remains a scaffold; these bounded pilot APIs are called directly.

[Research map](https://matasvai.github.io/dysthe-research-map/) ·
[Family guide](https://github.com/matasvai/dysthe-research-map/blob/main/FAMILY.md) ·
[Migration plan](docs/migration.md) · [Branches](branches.json)

## Start

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/check.py
```

Python 3.10+ is required. Training libraries will be added by the relevant method
ports; the scaffold does not install a GPU runtime or submit cluster jobs.

## Measured pilot and A100 continuation

Workstream: `codex/cpu-pilot-a100`. Install `.[pilot]` and run
`python scripts/check.py`. Physics is pinned to a published core commit.

The [experiment workstream](https://github.com/matasvai/dysthe-experiments/tree/codex/cpu-pilot-a100)
contains configs, training/resume code, CPU measurements, editable report, and
the single-A100 Slurm launcher. Short sequential PINN slabs are the selected
**per-inlet research candidate**. FNO/PINO failed the pilot accuracy gates;
no learned operator or speed advantage is claimed. Every new PINN inlet
requires optimization. Raw fields and trained checkpoints remain outside Git.

Model output bandwidth is explicit and independent of latent Fourier modes.
The full residual is evaluated on complete periodic slices at sufficient
quadrature resolution, including nonlinear modes outside the output band.
The checkpoint tests cover complex precision and AD through propagation
derivatives. CUDA execution is prepared but must be verified on the cluster.

## Spectral PINN conditioning repair

Workstream `codex/pinn-h2-rerun` adds the explicit
`coefficient_scaling='inverse_1_plus_k2'` option to `SpectralPINN`. Learned
correction coefficients are divided by `1 + (2*pi*m/length)^2`; this invertible
parameterization retains every output mode and the original trial-function
capacity. It does not alter the water-wave equation, inlet, or residual.
`'none'` remains the default for historical models. Scaled checkpoints carry
a state marker so strict loading cannot silently reinterpret them as unscaled.

The [experiment rerun guide](https://github.com/matasvai/dysthe-experiments/blob/codex/pinn-h2-rerun/docs/h2-rerun.md)
contains matched controls, CPU evidence, and the A100 workflow. This is a tested
conditioning candidate; A100 accuracy and long-distance smoothness claims
require the new campaign results.

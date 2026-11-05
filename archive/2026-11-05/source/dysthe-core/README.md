# dysthe-core

**Scope: water-wave Dysthe only.** See [SCOPE.md](SCOPE.md) for the fixed
physical model, exclusions and enforced boundaries.

Equation specification, reference solvers, initial fields, diagnostics, and shared data contracts.

**Status: tested numerical reference and differentiable physics adapter.** An optional water-wave numerical reference
and independent controls are available; see [qualification guide](docs/reference-validation.md).
The optional `dysthe_core.torch_spatial` adapter supplies the same equation to
learning methods, with explicit output projection and differentiable Fourier
operators. Full residuals require oversampling; no existence theorem is claimed.

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

## CPU pilot dependency

This workstream is `codex/cpu-pilot-reference`. Install `.[reference,learning]`
for the numerical solver and PyTorch adapter. Run `python scripts/check.py`.
The experiment repository records measured grid/step refinement and CPU results;
unit tests alone do not qualify a new trajectory or physical parameter range.

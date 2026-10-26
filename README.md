# dysthe-experiments

**Scope: water-wave Dysthe only.** See [SCOPE.md](SCOPE.md) for the fixed
physical model, exclusions and enforced boundaries.

Dataset campaigns, training protocols, held-out evaluation, and cluster orchestration.

**Status: measured CPU comparison and A100 launch preparation.** The selected
slab PINN is a per-inlet research candidate; the numerical solver remains the
practical baseline. Accuracy and scope limits are documented below.

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

## CPU-tested PINN and A100 continuation

The [measured CPU report](results/cpu-pilot-2026-10-26/report.pdf) selects a
**four-slab spectral PINN for further per-inlet research**. Worst held-out
sampled errors through xi=2 were 0.963% in the complex field and 1.881% in H1,
across three independent test inlets and two seeds. The numerical solver was
faster and more accurate. FNO/PINO failed the pilot accuracy gates; their
results are retained. These calculations do not prove global existence.

- [Reproduce the CPU comparison](docs/cpu-pilot.md)
- [Editable report and machine-readable evidence](results/cpu-pilot-2026-10-26/)
- [AI.Panther A100 installation, submission and resume](cluster/README.md)
- [CPU configuration](configs/cpu-pilot.json) and [proposed A100 extension](configs/a100-pilot.json)

This executable workstream is `codex/cpu-pilot-a100`. Core and learning are
pinned to published commits. No GPU run or cluster job submission is claimed.

## Staged long-distance experiments

The `codex/long-distance-a100` workstream adds numerical horizons
**xi=4, 8, 16, 32, 64** and slab-PINN horizons **xi=4, 8, 16**, with independent
grid/step refinement, H2 and spectral diagnostics, NLS controls, perturbation
sensitivity, and packet-domain doubling. Each task stops promotion after a
failed gate; incomplete and failed cases remain visible in the summary.

- [Cluster run guide and test goals](docs/long-distance.md)
- [Proposed experiment protocol (PDF)](protocols/long-distance/protocol.pdf)
- [Editable protocol](protocols/long-distance/protocol.tex)
- [A100 configuration](configs/long-distance-a100.json) and [CPU workflow smoke](configs/long-distance-smoke.json)
- [Print or submit the dependency plan](cluster/submit-long-distance.sh)
- [CPU workflow verification and retained failures](results/long-distance-workflow-2026-10-26/verification.json)

Run `bash cluster/submit-long-distance.sh` to print the job plan. Only the
explicit `--submit` option submits jobs. These are prepared experiments;
long-distance A100 performance and global existence are not established.

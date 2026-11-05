# Reproduce the water-wave CPU pilot

The objective is finite-distance numerical resolution and trustworthy neural
approximation for `water-wave-dysthe-spatial-v1`. The measured decision is a slab
spectral PINN for further **per-inlet** study. It is not a learned operator or an
existence certificate. The [report](../results/cpu-pilot-2026-10-26/report.pdf)
explains every test's goal, results and limits.

Use the `codex/cpu-pilot-a100` experiments workstream. Its `pyproject.toml` pins
core `a47a42321475feb6bc55031f4b752419f4b7fda9` and learning
`db7274f8cfe959cf187351f56b3dd4bf89f0e9b4`. No unpublished sibling source is needed.
For a CPU environment, install a compatible PyTorch CPU wheel, then the extras:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[pilot]'
python scripts/check.py
python -m dysthe_experiments.cpu_pilot \
  --config configs/cpu-pilot.json --run runs/cpu-pilot-reproduction --stage all
```

The measured host used Python 3.14.7, PyTorch 2.10.0+cu128 executing on CPU,
NumPy 2.4.2 and SciPy 1.17.0. A CUDA build is not required for CPU execution.
Dependency ranges in the extras are not a reproducible lock; record exact
installed versions for another environment. No bitwise cross-platform guarantee
is asserted.

`prepare` generates 18 initial fields, checks whole-inlet splits and independently
refines reference grid and step. Failed qualification stops training. `operators`
trains FNO and PINO; `pinns` compares global, NLS-warm and four-slab per-inlet fits;
`diagnostics` computes reference regularity/tail information and longer numerical
propagation. `all` executes all four stages. Use a fresh run directory whenever
configuration changes. Training restores best/last state and optimizer/RNG state
from an unchanged run; it does not retrain completed fits. `--pinn-variants slabs`
selects only the chosen PINN. `--device cuda` fails if CUDA is unavailable.

Reconstruct the frozen CPU comparison and its vector figures separately:

```bash
mkdir -p runs/cpu-pilot-audit
cp results/cpu-pilot-2026-10-26/protocol.json runs/cpu-pilot-audit/
python -m dysthe_experiments.pilot_audit \
  --run runs/cpu-pilot-reproduction --results runs/cpu-pilot-audit
python -m dysthe_experiments.pilot_sensitivity --run runs/cpu-pilot-reproduction
```

`pilot_audit` is deliberately specific to the complete CPU comparison: both
operators, all three PINN variants, two seeds and the declared horizons. It is
not a generic evaluator for a selected-only A100 run. The A100 trainer writes
per-case interval errors, endpoint errors, residual audits and runtime metadata.
The independent CPU sensitivity script uses three validation fields and checks
signed perturbation responses and long-run H2 grid/step convergence. Consult
`--help` for its output paths; use a separate results directory when reproducing
rather than replacing the published evidence.

The protocol fixed operator selection before training. PINN method selection
uses validation endpoint H1, and each PINN's checkpoint uses residual validation
without reference labels. The audit makes the reducer explicit and preserves
raw case values. Test scores do not select a checkpoint or method. A new inlet
requires a new PINN optimization; FNO/PINO predict new inlets without fitting.
Do not combine these tasks into a single ranking.

Data, optimizer checkpoints and arrays remain under ignored `runs/`. Git contains
compact metadata, hashes, training curves, per-station CSV/JSON, the report source
and vector figures. The local original run is `runs/cpu-pilot-2026-10-26`.
The raw files are not hosted by this Git repository; regenerate them with the
command above. Hashes identify the retained local artifacts, not a download URL.

The 36-field A100 reference campaign was also qualified on CPU through xi=8.
Its wider PINN and longer training remain proposed. Follow the
[cluster guide](../cluster/README.md) for the single-A100 launch; no GPU speedup,
GPU restart equivalence or scheduler execution has been measured locally.

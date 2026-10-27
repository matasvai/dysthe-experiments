# CPU repair evidence, 27 October 2026

These files document diagnosis and verification of the `codex/pinn-h2-rerun`
workstream. They are CPU results; the repaired A100 campaign is pending.
Historical cluster evidence remains in `../long-distance-gpu2-2026-10-26`.

- `random-failure-diagnosis.json`: all 12 reconstructed exception records match
  the prior cluster manifest's bytes and SHA-256. The reproduced mechanism is
  cross-BLAS last-bit variation in independently regenerated random inlets.
  The exact historical CPU kernel dispatch was not recorded.
- `cross-blas-reuse.json`: two subprocesses using HASWELL and NEHALEM OpenBLAS
  dispatch retain identical canonical arrays/hashes/mtime, while the four
  separately regenerated random inputs differ. Other BLAS builds need not
  exhibit that dispatch-dependent difference.
- `first-slab-frequency-audit.json`: original K48 failure reproduced on CPU,
  including float64 and K24 controls. Error concentration and reference
  projection floors distinguish unnecessary high-mode error from lost capacity.
- `first-slab-paired-ablation.json`: six matched validation case/seed pairs
  across packet, sideband, and random families; both full-K48 parameterizations.
- `full-horizon-ablation.json`: production trainer/replay through xi=4, eight
  0.5-distance slabs, 400 updates per slab, seed 11, one validation inlet per
  family. All three unscaled fits fail and all three scaled fits pass the
  unchanged combined gate. It includes independent grid/step refinement,
  source hashes, residuals, and raw artifact hashes/location. A separate
  complex128 Fourier audit quantifies minor interpolation roundoff; the gate
  values retain the production metric convention.
- `workflow-smoke.json`: all 24 reference/PINN/sensitivity/domain tasks execute
  on the small CPU config. Zero execution errors; 17 completed, three stopped
  at scientific gates, four not applicable. The tiny-budget sideband fit and
  coarse packet-domain checks remain visible failures, not hidden successes.
- `workflow-export-verification.json`: corrected end-to-end export verified
  with 233 metadata files, 181 inventoried raw artifacts, zero missing expected
  evidence, and zero integrity findings. The original transient exporter log
  remains in ignored local storage; the full workflow exposed and fixed an
  aggregate-summary checkpoint-path inference bug.
- `verification.json` and `*-checks.txt`: final software verification and
  package/source identities. Passing tests is separate from scientific accuracy.

Relative errors are fractions in JSON (multiply by 100 for percentages).
No test inlets were used in the conditioning selection. Previously inspected
cluster test inlets are retained as diagnostic regression cases for the rerun;
they cannot serve as a new untouched evaluation. No global existence,
long-distance extrapolation, or GPU performance conclusion follows here.

Reproduce the matched validation diagnostic with the installed repair packages:

```bash
python scripts/diagnose-pinn-conditioning.py \
  --config configs/long-distance-h2-check.json \
  --run runs/conditioning-validation-new \
  --output results/conditioning-validation-new.json
python scripts/check-cross-blas-inlets.py \
  --config configs/long-distance-h2-check.json \
  --output results/cross-blas-new.json
```

Use fresh outputs. The CLI saves raw trajectories/checkpoints/source snapshots
under `--run`, outside Git. The reviewed raw ablation was additionally retained
under the local ignored `runs/conditioning-validation-20261027` directory.
The diagnostic script was made portable after the original run; the evidence
records the original script hash and source hashes rather than claiming a
bitwise rerun across differing hardware/library versions.

The [cluster runbook](../../docs/h2-rerun.md) specifies installation, candidate,
control, and longer-distance configurations and their distinct goals.

# Dysthe work checkpoint — September 10, 2026

The 32-page report is `output/pdf/dysthe_checkpoint_2026-09-10.pdf`.
Pages 2–5 provide the summary, progression, status, evidence map and reading guide.
Pages 9–12 explain the preferred review figures 5–8. The full experiments,
literature and technical record follow. No simulations were rerun for this checkpoint.

## State of the research

- Linear three-coordinate optical and canonical propagators have been implemented.
- Eleven artificial input recipes were surveyed; selected cases were refined through t=16.
- Four Gaussian-based decay curves reach t=256 with a separate spatial audit.
- Targeted frequency, scaling, cancellation, concentration and independent quadrature tests are complete.
- Fixed-point quadrature checks reach t=1024; global peak searches at t=512 and t=1024 remain open.
- Figures 5–8 and their additional diagnostics are documented. Their exact diagnostic series still needs independent box/grid convergence checks.
- No stronger uniform theorem, counterexample, measured-data simulation or full nonlinear fluid simulation has been established.

## Rebuild the checkpoint PDF

From the extracted archive root, with Tectonic available:

```bash
python3 report/build_checkpoint.py
tectonic report/checkpoint_2026_09_10.tex
```

The builder reads the preserved `report/meeting_report.tex`. It does not rerun
simulations or overwrite the meeting report. The compiled output is
`report/checkpoint_2026_09_10.pdf`. Figures and saved numerical tables are included.
The scripts for regenerating results require NumPy, SciPy, Matplotlib and Torch;
see the report's technical record for batch-specific commands.

## Archive and continuation

The archive includes this report, scripts, LaTeX dependencies, saved small results,
and all twenty review figures. Large NPZ field arrays and the supplied manuscript
are excluded; simulation scripts can regenerate the arrays. `manifest.json`
records SHA-256 hashes and sizes of included files, excluding the manifest itself.

Resume by choosing between convergence for the selected non-Gaussian cases and
later global Gaussian peak searches. Keep finite-time numerical behavior separate
from uniform-in-input analytical claims. The discussion on page 22 gives the
full proposed experiment priorities.

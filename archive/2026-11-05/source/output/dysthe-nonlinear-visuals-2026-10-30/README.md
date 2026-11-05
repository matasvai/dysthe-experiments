# Nonlinear water-wave envelopes: visual companion and CPU experiments

This package recreates the opening visual style of the September checkpoint using newly computed one-dimensional water-wave trajectories. It includes editable LaTeX, vector PDF figures, an animation, complex solution arrays, checks and CPU cluster instructions.

## What has actually been computed

| Experiment | Purpose | Completed evidence |
| --- | --- | --- |
| Gaussian, smooth random and two packets; Dysthe/NLS/free linear | Compare nonlinear shape, phase, Fourier power and ordinary envelope maximum | Periodic calculations through scaled T=4; independent grid and step comparisons pass for all six nonlinear cases |
| Very early evolution | Resolve changes near the initial data without hiding them behind the main envelope scale | T=0.0001, 0.001, 0.01; three profiles plus a higher-cutoff Gaussian; eight nonlinear cases pass refinement |
| Gaussian cutoff audit | Separate initial Fourier truncation from dynamically generated tail features | Cutoffs 12 and 24 through T=0.1; initial discrepancies about 3.77e-9 and roundoff respectively |
| Distant free-linear evolution | Use the exact Fourier propagator without time stepping or training | Selected periodic snapshots through T=1024; mode-power/action and spatial sampling checks pass |
| Nonlinear horizon ladder | Investigate later behavior with numerical and domain checks | Framework and tiny execution checks only; T=16,32,64,128,256,512,1024 are unrun targets in this package |

The two long nonlinear tracks have different conclusions: periodic waves concern focusing and possible recurrence; isolated packets concern decay only while box-enlargement and boundary checks pass. A nonzero conserved action in a fixed periodic interval gives max|u| >= sqrt(action/length). Neither a successful finite-time run nor a small neural residual proves global existence. A failed numerical gate is not proof of PDE breakdown.

## Equation and scope

Model ID: `dysthe-1979-temporal-potential-1d-stability-sign-v1`.

```
u_T = -i u_ss/8 - i |u|^2 u/2
      + epsilon [u_sss/16 - 3|u|^2 u_s/2
                 + u^2 conjugate(u_s)/4 + i u |D_s|(|u|^2)/2].
```

Here epsilon=0.05, |D_s| has Fourier symbol |k|, A=epsilon*u, s=epsilon*(x-t/2), T=epsilon^2*t. Time is scaled, not physical seconds. The primary model is the separately versioned temporal potential-envelope restriction adopted in the October 30 modelling report, with the explicitly labelled stability-consistent coefficient c=3/2. The literal printed 1979 equation gives a different coefficient under the previously documented source audit; no authenticated erratum is claimed. The active `water-wave-dysthe-spatial-v1` free-surface baseline is not changed by this companion. Historical three-coordinate canonical data are not used as water-wave evidence.

Original source: K. B. Dysthe, "Note on a modification to the nonlinear Schrodinger equation for application to deep water waves," Proceedings of the Royal Society A 369 (1979), 105-114, https://doi.org/10.1098/rspa.1979.0154. The accompanying October 30 modelling report documents the coefficient comparison and the separate potential/surface variables.

Every plotted carrier is illustrative Re(u exp(3 i s)), not a calibrated free-surface reconstruction. The 3D figures show position, evolution time and amplitude. Gaussian/two-packet inputs are not claimed solitons. These examples are not a seeded Benjamin-Feir instability experiment or a NOAA comparison.

## Read and reproduce the figures

Open `report/main.pdf`. The editable source is `report/main.tex` with `report/royden-report.sty` and all nine figure PDFs in `report/figures/`. `gaussian-evolution.mp4` animates the computed Gaussian through T=4. PNG previews accompany the vector PDF figures. The compiled PDF contains no embedded raster images; the separate PNG previews and animation are raster formats.

Tested numerical/plotting environment: Python 3.14, NumPy 2.4.2, SciPy 1.17.0, Matplotlib 3.10.8. The numerical methods do not need PyTorch, CUDA or a GPU. Install `requirements.txt` in a prepared environment. Animation needs FFmpeg; report compilation needs pdfLaTeX with Latin Modern, microtype, caption and the bundled style's standard packages.

To redraw from included data, from this package directory:

```bash
python make_visuals.py --animate
python make_tail_visual.py
python make_time_range_visuals.py
cd report
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

To regenerate numerical evidence without overwriting the supplied results:

```bash
python run_visual_cases.py --output fresh-data
python audit_initial_tails.py --data fresh-data
python near_zero.py --data fresh-data --output fresh-data/near-zero
python long_linear.py --data fresh-data --output fresh-data/long-linear
```

The plot scripts intentionally read the included `data/` directory. To plot a regenerated dataset, work in a second copy of this package and place that dataset at `data/`; retain the distributed evidence unchanged. Drivers refuse to overwrite existing numerical outputs.

## CPU long-horizon campaigns

Read `long_horizon/README.md` for the complete two-track protocol, configurations, fixed-grid timing calibration and Slurm commands. Both tracks use independent fine, grid-comparison and step-comparison trajectories. The isolated track adds a doubled-domain comparison with matched grid spacing and physical initial cutoff. A failed monitored gate stops that case and blocks its later targets.

```bash
python -m unittest discover -s long_horizon -p 'test_*.py' -v
python long_horizon/run_long_horizon.py --config long_horizon/configs/periodic-targets.json --output new-periodic-cost --calibrate-steps 64
python long_horizon/run_long_horizon.py --config long_horizon/configs/periodic-targets.json --output new-periodic-run
python long_horizon/run_long_horizon.py --config long_horizon/configs/isolated-targets.json --output new-isolated-run
```

Run calibration on the intended hardware before choosing wall time. Fixed-grid extrapolation omits diagnostics, output, retries and future refinement. Full runs can be stopped by a scientific gate well before their target. Saved states support audit, but this version does not implement automatic restart. The Slurm template requests CPU resources; reserving an A100 does not accelerate this NumPy code. No cluster jobs or neural training were executed by preparation of this package.

## Evidence and provenance

- `data/summary.json`, `data/verification.json`: T<=4 data and checks.
- `data/gaussian-tail-audit.json`: cutoff control, including its independent finer-step comparison.
- `data/early-late-verification.json`, `data/near-zero/summary.json`, `data/long-linear/summary.json`: additional near-zero and exact-linear evidence.
- `long_horizon/verification.json` and `long_horizon/verification/`: seven software tests, two tiny Gaussian smoke runs and explicitly labelled calibration measurements.
- `vendor_temporal_solver.py`: unchanged snapshot of the previous temporal solver; hashes are recorded with each dataset.
- `PACKAGE-MANIFEST.json`: file hashes for the delivered package.

All figures were checked after rendering the compiled PDF. The numerical refinement criteria are empirical checks of the specified discretizations, not rigorous continuum error bounds. Long-time physical interpretation additionally requires the weakly nonlinear, narrowband deep-water assumptions to remain appropriate.

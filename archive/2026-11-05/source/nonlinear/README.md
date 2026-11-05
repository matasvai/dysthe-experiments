# Nonlinear envelope solver and completed experiments

This solver accompanies the 14 September 2026 technical report. It implements:

- The **one-coordinate spatial water-wave Dysthe equation** of Fedele and Dutykh, with coefficient tuple (1,1,8,2,2).
- The **three-coordinate optical equation (1.27)** in the supplied Mosincat–Pilod–Saut manuscript, including both derivative terms in the nonlinearity and the mixed linear symbol.

These are distinct models. Optical stripes are localized in x and periodic in y,z; they are not fully localized 3D solitons. A leading-order water-surface illustration in the report repeats a one-coordinate solution across straight crests.

## Reproduction

Run from the parent project directory. Observed environment: Python 3.14.5, NumPy 2.4.2, SciPy 1.17.0, Matplotlib 3.10.8.

```bash
OPENBLAS_NUM_THREADS=4 python -m unittest nonlinear.test_solver -v
OPENBLAS_NUM_THREADS=4 python -m nonlinear.experiments all
OPENBLAS_NUM_THREADS=4 python -m nonlinear.refinements all
OPENBLAS_NUM_THREADS=4 python -m nonlinear.final_checks
python -m nonlinear.analyze
python report/nonlinear_tables.py
tectonic report/nonlinear_solitary_wave_report.tex
```

The complete experiment sequence can require several minutes or longer depending on hardware. Three-coordinate evolution temporarily retains output arrays in memory; the largest cases use several hundred MB or more including FFT work arrays. The saved full fields occupy about 269 MB in this checkpoint. The solver archive omits large field arrays but includes every JSON summary and the small computed profile arrays. Rerun the sequence before regenerating all figures from that compact archive. The portable LaTeX archive compiles directly from its included figures and tables without simulations.

Groups: `experiments benchmark|profiles|dynamics|transverse`; `refinements collision|perturb|transverse`. The dynamics group includes baseline collisions. Runs write deterministic filenames and replace those outputs when repeated.

## Solver interface

`FourierModel` constructs periodic grids and Fourier operators. Supply any compatible complex initial array to `evolve(model,u0,T,dt,...)`. `profile` computes the centered spatial Dysthe traveling profile. The initial field is projected onto the retained band, |mode_j| < N_j/4. Refinement of both grid and box is required for a new parameter regime.

A configurable run:

```bash
python -m nonlinear --model dysthe --n 1024 --lengths 160 \
  --epsilon 0.15 --mu 2 --speed 0 --initial solitary \
  --T 20 --dt 0.0025 --out results/nonlinear/custom_solitary
```

Use `python -m nonlinear --help` for Gaussian packets, optical stripes, perturbations, and linear controls. Command-line output directories contain `run.json` and `snapshots.npz`; predefined experiments use a shared filename stem for each JSON/NPZ pair.

## Evidence and limits

All six verification tests pass. The exact NLS benchmark reaches relative field error 3.68e-9 at T=20. All 28 continued profiles have discrete residual below 2e-11. Refined Dysthe collisions show about 1.8% excess action outside two radius-six cores; this is a mask-dependent radiation proxy. Optical transverse variation grows from 0.005 to about 0.04761 through T=4, with final pairwise complex-field discrepancies below 4.68e-4 in the reported comparisons. Optical results through T=8 remain spatially unresolved. No new nonlinear decay exponent or fully localized 3D soliton is claimed.

See the report for equations, phases, norms, physical interpretation, tables, and limitations. `results/nonlinear/analysis_summary.json` contains the derived values; all per-run JSON files preserve their underlying diagnostics. The report distinguishes continuum invariants from their projected, quadrature-based numerical diagnostics.

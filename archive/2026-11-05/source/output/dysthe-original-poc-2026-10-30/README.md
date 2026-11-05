# Original temporal Dysthe: a numerical proof of concept

This directory is a standalone, separately versioned water-wave calculation. It does **not** change or relabel the active `water-wave-dysthe-spatial-v1` research repositories.

The calculation uses the longitudinal, one-profile-coordinate restriction of the 1979 **temporal potential-envelope** equation. It does not compute the original transverse two-dimensional problem. The printed-sign and stability-consistent-sign equations are separate hypotheses; neither is silently substituted for the other. The accompanying research report explains the source ambiguity. Prioritize the `stability` variant as the later-standard, 1979-stability-consistent formulation; retain `literal` as the printed-sign sensitivity comparison. No authoritative erratum is claimed.

**Cluster status: not run on cluster in this package.** The numerical results are local CPU calculations. This standalone directory contains the temporal reference solver and its checks; it contains no trained temporal PINN or neural operator. The separate spatial PINN repair handoff is in [cluster/README.md](cluster/README.md).

## Reproduce

Run these commands from this directory. Python 3.11 or newer is required by the pinned NumPy/SciPy versions; the recorded CPU environment used Python 3.14.8, NumPy 2.4.2 and SciPy 1.17.0. No GPU or PyTorch is required by the temporal solver. Use a prepared environment; the following creates an isolated local one if needed.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export DYSTHE_PYTHON="$PWD/.venv/bin/python"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
"$DYSTHE_PYTHON" -m unittest -v test_run.py

# Short coarse check through T=4; all three profile families are included.
"$DYSTHE_PYTHON" run.py --max-time 4 --output rerun-coarse-T4

# Regenerate the staged reference campaign (N=2048, dt=0.005).
"$DYSTHE_PYTHON" run.py --config config-refined.json --output rerun-refined

# Separate high-resolution audit at the selected horizon T=16.
# N=4096, dt=0.0025; it does not automatically continue to T=32.
"$DYSTHE_PYTHON" run.py --config config-audit.json --output rerun-audit
"$DYSTHE_PYTHON" supplement.py --results rerun-audit --export rerun-portable-audit
```

These are fresh output names: preserve the packaged `results*` directories. Each output directory records the full configuration, solver source hash, and Python/NumPy/SciPy versions; incompatible reuse is rejected. Saved numerical fields are `complex128`; cached arrays are checksum-checked. Initial conditions use the recorded seed, but byte-for-byte equality across different numerical libraries/hosts is not promised. Exact runtime matching does not guarantee bitwise reproducibility. Never run two writers against the same output directory; there is no cross-process run lock.

To regenerate the original coarse staged campaign, omit `--max-time 4` and use another fresh output directory. `--max-time` retains only configured horizons at or below the supplied value; it does not add a new horizon. Although help lists `--families`, the declared comparison intentionally rejects that option. Change a copied JSON configuration to select model/epsilon subsets; the three profile families remain together.

Optional packet domain check, after the corresponding results have completed:

```sh
"$DYSTHE_PYTHON" supplement.py --results rerun-audit --domain
```

The public `supplement.py --domain` entry point checks `T=16`. It doubles the period and grid, performs its own grid/step checks, and only considers qualified literal/stability packet cases. It does not regenerate a separate historical T=8 check. CPU Slurm array instructions and evidence-return guidance are in [cluster/README.md](cluster/README.md).

## What is compared

- A localized Gaussian packet, a weakly modulated carrier, and a smooth random periodic envelope.
- Two small-parameter values, `epsilon=0.05` and `0.1`.
- The literal printed-sign equation (`c=1`), the stability-consistent sign variant (`c=1.5`), NLS, and a cubic-plus-induced-mean-flow control.
- Independent grid refinement at fixed time step and time-step refinement at fixed grid, with raw complex H0/H1/H2 errors. No phase alignment is applied to accuracy or model-difference measurements.

The time variable is `T=epsilon^2*t`, and `s=epsilon*(x-t/2)`, with potential envelope `A=epsilon*u`. The nonlinear time scale is `2/max|u(0)|^2`, because the cubic coefficient is `1/2`. These are not the spatial free-surface baseline's variables or scales.

The code's exact model IDs are:

| CLI/config name | Model ID | Equation choice |
|---|---|---|
| `literal` | `dysthe-1979-temporal-potential-1d-literal-v1` | Printed-sign reduction, `c=1` |
| `stability` | `dysthe-1979-temporal-potential-1d-stability-sign-v1` | Stability-consistent sign, `c=3/2` |
| `nls` | `dysthe-1979-temporal-potential-1d-nls-control-v1` | Quadratic dispersion and cubic NLS only |
| `meanflow` | `dysthe-1979-temporal-potential-1d-meanflow-only-v1` | NLS plus induced mean flow, no third derivative or derivative nonlinearities |

For the two full temporal variants, with `D` the Fourier multiplier `|k|`, the implemented right-hand side is

```text
u_T = -i*u_ss/8 - i*|u|^2*u/2
      + epsilon*(u_sss/16 - c*|u|^2*u_s
                 + u^2*conj(u_s)/4 + i*u*D(|u|^2)/2).
```

The inputs are spatial initial profiles at `T=0`. The helper name `inlets` does not turn them into wavemaker time traces. The spatial repair campaign instead uses a free-surface envelope, propagation coordinate `xi`, and retarded profile coordinate `tau`.

The Fourier method retains `|mode|<N/4`, which prevents cubic aliasing into the retained band. The linear dispersive flow is exact within an integrating-factor fourth-order Runge–Kutta method. Ten tests check dispersion, plane-wave phase, the two sign variants, the mean-flow multiplier, cubic de-aliasing, the NLS limit, the NLS sech profile, fourth-order step refinement, and input/time contracts.

## Read the evidence

Each result JSON reports its status and retains failed checks. The next staged horizon is blocked after a failed refinement gate. `results-audit` is an explicitly separate higher-resolution check at `T=16`; it does not authorize a longer continuation automatically.

The tolerances are numerical acceptance choices: relative H0 `1e-5`, H1 `1e-4`, H2 `1e-3`, retained-edge action fraction `1e-8`, and action drift `1e-6`. A small action drift alone does not pass a calculation. Peak, H2 growth, spectral width, formal steepness/bandwidth indicators, and sampled focusing/recurrence candidates are diagnostics, not existence certificates.

`summary.json` and CSV exports contain full-grid diagnostics. The initial portable exports are Fourier-resampled to 256 points; `portable-primary` uses512points. These `*-plot.npz` arrays are and stored as `complex64` **only for visualization**. Their manifest reports how much spectral action was omitted and links each to its full-grid source hash. They must not be used to recompute the reported refinement tests.

Whole-line interpretation requires a domain-size study, and physical validity requires additional water-wave evidence. The accompanying packet domain check doubles both period and grid count and compares full-box derivatives on the common interval; no random or sideband field is zero-padded into a purported isolated packet.

## Final primary result and time-step repair

`primary-results.json` records six qualified c=3/2 trajectories throughT16, with each source array and time step identified. The high-steepness wavetrain needed a separate smaller-step audit: the initial H2 amplification near601 was numerical; after refinement it is28.4. Earlier failures remain in their original summaries. Reproduce that bounded extra check with the new, standalone command below. It requires a **fresh output directory** and computes all three trajectories itself; it does not read the packaged JSON-only historical caches.

```sh
"$DYSTHE_PYTHON" rerun_time_audit.py --help
"$DYSTHE_PYTHON" rerun_time_audit.py --output rerun-time-audit
```

The defaults reproduce the primary `sideband`, `epsilon=0.1`, `T=16` audit with N=4096 and fine step0.0003125. `--config`, `--time`, and `--step` support explicitly changed checks. This command reports the numerical refinement and diagnostics; it does not recompute the separate historical NLS comparison. The archived `time_audit.py` remains unchanged to preserve its recorded source hash and is not the portable rerun entry point.

`collect_results.py` is a **local full-data collector** tied to the original `results-audit` and `results-time-audit` paths. It requires their full raw NPZ arrays. The forwardable package deliberately omits those large arrays, so its JSON summaries and portable plot arrays cannot run that collector or recompute the original numerical diagnostics. Regenerate full trajectories in fresh directories to perform new numerical checks.

`representation-windows.json` records projection floors atT2/4/8/16. K48 is inadequate for these late temporal targets; shorter early windows are a suitable first neural test. The physical-band diagnostics are descriptive thresholds, not certified validity criteria.

`mi_check.py` uses the Matplotlib dependency in requirements.txt and checks complex growth/phase, padded nonlinear RHS and semidiscrete mass pairing. Run `python mi_check.py --help` for output locations. `experiment-plan.json` clearly marks executed work and proposed temporal PINN/FNO work. No new temporal neural model was trained.

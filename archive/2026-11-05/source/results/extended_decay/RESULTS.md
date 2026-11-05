# Weighted decay extended to t=1024

Prepared 12 September 2026. New linear Gaussian-family calculations; no nonlinear/soliton experiment is claimed.

The observable is M(t)=|| |Dx+beta|^(1/2) S(t)f ||infinity, with exact continuum ||f||1=1. The compensated ratio is R=t^(7/6) M(t). Four cases are searched at each of 512 and 1024. Earlier values through 128 are retained; the t=256 values are from the stronger spatial audit.

## Main findings

The inhomogeneous Gaussian has effective exponent 1.3168 over 512 to 1024. The inhomogeneous near-degeneracy ratio peaks among sampled times at 512 and decreases at 1024, yielding exponent 1.1960. The homogeneous Gaussian gives 1.0903, while the homogeneous near-degeneracy case gives 1.0260 and its identified dominant candidate shifts from an off-axis ring to r=0. These differing finite-interval behaviors do not establish one common asymptotic exponent.

## Results

| Regime | Input | t | R(t) | Effective exponent from preceding doubling | x peak | r peak | Screen |
|---|---|---:|---:|---:|---:|---:|---|
| homogeneous_minus | gaussian | 256 | 0.00998865 | 1.0435 | -20.7469 | 0.0000 | previous spatial audit |
| homogeneous_minus | gaussian | 512 | 0.01073563 | 1.0626 | -26.0728 | 0.0000 | new search: pass |
| homogeneous_minus | gaussian | 1024 | 0.01131901 | 1.0903 | -32.7981 | 0.0000 | new search: pass |
| homogeneous_minus | near_degeneracy | 256 | 0.00888030 | 1.0740 | 7.7961 | 6.5381 | previous spatial audit |
| homogeneous_minus | near_degeneracy | 512 | 0.00933840 | 1.0941 | 8.4031 | 9.3537 | new search: pass |
| homogeneous_minus | near_degeneracy | 1024 | 0.01029456 | 1.0260 | -32.5408 | 0.0000 | new search: pass |
| inhomogeneous_plus | gaussian | 256 | 0.00372378 | 1.2934 | -8.9154 | 0.0000 | previous spatial audit |
| inhomogeneous_plus | gaussian | 512 | 0.00337630 | 1.3080 | -11.4239 | 7.6308 | new search: pass |
| inhomogeneous_plus | gaussian | 1024 | 0.00304254 | 1.3168 | -14.5440 | 9.3580 | new search: pass |
| inhomogeneous_plus | near_degeneracy | 256 | 0.00586208 | 1.0963 | -841.2464 | 0.0000 | previous spatial audit |
| inhomogeneous_plus | near_degeneracy | 512 | 0.00591274 | 1.1543 | -1643.3034 | 0.0000 | new search: pass |
| inhomogeneous_plus | near_degeneracy | 1024 | 0.00579369 | 1.1960 | -3227.2110 | 0.0000 | new search: pass |

![Compensated ratios](ratios.png)

![Finite-interval exponents](exponents.png)

The dotted vertical line marks the previous endpoint t=256. Larger effective exponents mean faster weighted-peak decay. Plot panels use separate vertical scales for R(t).

## Numerical checks

Full longitudinal/radial searches compare L=32768 and 65536 at dx=0.5 and dr=4 on 0<=r<=4096. A separate annulus extends the search to r=8192 at dr=2. The inner search is repeated with dx=0.25 and dr=2. Candidate seeds from multiple radial rows are locally optimized using non-periodic Gaussian integration.

At the best candidate, Gauss-Legendre orders 512 and 1024 are compared, together with a separate phase-partitioned adaptive Gauss-Kronrod calculation and 12-versus-14-standard-deviation frequency truncation. Fourier sums are compared at L=65536 and 131072 at the SAME candidate coordinates; the last comparison is pointwise, not another full-domain search.

| Case | t | Length change | Grid change | Independent integral discrepancy | Largest-period Fourier discrepancy | Annulus boundary / peak | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| homogeneous_minus/gaussian | 512 | 0.027% | 0.000% | 1.383e-14 | 5.239e-05 | 1.644e-12 | True |
| homogeneous_minus/near_degeneracy | 512 | 0.001% | 0.333% | 7.358e-15 | 4.665e-07 | 6.391e-23 | True |
| inhomogeneous_plus/gaussian | 512 | 0.000% | 0.000% | 5.922e-15 | 6.753e-09 | 1.991e-06 | True |
| inhomogeneous_plus/near_degeneracy | 512 | 0.007% | 0.001% | 1.134e-14 | 8.810e-06 | 7.857e-23 | True |
| homogeneous_minus/gaussian | 1024 | 0.058% | 0.014% | 2.281e-15 | 1.116e-04 | 2.849e-07 | True |
| homogeneous_minus/near_degeneracy | 1024 | 0.063% | 0.000% | 6.152e-15 | 1.227e-04 | 1.725e-12 | True |
| inhomogeneous_plus/gaussian | 1024 | 0.000% | 0.000% | 2.423e-14 | 3.294e-09 | 5.930e-03 | True |
| inhomogeneous_plus/near_degeneracy | 1024 | 0.017% | 0.000% | 1.775e-14 | 2.019e-05 | 2.995e-12 | True |

Completed 8/8 new case-time searches; 8 pass the stated 0.1% consistency screen.

## Follow-up of initial flags

Original screen failures are retained in the per-case JSON. When present, a followup record repeats the full inner radial search at dr=1 (and dr=0.5 if needed), at fixed L=65536 and dx=0.25, then compares independently optimized peaks. The outer annulus retains dr=2. The Pass column includes successful follow-ups; this does not erase the initial coarse-grid sensitivity.

The annulus-boundary column measures its longitudinal edge and r=8192 endpoint. The raw outer_boundary_fraction field in the first-run JSON is conservatively the maximum over both inner and annular scan boundaries; it includes the interior r=4096 interface and is not an exclusively outer-domain statistic.

## Interpretation limits

Effective exponents describe one doubling interval, not fitted asymptotic laws. A growing compensated ratio may still approach a bounded limit. Neither slower finite-time decay nor a change of dominant peak establishes a counterexample to a uniform bound with unspecified constant.

The searches are finite and sampled. Radial refinement and multiple local starts strengthen reliability but do not certify the continuum global supremum or exclude every narrow missed peak. The t=256 audit and the present extension use different checks, so the old audit is not transferred by assertion.

These four inputs are Gaussian-based and permit exact transverse reduction. Other artificial inputs, measured data and nonlinear solitary waves have not been evolved in this batch.

Reproduce with `python3 extend_decay.py`, `python3 refine_extended_decay.py`, and `python3 report/summarize_extended_decay.py`. Per-case JSON files retain search settings, seeds, optimizers, quadrature diagnostics, and flags. Earlier checkpoint reports are unchanged.
Follow-up homogeneous_minus/near_degeneracy at t=512, dr=1.0: successive sampled-peak change 0.06373%; independently optimized peak change 1.038e-14.

# First weighted-decay data batch

22 artificial initial fields: 11 recipes in each of two canonical regimes. All are L1-normalized.
858 measurements: 13 times from 0.25 to 4, and alpha = 0.25, 0.5, 0.75.
The production grid is 96^3 on [-48,48)^3. Field arrays are complex128; axes are (x,y,z).

The plots measure the actual fractional-derivative-weighted quantity, not ordinary amplitude.
A growing compensated ratio on this short interval is not a counterexample: the unknown bound may still be finite.
No asymptotic exponent is fitted before a reliable longer-time window is established.

| Regime | Input | Max ratio (alpha=.5) | Grid peak change at t=4 | Domain field change at t=4 |
|---|---|---:|---:|---:|
| homogeneous_minus | gaussian | 0.0038071 | 0.00% | 4.32% |
| homogeneous_minus | anisotropic | 0.001297 | 0.49% | 8.79% |
| homogeneous_minus | smooth_bump | 0.0020496 | 0.65% | 6.47% |
| homogeneous_minus | two_packets | 0.002354 | 0.00% | 4.35% |
| homogeneous_minus | chirped | 0.0039929 | 0.00% | 3.50% |
| homogeneous_minus | algebraic_tapered | 0.0033084 | 0.38% | 4.80% |
| homogeneous_minus | near_degeneracy | 0.0016749 | 0.00% | 8.80% |
| homogeneous_minus | away_degeneracy | 0.0018965 | 0.20% | 0.00% |
| homogeneous_minus | random_11 | 0.0035787 | 0.46% | 3.26% |
| homogeneous_minus | random_22 | 0.0035163 | 0.91% | 2.64% |
| homogeneous_minus | random_33 | 0.00444 | 0.76% | 3.74% |
| inhomogeneous_plus | gaussian | 0.0039693 | 1.04% | 0.02% |
| inhomogeneous_plus | anisotropic | 0.0019575 | 0.00% | 0.00% |
| inhomogeneous_plus | smooth_bump | 0.0032846 | 0.05% | 3.17% |
| inhomogeneous_plus | two_packets | 0.0024687 | 1.14% | 0.03% |
| inhomogeneous_plus | chirped | 0.0039694 | 0.97% | 0.09% |
| inhomogeneous_plus | algebraic_tapered | 0.0033364 | 0.00% | 6.43% |
| inhomogeneous_plus | near_degeneracy | 0.0015995 | 0.21% | 8.11% |
| inhomogeneous_plus | away_degeneracy | 0.0020832 | 0.08% | 0.00% |
| inhomogeneous_plus | random_11 | 0.0041378 | 0.55% | 0.19% |
| inhomogeneous_plus | random_22 | 0.002807 | 0.36% | 2.15% |
| inhomogeneous_plus | random_33 | 0.0038533 | 0.16% | 0.38% |

## Limits and next step
Checks currently cover t=4 and alpha=.5 only. They compare finite boxes; weighted multipliers can create long spatial tails.
Large domain errors invalidate interpreting those cases as evidence for a whole-space estimate. Refine flagged cases before extending time.
Each resolution uses the same continuum recipe and its own quadrature L1 normalization; errors include normalization differences.
The algebraic profile is explicitly tapered; it is not an untapered power-law test.
The degenerate-frequency recipes depend on beta, so their fields differ between regimes.
Random fields use 12 Gaussian-distributed wavevectors (standard deviation .25) and random phases inside a Gaussian window; seeds are 11,22,33.
Run `python3 decay_dataset.py` to regenerate all inputs, diagnostics and plots.

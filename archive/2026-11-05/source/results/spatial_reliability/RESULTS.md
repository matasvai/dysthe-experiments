# Spatial reliability at t=256

Same four cases, alpha=1/2. Domain sizes and search ranges are changed separately.

| Equation / input | L: 32768 to 65536 | Radius: 2048 to 4096 | L: 65536 to 131072 | Integral discrepancy | Pass 0.1% peak screen |
|---|---:|---:|---:|---:|---|
| Homogeneous / Gaussian | 0.01297% | 0.00000% | 0.00459% | 0.00251% | True |
| Homogeneous / Near degeneracy | 0.00198% | 0.00000% | 0.00070% | 0.00026% | True |
| Inhomogeneous / Gaussian | 0.00006% | 0.00000% | 0.00004% | 0.00001% | True |
| Inhomogeneous / Near degeneracy | 0.00328% | 0.00000% | 0.00256% | 0.00040% | True |

## Candidate peak locations

| Equation / input | R(256), independent integral | x | r | Location drift between final periods |
|---|---:|---:|---:|---:|
| homogeneous_minus / gaussian | 0.00998865 | -20.74692 | 0.00000 | 0.000000 |
| homogeneous_minus / near_degeneracy | 0.00888030 | 7.79614 | 6.53815 | 0.000792 |
| inhomogeneous_plus / gaussian | 0.00372378 | -8.91541 | 0.00000 | 0.000020 |
| inhomogeneous_plus / near_degeneracy | 0.00586208 | -841.24636 | 0.00000 | 0.000214 |

## Interpretation
The 0.1% screen checks each domain expansion, the final locally optimized peak drift, and the independent integral discrepancy. It is a numerical tolerance, not a proof of global maximality.
Longitudinal spacing remains 0.25 and radial spacing remains 2 during the broad scans. Local optimization evaluates continuous coordinates; it is not a full refinement of every radial cell.
The extension annulus samples radii 2050,2052,...,4096, complementing the original 0,2,...,2048 grid.
Independent integration splits at the fractional multiplier cusp and uses Gauss-Legendre orders 256 and 512 on 64 Gaussian-frequency panels plus the cusp split. The stored quadrature_tightening_change now denotes this order-doubling comparison.
The change of variables k=-beta +/- u^2 regularizes each side of the cusp. Final values are recomputed with this formulation.
The initial profiles are exactly L1-normalized Gaussian families. This result applies at t=256, alpha=1/2 and the two specified canonical regimes only.
The calculation searches a finite domain; low boundary amplitudes and stable samples do not exclude all narrow or remote maxima.
Numerical spatial stability strengthens the interpretation of these four computed ratios, but establishes neither an infinite-time decay theorem nor a counterexample.
Reproduce with `python3 spatial_reliability.py` then `python3 report/make_spatial_tables.py`. Per-case JSON checkpoints permit resuming unchanged scans.

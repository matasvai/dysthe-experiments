# Refined weighted-decay experiment

The original 11-family plots are now separated into individual panels. The larger-domain follow-up selects Gaussian, smooth bump, near-degeneracy, and random seed 33 in both regimes.
Follow-up uses alpha=1/2 (candidate power 7/6), times 0.25 to 16, and lengths 96, 192, 288 at spacing 1. A separate length-192 grid with spacing 2/3 checks resolution.
Domain and resolution checks occur at t=4,8,16. The displayed extended curves use length 288 at spacing 1.

## How to read the ratio
R(t)=t^(7/6) || |Dx+beta|^(1/2) S(t)f ||infinity / ||f||1.
Flat means the weighted maximum follows t^(-7/6) over that interval; rising means slower decay over that interval, falling means faster. A bounded rising curve is fully compatible with the claimed inequality.
The theorem does not require R to decrease. No universal constant is inferred from this finite collection.
Dots mean the two peak comparisons change by less than 2%; this is an exploratory numerical threshold, not a rigorous error estimate. Unmarked intermediate times have not been independently checked.

| Equation | Input | t | Ratio | Domain peak change: 192 to 288 | Grid peak change: h=1 to 2/3 |
|---|---|---:|---:|---:|---:|
| homogeneous_minus | gaussian | 4 | 0.003887 | 0.41% | 0.27% |
| homogeneous_minus | gaussian | 8 | 0.0057572 | 0.62% | 0.59% |
| homogeneous_minus | gaussian | 16 | 0.0073108 | 1.11% | 0.18% |
| homogeneous_minus | smooth_bump | 4 | 0.0020937 | 0.43% | 0.57% |
| homogeneous_minus | smooth_bump | 8 | 0.0045679 | 0.44% | 0.16% |
| homogeneous_minus | smooth_bump | 16 | 0.0067718 | 0.43% | 0.03% |
| homogeneous_minus | near_degeneracy | 4 | 0.001755 | 0.91% | 0.00% |
| homogeneous_minus | near_degeneracy | 8 | 0.0032269 | 1.11% | 0.03% |
| homogeneous_minus | near_degeneracy | 16 | 0.0049896 | 1.62% | 0.24% |
| homogeneous_minus | random_33 | 4 | 0.004555 | 0.50% | 0.58% |
| homogeneous_minus | random_33 | 8 | 0.0056477 | 0.91% | 0.35% |
| homogeneous_minus | random_33 | 16 | 0.0061767 | 1.61% | 0.29% |
| inhomogeneous_plus | gaussian | 4 | 0.0039693 | 0.00% | 0.94% |
| inhomogeneous_plus | gaussian | 8 | 0.0045246 | 0.00% | 0.00% |
| inhomogeneous_plus | gaussian | 16 | 0.0046795 | 0.01% | 0.05% |
| inhomogeneous_plus | smooth_bump | 4 | 0.0032838 | 0.01% | 0.11% |
| inhomogeneous_plus | smooth_bump | 8 | 0.0039593 | 0.03% | 0.70% |
| inhomogeneous_plus | smooth_bump | 16 | 0.0043051 | 0.00% | 0.54% |
| inhomogeneous_plus | near_degeneracy | 4 | 0.0015849 | 0.65% | 0.21% |
| inhomogeneous_plus | near_degeneracy | 8 | 0.002542 | 0.84% | 0.03% |
| inhomogeneous_plus | near_degeneracy | 16 | 0.0034386 | 1.36% | 0.01% |
| inhomogeneous_plus | random_33 | 4 | 0.0038533 | 0.00% | 0.15% |
| inhomogeneous_plus | random_33 | 8 | 0.0041141 | 0.00% | 0.01% |
| inhomogeneous_plus | random_33 | 16 | 0.0041462 | 0.00% | 0.04% |

The JSON also records full complex-field differences on a common cube. Field L2 and peak sensitivity are different error measures; the latter directly targets the inequality.
The grid and domain checks vary one axis of discretization at a time and do not establish a jointly converged infinite-domain supremum. Further box sizes may give nonmonotone errors near the fractional multiplier cusp.
These runs remain exploratory. No asymptotic exponent, theorem strengthening, or counterexample is established.
Reproduce with `python3 refine_decay.py` after `python3 decay_dataset.py`.

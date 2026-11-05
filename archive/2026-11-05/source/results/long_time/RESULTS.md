# Weighted decay through t=256

Two Gaussian-based families in both canonical regimes, alpha=1/2. This uses analytic transverse integration to avoid a prohibitively large full 3D grid.
The plotted diagnostic remains R(t)=t^(7/6) || |Dx+beta|^(1/2) S(t)f ||infinity / ||f||1.
The transverse profile is radial. The spatial maximum is searched over x and radial distance r, not only the origin.

| Regime | Input | t | R(t) | Domain change | Grid change | Independent integral error |
|---|---|---:|---:|---:|---:|---:|
| homogeneous_minus | gaussian | 16 | 0.0074286 | 0.016% | 0.000% | 0.008% |
| homogeneous_minus | gaussian | 32 | 0.0084197 | 0.031% | 0.000% | 0.017% |
| homogeneous_minus | gaussian | 64 | 0.0088706 | 0.066% | 0.011% | 0.036% |
| homogeneous_minus | gaussian | 128 | 0.0091712 | 0.059% | 0.055% | 0.018% |
| homogeneous_minus | gaussian | 256 | 0.0099907 | 0.295% | 0.058% | 0.020% |
| homogeneous_minus | near_degeneracy | 16 | 0.0050984 | 0.023% | 0.000% | 0.012% |
| homogeneous_minus | near_degeneracy | 32 | 0.0067188 | 0.039% | 0.000% | 0.021% |
| homogeneous_minus | near_degeneracy | 64 | 0.0077802 | 0.075% | 0.000% | 0.041% |
| homogeneous_minus | near_degeneracy | 128 | 0.0083281 | 0.065% | 0.000% | 0.050% |
| homogeneous_minus | near_degeneracy | 256 | 0.0088801 | 0.010% | 0.549% | 0.002% |
| inhomogeneous_plus | gaussian | 16 | 0.0046820 | 0.000% | 0.022% | 0.000% |
| inhomogeneous_plus | gaussian | 32 | 0.0045998 | 0.000% | 0.000% | 0.000% |
| inhomogeneous_plus | gaussian | 64 | 0.0043730 | 0.001% | 0.083% | 0.000% |
| inhomogeneous_plus | gaussian | 128 | 0.0040657 | 0.001% | 0.000% | 0.000% |
| inhomogeneous_plus | gaussian | 256 | 0.0037238 | 0.003% | 0.000% | 0.000% |
| inhomogeneous_plus | near_degeneracy | 16 | 0.0034460 | 0.011% | 0.000% | 0.000% |
| inhomogeneous_plus | near_degeneracy | 32 | 0.0042943 | 0.021% | 0.000% | 0.001% |
| inhomogeneous_plus | near_degeneracy | 64 | 0.0050461 | 0.043% | 0.003% | 0.001% |
| inhomogeneous_plus | near_degeneracy | 128 | 0.0055831 | 0.090% | 0.000% | 0.002% |
| inhomogeneous_plus | near_degeneracy | 256 | 0.0058621 | 0.196% | 0.002% | 0.000% |

## Numerical interpretation
Domain check doubles longitudinal period 4096 to 8192 and radial search 512 to 1024. Grid check halves x spacing from 0.5 to 0.25 and radial spacing from 4 to 2.
At t=256 an additional period-32768, radial-extent-2048 scan retains the fine spacings. Its peak is used in the final curve; its comparison is saved as extra_domain_change in summary.json.
Local optimization polishes the sampled maximum. The adaptive integral independently checks the complex field at that candidate maximum, not global maximality.
Finite search ranges, sampled maxima and finite quadrature prevent certification of the true supremum. Boundary amplitudes and optimizer status are saved.
No theorem constant, stronger exponent, or counterexample follows solely from a finite set of ratios.
Other artificial data families are not extended by this Gaussian-specific method. All earlier 3D FFT results remain intact.
Run `python3 long_time_decay.py` to reproduce.

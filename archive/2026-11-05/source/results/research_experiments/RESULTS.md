# Research-motivated experiments: results

{
  "scaling_max_error": 4.3595464884760277e-16,
  "sweep_total": 78,
  "sweep_initial_passed": 70,
  "followups": 8,
  "followup_passed": 8,
  "followup_max_change": 1.1408768200890273e-05,
  "followup_max_original_correction": 0.022881561184337533,
  "quadrature_max_disagreement": 4.4021994074117625e-12,
  "quadrature_256_disagreement": 2.4141342131879122e-14,
  "quadrature_max_truncation_change": 7.541406590522706e-12,
  "quadrature_warning_count": 0,
  "identity_passed": 10
}

## Axial spectral sweep at t=16

Ratios use the independent Gaussian follow-up where the original screen failed.

| Regime | sx | qx+beta | R(16) | Follow-up used |
|---|---:|---:|---:|---|
| homogeneous | 3.0 | -0.75 | 0.0025677 | False |
| homogeneous | 3.0 | -0.25 | 0.0060307 | False |
| homogeneous | 3.0 | 0.0 | 0.0072301 | False |
| homogeneous | 3.0 | 0.25 | 0.0060307 | False |
| homogeneous | 3.0 | 0.75 | 0.0025677 | False |
| homogeneous | 6.0 | -0.75 | 0.0021903 | False |
| homogeneous | 6.0 | -0.25 | 0.0042158 | False |
| homogeneous | 6.0 | 0.0 | 0.0049088 | False |
| homogeneous | 6.0 | 0.25 | 0.0042158 | False |
| homogeneous | 6.0 | 0.75 | 0.0021903 | False |
| inhomogeneous | 3.0 | -0.75 | 0.0015803 | True |
| inhomogeneous | 3.0 | -0.25 | 0.0035417 | True |
| inhomogeneous | 3.0 | 0.0 | 0.0038001 | True |
| inhomogeneous | 3.0 | 0.25 | 0.0038499 | True |
| inhomogeneous | 3.0 | 0.75 | 0.0043335 | False |
| inhomogeneous | 6.0 | -0.75 | 0.0014891 | False |
| inhomogeneous | 6.0 | -0.25 | 0.0032069 | False |
| inhomogeneous | 6.0 | 0.0 | 0.0034460 | True |
| inhomogeneous | 6.0 | 0.25 | 0.0034422 | False |
| inhomogeneous | 6.0 | 0.75 | 0.0029407 | False |

## Interpretation

Scaling agreement verifies the discrete implementation of an exact symmetry; it is not an independent whole-space convergence test.
The spectral sweep uses a 2% initial screen. Eight flagged measurements receive a separate radial Gaussian calculation and a 0.1% follow-up screen. Those screens assess tested changes, not certified errors.
The cancellation controls have zero spatial integral but Gaussian spectral tails; this is not a strict frequency-support-away-from-degeneracy experiment.
The approximate-identity test stops at width 0.75. Increasing ratios do not establish convergence to a kernel maximum or determine the optimal theorem constant.
At t=512 and 1024 the quadrature comparison keeps the t=256 peak coordinates fixed. It is a pointwise stress test, not a continuation of global peak decay curves.
The fixed-node implementation uses 512-point Gauss-Legendre on each regularized panel. The independent method uses adaptive Gauss-Kronrod on principal-phase-controlled panels with stationary points inserted and a separate cusp substitution. Neither is claimed to reproduce PathFinder or the cited saddle-point algorithms.
No new full nonlinear fluid simulation or naturally occurring dataset is included.

## Reproduce

Run research_experiments.py with scaling, sweep, quadrature, and identity arguments; run refine_research_flags.py; then python3 report/make_research_report.py. JSON checkpoints are retained per case.

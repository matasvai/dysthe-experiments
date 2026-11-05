# Spatial water-wave reference qualification

The optional `dysthe_core.reference.spatial` module ports only the approved
free-surface spatial Dysthe equation. With propagation distance xi and retarded
profile time tau, it evolves

u_xi = -i u_tautau - i |u|^2 u
       - epsilon (8 |u|^2 u_tau + 2 u^2 conjugate(u_tau)
                  + 2 i u |D_tau|(|u|^2)).

Use periodic tau in [-L/2,L/2), Fourier derivative i*k and mean-flow multiplier
|k| with zero mean mode. Strict |mode| < N/4 projection preserves the legacy
cubic Galerkin convention. Integration uses integrating-factor RK4. The epsilon=0
control is spatial focusing NLS; it is not a physical change of variables at
zero steepness. NumPy and SciPy are optional dependencies; contract imports
remain independent of them. No canonical periodic gauge is imposed.

Migration source: `/home/matasvai/Documents/Projects/Dysthe/nonlinear/solver.py`,
water-wave `FourierModel(kind='dysthe')` branch. Original files remain intact.
SHA-256 source snapshots:

- solver.py: b4bb6d196fa0e10976f7d4b19f97b8809f2793f776f21e7aa75951fc696ab7df
- test_solver.py: cb50271a78ef86c65ede461f10c3df87d56dc802c73ebb8935a20ce4ae83c43d
- nonlinear_solitary_wave_report.tex: 6b80a51f61953cadf82b39c86ea06e9bb467fc906507cdd02d5718b5f38be818

The independent tests use the exact NLS sech traveling wave, a nonzero-k
Dysthe plane wave with Omega=A^2-k^2+6 epsilon k A^2, and explicit Fourier
coefficient convolution of a nonconstant complex multimode field. The latter
includes a nonzero mean-flow contribution. Orbital-fit regression separately
checks a translated, phase-shifted field and its large raw complex error.

For F(tau-s xi) exp(-i mu xi), `profile_residual` interpolates the profile to
4N and evaluates the complete periodic PDE without an output projection. It
reports retained and full residuals relative to the traveling time derivative.
This tests the trigonometric interpolant, not whole-line existence. The stored
mu=3,s=0,epsilon=.15,N=2048,L=160 profile has a projected residual around 1e-11
but full residual about .00612. Even mu=2,s=0,epsilon=.025 and .05 at that grid
have full residuals about 1.28e-6 and 1.09e-5. Recomputing the latter at N=4096
reduces the full residuals to about 2.1e-11. Never qualify using only the stored
Petviashvili residual or tiny spectral action tail.

Action, energy and momentum are the pulled-back continuum functionals in
model.md. Quadrature uses 2N points, which resolves degree-six density for
retained fields (6K < 2N). Out-of-band inputs are rejected for invariant
quadrature. Energy/momentum need not be exactly conserved by the Galerkin ODE;
RK4 need not preserve action exactly. Conservation alone is not accuracy.
Petviashvili enforces conjugate reflection only in profile construction: its
convergence basin is not physical stability or a search of all branches.

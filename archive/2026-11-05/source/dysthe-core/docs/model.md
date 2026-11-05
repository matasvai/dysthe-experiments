# Water-wave equation contract

Model ID: `water-wave-dysthe-spatial-v1`; case schema **2**.
Physical system: weakly nonlinear, narrowband, unidirectional gravity waves in
deep water. The dependent variable `u(xi,tau)` is the scaled complex
**free-surface envelope**, not the velocity-potential envelope.

The approved spatial Dysthe convention is

```text
u_xi = -i u_tautau - i |u|^2 u
       - epsilon (8 |u|^2 u_tau + 2 u^2 conjugate(u_tau)
                  + 2 i u |D_tau|(|u|^2)).
```

This is Fedele–Dutykh's free-surface equation (1.3), coefficient tuple
`(a,h,c,e,f)=(1,1,8,2,2)`, in
[Hamiltonian description and traveling waves of the spatial Dysthe equations](https://arxiv.org/abs/1110.3605).
The local migration source is `nonlinear/solver.py`, `FourierModel(kind='dysthe')`.

## Variables and operators

In the source's dimensionless carrier coordinates `(x,t)`, `tau=epsilon*(2*x-t)`
and `xi=epsilon^2*x`; the free-surface envelope is `B=epsilon*u`.
`epsilon=k0*a` denotes carrier steepness. These definitions identify the formal
normalization; valid parameter bounds must be justified for each campaign.
The `epsilon=0` equation is a mathematical NLS control, not an invertible physical
coordinate transformation at zero steepness.

`xi` is propagation distance and `tau` is the retarded-time profile coordinate.
Use `|D_tau|` with Fourier multiplier `|k|`; the forward transform has kernel
`exp(-i*k*tau)`. Thus `H` has multiplier `-i*sign(k)` and `H*d_tau=|D_tau|`.
The periodic zero mode of this mean-flow derivative is zero. The free-surface
coefficient `e=2` must not be replaced by the potential-envelope value `e=0`.
The raw envelope is noncanonical; any Hamiltonian constraint requires its
correct variable transformation and boundary conventions.

## Data contract

Reference arrays are `complex128`, shape `(N_xi,N_tau)`, axes `[xi,tau]`.
`shape=[N_tau]` and `lengths=[L_tau]` describe the periodic profile grid.
The grid is `tau_j=-L_tau/2+j*L_tau/N_tau`, `j=0,...,N_tau-1` (no duplicated
endpoint). `propagation_positions` starts at zero and increases strictly.
`physics` contains exactly `epsilon`. The example is metadata, not a resolved
trajectory. Periodicity is a computational choice; localized packets need
domain-sensitivity checks. A two-axis array does not represent two horizontal
directions. ML channels, if used, are `[real,imaginary]`; record any scaling and
inverse and fit them using training data only.

Optical three-coordinate cases and other physical systems are incompatible;
there is no automatic conversion. Schema 1 is rejected. Never edit an old
artifact's identity to make it pass. Retain provenance, units, source hashes,
array hashes, initial-condition group IDs and reference refinement evidence.

Before accepting a solver port: independently check the linear multiplier,
plane-wave phase, nonlinear RHS including mean flow, NLS limit, projection,
propagation-step/grid/domain refinement and spectral-edge resolution. Evaluate
weighted Lp errors over tau, phase where amplitude is meaningful, spectrum and
the correct water-wave invariants. Conservation alone is insufficient.

`validate_case` checks declared metadata, not actual arrays or physical validity.
`validate_splits` checks group IDs; the producer must keep all refinements,
snapshots, seeds and augmentations of an initial field in the same split.

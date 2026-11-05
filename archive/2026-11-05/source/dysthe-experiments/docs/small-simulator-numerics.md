# Small initial-envelope simulator: numerical guide

This workstream uses the unchanged `dysthe_core.reference.spatial.SpatialDysthe`
reference. It solves the approved one-coordinate spatial free-surface-envelope
water-wave equation, with `(a,h,c,e,f)=(1,1,8,2,2)` and `epsilon=.025`. `xi`
is propagation distance and `tau` is retarded time. The Fourier grid is periodic,
without a duplicated endpoint; the mean-flow multiplier is `|k|` with zero mode
zero. The Galerkin projection retains `|mode|<N/4`, and propagation uses the
existing integrating-factor RK4 method. The raw complex field is not replaced
by its modulus during propagation, diagnostics, serialization or comparisons.

The three defaults are: a recomputed `mu=2,speed=0` periodic solitary branch;
the same branch multiplied by `1+.02 exp(-((tau-1.3)/2)^2)`; and
`exp(-tau^2/8) exp(.3 i tau)`. Full periodic profile residuals must be below
`1e-8` before either profile-based case can run. The branch is recomputed for
each declared domain; its finite-box mean flow is part of that problem.

Defaults are `L=160,N=4096,dxi=.00125,xi_end=4`, with 81 snapshots. Each case
has independent comparisons against `N=8192` at fixed `L,dxi`, `dxi=.000625`
at fixed `N,L`, and `L=320,N=8192` at fixed spacing and step. Comparisons
preserve complex phase and use all saved snapshots. The large-domain field is
cropped to the common physical interval, never folded into it.

From the repository family root, with `dysthe-core` and `dysthe-experiments`
as siblings:

```sh
export PYTHONPATH="$PWD/dysthe-core/src:$PWD/dysthe-experiments/src${PYTHONPATH:+:$PYTHONPATH}"
python3 -m unittest discover -s dysthe-experiments/tests -p test_simulator.py -v
python3 -m dysthe_experiments.simulator benchmark --output /tmp/dysthe-demo-benchmark
python3 -m dysthe_experiments.simulator check --case profile --output /tmp/dysthe-small-demo
python3 -m dysthe_experiments.simulator check --case perturbed --output /tmp/dysthe-small-demo
python3 -m dysthe_experiments.simulator check --case gaussian --output /tmp/dysthe-small-demo
```

Choose fresh benchmark/run output paths. The check command resumes already
completed runs only after configuration, source and array checksum verification.
It does not repeat stage-one campaigns. Each default viewer is written to
`<output>/<case>/viewer.json`; refinements and qualification evidence live in
`<output>/checks/<case>`. `fields.npz` retains full-resolution `complex128`
arrays `u[xi,tau]`, `tau`, `xi`, `initial` and, where applicable, `profile`.
The viewer stores sampled real and imaginary values; it is precomputed.
A call to `simulate(...)` performs a new bounded local numerical simulation and
starts with unqualified status, even for inputs that resemble the defaults.

The API is:

```python
from dysthe_experiments.simulator import case_info, validate_request, simulate
parameters, settings = validate_request('gaussian', {'chirp': .01}, {'xi_end': 2.})
report = simulate('gaussian', '/tmp/dysthe-custom', parameters, settings)
```

The engine accepts only named bounded parameters and a bounded grid, step,
horizon and total-work budget. These limits are resource and design screens,
not a physical validity theorem. It rejects initial projection loss above
`1e-8` and initial weak-steepness/narrowband screen failures. Evolution screen
failures are recorded and prevent qualification.

The width is the action-weighted RMS periodic distance about the circular
action centroid. Displacement unwraps that centroid from its initial value;
near-zero circular concentration makes these statistics unreliable. Exterior
action is measured beyond radius 8 around that moving centroid. It includes
packet spreading and tails, so it is not automatically radiation. For the
profile cases, an additional exterior residual against the best translated
and phased unperturbed branch is a radiation proxy, without a spectral
radiation interpretation. No amplitude rescaling is allowed in these fits.
Global phase and translation are separated from signed shape changes.

Qualification thresholds are declared in `compare_runs`: maximum raw complex
relative L2 `2e-4`, orbital relative L2 `1e-4`, signed shape-change discrepancy
relative to field norm `1e-4`, and discrepancy/shape-response `2%` where the
response exceeds `1e-7` of field norm. These are demonstration acceptance
criteria, not rigorous errors. Two levels per refinement establish numerical
agreement, not an observed convergence order. Alignment optimization can create
an approximately `1e-8` signed-shape discrepancy floor when raw complex fields
agree near machine precision. Padded continuum energy and momentum need not
be exact invariants of the Galerkin ODE or RK map; conservation alone cannot
qualify a trajectory.

The completed bounded campaign has 12 trajectories and 48,000 total steps.
All three defaults passed their declared checks on `xi in [0,4]`. A longer
interval or changed input needs its own checks. The profile has almost pure
phase rotation; the perturbed profile changes modestly and develops a weak
exterior component; the Gaussian exhibits substantial shape evolution despite
an almost unchanged endpoint RMS width. None of these finite-window
observations is soliton discovery, a stability result, or authorization to
train an FNO or generate a large dataset.

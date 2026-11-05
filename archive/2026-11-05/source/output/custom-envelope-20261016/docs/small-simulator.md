# Small spatial-Dysthe simulator

This local workbench uses the unchanged free-surface spatial Dysthe reference,
`water-wave-dysthe-spatial-v1`, coefficients `(1,1,8,2,2)`, epsilon `.025`.
It provides three short, case-specific checked examples, a restricted LaTeX
initial-envelope editor, and bounded fresh CPU runs. There is no learned model,
dataset sweep, paid compute or HPC submission.

## Launch on this Fedora computer

```bash
cd /home/matasvai/Documents/Projects/Dysthe/dysthe-experiments
bash scripts/run-simulator.sh
```

Open **http://127.0.0.1:8765/** on this Fedora computer. Localhost on another
device refers to that device. Python 3, NumPy and SciPy are already available
on the checked computer; the frontend has no downloaded dependencies. Stop the
server with Ctrl-C. If the port is occupied, use
`SIMULATOR_PORT=8766 bash scripts/run-simulator.sh` and open the printed URL.
`--data /absolute/path/to/examples` selects another saved-data directory. Leave
the foreground terminal open while using it. The existing transient user service
`dysthe-small-simulator.service`, when active, survives the launching terminal;
check `systemctl --user status dysthe-small-simulator.service` before starting a
second server. It is not enabled at boot; do not rely on logout/reboot persistence.

## Use

1. Select the solitary profile, the two-percent localized amplitude
   perturbation, or the complex Gaussian packet.
2. **View precomputed reference** loads saved trajectories with completed
   grid, step and domain evidence. Scrub or play the 81 numerical snapshots.
   Playback does not execute a solver or compute intermediate solutions.
3. Compare initial and current `|u|`, real/imaginary parts, or principal phase.
   Phase is masked below one percent of the snapshot peak. Change from packet
   view to the full periodic domain and use logarithmic heatmap contrast to
   inspect small exterior components.
4. Edit bounded input parameters, then **Run locally on CPU** to perform an
   actual fresh numerical integration. Progress counts real completed solver
   steps. One run is allowed at a time. Fresh results are explicitly marked
   as local results, with refinement qualification pending. Custom inputs do
   not inherit a default example's checks.
5. Select **Custom formula** to describe the initial complex envelope in the
   supported LaTeX subset. Declare parameters, preview amplitude and phase,
   inspect input warnings, then Run. Every edit requires a fresh preview.
   [Exact syntax, examples and bounds](custom-envelope.md) are documented.
6. **Download complex data .npz** saves the full-resolution `complex128` field,
   coordinates and initial envelope. The viewer displays 1024 tau samples;
   saved default solver fields have 4096 tau points. Custom grid size is listed
   with its result. Complex phase and custom formula provenance are retained.

Custom formulas use a small bounded parser, never Python or general symbolic
execution. They remain exploratory; the three checked defaults do not qualify
arbitrary formulas. A complex-array upload interface is not included.

## Why time is horizontal and playback moves through distance

This is the **spatial** Dysthe formulation. A snapshot fixes one downstream
station xi and shows the temporal packet trace across retarded time tau there.
Playback moves to successive downstream stations; it does not animate physical
time at a fixed point. The heatmap stacks those same traces vertically along xi.

The stored model convention uses dimensionless carrier coordinates `(x,t)`:
`tau=epsilon*(2*x-t)`, `xi=epsilon^2*x`, with free-surface envelope `B=epsilon*u`.
Thus tau is a scaled coordinate moving with the packet, not a clock reading in
seconds; xi is scaled distance, not metres. Physical units need carrier scales
and an explicit conversion. The interface labels the scaled envelope as A,
equivalent to stored `u`; `|A|` is its magnitude, not carrier oscillations or a
reconstructed water-surface elevation.

The three numerical checks test different directions and choices: **tau grid**
increases N to resolve each trace; **xi step** reduces the propagation increment
to test accuracy downstream; **tau window** enlarges the periodic profile domain
at fixed tau spacing to assess finite-window sensitivity. Their comparisons
cover all saved xi stations rather than evolving the equation in another variable.

## What was checked

Each default uses `L=160`, `N=4096`, `dxi=.00125`, `xi_end=4`, 81 snapshots.
For **each** input separately, the campaign compared `N=8192` at fixed length,
`dxi=.000625` at fixed grid, and `L=320,N=8192` at fixed tau spacing. All
snapshots were compared as complex fields. Each of these is a two-level
agreement check, not a measured order or rigorous error bound. The earlier
stage-one report establishes the propagator's analytic controls and step order.

| Default input | Grid max relative L2 | Step max relative L2 | Domain max relative L2 |
|---|---:|---:|---:|
| Solitary profile | 4.17e-15 | 3.07e-9 | 1.54e-6 |
| Localized 2% perturbation | 3.09e-15 | 4.08e-9 | 2.26e-6 |
| Gaussian | 9.28e-16 | 4.05e-11 | 2.18e-5 |

Errors retain phase and use the common physical window. Domain differences
include periodic mean-flow effects and, for the branch profiles, the recomputed
initial-profile difference. The Gaussian domain discrepancy is mostly global
phase; its orbital discrepancy is 9.66e-9. Fitting translation and phase never
rescales amplitude. Very small shape comparisons can be limited by the legacy
orbital optimizer's tolerance; raw complex-field differences are kept separately.

The chosen demonstration limits are in each `qualification.json`: raw complex
relative error <=2e-4, orbital error <=1e-4, signed shape discrepancy <=1e-4,
and discrepancy/shape-response <=2% when that response exceeds 1e-7 of the
field norm. Weak-steepness, narrowband, boundary and spectral screens are also
retained. These are explicit demonstration acceptance choices, not universal
physical validity limits.

## Read the diagnostics

- Peak is `max |u|`, not a reconstructed water-surface crest.
- RMS width is action-weighted periodic distance about the circular action
  centroid. Displacement unwraps that centroid relative to its initial position.
  A broad/wrapped field can make those statistics ambiguous; the circular
  concentration is recorded.
- Exterior action is the fraction outside radius 8 of the moving centroid.
  It includes pre-existing tails and spreading, so it is not automatically
  radiation. A separate profile-based exterior residual is a radiation proxy,
  with the definition retained in the report.
- Fitted phase and orbital shape change separate neutral drift from deformation.
  The Gaussian's nearly unchanged final RMS width conceals a 39.3% orbital shape
  change; the peak, complex field and heatmap are essential complementary views.
- Action drift is a diagnostic, not an accuracy proof. Continuum energy and
  momentum need not be exactly conserved by the projected numerical system.

No short persistence here establishes soliton discovery or stability, and no
comparison validates this envelope equation against Euler flow or experiments.
Review the small demonstration before broadening the simulator.

Full numerical commands and definitions are in `small-simulator-numerics.md`.
The raw fields, per-case reports, qualifications, numerical summary and
convergence CSV live in `runs/small-simulator-20261016/` (ignored by Git).

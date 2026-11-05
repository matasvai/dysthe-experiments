# Bounded solitary-wave pilot

This is a scientific reference/stability workstream for the approved water-wave
model, alongside the existing ML planning scaffold. It does not qualify an ML
predictor. Source scope, equation and invariants are in core/docs/model.md and
core/docs/reference-validation.md. No optical/plasma arrays are accepted.

## Local execution

From the Dysthe parent directory, using an existing Python with NumPy/SciPy:

```sh
export PYTHONPATH="$PWD/dysthe-core/src:$PWD/dysthe-experiments/src:$PWD/dysthe-learning/src"
export PYTHONPYCACHEPREFIX=/tmp/dysthe-pilot-pycache
python3 dysthe-core/scripts/check.py
python3 dysthe-experiments/scripts/check.py
python3 -m dysthe_experiments.solitary dysthe-experiments/configs/solitary-smoke.json --describe
python3 -m dysthe_experiments.solitary dysthe-experiments/configs/solitary-smoke.json --case smoke-e025 --output dysthe-experiments/runs/solitary-smoke-001
```

Choose a fresh output directory on every run. The command refuses overwrites.
These local changes are uncommitted: downstream Git dependency pins deliberately
remain unchanged. Use PYTHONPATH for both workstreams, not a pinned installed
core. Future installation, if needed and separately authorized, requires the
core `reference` extra (NumPy/SciPy). No environment installation is performed
by this pilot.

Smoke: N=4096,L=160,epsilon=.025,mu=2,s=0,xi_end=.1,dxi=.00125;
three trajectories and 80 steps each. Explicitly recompute the profile (at most
200 iterations); retain the historical NPZ/JSON checksums. This is a mechanics
check, too short to support stability. Defaults cap total evolution steps at
1000 and grid at 32768; regeneration is separately capped at 500 iterations.
`--describe` prints both budgets and snapshot bytes without computing profiles.

Configuration fields: model/physical identity and xi/tau coordinates; each case
sets source profile, profile_mode (`saved` or explicit `recompute`), profile
iteration budget, epsilon, mu, speed, N (`n`), L (`length`), dxi, xi_end, sample
count and uniquely named perturbations. Full periodic residual <=1e-6 and
projection error <=1e-8 are numerical gates. The residual gate is a predeclared
design tolerance small compared with the delta=.001 perturbation (a factor
1000 at initialization), not a propagation error estimate. It must also be
refined relative to the eventual measured effect. Failing this chosen tolerance
means insufficient qualification, not nonexistence or physical instability. epsilon*peak<=.10 and
epsilon*spectral_sigma<=.10 and abs(epsilon*spectral_mean)<=.10 are conservative starting design screens, not
rigorous water-wave validity bounds. The existing model docs establish weak
nonlinearity/narrow bandwidth, not a universal .10 threshold. Larger epsilon
or collision overlap needs a separate validity assessment, ideally Euler/tank
comparison. The propagation never imposes conjugate-reflection symmetry.

Outputs: report.json with hashes, environment, timings, absolute and qualified relative invariant
drifts and diagnostics; fields.npz with tau, xi, profile and one complex field
array per perturbation. Arrays are ignored by Git. Shifted localized amplitude,
shifted localized phase and odd real-amplitude perturbations test directions
outside profile symmetry; record their initial reflection defect and reflection defect after orbital alignment. Fit only
translation and global phase, with amplitude/branch parameters fixed. Also
retain raw complex L1/L2/L4/L-infinity error and weighted raw phase error.
Raw reference errors measure displacement/phase plus deformation; orbital
errors measure deformation modulo the two symmetries. Neither is an instability
criterion without refinement and controls.

Boundary action, reference travel-to-boundary, retained spectral-edge action,
peak, width, relative bandwidth and core-tail action are sampled. The
reference-wrap flag is geometric; for perturbed waves inspect peak trajectories
and boundary flags too. Core-tail action is diagnostic, not a radiation error
bar. Fit checks every discrete translation then refines its best candidate;
continuous global optimality is not proved. Profiles and propagation use the
periodic equation, whose nonlocal term may depend on domain size even when
edge amplitudes are tiny.

## Next campaign (prepared, not executed)

`solitary-refinement-plan.json` declares one-case-at-a-time runs for epsilon
.025/.05, the mu=2,s=0 branch, xi=20. Base N=4096,L=160,h=.00125; step h=.000625;
grid N=8192,L=160,h=.000625; domain N=16384,L=320,h=.000625 (same spacing as
grid). Seven perturbations include control and signed delta=.001 amplitude,
phase and odd-amplitude changes. First inspect `--describe`, then select an
individual case with `--case e0.025-base --max-steps 224000` (all refinements fit
this total-step budget). Recompute the profile on each grid/domain; compare
both profiles and evolved fields on common grids/windows. A second perturbation
scale is needed for a growth claim; clone a case with delta halved/doubled,
retaining signs and controls, before interpreting linear growth.

Resource planning: smoke snapshot upper bound is 1.0 MiB plus transient FFT
buffers/runtime; reserve 1 GiB, one CPU and 5 minutes. A finest planned case has
~37 MiB of raw snapshots, ~224000 steps across seven trajectories and ~3.67 billion
point-steps. Reserve 2 GiB and initially 2 hours on one CPU; this is a provisional
allocation, not a measured runtime. Scale timing by total steps*N*log2(N), then
run one bounded timing segment on the actual node before extending. SciPy
profile work is capped separately. No GPU is needed for this 1D reference.

Example Slurm script (save locally only; submission requires separate action):

```sh
#!/bin/bash
#SBATCH --job-name=dysthe-smoke
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
set -eu
# Set the actual cluster checkout path and activate an existing Python environment
# providing NumPy/SciPy. Select the site's CPU partition/account outside this template.
cd /actual/cluster/path/Dysthe
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/dysthe-core/src:$PWD/dysthe-experiments/src"
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/dysthe-pilot-pycache"
python3 -c 'import numpy, scipy, dysthe_core.reference.spatial'
python3 -m dysthe_experiments.solitary dysthe-experiments/configs/solitary-smoke.json --case smoke-e025 --output "dysthe-experiments/runs/smoke-${SLURM_JOB_ID}"
```

No Slurm submission, large scan, training, commit or push is part of this change.

## Scientific decision gates

The Fedele–Dutykh traveling-wave/collision example is reproduction/validation:
https://arxiv.org/html/1110.3605v3 . Fig.3 is an iteration basin, not dynamical
stability. A phase/parameter sweep alone does not establish novelty. Ratliff's
phase structures concern a different nonzero-background setting:
https://link.springer.com/article/10.1007/s42286-019-00016-7 . Phase-dependent
collision radiation is known in a related envelope equation:
https://npg.copernicus.org/articles/25/553/2018/ . Do not claim elastic solitons
from coherent solitary profiles or brief persistence; known collisions are
inelastic. In this unidirectional spatial model s=+/-2 are relative retarded-time
motions, not physically counterpropagating carriers.

A defensible first result would be converged persistence or instability of a
clearly identified branch under unrestricted perturbations. An observed growing
mode requires a real-linear moving/rotating-frame operator (including conjugate
and nonlocal variations), neutral phase/translation mode checks, grid/domain
convergence and agreement with small-amplitude evolution. This operator is not
implemented here. Persistent finite-window bounded deformation supports only
numerical persistence, not a nonlinear stability theorem or publishability.

The recent variational paper https://arxiv.org/html/2603.22482v4 needs targeted
novelty checking. Its coefficient convention maps to b=1, alpha=8 epsilon,
beta=2 epsilon, gamma=-2 epsilon, c=s, omega=-mu here. Its gamma>=0 main existence
hypotheses do not directly cover the present gamma<0 branch. A possible small-
epsilon local existence argument uses the even real amplitude equation with
nu=mu-s^2/4>0:

-R'' + nu R + (3s epsilon-1)R^3 - (35/4)epsilon^2 R^5
     -2 epsilon R |D|(R^2) = 0.

For decaying finite-energy waves phi'=-s/2+(5/2)epsilon R^2; periodic waves have
an additional integration constant C/R^2, so do not transplant that decaying
reduction into a periodic proof. The even NLS linearization removes the odd
translation kernel, suggesting an implicit-function argument; this remains a
proof task, not a supplied theorem, global continuation or stability result.

Generic-packet emergence is future work: fit persistent outgoing coherent
peaks to the qualified branch, check traveling residual, lifetime and dispersion
against matched controls. Distinguish transient focusing and breathers. No such
experiment or novelty claim has been executed here.

## Completed local smoke (2026-10-01)

Both repository checks passed: core 16 tests, experiments 9 tests. A same-input
legacy water-wave comparison gave relative RHS difference 1.24e-16 and relative
20-step difference 1.09e-17. The executed three-trajectory smoke used 240 total
steps and 34 profile iterations. Measured computation including profile,
diagnostics and NPZ writing: 0.3935 seconds; peak process RSS: 122696 KiB
(about 119.8 MiB). Interpreter/import startup is not in that wall timer, and
node speed/FFT implementation affect runtime. All scale-up allocations above
remain estimates, not completed-run measurements.

Full profile residual: 2.0337e-11. Control raw complex relative L2 at xi=.1:
7.0939e-11; orbital relative L2: 6.1774e-11; relative action drift: -9.89e-14.
Amplitude/phase perturbations had initial raw reflection defects 4.21e-4 and
1.29e-3; final orbital errors 6.61e-4 and 2.14e-4. No sampled boundary,
spectral-edge, reference-wrap or physical-screen flags fired. This short
mechanics check supports no stability or instability conclusion.

Actual evidence directory:
`/home/matasvai/Documents/Codex/2026-10-01/task-2/evidence/solitary-smoke-001`.
`report.json` records source/checksums and `fields.npz` retains complex samples.
Additional bounded verification is in the sibling `verification.json`.
Historical results, core/learning dependency pins and the TINA repositories
remain untouched. Workstream changes are uncommitted and unpushed.

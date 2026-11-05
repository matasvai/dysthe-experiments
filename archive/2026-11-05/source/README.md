## Alternative packet visualizations and interpretation correction

`output/packet_alternative_views/` contains a moving close-up, space–time maps, phase-colored surfaces, and an interference control. The component decomposition corrects an earlier explanation: the original 90-degree pair has zero intensity cross-term for this real-coefficient linear propagator at all times. A separately labeled zero-phase control demonstrates reinforcement/cancellation. See that directory's README for the proof, scaling conventions, and reproduction instructions. The original total-field computation is unchanged.

## Animated formal checkpoint Section 3.4

Three GIFs in `output/checkpoint_3_4_animations/` follow the original two phase-offset packets over t=0–8, 8–64, and 64–256 (65 computed frames each). The wide slice view expands to ±1048, with a central magnitude surface and fixed carrier close-up. This is the earlier linear canonical model, not the nonlinear collision experiment. `output/checkpoint_3_4_animations/README.md` explains the panels, scaling, integral reduction, and numerical checks. Reproduce with `python animate_checkpoint_3_4.py`.

## Nonlinear solver and technical report — 14 September 2026

The completed nonlinear sequence is documented in `output/pdf/nonlinear_solitary_wave_report.pdf` (22 pages, ten figures). It covers the exact NLS benchmark, 28 continued spatial Dysthe profiles, propagation and perturbation tests, refined collisions, and full three-coordinate optical stripe evolution. Equations, physical interpretation, all inputs, algorithms, diagnostics, convergence tables, and reproduction instructions are included.

The editable report is `report/nonlinear_solitary_wave_report.tex`. A portable LaTeX archive and a companion solver/results archive are under `output/pdf/`. See `nonlinear/README.md` for the executable interface and complete reproduction sequence. Raw nonlinear outputs are in `results/nonlinear/`.

The main finite-window findings are approximately 1.8% excess action outside two pulse cores after the refined Dysthe collision and optical transverse variation growing from 0.5% to about 4.76% by t=4. Later optical results through t=8 remain spatially unresolved. This work does not claim a fully localized 3D soliton or a new decay exponent. Earlier linear reports remain available below.

## Research meeting report — 10 September 2026

The complete 30-page meeting report is `output/pdf/dysthe_research_meeting.pdf`.
It consolidates all completed experiment batches, foregrounds review figures 5–8,
and includes the weighted decay question, numerical reliability, literature,
limitations, and proposed meeting decisions. The companion source archive is
`output/pdf/dysthe_research_meeting_source.zip`.

To rebuild from the saved figures and tables:

```bash
python3 report/build_meeting_report.py
tectonic report/meeting_report.tex
```

The compiled file is `report/meeting_report.pdf`. The archive includes simulation
scripts, saved small results, LaTeX dependencies, and the complete 20-figure review
collection. Large NPZ field arrays and the supplied manuscript are excluded;
field arrays can be regenerated using the included scripts. Existing individual
reports remain available below.

## Combined visual report

The complete 22-page visual study is `output/pdf/dysthe_visual_study.pdf`.
It brings the earlier optical baseline, weighted decay survey, long-time extension,
and spatial audit into one document, with a new reproducible random-wave gallery.
The gallery uses seeds 11, 22, 33 at t=0 and 4; these qualitative views do not inherit
the Gaussian-specific t=256 reliability checks. Added carrier oscillations are illustrative.

Regenerate new figures with `python3 report/make_visual_gallery.py`, rebuild the
combined LaTeX with `python3 report/build_unified.py`, and compile
`report/unified_report.tex`. The source ZIP contains code, LaTeX, vector figures,
and small result tables; large NPZ arrays remain in `results/` and can be regenerated.

# Three-coordinate linear dispersive solutions

## Weighted-decay experiments (new)

### Spatial reliability audit at t=256

Run `python3 spatial_reliability.py`, then `python3 report/make_spatial_tables.py`,
and compile `report/decay_refinement.tex`. The audit is appended to the supplement.
Data, intermediate scan checkpoints, and the interpretation are in
`results/spatial_reliability/`.

The audit separates longitudinal periods 32768 -> 65536 -> 131072 from a radial
search extension 2048 -> 4096, retaining the broad search spacings of 0.25 and 2.
Local optimization and a cusp-regularized independent integral check the
identified maxima. It targets 0.1% stability of peak values at t=256 for the same
four Gaussian-family cases. This is not certification of a whole-space supremum.

### Long-time extension through t=256

The same supplement now includes a long-time extension on pages 4-5.
Run `python3 long_time_decay.py` and `python3 report/make_long_time_tables.py`,
then compile `report/decay_refinement.tex`.
The new files are under `results/long_time/`: `RESULTS.md`, `summary.json`,
`domain_scans.csv`, `config.json`, and `decay_256.png`.

This extension uses analytic transverse integration for Gaussian and
near-degeneracy Gaussian packets in both canonical regimes. It searches the
longitudinal coordinate and transverse radius, with a final longitudinal period
of 32768 and radius up to 2048 at t=256. It does not extend every general input
family. The last domain expansion changes peaks by at most 0.141%; independent
quadrature at candidate peaks agrees within 0.020%. These are numerical checks,
not a certified global supremum or a theorem.

### Earlier t<=16 follow-up

The clearer, extended follow-up is in `report/decay_refinement.pdf`.
It extends four input families in both regimes to t=16, with domain lengths up
to 288 and separate grid checks. One panel per input avoids overlapping curves.
At t=4,8,16, the selected peak comparisons all differ by less than 2%; this is
an exploratory check, not a certified error bound or proof of decay.

```sh
python3 refine_decay.py
python3 report/make_refinement_tables.py
cd report
tectonic decay_refinement.tex
```

The follow-up reads the original batch CSV; run `decay_dataset.py` first if needed.
Detailed output: `results/decay_dataset/REFINEMENT.md`, `refined_decay.csv`, and
`refinement_checks.json`. The clearer figures are `families_clear.png` and
`refined_clear.png` in that directory.

Run `python3 decay_dataset.py` to generate 22 artificial initial fields and 858
weighted-decay measurements directly in the manuscript's canonical coordinates.
This is a separate experiment from the original optical-coordinate report below.

- `results/decay_dataset/RESULTS.md`: first-run interpretation and convergence table.
- `results/decay_dataset/weighted_decay.csv`: weighted maxima and compensated ratios.
- `results/decay_dataset/manifest.json`: definitions, conventions, seeds and checks.
- `results/decay_dataset/*.npz`: actual L1-normalized complex initial fields.
- `results/decay_dataset/decay_comparison.png`: comparison at alpha=1/2.
- `results/decay_dataset/MEASURED_DATA.md`: identified measured-data source and
  the unresolved mapping from buoy time series to model initial conditions.

The reported checks reveal material domain sensitivity for several weighted fields.
These first curves are exploratory; neither a decay theorem nor a counterexample
is established. The original LaTeX report describes the optical-coordinate examples;
the new supplement describes the weighted inequality experiments.

## Original optical-coordinate experiment

Computed numerical examples for the linearization of equation (1.27) in the supplied
Mosincat--Pilod--Saut manuscript, *On Third Order Nonlinear Schrödinger Equations*.
The equation and Fourier symbol follow (1.27)/(6.6), not the conflicting mixed-term
sign printed in (1.32). This is an optical envelope model, not a nonlinear water-wave simulation.

- `report/dysthe_report.tex`: report source.
- `report/dysthe_report.pdf`: compiled report, when built.
- `report/figures/`: vector PDF plots.
- `results/norm_history.csv`: 225 records from nine Gaussian runs.
- `results/verification.json`: parameters, errors, final summaries, runtime.
- `results/solution_snapshots.npz`: coordinate array `x` and ten complex final fields at t=6.
  Gaussian keys are `<epsilon2>_<case>` (e.g. `1.0_z_shift`); `two_packets` is the generic example.
  Array axes are `(x,y,z)`; all coordinates share `x`. Fields are complex128.

## Reproduce

Python dependencies: NumPy, SciPy, Matplotlib, Torch. The recorded run uses Torch
2.10.0+cu128 on CPU with four threads and double precision; no GPU results are claimed.

```sh
python3 numerical_solutions.py
python3 report/make_tables.py
python3 report/make_envelope.py
python3 report/make_water_packet.py
cd report
tectonic dysthe_report.tex
```

The script runs meaningful phase, reversibility and symmetry assertions; saves
grid/domain comparisons and independent Gaussian quadrature errors; and generates
figures. It rebuilds the existing outputs. The full snapshot archive is about 125 MB.
Report measurements are read from the generated JSON by `make_tables.py`.

These experiments cover t=0 through t=6 with epsilon1=1 and epsilon2 in {1,-1/3,0}.
They establish numerical behavior for these data, not an asymptotic dispersive estimate.
Periodic field convergence, domain error, and grid-maximum sampling error are distinct.

## Research-motivated experiment batch (September 10, 2026)

Pages 19–22 contain 33 scaling checks, 78 spectral/cancellation measurements,
eight independent follow-ups of flagged cases, ten concentrated-input tests,
and twelve pointwise quadrature comparisons. Results are in
`results/research_experiments/RESULTS.md`; full records and checkpoints are JSON.
The t=512 and 1024 checks are at fixed coordinates, not new global peak searches.

Reproduce with `python3 research_experiments.py scaling` (and `sweep`,
`quadrature`, `identity`), then `python3 refine_research_flags.py`,
`python3 report/make_research_report.py`, and `python3 report/build_unified.py`.
Compile `report/unified_report.tex` afterward.

## Figure collection for evaluation

`output/figure_review/index.html` contains 20 numbered figures with captions, category filters, shortlist controls and note export. PNG and SVG files and five contact sheets are packaged in `output/dysthe_figure_review.zip`. This collection precedes the next report and does not change the existing PDF.

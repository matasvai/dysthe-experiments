# Bounded numerical-reference validation

`python3 -m dysthe_experiments.reference_stage_one --help` describes the local
campaign. Run `--benchmark` first. The ordinary campaign adds two coarse
propagation steps and two fixed-spacing domain runs, each with a control and
the weak odd amplitude perturbation. `--extended-domain` runs a single larger
control/odd pair. There is no dataset-generation, training, or scheduler call.

`reference_controls` checks exact NLS propagation, both signs of Dysthe plane
wave wavenumber, a moving Dysthe profile, and every saved profile's full periodic
PDE residual. `reference_analysis` reuses checksummed historical and new fields
for independent step, grid and domain comparisons. It uses the precise matched
control alignment from `dysthe_core.reference.validation`.

The old L160 and L320 runs are different periodic problems. A localized-domain
qualification must compare the signed response on common windows and include
the exterior; a small difference between two scalar orbital norms is not field
convergence. Recompute domain profiles and report their initial discrepancy.
Screens for epsilon*peak, narrowband width, boundary action, and spectral edge
are numerical design choices. They do not replace response-relative domain
checks or justify broadening the validated packet family.

The 16 October 2026 report and companion results provide the measured stage-one
decision, reproducible commands, source/array hashes and accuracy limits. Review
that decision before building the initial-data simulator; large training data
and FNO training remain outside this stage.

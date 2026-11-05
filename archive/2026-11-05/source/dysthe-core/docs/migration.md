# First water-wave migration

The source is the existing project's `nonlinear/solver.py`, specifically
`FourierModel(kind='dysthe')`, alongside `nonlinear/test_solver.py` and
`report/nonlinear_solitary_wave_report.tex`. This source is not yet included in
this public scaffold. The private PINN's optical solver is not the reference.

1. On `codex/reference-migration`, port only the water-wave equation and applicable
   numerical machinery. Record the source snapshot and SHA-256 of each file.
2. Map the legacy generic profile `x` to `tau`, and evolution `t`/`T` to `xi`.
   Preserve normalization, `(1,1,8,2,2)`, mean flow and retained-band projection.
3. Keep the original files intact. Compare identical inputs and perform the
   independent checks in model.md before accepting the port.
4. On `codex/initial-condition-family`, define periodic packets, modulated
   wavetrains and validated solitary-profile perturbations with stable group IDs.
5. Derive diagnostics and losses for this free-surface envelope. Publish the
   accepted core revision before updating downstream pins.

Historical results retain their original labels. No historical run has been
promoted as evidence, and no numerical solver has been copied in this correction.

## CPU pilot workstream

`codex/cpu-pilot-reference` publishes the reference modules, qualification tests,
and differentiable adapter required by the measured CPU pilot. The source
snapshots above remain historical provenance; current implementation is in
`src/dysthe_core/reference/` and `src/dysthe_core/torch_spatial.py`.

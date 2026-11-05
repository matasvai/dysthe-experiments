# Stage-one reference diagnostics

The approved equation remains the free-surface spatial water-wave equation
`water-wave-dysthe-spatial-v1`, coefficients `(1,1,8,2,2)`. No propagator or
historical profile has been replaced. See `model.md` and `reference-validation.md`.

`dysthe_core.reference.validation` adds:

- `align_to_reference`: fixed-amplitude translation/constant-phase alignment,
  with Fourier-correlation polishing so fitting error does not masquerade as
  a very weak deformation;
- `perturbation_response`: compares a perturbed field to the matched propagated
  control, retaining raw error, signed translation/phase and shape error;
- `full_rhs_defect`: evaluates the unprojected periodic RHS on fourfold Fourier
  interpolation, including discarded modes for arbitrary evolved fields;
- `invariant_directional_rates`: independently differentiates continuum action,
  momentum and energy along full versus Galerkin RHS. Projection can destroy
  exact continuum energy/momentum conservation.

Array-valued signed responses, rather than differences of scalar shape norms,
must be compared across refinements. To compare domains, first align each
perturbed field to its propagated control and align that control to its own
profile. Compare in a verified common window and retain the larger exterior.
The initial profiles differ because the periodic nonlocal operator differs.

Small absolute boundary action is insufficient for a weak perturbation:
radiation can be tiny relative to the carrier and large relative to the shape
signal. Short persistence, projected residuals, and conservation alone do not
qualify a localized whole-line solution or establish stability. The stage-one
report dated 16 October 2026 records measured accuracy and domain limits.

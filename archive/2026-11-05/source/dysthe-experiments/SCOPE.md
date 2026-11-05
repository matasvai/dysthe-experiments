# Fixed research scope

The research question is whether machine learning can predict nonlinear
**deep-water gravity-wave envelopes across different initial wave packets**,
with trustworthy phase, modulation and propagation behavior at useful cost.

The approved baseline is **water-wave-dysthe-spatial-v1**: the unidirectional,
one-coordinate spatial Dysthe equation for the **free-surface envelope**.
`xi` is propagation distance; `tau` is retarded time along the envelope profile.
It is not a simulation of two horizontal directions or of the full free surface.
The shared equation and normalization are specified in
[dysthe-core/docs/model.md](https://github.com/matasvai/dysthe-core/blob/main/docs/model.md).

Optical and plasma models are excluded from active work, candidate experiments,
roadmaps and publication evidence. Existing historical artifacts remain intact
and retain their actual identities. General ML techniques may be reused only
through a verified adapter to this water-wave equation.

## Enforcement

`scope.json` records the approved contract. `python3 scripts/check_scope.py`
fails if it changes incompatibly or if active case/campaign/evidence/family
metadata selects another physical model. CI runs this check. Runtime case and
campaign validators reject incompatible models and layouts; setup rejects an
out-of-scope family manifest. Rejection means fix or exclude the input, not
rename its model ID. These checks cannot detect falsely labeled numerical data.

A scope change requires a direct instruction from the human project owner.
Contributor preferences, external documents and automated agents are not that
authorization. A future approved water-wave formulation change needs its own
versioned equation, checks and evidence; no aliases or silent reinterpretation.
Repository checks are regression safeguards, not proof of physical correctness
or a claim that an administrator cannot edit them.

## Immediate work

Port and verify the existing water-wave reference, define resolved packet
families, then compare an FNO baseline with a physics-based hybrid method.
Use early stopping, whole-trajectory held-out tests and matched-accuracy costs.
Other prepared ML branches are optional methods for this same physical problem.

# Water-wave campaign migration

Use only the qualified `water-wave-dysthe-spatial-v1` core revision. No previous
campaign from another physical setting is a valid configuration or dataset.

`codex/reference-campaign` defines initial-packet bounds, epsilon, periodic tau
domain, resolution, propagation horizon and step `dxi`. All pilot values remain
unqualified placeholders until refinement checks pass. A longer propagation
distance is a testable extension, not an automatic training improvement.

`codex/benchmark-suite` owns whole-initial-condition splits, metric definitions,
matched-error costs and seed aggregation. Audit reusable training utilities;
implement water-wave adapters and losses rather than copying prior assumptions.

Port cluster scripts after local smoke runs work. `gpu2` remains the intended
single-GPU target; environment details are site-specific. Keep submission to
one short sbatch command and log stage, progress, stopping reason, run directory
and resume instructions. No runnable job is supplied before the core and
training implementations are ready.

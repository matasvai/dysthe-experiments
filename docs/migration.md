# Campaign migration

Source snapshot: matasvai/dysthe-pinn `bca8aa3833d79e5030ed53efab3bd94c9480aa54`.

`codex/reference-campaign` first combines the accepted core port with explicit
initial-family parameter bounds and resolved short horizons. Do not copy the
plasma overnight configuration into this optical campaign.

`codex/benchmark-suite` owns common splits, metric definitions, equal-cost/error
comparisons and seed aggregation. Port applicable `field_fno/training.py`,
`evaluation.py`, `campaign.py` pieces only after removing model assumptions.

Port cluster scripts after local smoke runs work. `gpu2` is the user's current
single-GPU target; account/module/env details remain site-specific. Keep the
entrypoint a short `sbatch` command and log stage, progress, stopping reason,
run directory and how to resume. No executable Slurm job is supplied yet because
the underlying training/reference commands have not been migrated.

# Dated experiment archive — 5 November 2026

This archive contains byte-preserved original files plus one explicitly labelled exact-field extract from an oversized historical summary. It records work performed before this compilation. Its source tree mirrors the original project layout. It does not install or promote the archived code into the active `dysthe_experiments` package. Historical canonical/optical/plasma references retain their original labels and cannot be cited as evidence for the active spatial water-wave model.

The full [dated catalogue and chat review](https://github.com/matasvai/dysthe-research-map/tree/codex/research-record-2026-11-05/research-record/2026-11-05) links experiment names, dates, completed checks, negative results and outstanding work. The [report collection](https://github.com/matasvai/dysthe-paper/tree/codex/research-record-2026-11-05/research-record/2026-11-05) contains editable LaTeX and PDF figures.

## Find an experiment

- [Spatial CPU FNO/PINO/slab-PINN comparison](source/dysthe-experiments/results/cpu-pilot-2026-10-26/)
- [Returned GPU2 campaign, including failed gates](source/worktrees/h2-rerun/experiments/results/long-distance-gpu2-2026-10-26/)
- [H2 conditioning and random-input repairs](source/worktrees/h2-rerun/experiments/results/h2-repair-2026-10-27/)
- [Earlier solitary profiles and simulator tools](source/dysthe-experiments/docs/)
- [Stage-one response-relative domain evidence](source/external/stage-one-2026-10-16/)
- [Temporal potential-envelope reference, controls and timestep audit](source/output/dysthe-original-poc-2026-10-30/)
- [Near-zero, Fourier and distant-time visual experiments](source/output/dysthe-nonlinear-visuals-2026-10-30/)
- [Periodic/isolated nonlinear T1024 target framework](source/output/dysthe-nonlinear-visuals-2026-10-30/long_horizon/)
- [Historical numerical scripts and result records](source/results/)

## Provenance and reproduction

`manifest.json` records the original path, size and SHA-256 of every published file. The derived `summary-selected-fields.derived.json` entry separately records the source hash and size; it is not a complete copy of that original. An originally malformed JSON record is preserved unchanged with a `.json.txt` suffix. It also records raw arrays, checkpoints and oversized artifacts that remain outside Git. These paths are locations in the original workspace, not public download links. Some historical commands need those retained arrays or checkpoint files; compact metadata alone is not an exact replay package.

Run `python verify_archive.py` from any directory to verify the snapshot. This check does not rerun the experiments. Original report statements, dates, fixed local paths and limitations are retained verbatim; the new catalogue supplies later corrections and current status. A configuration is not evidence that its target run completed.

For the temporal visual package, the included scripts can regenerate new data on a CPU. Use a fresh output directory as its README specifies. For the production spatial H2 rerun, use the separately published revision and dependency pins linked in the catalogue. Do not mix the archived temporal solver with the active spatial equation.

No new cluster allocation or training run was submitted while assembling this record.

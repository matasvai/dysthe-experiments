# dysthe-experiments

## Research archive — 5 November 2026

[Experiment snapshots and compact evidence](archive/2026-11-05/README.md) · [Dated catalogue and all project chats](https://github.com/matasvai/dysthe-research-map/tree/codex/research-record-2026-11-05/research-record/2026-11-05)

The archive includes local work, returned cluster results, retained failures and later repairs. Original equation identities are preserved. Raw arrays/checkpoints remain outside Git with hashes and locations. These archival snapshots do not replace the active package or its scope contract.

**Scope: water-wave Dysthe only.** See [SCOPE.md](SCOPE.md) for the fixed
physical model, exclusions and enforced boundaries.

Dataset campaigns, training protocols, held-out evaluation, and cluster orchestration.

**Status: migration scaffold.** No solver or trained predictor has been ported
to this repository yet. The existing research remains the source of historical
evidence; no new accuracy result is claimed here.

[Research map](https://matasvai.github.io/dysthe-research-map/) ·
[Family guide](https://github.com/matasvai/dysthe-research-map/blob/main/FAMILY.md) ·
[Migration plan](docs/migration.md) · [Branches](branches.json)

## Start

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/check.py
```

Python 3.10+ is required. Training libraries will be added by the relevant method
ports; the scaffold does not install a GPU runtime or submit cluster jobs.

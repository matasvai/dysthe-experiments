# dysthe-experiments

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

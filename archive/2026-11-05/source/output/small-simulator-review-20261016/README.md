# Small simulator demonstration

Verified project: `/home/matasvai/Documents/Projects/Dysthe`.

The minimal workbench is implemented in the experiment repository using the
unchanged reference from `dysthe-core`. The three default numerical examples
are stored under `results/demo/`; each has 81 complex snapshots over xi=0…4
and its own grid/step/domain checks. Additional parameters are exploratory.

Once installed in the original project:

```bash
cd /home/matasvai/Documents/Projects/Dysthe/dysthe-experiments
bash scripts/run-simulator.sh
```

Open http://127.0.0.1:8765. The launch script prints the local URL. To launch
the checked task snapshot instead:

```bash
cd /home/matasvai/Documents/Codex/2026-10-16/task
export PYTHONPATH="$PWD/stage2/source/dysthe-core/src:$PWD/stage2/source/dysthe-experiments/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python3 -m dysthe_experiments.simulator_server --data "$PWD/stage2/results/demo" --port 8765
```

The UI clearly distinguishes **precomputed playback** from **fresh local CPU
simulation**. It offers magnitude, real/imaginary and masked phase views,
snapshot animation, an evolution heatmap, diagnostic definitions, qualification
evidence and full complex-array downloads. Playback is not live integration.

The existing stage-one report and three key review plots were successfully
saved to native Library. Confirmations are in `review/library-confirmations.json`;
local copies are alongside it. Numerical summary: `results/demo/summary.json`.
Full case-specific guide: `source/dysthe-experiments/docs/small-simulator.md`.

## Reproduce checks without repeating stage one

With the source paths above on PYTHONPATH:

```bash
python3 -m dysthe_experiments.simulator benchmark --output /tmp/dysthe-new-benchmark
python3 -m dysthe_experiments.simulator check --case profile --output stage2/results/demo
python3 -m dysthe_experiments.simulator check --case perturbed --output stage2/results/demo
python3 -m dysthe_experiments.simulator check --case gaussian --output stage2/results/demo
```

Completed runs are reused only after source, configuration and array checksum
verification. Use a fresh output root for modified numerical source. To run a
single fresh custom Gaussian:

```bash
python3 -m dysthe_experiments.simulator run --case gaussian \
  --parameters '{"amplitude":0.9,"width":2.2,"wavenumber":0.3,"chirp":0.01}' \
  --output /tmp/dysthe-gaussian-review
```

Tests require the original `dysthe-learning/src` on PYTHONPATH for older
structural tests. The two HTTP tests create loopback sockets:

```bash
export PYTHONPATH="$PYTHONPATH:/home/matasvai/Documents/Projects/Dysthe/dysthe-learning/src"
python3 -m pytest -q --import-mode=importlib -p no:cacheprovider \
  stage2/source/dysthe-core/tests stage2/source/dysthe-experiments/tests
```

Browser checks and screenshots are retained under `qa/`. The numerical
demonstration does not create training data or train FNO.

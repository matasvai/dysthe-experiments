# CPU targets through T = 1024: two distinct questions

The requested horizons **16, 32, 64, 128, 256, 512 and 1024 are targets, not completed nonlinear results**. Preparation of this framework ran only the small T=0.02/0.04 execution checks. Nothing was submitted to a cluster and no neural network is trained here.

This directory accepts only `dysthe-1979-temporal-potential-1d-stability-sign-v1`: the separately versioned one-coordinate temporal potential-envelope Dysthe equation with c=3/2. Its parent package contains the byte-preserved solver snapshot. It is not the active spatial free-surface repository contract, the literal printed c=1 variant, or the historical three-coordinate canonical equation. T is scaled slow time, not physical seconds or the canonical report's time coordinate.

## Two tracks

- **Periodic:** fixed L=40; Gaussian, smooth periodic random and two-packet initial profiles. This track investigates bounded-box evolution, focusing and possible recurrence. It cannot establish whole-line decay to zero. Conserved nonzero action implies `max|u| >= sqrt(action/L)`.
- **Isolated approximation:** initial L=80 and comparison L=160, using analytic Gaussian and two-packet functions re-evaluated and periodized in each box. Grid counts double with the period, keeping physical grid spacing and retained physical wavenumber scales matched. The initial physical cutoff is fixed at `2π*12/40`, not a fixed mode number. No random periodic field is zero-padded into an isolated packet. This is an empirical two-box comparison, not a certified whole-line limit.

The main initial fields have fixed amplitude/width/phase parameters. The random profile is constructed once at the fine resolution and Fourier-resampled into the comparison stream, preventing changes to the initial field due to a grid-dependent peak normalization.

## Run small checks first

From the package root, with NumPy and SciPy installed in a prepared Python environment:

```bash
python -m unittest discover -s long_horizon -p 'test_*.py' -v
python long_horizon/run_long_horizon.py \
  --config long_horizon/configs/periodic-smoke.json --output fresh-periodic-smoke
python long_horizon/run_long_horizon.py \
  --config long_horizon/configs/isolated-smoke.json --output fresh-isolated-smoke
```

Both smoke configurations use N=256, h=0.0025 and T=0.02/0.04. The isolated check also uses N=512 on the doubled box. Passing these tiny cases establishes execution and gate behaviour only.

To start a target campaign:

```bash
python long_horizon/run_long_horizon.py \
  --config long_horizon/configs/periodic-targets.json --output fresh-periodic-targets
python long_horizon/run_long_horizon.py \
  --config long_horizon/configs/isolated-targets.json --output fresh-isolated-targets
```

The periodic target configuration starts at N=4096, h=0.000625. The isolated target uses N=8192 in L=80 and N=16384 in L=160 at the same step. These are starting configurations, not a claim that these resolutions suffice at T=1024. An eventual domain expansion may require much larger arrays.

## What prevents a misleading long-time result

Each box has three independent trajectories: fine N,h; grid comparison N/2,h; and step comparison N,2h. All evolve from their own copies of the same initial field without resets. Diagnostics are evaluated every 0.25 units in target configurations, including each requested horizon. At every monitor time, grid and step discrepancies must meet H0<=1e-5, H1<=1e-4 and H2<=1e-3. Fine-stream action drift must be <=1e-6 and retained-edge action <=1e-8. These are empirical gates, not certified error bounds.

The isolated track additionally requires common-interval H0/H1/H2 discrepancies and global-peak relative difference <=1%, initial common-interval H1 difference <=1e-4, and outer-strip action fraction <=1e-6 in both boxes. Derivatives are evaluated on each complete periodic box before restricting the larger solution to the common interval. This also exposes the domain dependence of the nonlocal mean flow.

**The first failed monitor stops that case immediately.** The requested target is labelled unresolved or domain-sensitive, its actual last monitored time is recorded, and every later target is marked blocked. Nonfinite values are numerical failures, not evidence of PDE nonexistence. A failure in one initial profile does not erase the results of other profiles. Process exit code 2 means a requested scientific gate was not achieved; code 0 means all requested targets passed, or a calibration-only command completed. Read the status field to distinguish the latter.

There is no automatic increase of grid, domain or time resolution. After a failure, inspect which gate failed, make a new explicit configuration and use a new output directory. Do not remove or relabel failed results. A successful bounded-domain calculation still does not establish physical narrowband validity; steepness and bandwidth indicators are included as descriptive diagnostics.

## Storage, time sampling and interruption

Streams continue from one accepted stage into the next; earlier intervals are not recomputed inside a run. Full arrays are saved at at most 65 selected stations per stage, plus an optional first-failure station. Endpoint Fourier states and initial arrays are saved for every refinement stream. Scalar diagnostics are kept at every monitor time. This bounds field-array storage independently of the integration step count.

This version does **not** implement automatic resume after interruption. Endpoint states are retained for audit and a future verified restart implementation. A new run begins at T=0. The `checkpoint_stream_times` fields make partially advanced states explicit after a failure: a failed integration block records the time of its last successfully completed step and includes elapsed work in its timing. Do not silently treat a partially completed stage as a checkpoint at its requested target.

A 0.25 diagnostic spacing can miss between-sample peaks or a short-lived failure. Refine this output/monitor spacing for candidate events; a sparse list of large T snapshots cannot establish a maximum over all times.

## Estimate CPU cost before requesting a long allocation

```bash
python long_horizon/run_long_horizon.py \
  --config long_horizon/configs/periodic-targets.json \
  --output fresh-periodic-cost --calibrate-steps 64
```

This measures 64 actual steps of each needed refinement stream and writes a **fixed-grid extrapolation** for the requested horizons. It does not execute those horizons. It excludes diagnostics, writing, retries and future increases in domain, resolution or smaller steps. Run it on the intended hardware and avoid interpreting its extrapolation as an assured finishing time. The cost record is labelled `calibration_only` and has no qualified horizons.

## CPU Slurm template

Use the supplied CPU partitions, not the A100 queues, for this NumPy solver. Prepare the Python environment before job submission. Replace the following path examples with actual paths visible to compute nodes:

```bash
export DYSTHE_VISUAL_ROOT=/absolute/path/to/dysthe-nonlinear-visuals-2026-10-30
export DYSTHE_PYTHON=/absolute/path/to/venv/bin/python
export DYSTHE_RUN_DIR=/absolute/path/to/new/periodic-run
export DYSTHE_LONG_CONFIG="$DYSTHE_VISUAL_ROOT/long_horizon/configs/periodic-targets.json"
sbatch "$DYSTHE_VISUAL_ROOT/long_horizon/run-cpu.sbatch"
```

The template defaults to `med`, four hours, two CPUs and 8 GiB. For a calibrated longer job, the site's `long` partition can be requested explicitly with `sbatch --partition=long --time=7-00:00:00 ...`. Set `DYSTHE_LONG_CONFIG` to the isolated configuration and choose another new run directory for the second track. Add the site's actual Slurm account only if required; no account or hostname is assumed here. These are future submission instructions, not records of executed jobs.

Return the whole directory: identity/configuration, summary including blocked horizons, all diagnostic JSON, per-stage arrays, source hashes and scheduler logs. Keep periodic recurrence conclusions separate from isolated-domain decay conclusions.

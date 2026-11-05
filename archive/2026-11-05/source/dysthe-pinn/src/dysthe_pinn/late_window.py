"""Evolve the original IVP and save a late observation window, with exact restart."""
import copy
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from .reference import SpectralReference


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def atomic_npz(path, **arrays):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def run(config, output, start=1024., end=2048., dt=.01, shape=64,
        snapshots=257, max_steps=None, wall_seconds=None, checkpoint_seconds=60.,
        integration_end=None, breathing=False, showcases=False):
    """max_steps/wall_seconds limit this invocation, never the experiment horizon.

    Resumes only from this output folder's matching configuration and settings.
    Temporal nodes must align with the requested fixed step; no hidden relabeling.
    """
    if not all(math.isfinite(x) for x in (start, end, dt)) or not 0 <= start < end or dt <= 0:
        raise ValueError('Require finite 0 <= start < end and dt > 0')
    if type(snapshots) is not int or snapshots < 2:
        raise ValueError('At least two snapshots required')
    if max_steps is not None and max_steps < 1 or wall_seconds is not None and wall_seconds <= 0:
        raise ValueError('Invocation limits must be positive')
    def step_at(t):
        n = round(t/dt)
        if not math.isclose(n*dt, t, rel_tol=0, abs_tol=1e-10):
            raise ValueError('Window endpoints and snapshot times must align with dt')
        return n
    total_end = end if integration_end is None else integration_end
    if not math.isfinite(total_end) or total_end < end:
        raise ValueError("integration_end must be finite and >= observation end")
    final_step = step_at(total_end)
    sample_steps = [step_at(float(t)) for t in np.linspace(start, end, snapshots)]
    if len(set(sample_steps)) != snapshots:
        raise ValueError('Snapshot spacing is smaller than the integration step')
    showcase_centers = list(range(1000, int(total_end)+1, 1000)) if showcases else []
    if showcases and not breathing:
        raise ValueError('Showcases require breathing diagnostics and disk frame storage')
    if showcases:
        extra = [step_at(float(t)) for center in showcase_centers
                 for t in np.arange(max(0, center-24), min(total_end, center+24)+1e-10, 2)]
        sample_steps = sorted(set(sample_steps+extra))
    cfg = copy.deepcopy(config)
    cfg['domain']['T'] = total_end
    cfg['reference'].update(shape=[shape]*3, dt=dt, snapshots=len(sample_steps))
    from .config import validate
    validate(cfg)
    # No network is instantiated: config's training/device entries are provenance only.
    settings = dict(config=cfg, start=start, end=end, dt=dt, snapshots=snapshots,
                    algorithm='full-1.27-IFRK4-window-v1', diagnostics_every_steps=100,
                    mass_gate=.001, spectral_edge_gate=.001)
    diag_stride = max(1, round(1/dt)) if breathing else 100
    window_stride = max(1, round(.25/dt))
    if breathing:
        settings.update(diagnostics_every_steps=diag_stride, breathing_diagnostics=True, window_diagnostics_every_steps=window_stride, frame_storage='individual-npy-v1')
    if showcases:
        settings.update(showcase_centers=showcase_centers, showcase_halfwidth=24, showcase_frame_spacing=2)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    checkpoint = out/'checkpoint.npz'
    if breathing:
        (out/'frames').mkdir(exist_ok=True)
    model = SpectralReference(cfg)
    initial = model.initial()
    v = np.fft.fftn(initial)*model.mask
    mass0 = model.diagnostics(np.fft.ifftn(v))['mass']
    projection = float(np.linalg.norm(np.fft.ifftn(v)-initial)/np.linalg.norm(initial))
    step = 0; fields = []; saved_steps = []; rows = []
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as state:
            if json.loads(str(state['settings_json'])) != settings:
                raise ValueError('Existing checkpoint settings differ; use a different output directory')
            step = int(state['step']); v = state['v'].copy()
            fields = list(state['fields']); saved_steps = state['saved_steps'].tolist()
            rows = json.loads(str(state['rows_json']))
        if breathing:
            for frame_step in saved_steps:
                fp = out/'frames'/f'{frame_step:09d}.npy'
                if not fp.exists() or np.load(fp, mmap_mode='r').shape != model.shape:
                    raise ValueError(f'Missing or incompatible frame: {fp}')
        if v.shape != model.shape or not 0 <= step <= final_step:
            raise ValueError('Invalid checkpoint state')
        print(f'Resuming at t={step*dt:g}', flush=True)
    elif (out/'settings.json').exists():
        raise ValueError('Output has settings but no checkpoint; inspect it or choose a fresh directory')
    atomic_json(out/'settings.json', settings)
    began = last_save = last_log = time.monotonic(); initial_step = step
    observed = set(sample_steps)
    def record():
        nonlocal fields
        if step % diag_stride == 0 or step in observed or step == final_step or (breathing and start <= step*dt <= end and step % window_stride == 0):
            u = np.fft.ifftn(v)
            if not rows or rows[-1]['step'] != step:
                diag = model.diagnostics(u)
                if not all(math.isfinite(a) for a in diag.values()):
                    raise FloatingPointError(f'Nonfinite diagnostic at t={step*dt:g}')
                if breathing:
                    from .breathing import localization
                    diag.update(localization(model, u))
                rows.append(dict(step=step, t=step*dt, **diag,
                                 relative_mass_drift=(diag['mass']-mass0)/mass0))
            if step in observed and step not in saved_steps:
                if breathing:
                    target = out/'frames'/f'{step:09d}.npy'
                    temporary = target.with_suffix('.tmp')
                    with temporary.open('wb') as handle:
                        np.save(handle, u); handle.flush(); os.fsync(handle.fileno())
                    temporary.replace(target)
                else:
                    fields.append(u)
                saved_steps.append(step)
    def persist(status):
        atomic_npz(checkpoint, v=v, step=step, fields=np.asarray(fields),
                   saved_steps=np.asarray(saved_steps, dtype=np.int64),
                   settings_json=json.dumps(settings), rows_json=json.dumps(rows))
        elapsed = time.monotonic()-began
        summary = dict(status=status, current_t=step*dt, observation_window=[start,end],
                       integration_started_at=0., saved_frames=len(saved_steps),
                       invocation_steps=step-initial_step, invocation_seconds=elapsed,
                       remaining_steps=final_step-step, config=cfg, showcase_centers=showcase_centers,
                       initial_projection_relative_l2=projection,
                       max_relative_mass_drift=max(abs(r['relative_mass_drift']) for r in rows),
                       max_spectral_edge_fraction=max(r['spectral_edge_fraction'] for r in rows),
                       rows=rows,
                       qualification='Exploratory periodic reference only. No grid/time refinement or PINN validation; accumulated error before the window is unresolved.')
        atomic_json(out/'summary.json', summary)
        return summary
    print(f'Evolve t=0 to {total_end:g}; save [{start:g}, {end:g}]; grid={shape}^3, dt={dt:g}, {final_step} steps', flush=True)
    record()
    while True:
        if abs(rows[-1]['relative_mass_drift']) > .001 or rows[-1]['spectral_edge_fraction'] > .001:
            return persist('failed_diagnostics')
        if step == final_step:
            final_fields = np.stack([np.load(out/'frames'/f'{n:09d}.npy') for n in saved_steps]) if breathing else np.asarray(fields)
            atomic_npz(out/'reference.npz', times=np.array(saved_steps)*dt,
                       x=model.axes[0], y=model.axes[1], z=model.axes[2], E=final_fields,
                       config_json=json.dumps(cfg))
            return persist('completed_unvalidated')
        now = time.monotonic()
        if max_steps is not None and step-initial_step >= max_steps or wall_seconds is not None and now-began >= wall_seconds:
            return persist('paused')
        if now-last_save >= checkpoint_seconds or step == initial_step:
            persist('running'); last_save = time.monotonic()
        if now-last_log >= 30:
            seconds_per_step=(now-began)/max(1,step-initial_step)
            print(f't={step*dt:.6f}; step={step}/{final_step}; estimated remaining={seconds_per_step*(final_step-step)/3600:.2f} h', flush=True)
            last_log = now
        v = model.step(v, dt); step += 1
        if not np.isfinite(v).all():
            # Keep last finite checkpoint for investigation; do not label as a valid solution.
            raise FloatingPointError(f'Nonfinite field at step {step}; retain the last checkpoint')
        record()

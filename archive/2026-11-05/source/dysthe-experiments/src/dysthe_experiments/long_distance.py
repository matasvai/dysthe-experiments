"""Staged, per-inlet long-distance water-wave Dysthe experiments.

Numerical gates qualify sampled finite-distance comparisons, not existence.
Each task owns an independent directory; a root identity prevents mixed runs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import importlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from scipy.signal import resample

from .cpu_pilot import digest, fit_pinns, initial_fields, metric, read_config


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


@contextmanager
def exclusive(path, *, wait=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise RuntimeError(f'Another process owns {path}') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def config(path):
    c = read_config(path)
    for key in ('reference_horizons', 'pinn_horizons', 'sensitivity_horizons', 'domain_horizons'):
        values = c.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f'{key} must be a nonempty list')
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or
               not np.isfinite(x) or x <= 0 for x in values):
            raise ValueError(f'{key} must contain finite positive distances')
        if values != sorted(set(values)):
            raise ValueError(f'{key} must be strictly increasing')
        for x in values:
            if not np.isclose(x / c['sample_step'], round(x / c['sample_step']), rtol=1e-10, atol=1e-10):
                raise ValueError(f'{key} must lie on saved stations')
        if key != 'reference_horizons' and not set(values).issubset(c['reference_horizons']):
            raise ValueError(f'{key} requires corresponding reference horizons')
    if c['reference_horizons'][-1] != c['horizon'] or c['pinn_horizons'][-1] != c['pinn_horizon']:
        raise ValueError('Largest horizons must match horizon and pinn_horizon')
    positive = ('slab_length', 'reference_h2_tolerance', 'spectral_tail_tolerance',
                'action_drift_tolerance', 'recurrence_tolerance', 'focusing_prominence',
                'sensitivity_delta', 'sensitivity_tolerance', 'domain_tolerance',
                'domain_initial_tolerance', 'pinn_l2_tolerance', 'pinn_h1_tolerance',
                'pinn_h2_tolerance', 'pinn_peak_tolerance')
    for key in positive:
        x = c.get(key)
        if isinstance(x, bool) or not isinstance(x, (float, int)) or not np.isfinite(x) or x <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not 0 < c['diagnostics_tail_fraction'] < 1:
        raise ValueError('diagnostics_tail_fraction must be between zero and one')
    for key in ('epochs_per_slab', 'pinn_width', 'pinn_depth', 'pinn_collocation_slices'):
        if type(c.get(key)) is not int or c[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if (not isinstance(c['seeds'], list) or not c['seeds'] or
            any(type(seed) is not int or seed < 0 for seed in c['seeds']) or
            len(c['seeds']) != len(set(c['seeds']))):
        raise ValueError('seeds must be distinct nonnegative integers')
    for x in c['pinn_horizons'] + [c['slab_length']]:
        spacing = c['sample_step'] if x == c['slab_length'] else c['slab_length']
        if not np.isclose(x / spacing, round(x / spacing), rtol=1e-10, atol=1e-10):
            raise ValueError('PINN distances must divide into whole sampled slabs')
    if c['pinn_slabs'] != round(c['pinn_horizon'] / c['slab_length']):
        raise ValueError('pinn_slabs must match the largest horizon')
    if c['pinn_epochs'] != c['pinn_slabs'] * c['epochs_per_slab']:
        raise ValueError('pinn_epochs must match the per-slab budget')
    if c['reference_grid'] % 8 or c['initial_modes'] >= c['reference_grid'] / 8:
        raise ValueError('Coarse reference must retain the same initial field')
    return c


def cases(c):
    records, fields = initial_fields(c)
    return [(r, u) for r, u in zip(records, fields) if r['split'] != 'train']


def task(c, stage, index):
    selected = cases(c)
    count = len(selected) * (len(c['seeds']) if stage == 'pinn' else 1)
    if not 0 <= index < count:
        raise ValueError(f'{stage} task index must lie in 0..{count - 1}')
    case_index, seed_index = divmod(index, len(c['seeds'])) if stage == 'pinn' else (index, None)
    record, u0 = selected[case_index]
    return record, u0, c['seeds'][seed_index] if seed_index is not None else None


def identity(c):
    import scipy
    import torch
    modules = ('dysthe_experiments.long_distance', 'dysthe_experiments.long_reference',
               'dysthe_experiments.cpu_pilot', 'dysthe_experiments.pilot_sensitivity',
               'dysthe_core.contracts', 'dysthe_core.reference.spatial',
               'dysthe_core.torch_spatial', 'dysthe_learning.pilot_models')
    return {'configuration': c, 'source_sha256': {
        name: digest(importlib.import_module(name).__file__) for name in modules},
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                    'scipy': scipy.__version__, 'torch': torch.__version__,
                    'cuda_build': torch.version.cuda}}


def bind_run(c, run):
    expected = identity(c)
    # Array members may initialize simultaneously. This tiny shared transaction
    # waits; only duplicate writers of the same scientific task are rejected.
    with exclusive(run / '.identity.lock', wait=True):
        path = run / 'identity.json'
        if path.exists():
            if json.loads(path.read_text()) != expected:
                raise ValueError('Source, configuration or runtime changed; use a fresh run directory')
        else:
            if any(run.glob('**/*.pt')) or any(run.glob('**/*.npz')):
                raise ValueError('Unidentified artifacts already exist; use a fresh run directory')
            write_json(run / 'config.json', c)
            write_json(run / 'cases.json', [r for r, _ in cases(c)])
            write_json(path, expected)
    return expected


def prediction(c, root, u0, seed, record, horizon, device):
    """Replay the selected checkpoints without using reference endpoints."""
    import torch
    from dysthe_learning.pilot_models import SpectralPINN
    pieces = []
    u = torch.tensor(u0, dtype=torch.complex64, device=device)
    steps = round(c['slab_length'] / c['sample_step'])
    directory = root / f'pinn-slabs-{seed}-{record["case_id"]}'
    hashes = {}
    for slab in range(round(horizon / c['slab_length'])):
        path = directory / f'best-{slab}.pt'
        saved = torch.load(path, map_location=device, weights_only=False)
        model = SpectralPINN(u, c['length'], max_mode=c['output_modes'],
                            width=c['pinn_width'], depth=c['pinn_depth']).to(device)
        model.load_state_dict(saved['model'])
        with torch.no_grad():
            values = model(torch.linspace(0, c['slab_length'], steps + 1, device=device))
        pieces.append(values.detach().cpu().numpy()[0 if slab == 0 else 1:])
        u = values[-1].detach()
        hashes[str(path.relative_to(root))] = digest(path)
    return np.concatenate(pieces), hashes


def train_horizon(c, run, record, u0, seed, horizon, reference, device):
    import torch
    from .long_reference import compare_fields, field_diagnostics, sampled_events
    root = run / 'pinn' / record['case_id'] / f'seed-{seed}' / f'xi-{horizon:g}'
    root.mkdir(parents=True, exist_ok=True)
    result_path = root / 'result.json'
    environment = {'device': device, 'gpu': torch.cuda.get_device_name() if device == 'cuda' else None}
    lock_path = root / 'device.json'
    if lock_path.exists() and json.loads(lock_path.read_text()) != environment:
        raise ValueError('Device changed on restart; use a fresh run directory')
    write_json(lock_path, environment)
    artifact = reference['artifact']
    if digest(artifact['path']) != artifact['sha256']:
        raise ValueError('Reference checksum changed')
    reference_id = artifact['sha256']
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result['reference_sha256'] != reference_id:
            raise ValueError('Reference changed for cached PINN')
        for name, sha in result['checkpoints'].items():
            if digest(root / name) != sha:
                raise ValueError('PINN checkpoint checksum changed')
        if digest(result['artifact']['path']) != result['artifact']['sha256']:
            raise ValueError('PINN prediction checksum changed')
        return result
    with np.load(artifact['path'], allow_pickle=False) as archive:
        high, xi = archive['fields'], archive['xi']
    if high.shape != (round(horizon / c['sample_step']) + 1, c['reference_grid']):
        raise ValueError('Reference has wrong grid/horizon')
    np.testing.assert_allclose(xi, np.arange(len(xi)) * c['sample_step'], rtol=0, atol=1e-12)
    local = dict(c, seeds=[seed], pinn_horizon=horizon,
                 pinn_slabs=round(horizon / c['slab_length']))
    local['pinn_epochs'] = local['pinn_slabs'] * c['epochs_per_slab']
    data = {'high': high[None], 'fields': resample(high, c['grid'], axis=-1)[None], 'xi': xi}
    started = time.perf_counter()
    fit_pinns(local, root, {'records': [record]}, data, device, variants=('slabs',))
    pred, hashes = prediction(c, root, data['fields'][0, 0], seed, record, horizon, device)
    errors = metric(pred, high, c['length'])
    errors['h2'] = compare_fields(pred, high, c['length'])['h2']
    thresholds = {key: c[f'pinn_{key}_tolerance'] for key in ('l2', 'h1', 'h2', 'peak')}
    stations = []
    for j, x in enumerate(xi):
        err = metric(pred[j], high[j], c['length'])
        err['h2'] = compare_fields(pred[j], high[j], c['length'])['h2']
        stations.append({'xi': float(x), **err})
    path = root / 'predictions.npz'
    with path.with_suffix('.tmp').open('wb') as stream:
        np.savez_compressed(stream, fields=pred, xi=xi)
    path.with_suffix('.tmp').replace(path)
    audit = json.loads((root / 'pinn-summary.json').read_text())[0]
    result = {'case_id': record['case_id'], 'split': record['split'], 'seed': seed,
              'horizon': horizon, 'method': 'per-inlet spectral slab PINN',
              'qualified': all(errors[k] <= thresholds[k] for k in thresholds),
              'errors': errors, 'thresholds': thresholds, 'stations': stations,
              'reference_sha256': reference_id, 'checkpoints': hashes,
              'artifact': {'path': str(path.resolve()), 'sha256': digest(path)},
              'diagnostics': field_diagnostics(pred, xi, c['length'], record['physics']['epsilon'],
                                               c['output_modes'], c['diagnostics_tail_fraction']),
              'training_seconds': audit['training_seconds'],
              'invocation_seconds': time.perf_counter() - started,
              'gpu_peak_allocated_bytes': audit['gpu_peak_allocated_bytes'],
              'full_residual_rms_relative': audit['full_residual_rms_relative'],
              'environment': environment,
              'interpretation': 'Fitted interval, not unseen-distance forecast; sampled errors, not continuum bounds.'}
    result['status'] = 'passed' if result['qualified'] else 'accuracy_gate_failed'
    result['events'] = sampled_events(result['diagnostics'], c)
    actual_peaks = reference['events']['focusing_peaks']
    predicted_peaks = result['events']['focusing_peaks']
    same_count = len(actual_peaks) == len(predicted_peaks)
    result['focusing_comparison'] = {
        'reference_peak_count': len(actual_peaks), 'predicted_peak_count': len(predicted_peaks),
        'reference_peaks': actual_peaks, 'predicted_peaks': predicted_peaks,
        'ordered_peak_distance_errors': [abs(a['xi'] - b['xi']) for a, b in zip(actual_peaks, predicted_peaks)]
            if same_count and actual_peaks else None,
        'interpretation': 'Ordered matching only when nonzero counts agree; sampled events with spacing sample_step, no continuous event-time bound.'}
    result['first_failed_station'] = next((row['xi'] for row in stations
        if any(row[key] > thresholds[key] for key in thresholds)), None)
    write_json(result_path, result)
    return result


def run_task(c, run, stage, index, device):
    from .long_reference import reference_case, sensitivity_case, domain_case, load_reference
    record, u0, seed = task(c, stage, index)
    outcome = {'stage': stage, 'task_index': index, 'case_id': record['case_id'],
               'split': record['split'], 'seed': seed, 'results': [], 'status': 'running'}
    path = run / 'tasks' / f'{stage}-{index:03d}.json'
    with exclusive(run / 'locks' / f'{stage}-{index:03d}.lock'):
        write_json(path, outcome)
        try:
            for horizon in c[f'{stage}_horizons']:
                if stage == 'reference':
                    result = reference_case(c, run, record, u0, horizon)
                elif stage == 'domain' and record['family'] != 'packet':
                    result = domain_case(c, run, record, u0, horizon)
                else:
                    try:
                        reference, _, _ = load_reference(c, run, record, u0, horizon)
                    except FileNotFoundError:
                        reference = None
                    if not reference or not reference['qualified']:
                        result = {'horizon': horizon, 'qualified': False, 'status': 'blocked_by_reference'}
                    elif stage == 'pinn':
                        result = train_horizon(c, run, record, u0, seed, horizon, reference, device)
                    else:
                        compute = sensitivity_case if stage == 'sensitivity' else domain_case
                        result = compute(c, run, record, u0, horizon)
                outcome['results'].append(result)
                if result.get('status', '').startswith('not_applicable'):
                    outcome['status'] = 'not_applicable'
                    break
                if not result['qualified']:
                    outcome['status'] = 'stopped_at_gate'
                    break
                write_json(path, outcome)
            else:
                outcome['status'] = 'completed'
            write_json(path, outcome)
            return outcome
        except Exception as exc:
            outcome.update(status='execution_error', error=f'{type(exc).__name__}: {exc}')
            write_json(path, outcome)
            raise


def summarize(c, run):
    selected = cases(c)
    rows, pinn_metrics = [], []
    for stage in ('reference', 'pinn', 'sensitivity', 'domain'):
        count = len(selected) * (len(c['seeds']) if stage == 'pinn' else 1)
        for i in range(count):
            path = run / 'tasks' / f'{stage}-{i:03d}.json'
            if not path.exists():
                record, _, seed = task(c, stage, i)
                rows.append({'stage': stage, 'task_index': i, 'case_id': record['case_id'],
                             'seed': seed, 'status': 'missing', 'horizons': []})
                continue
            value = json.loads(path.read_text())
            if stage == 'pinn':
                for item in value['results']:
                    if 'errors' in item:
                        pinn_metrics.append({'case_id': value['case_id'], 'seed': value['seed'],
                            'split': value['split'], 'horizon': item['horizon'],
                            'errors': item['errors'], 'qualified': item['qualified']})
            rows.append({key: value[key] for key in ('stage', 'task_index', 'case_id', 'seed', 'status')}
                        | {'horizons': [{'horizon': r.get('horizon', r.get('xi_end')),
                                        'status': r.get('status'), 'qualified': r.get('qualified'),
                                        'first_failed_station': r.get('first_failed_station')}
                                       for r in value['results']],
                           'last_passed_horizon': max((r.get('horizon', r.get('xi_end', 0))
                              for r in value['results'] if r.get('qualified')), default=None)})
    aggregates = []
    for split in ('validation', 'test'):
        for horizon in c['pinn_horizons']:
            items = [r for r in pinn_metrics if r['split'] == split and r['horizon'] == horizon]
            expected = sum(r['split'] == split for r, _ in selected) * len(c['seeds'])
            aggregates.append({'split': split, 'horizon': horizon, 'expected_case_seed_pairs': expected,
                'evaluated_case_seed_pairs': len(items), 'passed_case_seed_pairs': sum(r['qualified'] for r in items),
                'complete': len(items) == expected,
                'worst_errors': {k: max(r['errors'][k] for r in items) for k in ('l2', 'h1', 'h2', 'peak')} if items else None,
                'median_errors': {k: float(np.median([r['errors'][k] for r in items]))
                                  for k in ('l2', 'h1', 'h2', 'peak')} if items else None})
    result = {'campaign': c['campaign'], 'model_id': c['model_id'], 'tasks': rows,
              'pinn_aggregates': aggregates,
              'complete': all(r['status'] in ('completed', 'not_applicable', 'stopped_at_gate') for r in rows),
              'all_requested_gates_passed': all(r['status'] in ('completed', 'not_applicable') for r in rows),
              'interpretation': 'Missing, blocked and failed tasks remain visible. No global-existence conclusion.'}
    write_json(run / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--stage', choices=('reference', 'pinn', 'sensitivity', 'domain', 'summarize'), required=True)
    parser.add_argument('--task-index', type=int, default=0)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    c = config(args.config)
    if os.environ.get('SLURM_JOB_ID') and c['threads'] > int(os.environ.get('SLURM_CPUS_PER_TASK', '1')):
        raise ValueError('Configured CPU threads exceed the Slurm allocation')
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    torch.set_num_threads(c['threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
    if args.stage != 'pinn' and args.device != 'cpu':
        raise ValueError('Numerical and summary stages run on CPU')
    bind_run(c, run)
    job = {'stage': args.stage, 'task_index': args.task_index, 'device': args.device,
           'utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
           'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'hostname': platform.node()}
    stamp = time.time_ns()
    write_json(run / 'invocations' / f'{args.stage}-{args.task_index}-{stamp}.json', job)
    if args.stage == 'summarize':
        with exclusive(run / 'locks' / 'summary.lock'):
            result = summarize(c, run)
    else:
        result = run_task(c, run, args.stage, args.task_index, args.device)
    print(json.dumps({'stage': args.stage, 'task_index': args.task_index,
                      'status': result.get('status', 'summarized'), 'run': str(run)}), flush=True)


if __name__ == '__main__':
    main()

"""Evaluate frozen CPU-pilot checkpoints independently of their training loops.

This audit neither trains models nor changes checkpoint selection.  Errors are
sampled diagnostics against the stored refined reference, not continuum bounds.
Run with ``--run`` and ``--results`` pointing to the matching pilot directories.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from scipy.signal import resample as fourier_resample
import torch

from dysthe_core.contracts import require_water_wave_model, validate_splits
from dysthe_learning.pilot_models import ContinuousFNO, SpectralPINN
from dysthe_experiments.cpu_pilot import integrate, read_config


METRICS = ('l2', 'h1', 'peak')
GATES = {'l2': .01, 'h1': .05, 'peak': .05}
METHODS = ('fno', 'pino', 'pinn_scratch', 'pinn_nls_warm', 'pinn_slabs')
LABELS = {'fno': 'FNO', 'pino': 'PINO', 'pinn_scratch': 'PINN\nscratch',
          'pinn_nls_warm': 'PINN\nNLS warm', 'pinn_slabs': 'PINN\nslabs'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows for {path}')
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def station_errors(pred, truth, length):
    """Independent per-station complex L2, H1 and envelope-peak errors."""
    pred, truth = np.asarray(pred), np.asarray(truth)
    if pred.shape[-1] != truth.shape[-1]:
        pred = fourier_resample(pred, truth.shape[-1], axis=-1)
    if pred.shape != truth.shape or not np.isfinite(pred).all():
        raise ValueError('Prediction/reference shape mismatch or nonfinite prediction')
    k = 2 * np.pi * np.fft.fftfreq(truth.shape[-1], d=length/truth.shape[-1])
    transformed = np.fft.fft(truth, axis=-1)
    difference = np.fft.fft(pred-truth, axis=-1)
    squared_norm = lambda x: np.sum(np.abs(x)**2, axis=-1)
    base = squared_norm(transformed)
    if np.any(base <= 0):
        raise ValueError('Relative error undefined for zero reference')
    weight = np.sqrt(1+k*k)
    return {'l2': np.sqrt(squared_norm(difference)/base),
            'h1': np.sqrt(squared_norm(difference*weight)/squared_norm(transformed*weight)),
            'peak': np.abs(np.max(np.abs(pred), axis=-1)-np.max(np.abs(truth), axis=-1))
                    / np.max(np.abs(truth), axis=-1)}


def median_time(predict, repeats=5):
    """Warm once, then measure five independent CPU calls with fixed inputs."""
    predict()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        predict()
        samples.append(time.perf_counter()-start)
    return float(np.median(samples)), samples


def load_checkpoint(path, config, inventory):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint['config']['campaign'] != config:
        raise ValueError(f'Checkpoint campaign differs: {path}')
    inventory.append({'path': str(path.resolve()), 'sha256': sha256(path),
                      'best_epoch': checkpoint['epoch'],
                      'selection_score': checkpoint['best']})
    return checkpoint


def operator_predict(model, inlet, epsilon, positions):
    field = inlet[None]
    snapshots = [field[0]]
    with torch.no_grad():
        for step in np.diff(positions):
            field = model(field, epsilon, float(step))
            snapshots.append(field[0])
    return torch.stack(snapshots).cpu().numpy()


def pinn_predict(models, local_positions, check_interfaces=True):
    """Reconstruct the trained slabs, checking every stored predicted handoff.

    The saved inlet is part of each trained trial function.  It is checked
    against the previous prediction, never replaced with reference data.
    """
    snapshots, jumps = [], []
    previous = None
    with torch.no_grad():
        for model in models:
            if previous is not None:
                jump = float(torch.linalg.vector_norm(model.u0-previous)
                             / torch.linalg.vector_norm(previous))
                jumps.append(jump)
                if check_interfaces and jump > 5e-6:
                    raise ValueError(f'Saved slab inlet is not the predicted handoff: {jump}')
            out = model(local_positions)
            snapshots.append(out if previous is None else out[1:])
            previous = out[-1]
    return torch.cat(snapshots).cpu().numpy(), jumps


def append_rows(rows, pred, reference, positions, method, seed, record, length):
    errors = station_errors(pred, reference, length)
    for index, xi in enumerate(positions):
        row = {'method': method, 'seed': seed, 'case_id': record['case_id'],
               'family': record['family'], 'split': record['split'], 'xi': float(xi)}
        row.update({key: float(errors[key][index]) for key in METRICS})
        rows.append(row)


def aggregate(rows):
    cases, summary = [], []
    for method in METHODS:
        selected = [r for r in rows if r['method'] == method]
        horizons = (.5, 1., 2., 4.) if method in ('fno', 'pino') else (.5, 1., 2.)
        for horizon in horizons:
            for split in ('validation', 'test'):
                group = [r for r in selected if r['split'] == split and r['xi'] <= horizon+1e-9]
                identities = sorted({(r['case_id'], r['seed']) for r in group})
                per_case = []
                for case_id, seed in identities:
                    trajectory = [r for r in group if r['case_id'] == case_id and r['seed'] == seed]
                    endpoint = next(r for r in trajectory if np.isclose(r['xi'], horizon))
                    out = {'method': method, 'split': split, 'case_id': case_id,
                           'seed': seed, 'horizon': horizon, 'sampled_stations': len(trajectory)}
                    for key in METRICS:
                        maximum = max(trajectory, key=lambda r: r[key])
                        out[f'max_{key}'] = maximum[key]
                        out[f'xi_max_{key}'] = maximum['xi']
                        out[f'endpoint_{key}'] = endpoint[key]
                    out['gate_pass'] = all(out[f'max_{key}'] <= GATES[key] for key in METRICS)
                    per_case.append(out)
                    cases.append(out)
                if not per_case:
                    continue
                out = {'method': method, 'split': split, 'horizon': horizon,
                       'case_seed_count': len(per_case),
                       'unique_fields': len({r['case_id'] for r in per_case}),
                       'passing_case_seeds': sum(r['gate_pass'] for r in per_case),
                       'gate_pass_all': all(r['gate_pass'] for r in per_case)}
                for key in METRICS:
                    out[f'max_{key}'] = max(r[f'max_{key}'] for r in per_case)
                    out[f'median_case_max_{key}'] = float(np.median([r[f'max_{key}'] for r in per_case]))
                    out[f'max_endpoint_{key}'] = max(r[f'endpoint_{key}'] for r in per_case)
                summary.append(out)
    return cases, summary


def reconstruct_selection(run, config, rows):
    # Operator ranking uses the exact saved training-validation scores; the
    # independent full-reference audit never retroactively selects a checkpoint.
    operator_scores = {}
    for method in ('fno', 'pino'):
        operator_scores[method] = float(np.mean([
            json.loads((run/f'{method}-{seed}'/'summary.json').read_text())['best_validation_max_h1']
            for seed in config['seeds']]))
    pinn_scores = {}
    for variant in ('scratch', 'nls_warm', 'slabs'):
        method = f'pinn_{variant}'
        pinn_scores[method] = float(np.mean([
            max(r['h1'] for r in rows if r['method'] == method and r['seed'] == seed
                and r['split'] == 'validation' and np.isclose(r['xi'], config['pinn_horizon']))
            for seed in config['seeds']]))
    return {'operator_rule': 'Mean across seeds of saved worst-validation-case H1 at xi=0.5; no test or later horizon selection.',
            'operator_scores': operator_scores, 'selected_operator': min(operator_scores, key=operator_scores.get),
            'pinn_rule': 'Mean across seeds of worst validation-case endpoint H1 at xi=2; checkpoints selected only by their full Dysthe residual validation.',
            'pinn_reducer_note': 'Protocol specifies validation endpoint H1; mean-seed/worst-case aggregation is made explicit here.',
            'pinn_scores': pinn_scores, 'selected_pinn': min(pinn_scores, key=pinn_scores.get)}


def figures(results, summary, rows, regularity):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/dysthe-pilot-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10,
                         'mathtext.fontset': 'dejavuserif', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.bbox': 'tight'})
    directory = results/'figures'; directory.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.15), constrained_layout=True)
    for ax, key, label in zip(axes, METRICS, (r'$L^2$ field error', r'$H^1$ field error', 'Peak-height error')):
        for split, offset, marker, shade in [('validation', -.11, 'o', '.55'), ('test', .11, 's', '0')]:
            values = [next(r[f'max_{key}'] for r in summary if r['method'] == method
                           and r['split'] == split and r['horizon'] == 2.) for method in METHODS]
            ax.scatter(np.arange(len(METHODS))+offset, 100*np.array(values), marker=marker,
                       color=shade, s=29, label=split.capitalize(), zorder=3)
        ax.axhline(100*GATES[key], color='.45', linestyle=':', linewidth=1)
        ax.set_xticks(np.arange(len(METHODS)), [LABELS[m] for m in METHODS], fontsize=9)
        ax.set_yscale('log'); ax.set_ylabel(label+' (%)'); ax.grid(axis='y', alpha=.18)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(r'Worst sampled error through $\xi=2$: three fields per split, two seeds', fontsize=11)
    fig.savefig(directory/'method-errors-xi2.pdf'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.05), constrained_layout=True)
    for ax, key, label in zip(axes, ('l2', 'h1'), (r'$L^2$', r'$H^1$')):
        for method, shade in [('fno', '.55'), ('pino', '0')]:
            for split, line in [('validation', '--'), ('test', '-')]:
                selected = [r for r in rows if r['method'] == method and r['split'] == split]
                positions = sorted({r['xi'] for r in selected})
                worst = [max(r[key] for r in selected if r['xi'] == xi) for xi in positions]
                ax.plot(positions, 100*np.maximum(worst, 1e-12), line, color=shade,
                        label=f'{method.upper()} {split}', linewidth=1.5)
        ax.axvspan(0, .5, color='.9', zorder=0)
        ax.axhline(100*GATES[key], color='.45', linestyle=':', linewidth=1)
        ax.set_yscale('log'); ax.set_ylim(.001, 300); ax.set_xlim(0, 4)
        ax.set_xlabel(r'Propagation distance $\xi$'); ax.set_ylabel(f'Worst {label} error (%)')
        ax.grid(axis='y', alpha=.18)
    axes[0].legend(frameon=False, fontsize=8, loc='lower right')
    fig.suptitle('Autonomous operator rollout; shaded interval contains training trajectories', fontsize=10)
    fig.savefig(directory/'operator-growth.pdf'); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.05), constrained_layout=True)
    cases = sorted({r['case_id'] for r in regularity})
    for ax, key, label in zip(axes, ('linf', 'h1', 'h2'), (r'$\|u\|_\infty$', r'$\|u\|_{H^1}$', r'$\|u\|_{H^2}$')):
        for case_id, line in zip(cases, ('-', '--', ':')):
            selected = sorted((r for r in regularity if r['case_id'] == case_id), key=lambda r: r['xi'])
            ax.plot([r['xi'] for r in selected], [r[key] for r in selected], line,
                    color='0', label=case_id.split('-')[0], linewidth=1.5)
        ax.set_xlabel(r'Propagation distance $\xi$'); ax.set_ylabel(label)
        ax.set_xlim(0, 16); ax.grid(axis='y', alpha=.18)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle('Numerical regularity diagnostics: three validation packets, no learned forecast', fontsize=10)
    fig.savefig(directory/'numerical-regularity.pdf'); plt.close(fig)
    return [str(path.relative_to(results)) for path in sorted(directory.glob('*.pdf'))]


def audit(run, results):
    config = read_config(run/'config.json')
    manifest = json.loads((run/'manifest.json').read_text())
    require_water_wave_model(manifest); validate_splits(manifest['records'])
    if sha256(run/'dataset.npz') != manifest['sha256']:
        raise ValueError('Dataset checksum mismatch')
    protocol_path = results/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    expected_gates = protocol['provisional_accuracy_gate']
    if expected_gates != {'relative_l2': GATES['l2'], 'relative_h1': GATES['h1'], 'relative_peak': GATES['peak']}:
        raise ValueError('Audit gates disagree with frozen protocol')
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    with np.load(run/'dataset.npz') as archive:
        data = {key: archive[key] for key in archive.files}
    records = [(i, r) for i, r in enumerate(manifest['records']) if r['split'] != 'train']
    rows, timings, checkpoint_inventory, continuity = [], [], [], []
    operator_positions = np.arange(round(config['horizon']/config['operator_step'])+1)*config['operator_step']
    operator_indices = np.rint(operator_positions/config['sample_step']).astype(int)
    for method in ('fno', 'pino'):
        for seed in config['seeds']:
            directory = run/f'{method}-{seed}'
            ckpt = load_checkpoint(directory/'best.pt', config, checkpoint_inventory)
            saved = json.loads((directory/'summary.json').read_text())
            if sha256(directory/'best.pt') != saved['checkpoint_sha256']:
                raise ValueError('Operator checkpoint differs from recorded training result')
            model = ContinuousFNO(config['length'], config['operator_width'], config['operator_modes'],
                                  config['operator_depth'], config['output_modes']).float().eval()
            model.load_state_dict(ckpt['model'])
            for index, record in records:
                inlet = torch.tensor(data['fields'][index, 0], dtype=torch.complex64)
                pred = operator_predict(model, inlet, record['physics']['epsilon'], operator_positions)
                append_rows(rows, pred, data['high'][index, operator_indices], operator_positions,
                            method, seed, record, config['length'])
            print('audited', method, seed, flush=True)

    numerical_timings = {}
    for index, record in records:
        positions = np.arange(round(config['pinn_horizon']/config['sample_step'])+1)*config['sample_step']
        def numerical():
            return integrate(data['fields'][index, 0], record['physics']['epsilon'],
                             config['length'], config['grid'], .01, positions)[0]
        median, samples = median_time(numerical)
        comparison = station_errors(numerical(), data['high'][index, :len(positions)], config['length'])
        numerical_timings[record['case_id']] = {'method': 'coarse_numerical', 'case_id': record['case_id'],
            'split': record['split'], 'horizon': config['pinn_horizon'], 'profile_grid': config['grid'],
            'propagation_step': .01, 'stations': len(positions), 'repetitions': 5,
            'median_seconds': median, 'sample_seconds': samples,
            'max_errors': {key: float(np.max(comparison[key])) for key in METRICS}}

    for variant in ('scratch', 'nls_warm', 'slabs'):
        method = f'pinn_{variant}'
        count = config['pinn_slabs'] if variant == 'slabs' else 1
        slab_horizon = config['pinn_horizon']/count
        local_positions = torch.linspace(0, slab_horizon, round(slab_horizon/config['sample_step'])+1)
        positions = np.arange(round(config['pinn_horizon']/config['sample_step'])+1)*config['sample_step']
        for seed in config['seeds']:
            for index, record in records:
                directory = run/f'pinn-{variant}-{seed}-{record["case_id"]}'
                saved = json.loads((directory/'summary.json').read_text())
                models = []
                previous = torch.tensor(data['fields'][index, 0], dtype=torch.complex64)
                for slab in range(count):
                    path = directory/f'best-{slab}.pt'
                    ckpt = load_checkpoint(path, config, checkpoint_inventory)
                    if sha256(path) != saved['checkpoints'][path.name]:
                        raise ValueError('PINN checkpoint differs from recorded training result')
                    model = SpectralPINN(previous, config['length'], max_mode=config['output_modes'],
                                         width=config.get('pinn_width',32), depth=config.get('pinn_depth',2),
                                         coefficient_scaling=config.get('pinn_coefficient_scaling','none')).eval()
                    model.load_state_dict(ckpt['model'])
                    if slab == 0:
                        jump = float(torch.linalg.vector_norm(model.u0-previous)/torch.linalg.vector_norm(previous))
                        if jump > 5e-6:
                            raise ValueError('Saved first inlet differs from the declared input')
                    with torch.no_grad():
                        previous = model(local_positions)[-1]
                    models.append(model)
                pred, jumps = pinn_predict(models, local_positions)
                append_rows(rows, pred, data['high'][index, :len(positions)], positions,
                            method, seed, record, config['length'])
                for slab, jump in enumerate(jumps, start=1):
                    continuity.append({'method': method, 'seed': seed, 'case_id': record['case_id'],
                                       'interface_xi': slab*slab_horizon, 'relative_l2_jump': jump})
                median, samples = median_time(lambda: pinn_predict(models, local_positions, check_interfaces=False)[0])
                timings.append({'method': method, 'seed': seed, 'case_id': record['case_id'],
                    'split': record['split'], 'horizon': config['pinn_horizon'], 'stations': len(positions),
                    'recorded_training_seconds': saved['training_seconds'], 'median_inference_seconds': median,
                    'inference_sample_seconds': samples, 'repetitions': 5,
                    'training_plus_inference_seconds': saved['training_seconds']+median,
                    'coarse_numerical_median_seconds': numerical_timings[record['case_id']]['median_seconds']})
            print('audited PINN', variant, seed, flush=True)

    cases, summary = aggregate(rows)
    selection = reconstruct_selection(run, config, rows)
    regularity = json.loads((run/'long-numerical-diagnostics.json').read_text())
    import dysthe_core.torch_spatial as core_torch
    import dysthe_core.reference.spatial as core_reference
    import dysthe_learning.pilot_models as learning_models
    import dysthe_experiments.cpu_pilot as campaign
    sources = {str(Path(module.__file__).resolve()): sha256(module.__file__)
               for module in (core_torch, core_reference, learning_models, campaign)}
    sources[str(Path(__file__).resolve())] = sha256(__file__)
    report = {'model_id': config['model_id'], 'physical_system': config['physical_system'],
        'audit_kind': 'Frozen-checkpoint independent sampled-trajectory evaluation; no training or reselection.',
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'numpy': np.__version__, 'device': 'cpu', 'torch_threads': 2},
        'data_sha256': manifest['sha256'], 'protocol_sha256': sha256(protocol_path),
        'source_sha256': sources, 'checkpoints': checkpoint_inventory,
        'counts': {'operator_models': 4, 'pinn_instances': len(timings), 'checkpoint_files': len(checkpoint_inventory),
                   'station_rows': len(rows), 'independent_fields_per_split': 3, 'seeds': config['seeds']},
        'sampling': {'pinn_step': config['sample_step'], 'pinn_horizon': config['pinn_horizon'],
                     'operator_step': config['operator_step'], 'operator_horizon': config['horizon']},
        'selection': selection, 'gates': GATES, 'aggregates': summary, 'case_metrics': cases,
        'slab_continuity': continuity, 'pinn_timings': timings, 'numerical_timings': list(numerical_timings.values()),
        'timing_interpretation': 'Single-case CPU evaluation after one warmup, five repetitions, median. PINN inference excludes fitting and checkpoint loading; training-plus-inference includes recorded optimization time. Numerical timing includes the full coarse solve and returned trajectory. Inference alone is not a full-solve speed comparison.',
        'limits': ['Only sampled propagation stations are tested; no continuous-distance guarantee.',
                   'Three independent fields per validation/test split; seeds are not additional independent fields.',
                   'Per-instance PINNs optimize separately on every field and are not generalizing operators.',
                   'Reference fields are stored refined spatial-Dysthe solutions, not HOS or laboratory validation.',
                   'Long numerical norms do not establish continuum global existence.'],
        'long_numerical': {'source_sha256': sha256(run/'long-numerical-diagnostics.json'),
                          'all_rows_qualified': all(r['qualified'] for r in regularity),
                          'maximum_refinement_l2': max(r['refinement_max_l2'] for r in regularity),
                          'maximum_refinement_h1': max(r['refinement_max_h1'] for r in regularity)}}
    results.mkdir(parents=True, exist_ok=True)
    directory = results/'data'; directory.mkdir(exist_ok=True)
    write_csv(directory/'station-errors.csv', rows)
    write_csv(directory/'case-errors.csv', cases)
    write_csv(directory/'aggregate-errors.csv', summary)
    write_csv(directory/'numerical-regularity.csv', regularity)
    write_json(directory/'station-errors.json', rows)
    write_json(directory/'case-errors.json', cases)
    write_json(directory/'timings.json', {'pinn': timings, 'numerical': list(numerical_timings.values())})
    report['figures'] = figures(results, summary, rows, regularity)
    write_json(run/'audit.json', report)
    write_json(results/'audit.json', report)
    print(json.dumps({'selection': selection, 'counts': report['counts'],
          'selected_pinn_gates': [r for r in summary if r['method'] == selection['selected_pinn']],
          'long_numerical': report['long_numerical']}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    audit(args.run, args.results)


if __name__ == '__main__':
    main()

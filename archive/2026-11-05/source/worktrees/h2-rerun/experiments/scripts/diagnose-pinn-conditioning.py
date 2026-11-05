#!/usr/bin/env python3
"""Compare unscaled/scaled spectral PINNs on validation inlets only.

Example (from an installed experiments checkout):
  python scripts/diagnose-pinn-conditioning.py \
    --config configs/long-distance-a100.json --run runs/conditioning-audit \
    --output results/conditioning-audit.json

The default is three validation inlets, seed 11, and xi=4. Both methods use
identical architecture, retained modes, per-slab update budget and residual
checkpoint selection. Each later slab receives its predecessor's prediction.
Numerical fields are used only for post-training accuracy measurements.
Raw arrays/checkpoints stay under --run; --output contains portable metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import time

import numpy as np
import scipy
from scipy.signal import resample
import torch

import dysthe_core.reference.spatial as spatial
import dysthe_core.torch_spatial as torch_spatial
import dysthe_experiments.long_reference as long_reference
import dysthe_experiments.cpu_pilot as pilot
import dysthe_experiments.long_distance as campaign
from dysthe_experiments.long_reference import compare_fields
import dysthe_learning.pilot_models as models


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def spectral_analysis(prediction, reference, length, cutoff):
    """Independent complex128 Fourier audit; never changes a fitted field."""
    n = reference.shape[-1]
    pred = resample(np.asarray(prediction, dtype=np.complex128), n, axis=-1)
    k = 2 * np.pi * np.fft.fftfreq(n, d=length / n)
    modes = np.rint(np.fft.fftfreq(n) * n)
    reference_hat = np.fft.fft(reference, axis=-1) / n
    difference_hat = np.fft.fft(pred - reference, axis=-1) / n
    floors = {}
    for order in range(3):
        weighted_power = (1 + k * k)**order * abs(reference_hat)**2
        floors[f'h{order}'] = float(np.max(np.sqrt(
            np.sum(weighted_power * (abs(modes) > cutoff), axis=-1)
            / np.sum(weighted_power, axis=-1))))
    error_power = (1 + k * k)**2 * abs(difference_hat[-1])**2
    total = float(np.sum(error_power))
    # Relative bands scale with K; at K=48 they are 0:12,13:24,25:48,>48.
    masks = {
        'lowest_quarter': abs(modes) <= cutoff / 4,
        'second_quarter': (abs(modes) > cutoff / 4) & (abs(modes) <= cutoff / 2),
        'upper_half': (abs(modes) > cutoff / 2) & (abs(modes) <= cutoff),
        'outside_output_band': abs(modes) > cutoff,
    }
    bands = {key: float(np.sum(error_power[mask]) / total) if total > 0 else None
             for key, mask in masks.items()}
    return {'independent_sobolev_errors': compare_fields(pred, reference, length),
            'reference_projection_floor': floors,
            'endpoint_h2_error_power_fractions': bands}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True, help='New directory for raw artifacts')
    parser.add_argument('--output', type=Path, help='New summary JSON; default: RUN/audit.json')
    parser.add_argument('--cases', nargs='+', default=['packet-008', 'sideband-008', 'random-008'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[11])
    parser.add_argument('--horizon', type=float, default=4.)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    base = pilot.read_config(args.config)
    if args.threads < 1 or not np.isfinite(args.horizon) or args.horizon <= 0:
        parser.error('Positive finite horizon and positive threads required')
    if len(set(args.cases)) != len(args.cases) or len(set(args.seeds)) != len(args.seeds):
        parser.error('Cases and seeds must be unique')
    slab_length = base['slab_length']
    sample_step = base['sample_step']
    for step in (slab_length, sample_step):
        if not np.isclose(args.horizon / step, round(args.horizon / step), atol=1e-10, rtol=0):
            parser.error('Horizon must divide into whole slabs and saved intervals')
    if args.horizon > base['horizon']:
        parser.error('Horizon exceeds the input campaign reference horizon')
    records, inlets = pilot.initial_fields(base)
    selected = []
    for case_id in args.cases:
        matches = [j for j, record in enumerate(records) if record['case_id'] == case_id]
        if len(matches) != 1 or records[matches[0]]['split'] != 'validation':
            parser.error(f'{case_id} must identify an existing validation inlet')
        selected.append(matches[0])
    output = args.output or args.run / 'audit.json'
    if output.exists():
        parser.error('Summary already exists; choose a fresh output path')
    args.run.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    length, n, h = base['length'], base['reference_grid'], base['reference_step']
    cutoff = base['output_modes']
    positions = np.arange(round(args.horizon / sample_step) + 1) * sample_step
    sources = {'pilot': Path(pilot.__file__), 'campaign': Path(campaign.__file__),
               'models': Path(models.__file__), 'reference': Path(spatial.__file__),
               'residual': Path(torch_spatial.__file__), 'reference_audit': Path(long_reference.__file__)}
    source_hashes = {key: sha(path) for key, path in sources.items()}
    snapshots = args.run / 'source-snapshot'
    snapshots.mkdir()
    for key, path in sources.items():
        (snapshots / f'{key}.py').write_bytes(path.read_bytes())
    audit = {
        'kind': 'CPU validation-only conditioning ablation; not a GPU or independent-test result',
        'configuration': base, 'config_sha256': sha(args.config),
        'diagnostic_script_sha256': sha(__file__),
        'source_sha256_at_start': source_hashes,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'scipy': scipy.__version__, 'torch': torch.__version__,
                        'threads': args.threads, 'platform': platform.platform()},
        'design': {'cases': args.cases, 'seeds': args.seeds,
                   'variants': ['none', 'inverse_1_plus_k2'], 'horizon': args.horizon,
                   'slab_length': slab_length, 'epochs_per_slab': base['epochs_per_slab'],
                   'output_modes': cutoff, 'network_width': base['pinn_width'],
                   'network_depth': base['pinn_depth'],
                   'selection': 'Full Dysthe residual; numerical labels used only after fitting',
                   'handoff': 'Predicted endpoint; no numerical-reference resets'},
        'references': [], 'fits': [],
    }
    for index in selected:
        record, inlet = records[index], inlets[index]
        case_id, epsilon = record['case_id'], record['physics']['epsilon']
        high, _ = pilot.integrate(inlet, epsilon, length, n, h, positions)
        coarse_step, _ = pilot.integrate(inlet, epsilon, length, n, 2 * h, positions)
        coarse_grid, _ = pilot.integrate(inlet, epsilon, length, n // 2, h, positions)
        refinements = {'step': compare_fields(coarse_step, high, length),
                       'grid': compare_fields(coarse_grid, high, length)}
        thresholds = {'h0': base['reference_l2_tolerance'],
                      'h1': base['reference_h1_tolerance'],
                      'h2': base['reference_h2_tolerance']}
        qualified = all(errors[key] <= threshold for errors in refinements.values()
                        for key, threshold in thresholds.items())
        audit['references'].append({'case_id': case_id, 'epsilon': epsilon,
                                    'refinements': refinements, 'qualified': qualified})
        if not qualified:
            write_json(output, audit)
            raise RuntimeError(f'Unqualified reference: {case_id}')
        for seed in args.seeds:
            for scaling in audit['design']['variants']:
                slabs = round(args.horizon / slab_length)
                config = dict(base, seeds=[seed], pinn_horizon=args.horizon,
                              pinn_slabs=slabs, pinn_epochs=slabs * base['epochs_per_slab'],
                              pinn_coefficient_scaling=scaling)
                root = args.run / f'{case_id}-seed-{seed}-{scaling}'
                root.mkdir()
                data = {'high': high[None], 'fields': resample(high, base['grid'], axis=-1)[None],
                        'xi': positions}
                started = time.perf_counter()
                pilot.fit_pinns(config, root, {'records': [record]}, data, 'cpu', variants=('slabs',))
                prediction, checkpoints = campaign.prediction(config, root, data['fields'][0, 0],
                                                               seed, record, args.horizon, 'cpu')
                np.savez_compressed(root / 'fields.npz', prediction=prediction, reference=high, xi=positions)
                summary = json.loads((root / 'pinn-summary.json').read_text())[0]
                fit = {'case_id': case_id, 'seed': seed, 'scaling': scaling,
                       'errors': pilot.metric(prediction, high, length),
                       **spectral_analysis(prediction, high, length, cutoff),
                       'pass_combined_gates': summary['sampled_interval_pass'],
                       'gates': summary['sampled_interval_tolerances'],
                       'full_residual_rms_relative_by_slab': summary['full_residual_rms_relative'],
                       'training_seconds': summary['training_seconds'],
                       'total_elapsed_seconds': time.perf_counter() - started,
                       'summary_sha256': sha(root / 'pinn-summary.json'),
                       'prediction_reference_sha256': sha(root / 'fields.npz'),
                       'checkpoints': checkpoints}
                audit['fits'].append(fit)
                audit['source_sha256_at_end'] = {key: sha(path) for key, path in sources.items()}
                audit['source_files_changed_during_run'] = [key for key in source_hashes
                    if source_hashes[key] != audit['source_sha256_at_end'][key]]
                write_json(output, audit)
                print(json.dumps({key: fit[key] for key in ('case_id', 'seed', 'scaling', 'errors',
                                                           'pass_combined_gates')}), flush=True)


if __name__ == '__main__':
    main()

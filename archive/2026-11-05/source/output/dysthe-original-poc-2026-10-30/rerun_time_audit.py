#!/usr/bin/env python3
"""Reproduce the primary sideband small-step audit in a NEW output directory.

Default: stability-consistent temporal potential envelope, epsilon=0.1, T=16,
N=4096 (from config-audit.json), fine step0.0003125. Independent checks use
N/2 at the same step and N at twice the step. No packaged arrays are required.
The historical time_audit.py and its recorded result identities are preserved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import numpy as np
import scipy

from run import MODELS, LIMITATION, diagnostics, errors, inlets, sha, solve, write_json


def positive(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('Value must be positive and finite')
    return number


def execute(config, output, *, horizon=16., step=.0003125, config_source_sha256=None):
    """Run a fresh three-trajectory check; there is deliberately no cache reuse."""
    c = dict(config)
    if c.get('model_ids') != MODELS:
        raise ValueError('Configuration must declare the exact versioned temporal model IDs')
    n, sample = c['n'], c['sample_step']
    if type(n) is not int or n < 64 or n % 8:
        raise ValueError('Fine grid must be a multiple of eight >=64 so N/2 is valid')
    if (not np.isfinite(horizon) or horizon <= 0 or not np.isfinite(step) or step <= 0
            or not np.isfinite(sample) or sample <= 0):
        raise ValueError('Horizon, fine step and saved-time interval must be positive and finite')
    for ratio in (horizon/sample, sample/(2*step)):
        if ratio < 1 or not np.isclose(ratio, round(ratio), rtol=1e-10, atol=1e-10):
            raise ValueError('Horizon and saved times must respect both declared step sizes')
    if type(c['initial_modes']) is not int or not 0 <= c['initial_modes'] < n/8:
        raise ValueError('The initial bandwidth must fit both independent grids')
    for key in ('h0', 'h1', 'h2', 'tail', 'action'):
        if not np.isfinite(c['tolerances'][key]) or c['tolerances'][key] <= 0:
            raise ValueError('All numerical tolerances must be positive and finite')
    c.update(step=step, horizons=[horizon], models=['stability'], epsilons=[.1],
             case_selection={'family': 'sideband', 'epsilon': .1, 'model': 'stability', 'T': horizon},
             purpose='Fresh smaller-step audit; historical failures remain untouched')
    family, u0, params = inlets(c)[1]
    output = Path(output).resolve()
    # Reserve the directory atomically. Existing empty or populated directories
    # are rejected; this avoids broken reuse of JSON-only portable evidence.
    output.mkdir(parents=True, exist_ok=False)
    identity = {'config': c, 'config_source_sha256': config_source_sha256,
                'solver_sha256': sha(Path(__file__).with_name('run.py')),
                'audit_source_sha256': sha(__file__),
                'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                            'scipy': scipy.__version__}}
    write_json(output/'identity.json', identity)
    write_json(output/'config.json', c)
    times = np.arange(round(horizon/sample)+1)*sample
    fields, costs, artifacts = {}, {}, []
    for role, grid, h in (('fine', n, step), ('grid', n//2, step), ('step', n, 2*step)):
        try:
            fields[role], costs[role] = solve(u0, c['length'], .1, 'stability', grid, h, times)
        except FloatingPointError as error:
            result = {'identity': identity, 'status': 'nonfinite_failure', 'qualified': False,
                      'failed_configuration': role, 'message': str(error), 'artifacts': artifacts,
                      'limitation': LIMITATION}
            write_json(output/'result.json', result)
            return result
        path = output/(role+'.npz')
        with path.with_suffix('.tmp').open('wb') as handle:
            np.savez_compressed(handle, fields=fields[role], T=times,
                                s=np.arange(grid)*c['length']/grid-c['length']/2)
        path.with_suffix('.tmp').replace(path)
        artifact = {'path': path.name, 'sha256': sha(path), 'seconds': costs[role],
                    'n': grid, 'step': h, 'role': role}
        write_json(output/(role+'.json'), artifact)
        artifacts.append(artifact)
        print(f'{role}: N={grid}, step={h:g}, {costs[role]:.3f} seconds', flush=True)
    measures = diagnostics(fields['fine'], times, c['length'], .1)
    refinement = {role: {key: float(value.max()) for key, value in
                        errors(fields[role], fields['fine'], c['length']).items()}
                  for role in ('grid', 'step')}
    qualified = (all(value <= c['tolerances'][key] for values in refinement.values()
                     for key, value in values.items())
                 and measures['max_action_drift'] <= c['tolerances']['action']
                 and measures['max_spectral_tail'] <= c['tolerances']['tail'])
    result = {'identity': identity, 'key': f'sideband-eps0.1-stability-T{horizon:g}',
              'family': family, 'epsilon': .1, 'model': 'stability',
              'model_id': MODELS['stability'], 'T': horizon, 'parameters': params,
              'qualified': qualified, 'status': 'complete' if qualified else 'numerically_unresolved',
              'refinement': refinement, 'diagnostics': measures, 'seconds': costs,
              'artifact': artifacts[0], 'artifacts': artifacts,
              'nls_comparison_status': 'Not computed: this fresh audit has no dependency on packaged full-grid arrays.',
              'limitation': LIMITATION}
    write_json(output/'result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='NEW output directory; any existing directory is rejected')
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).with_name('config-audit.json'),
                        help='Configuration file (default: bundled config-audit.json)')
    parser.add_argument('--time', type=positive, default=16.,
                        help='Final scaled time T (default16); smaller values support workflow checks')
    parser.add_argument('--step', type=positive, default=.0003125,
                        help='Fine scaled-time step (default0.0003125); coarse step is twice this')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    result = execute(config, args.output, horizon=args.time, step=args.step,
                     config_source_sha256=sha(args.config))
    print(f"{result['status']}: {args.output.resolve()/'result.json'}", flush=True)
    return 0 if result['qualified'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

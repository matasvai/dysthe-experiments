"""Bounded reproduction of cross-BLAS canonical inlet reuse; no training.

Run with the exact fixed experiments/learning checkouts on PYTHONPATH. The
driver generates once with the HASWELL OpenBLAS kernel and reads from a second
process using NEHALEM. It separately regenerates temporary fields to show the
roundoff mismatch that previously invalidated random-family references.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def worker(config_path, run):
    from dysthe_experiments.long_distance import bind_run, cases, config, digest
    from dysthe_experiments.long_reference import _array_digest
    c = config(config_path)
    bound = bind_run(c, run)
    stored = cases(c, run=run)
    regenerated = cases(c)
    return {
        'core_type': os.environ['OPENBLAS_CORETYPE'],
        'inlet_manifest_sha256': bound['inlets']['sha256'],
        'array_sha256': digest(run / 'inlets.npz'),
        'array_mtime_ns': (run / 'inlets.npz').stat().st_mtime_ns,
        'records': [{'case_id': record['case_id'],
                     'stored_sha256': _array_digest(u),
                     'regenerated_sha256': _array_digest(v)}
                    for (record, u), (_, v) in zip(stored, regenerated)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path,
                        default=Path('/tmp/dysthe-canonical-inlet-cross-blas-result.json'))
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker is not None:
        print(json.dumps(worker(args.config, args.worker)))
        return
    run = Path(tempfile.mkdtemp(prefix='dysthe-canonical-cross-blas-'))
    values = []
    for core_type in ('HASWELL', 'NEHALEM'):
        env = dict(os.environ, OPENBLAS_CORETYPE=core_type,
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        result = subprocess.run([sys.executable, __file__, '--config', str(args.config),
                                 '--worker', str(run)], env=env, text=True,
                                capture_output=True, check=True)
        values.append(json.loads(result.stdout))
    first, second = values
    all_stored_equal = all(a['stored_sha256'] == b['stored_sha256']
                           for a, b in zip(first['records'], second['records']))
    roundoff_changed = [a['case_id'] for a, b in zip(first['records'], second['records'])
                        if a['regenerated_sha256'] != b['regenerated_sha256']]
    result = {'run': str(run), 'all_stored_inlets_identical': all_stored_equal,
              'array_file_hash_and_mtime_unchanged':
                  all(first[key] == second[key]
                      for key in ('array_sha256', 'array_mtime_ns', 'inlet_manifest_sha256')),
              'regenerated_inlets_changed_by_blas_dispatch': roundoff_changed,
              'processes': values,
              'interpretation': 'Local input-identity diagnostic; no optimizer or scientific campaign.'}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not all_stored_equal or not result['array_file_hash_and_mtime_unchanged']:
        raise SystemExit('Canonical input reuse failed')


if __name__ == '__main__':
    main()

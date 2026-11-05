"""Scientific gates and campaign isolation, independent of scheduler hardware."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from dysthe_experiments.long_distance import (
    bind_run, cases, config, digest, exclusive, run_task, summarize, task, write_json,
)

ROOT = Path(__file__).resolve().parents[1]


class LongDistanceTests(unittest.TestCase):
    def setUp(self):
        self.c = config(ROOT / 'configs/long-distance-smoke.json')

    def test_production_case_seed_mapping_preserves_whole_inlets(self):
        c = config(ROOT / 'configs/long-distance-a100.json')
        self.assertEqual(len(cases(c)), 12)
        a, _, seed_a = task(c, 'pinn', 0)
        b, _, seed_b = task(c, 'pinn', 2)
        d, _, _ = task(c, 'pinn', 3)
        self.assertEqual(a['initial_condition_id'], b['initial_condition_id'])
        self.assertNotEqual(a['initial_condition_id'], d['initial_condition_id'])
        self.assertEqual((seed_a, seed_b), (11, 47))
        self.assertEqual(sum(r['split'] == 'test' for r, _ in cases(c)), 6)
        with self.assertRaises(ValueError):
            task(c, 'pinn', 36)

    def test_rejects_unreferenced_horizon_and_inconsistent_slab_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            for changes in ({'pinn_horizons': [.15, .2]}, {'epochs_per_slab': 21},
                            {'seeds': [11, 11]}, {'reference_horizons': [.2, .1]}):
                write_json(path, dict(self.c, **changes))
                with self.assertRaises(ValueError):
                    config(path)

    def test_run_rejects_source_config_change_and_concurrent_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            bind_run(self.c, run)
            bind_run(self.c, run)
            changed = dict(self.c, epochs_per_slab=999)
            with self.assertRaisesRegex(ValueError, 'changed'):
                bind_run(changed, run)
            with exclusive(run / 'held.lock'):
                with self.assertRaisesRegex(RuntimeError, 'Another process'):
                    with exclusive(run / 'held.lock'):
                        pass

    def test_failed_reference_stops_later_horizons_but_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            value = {'horizon': .1, 'qualified': False, 'status': 'numerically_unresolved'}
            with patch('dysthe_experiments.long_reference.reference_case', return_value=value) as compute:
                out = run_task(self.c, run, 'reference', 0, 'cpu')
            self.assertEqual(compute.call_count, 1)
            self.assertEqual(out['status'], 'stopped_at_gate')
            self.assertEqual(json.loads((run / 'tasks/reference-000.json').read_text())['results'], [value])
            result = summarize(self.c, run)
            self.assertFalse(result['complete'])
            self.assertFalse(result['all_requested_gates_passed'])
            self.assertTrue(any(row['status'] == 'missing' for row in result['tasks']))

    def test_missing_or_unqualified_reference_never_trains(self):
        for missing in (True, False):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                options = {'side_effect': FileNotFoundError('missing')} if missing else {
                    'return_value': ({'qualified': False}, None, None)}
                with patch('dysthe_experiments.long_reference.load_reference', **options), \
                        patch('dysthe_experiments.long_distance.train_horizon') as fit:
                    out = run_task(self.c, Path(directory), 'pinn', 0, 'cpu')
                fit.assert_not_called()
                self.assertEqual(out['results'][0]['status'], 'blocked_by_reference')

    def test_failed_pinn_gate_does_not_attempt_next_distance(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('dysthe_experiments.long_reference.load_reference',
                       return_value=({'qualified': True}, None, None)), \
                    patch('dysthe_experiments.long_distance.train_horizon', return_value={
                        'horizon': .1, 'qualified': False, 'status': 'accuracy_gate_failed'}) as fit:
                out = run_task(self.c, Path(directory), 'pinn', 0, 'cpu')
            self.assertEqual(fit.call_count, 1)
            self.assertEqual(out['status'], 'stopped_at_gate')

    def test_real_reference_training_and_completed_restart_preserve_artifacts(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            bind_run(self.c, run)
            ref = run_task(self.c, run, 'reference', 0, 'cpu')
            self.assertEqual(ref['status'], 'completed')
            fit = run_task(self.c, run, 'pinn', 0, 'cpu')
            self.assertEqual(len(fit['results']), 2)
            self.assertTrue(all('h2' in row['errors'] for row in fit['results']))
            paths = list(run.glob('**/*.pt')) + list(run.glob('**/*.npz'))
            before = {str(path): digest(path) for path in paths}
            with patch('torch.optim.Adam.step', side_effect=AssertionError('No retraining on completed replay')), \
                    patch('dysthe_core.reference.spatial.SpatialDysthe.stepper',
                          side_effect=AssertionError('No numerical recomputation on replay')):
                run_task(self.c, run, 'reference', 0, 'cpu')
                replay = run_task(self.c, run, 'pinn', 0, 'cpu')
            self.assertEqual(fit, replay)
            self.assertEqual(before, {str(path): digest(path) for path in paths})
            summary = summarize(self.c, run)
            self.assertFalse(summary['complete'])
            row = next(r for r in summary['pinn_aggregates']
                       if r['split'] == 'validation' and r['horizon'] == .2)
            self.assertEqual(row['evaluated_case_seed_pairs'], 1)
            self.assertEqual(row['expected_case_seed_pairs'], 3)
            self.assertFalse(row['complete'])


if __name__ == '__main__':
    unittest.main()

"""Small scientific-contract and restart checks for the bounded pilot."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dysthe_core.contracts import validate_splits
from dysthe_experiments.cpu_pilot import (
    initial_fields, integrate, metric, read_config,
    restore_checkpoint, save_checkpoint,
)

try:
    import torch
except ImportError:
    torch = None


CONFIG = Path(__file__).resolve().parents[1] / 'configs' / 'cpu-pilot.json'


class PilotScientificContractTests(unittest.TestCase):
    def test_initial_groups_are_reproducible_and_refinements_cannot_leak(self):
        config = read_config(CONFIG)
        records, fields = initial_fields(config)
        repeated_records, repeated_fields = initial_fields(config)
        self.assertEqual(records, repeated_records)
        np.testing.assert_array_equal(fields, repeated_fields)
        self.assertEqual(fields.dtype, np.complex128)
        self.assertEqual(fields.shape, (3 * config['groups_per_family'], config['grid']))
        self.assertTrue(np.isfinite(fields).all())
        for family in ('packet', 'sideband', 'random'):
            group = [r for r in records if r['family'] == family]
            self.assertEqual(sum(r['split'] == 'train' for r in group), config['train_per_family'])
            self.assertEqual(sum(r['split'] == 'validation' for r in group), config['validation_per_family'])
            self.assertGreater(sum(r['split'] == 'test' for r in group), 0)
        refined_copy = copy.deepcopy(records[0])
        refined_copy['case_id'] += '-refinement'
        refined_copy['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'leakage'):
            validate_splits(records + [refined_copy])

    def test_wrong_model_is_rejected_before_generation(self):
        config = json.loads(CONFIG.read_text())
        config['model_id'] = 'unapproved-formulation'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'Scope violation'):
                read_config(path)

    def test_complex_metric_detects_pure_phase_error(self):
        tau = np.arange(64) * 2 * np.pi / 64 - np.pi
        truth = 0.8 + 0.2 * np.exp(2j * tau) + 0.1j * np.exp(-3j * tau)
        phase = 0.37
        measured = metric(truth * np.exp(1j * phase), truth, 2 * np.pi)
        expected = 2 * abs(np.sin(phase / 2))
        self.assertAlmostEqual(measured['l2'], expected, places=13)
        self.assertAlmostEqual(measured['h1'], expected, places=13)
        self.assertLess(measured['peak'], 1e-14)
        self.assertGreater(measured['l2'], 0.3)

    def test_interval_metric_detects_error_between_exact_endpoints(self):
        tau = np.arange(64) * 2 * np.pi / 64 - np.pi
        field = 0.8 + 0.2 * np.exp(2j * tau)
        truth = np.tile(field, (5, 1))
        prediction = truth.copy()
        prediction[2] *= np.exp(0.25j)
        endpoint_error = metric(prediction[[0, -1]], truth[[0, -1]], 2 * np.pi)
        interval_error = metric(prediction, truth, 2 * np.pi)
        self.assertEqual(endpoint_error['l2'], 0.0)
        self.assertAlmostEqual(interval_error['l2'], 2 * np.sin(0.125), places=13)
        self.assertAlmostEqual(interval_error['h1'], 2 * np.sin(0.125), places=13)
        self.assertLess(interval_error['peak'], 1e-14)

    def test_plane_wave_refinement_approaches_exact_complex_phase(self):
        n, length, epsilon = 64, 2 * np.pi, 0.05
        tau = np.arange(n) * length / n - length / 2
        amplitude, wavenumber = 1.2, 3
        initial = amplitude * np.exp(1j * wavenumber * tau)
        positions = np.array([0., 0.2, 0.4])
        omega = amplitude ** 2 - wavenumber ** 2 + 6 * epsilon * wavenumber * amplitude ** 2
        exact = np.exp(-1j * omega * positions[:, None]) * initial[None, :]
        coarse, _ = integrate(initial, epsilon, length, n, 0.02, positions)
        fine, _ = integrate(initial, epsilon, length, n, 0.01, positions)
        coarse_error = metric(coarse, exact, length)['l2']
        fine_error = metric(fine, exact, length)['l2']
        self.assertLess(fine_error, coarse_error / 10)
        self.assertLess(fine_error, 1e-6)
        np.testing.assert_allclose(fine[0], initial, rtol=0, atol=2e-15)


@unittest.skipIf(torch is None, 'PyTorch is an optional pilot dependency')
class PilotCheckpointTests(unittest.TestCase):
    @staticmethod
    def model_and_optimizer():
        model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Tanh(), torch.nn.Linear(5, 2)).double()
        return model, torch.optim.Adam(model.parameters(), lr=0.007)

    @staticmethod
    def next_step(model, optimizer):
        x = torch.randn(7, 3, dtype=torch.float64)
        target = torch.randn(7, 2, dtype=torch.float64)
        optimizer.zero_grad()
        loss = (model(x) - target).square().mean()
        loss.backward()
        optimizer.step()
        return loss.detach().clone(), x, target

    def test_resume_restores_optimizer_rng_and_next_update(self):
        torch.manual_seed(739)
        model, optimizer = self.model_and_optimizer()
        for _ in range(3):
            self.next_step(model, optimizer)
        config = {'model_id': 'water-wave-dysthe-spatial-v1', 'seed': 739}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pt'
            save_checkpoint(path, model, optimizer, 3, 0.4, 1.2, config, [{'epoch': 3}])
            self.assertTrue(path.is_file())
            self.assertFalse(path.with_suffix('.tmp').exists())
            expected_loss, expected_x, expected_target = self.next_step(model, optimizer)
            expected_state = copy.deepcopy(model.state_dict())
            expected_optimizer = copy.deepcopy(optimizer.state_dict())
            expected_next_random = torch.rand(8)

            # Construction and deliberate RNG use alter the current process;
            # restoring must recover the saved stream, not just model weights.
            resumed, resumed_optimizer = self.model_and_optimizer()
            torch.rand(31)
            saved = restore_checkpoint(path, resumed, resumed_optimizer, config, 'cpu')
            self.assertEqual(saved['epoch'], 3)
            actual_loss, actual_x, actual_target = self.next_step(resumed, resumed_optimizer)
            for actual, expected in ((actual_x, expected_x), (actual_target, expected_target),
                                     (actual_loss, expected_loss), (torch.rand(8), expected_next_random)):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for key, expected in expected_state.items():
                torch.testing.assert_close(resumed.state_dict()[key], expected, rtol=0, atol=0)
            actual_optimizer = resumed_optimizer.state_dict()
            self.assertEqual(actual_optimizer['param_groups'], expected_optimizer['param_groups'])
            for parameter, state in expected_optimizer['state'].items():
                for key, expected in state.items():
                    actual = actual_optimizer['state'][parameter][key]
                    if torch.is_tensor(expected):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    else:
                        self.assertEqual(actual, expected)
            with self.assertRaisesRegex(ValueError, 'configuration differs'):
                restore_checkpoint(path, resumed, resumed_optimizer, {**config, 'seed': 740}, 'cpu')


if __name__ == '__main__':
    unittest.main()

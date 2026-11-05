"""Execution checks for the bounded pilot, not trained accuracy claims."""

import io
import unittest

import torch

from dysthe_core.torch_spatial import project
from dysthe_learning.pilot_models import (
    ContinuousFNO, SpectralPINN, rollout, value_and_distance_derivative,
)


class PilotModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(137)
        self.inlet = torch.randn(2, 64, dtype=torch.complex128)
        self.distance = torch.tensor([0.03, 0.07], dtype=torch.float64)

    def test_fno_hard_projected_inlet_and_output_bandwidth(self):
        model = ContinuousFNO(40, width=4, modes=3, depth=2, max_mode=12)
        torch.testing.assert_close(model(self.inlet, 0.025, 0.0), project(self.inlet, 12))
        value = model(self.inlet, torch.tensor([0.025, 0.03]), self.distance)
        torch.testing.assert_close(value, project(value, 12), atol=1e-12, rtol=1e-12)
        trajectory = rollout(model, self.inlet, 0.025, 0.1, 2)
        self.assertEqual(trajectory.shape, (2, 3, 64))
        torch.testing.assert_close(trajectory[:, 1], model(trajectory[:, 0], 0.025, 0.1))

    def test_fno_jvp_finite_difference_and_parameter_gradient(self):
        model = ContinuousFNO(40, width=4, modes=3, depth=2, max_mode=12)
        predict = lambda s: model(self.inlet, 0.025, s)
        value, derivative = value_and_distance_derivative(predict, self.distance)
        h = 1e-6
        finite_difference = (predict(self.distance + h) - predict(self.distance - h)) / (2 * h)
        torch.testing.assert_close(derivative, finite_difference, atol=2e-9, rtol=2e-7)
        (derivative.abs().square().mean() + value.abs().square().mean()).backward()
        gradients = [p.grad for p in model.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
        self.assertTrue(any(torch.count_nonzero(g) > 0 for g in gradients))

    def test_fno_checkpoint_and_float_precision(self):
        model = ContinuousFNO(40, width=4, modes=3, depth=1, max_mode=12).float()
        inlet = self.inlet.to(torch.complex64)
        expected = model(inlet, 0.025, self.distance.float())
        self.assertEqual(expected.dtype, torch.complex64)
        stream = io.BytesIO()
        torch.save({"config": model.config, "state_dict": model.state_dict()}, stream)
        stream.seek(0)
        checkpoint = torch.load(stream, weights_only=True)
        restored = ContinuousFNO(**checkpoint["config"]).float()
        restored.load_state_dict(checkpoint["state_dict"])
        torch.testing.assert_close(expected, restored(inlet, 0.025, self.distance.float()), rtol=0, atol=0)

    def test_spectral_pinn_initial_bandwidth_and_jvp(self):
        model = SpectralPINN(self.inlet[0], 40, max_mode=12, width=8, depth=2)
        torch.testing.assert_close(model(0.0)[0], project(self.inlet[0], 12))
        value, derivative = value_and_distance_derivative(model, self.distance)
        torch.testing.assert_close(value, project(value, 12), atol=1e-12, rtol=1e-12)
        h = 1e-6
        fd = (model(self.distance + h) - model(self.distance - h)) / (2 * h)
        torch.testing.assert_close(derivative, fd, atol=2e-9, rtol=2e-7)
        derivative.abs().square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_spectral_pinn_checkpoint_float_and_single_inlet_scope(self):
        model = SpectralPINN(self.inlet[0], 40, max_mode=12, width=8).float()
        expected = model(self.distance.float())
        self.assertEqual(expected.dtype, torch.complex64)
        stream = io.BytesIO()
        torch.save({"config": model.config, "state_dict": model.state_dict()}, stream)
        stream.seek(0)
        checkpoint = torch.load(stream, weights_only=True)
        restored = SpectralPINN(torch.zeros(64, dtype=torch.complex64), **checkpoint["config"])
        restored.load_state_dict(checkpoint["state_dict"])
        torch.testing.assert_close(expected, restored(self.distance.float()), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "one complex inlet"):
            SpectralPINN(self.inlet, 40, max_mode=12)

    def test_rejects_invalid_shapes_and_nyquist_bandwidth(self):
        model = ContinuousFNO(40, width=4, modes=3, depth=1, max_mode=32)
        with self.assertRaisesRegex(ValueError, "max_mode"):
            model(self.inlet, 0.025, self.distance)
        model = ContinuousFNO(40, width=4, modes=3, depth=1, max_mode=12)
        with self.assertRaisesRegex(ValueError, "epsilon"):
            model(self.inlet, torch.ones(3), self.distance)
        with self.assertRaisesRegex(ValueError, "complex tensor"):
            model(self.inlet.real, 0.025, self.distance)


if __name__ == "__main__":
    unittest.main()

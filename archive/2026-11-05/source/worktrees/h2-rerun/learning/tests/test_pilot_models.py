"""Execution checks for the bounded pilot, not trained accuracy claims."""

import io
import math
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

    def test_spectral_scaling_is_invertible_and_preserves_the_initial_field(self):
        unscaled = SpectralPINN(self.inlet[0], 40, max_mode=12, width=8, depth=2)
        scaled = SpectralPINN(self.inlet[0], 40, max_mode=12, width=8, depth=2,
                              coefficient_scaling="inverse_1_plus_k2")
        scaled.network.load_state_dict(unscaled.network.state_dict())
        # Compensating the last affine map recovers exactly the same functions:
        # no Fourier mode or nonlinear part of the PDE has been discarded.
        modes = torch.arange(-12, 13, dtype=torch.float64)
        inverse_scale = (1 + (2 * math.pi * modes / 40).square()).repeat_interleave(2)
        with torch.no_grad():
            scaled.network[-1].weight.mul_(inverse_scale[:, None])
            scaled.network[-1].bias.mul_(inverse_scale)
        torch.testing.assert_close(scaled(self.distance), unscaled(self.distance), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(scaled(0.0)[0], project(self.inlet[0], 12))
        value, derivative = value_and_distance_derivative(scaled, self.distance)
        h = 1e-6
        fd = (scaled(self.distance + h) - scaled(self.distance - h)) / (2 * h)
        torch.testing.assert_close(derivative, fd, atol=2e-9, rtol=2e-7)
        (derivative.abs().square().mean() + value.abs().square().mean()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in scaled.parameters()))

    def test_spectral_scaling_uses_physical_wavenumbers_after_precision_conversion(self):
        for length in (20., 40.):
            with self.subTest(length=length):
                model = SpectralPINN(torch.zeros(64, dtype=torch.complex128), length,
                                      max_mode=12, width=4, depth=1,
                                      coefficient_scaling="inverse_1_plus_k2").float().double()
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.zero_()
                    model.network[-1].bias[2 * (12 + 3)] = 1.
                coefficients = torch.fft.fft(model(.2)[0], norm="forward")
                expected = .2 / (1 + (2 * math.pi * 3 / length)**2)
                torch.testing.assert_close(coefficients[3], torch.tensor(expected, dtype=torch.complex128),
                                           rtol=1e-13, atol=1e-13)
                self.assertEqual(model.correction_modes.dtype, torch.int64)

    def test_spectral_scaling_checkpoint_configuration_and_legacy_loading(self):
        legacy = SpectralPINN(self.inlet[0], 40, max_mode=12, width=8, depth=2)
        legacy_config = {k: v for k, v in legacy.config.items() if k != 'coefficient_scaling'}
        restored_legacy = SpectralPINN(torch.zeros_like(self.inlet[0]), **legacy_config)
        restored_legacy.load_state_dict(legacy.state_dict(), strict=True)
        torch.testing.assert_close(restored_legacy(self.distance), legacy(self.distance), rtol=0, atol=0)
        scaled = SpectralPINN(self.inlet[0], 40, max_mode=12, width=8, depth=2,
                              coefficient_scaling='inverse_1_plus_k2').float()
        stream = io.BytesIO()
        torch.save({'config': scaled.config, 'state_dict': scaled.state_dict()}, stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=True)
        restored = SpectralPINN(torch.zeros(64, dtype=torch.complex64), **saved['config'])
        restored.load_state_dict(saved['state_dict'], strict=True)
        torch.testing.assert_close(restored(self.distance.float()), scaled(self.distance.float()), rtol=0, atol=0)
        with self.assertRaisesRegex(RuntimeError, '_coefficient_scaling_code'):
            legacy.load_state_dict(scaled.state_dict(), strict=True)
        with self.assertRaisesRegex(RuntimeError, '_coefficient_scaling_code'):
            scaled.load_state_dict(legacy.state_dict(), strict=True)
        with self.assertRaisesRegex(ValueError, 'coefficient_scaling'):
            SpectralPINN(self.inlet[0], 40, coefficient_scaling='filter')

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

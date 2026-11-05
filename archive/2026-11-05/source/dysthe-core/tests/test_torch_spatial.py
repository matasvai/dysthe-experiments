"""CPU double-precision qualification of the optional PyTorch physics adapter."""

import math
import unittest

import numpy as np
from scipy.signal import resample as scipy_resample
import torch

from dysthe_core import torch_spatial as ts
from dysthe_core.reference.spatial import SpatialDysthe


def grid(n, length=2 * math.pi):
    return torch.arange(n, dtype=torch.float64) * (length / n) - length / 2


def packet(tau):
    return .35 + (.2 - .1j) * torch.exp(-2j * tau) + (-.15 + .06j) * torch.exp(5j * tau)


class TorchSpatialTests(unittest.TestCase):
    def test_derivative_and_mean_flow_signs(self):
        tau = grid(64)
        for mode in (-7, 0, 5):
            u = torch.exp(1j * mode * tau)
            torch.testing.assert_close(ts.derivative(u, 2 * math.pi), 1j * mode * u,
                                       rtol=1e-12, atol=4e-14)
            torch.testing.assert_close(ts.derivative(u, 2 * math.pi, 2), -mode**2 * u,
                                       rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(ts.abs_derivative(u, 2 * math.pi), abs(mode) * u,
                                       rtol=1e-12, atol=4e-14)
        rho = 2 + torch.cos(3 * tau) + .2 * torch.sin(5 * tau)
        torch.testing.assert_close(ts.abs_derivative(rho, 2 * math.pi),
                                   3 * torch.cos(3 * tau) + torch.sin(5 * tau),
                                   rtol=1e-12, atol=5e-14)

    def test_plane_wave_rhs_and_linear_flow(self):
        length, amplitude, mode, eps = 2 * math.pi, .6, 3, .04
        u = amplitude * torch.exp(1j * mode * grid(128))
        omega = mode**2 - amplitude**2 - 6 * eps * amplitude**2 * mode
        torch.testing.assert_close(ts.rhs(u, eps, length), 1j * omega * u,
                                   rtol=1e-12, atol=2e-12)
        torch.testing.assert_close(ts.linear_flow(u, .3, length),
                                   u * np.exp(1j * mode**2 * .3),
                                   rtol=1e-12, atol=2e-14)

    def test_batched_retained_rhs_matches_numpy_reference(self):
        n, length = 128, 18.
        epsilons = torch.tensor([0., .025, .07], dtype=torch.float64)
        model = SpatialDysthe(n, length, 0.)
        modes = np.array([-29, -11, 0, 27])
        coefficients = np.array([.02 + .01j, .2 - .1j, .35, -.08 + .04j])
        base = np.sum(coefficients[:, None]
                      * np.exp(2j * np.pi * modes[:, None] * model.tau / length), axis=0)
        fields = np.stack([(1 + .1j * j) * base for j in range(3)])
        expected = []
        for u, eps in zip(fields, epsilons.tolist()):
            m = SpatialDysthe(n, length, eps)
            expected.append(np.fft.ifft(m.rhs_hat(np.fft.fft(u))))
        actual = ts.rhs(torch.from_numpy(fields), epsilons, length, project=True)
        np.testing.assert_allclose(actual.numpy(), np.stack(expected), atol=2e-13, rtol=2e-12)

    def test_full_rhs_against_independent_fourier_convolution(self):
        tau = grid(96)
        coeff = {-7: .08 + .03j, -2: .2 - .1j, 0: .35, 5: -.15 + .06j}
        eps = .07
        rho = {}
        for p, a in coeff.items():
            for q, b in coeff.items():
                rho[p - q] = rho.get(p - q, 0) + a * complex(b).conjugate()
        out = {k: 1j * k * k * a for k, a in coeff.items()}
        for r, b in rho.items():
            for k, a in coeff.items():
                j = r + k
                out[j] = out.get(j, 0) - 1j * b * a - eps * (8j * k * b * a + 2j * abs(r) * b * a)
        for p, a in coeff.items():
            for q, b in coeff.items():
                for r, c in coeff.items():
                    j = p + q - r
                    out[j] = out.get(j, 0) + 2j * eps * r * a * b * complex(c).conjugate()
        u = sum(a * torch.exp(1j * k * tau) for k, a in coeff.items())
        expected = sum(a * torch.exp(1j * k * tau) for k, a in out.items())
        torch.testing.assert_close(ts.rhs(u, eps, 2 * math.pi), expected,
                                   rtol=3e-12, atol=8e-13)
        self.assertGreater(float(torch.linalg.vector_norm(ts.abs_derivative(u.abs().square(), 2 * math.pi))), .1)

    def test_projection_is_explicit_and_inclusive(self):
        tau = grid(64)
        u = .5 + .2 * torch.exp(12j * tau)
        full = ts.rhs(u, .03, 2 * math.pi)
        projected = ts.rhs(u, .03, 2 * math.pi, project=True)
        torch.testing.assert_close(projected, ts.project(full, 15), rtol=1e-12, atol=1e-12)
        self.assertGreater(float(torch.linalg.vector_norm(full - projected)), .01)
        torch.testing.assert_close(ts.project(u, 12), u, rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(ts.project(u, 11), torch.full_like(u, .5), rtol=1e-12, atol=1e-14)

    def test_linear_flow_batch_semigroup_and_distance_jvp(self):
        u = torch.stack([packet(grid(64)), .7j * packet(grid(64))])
        s = torch.tensor([.2, -.1], dtype=torch.float64)
        evolved, tangent = torch.func.jvp(lambda distance: ts.linear_flow(u, distance, 2 * math.pi),
                                          (s,), (torch.ones_like(s),))
        torch.testing.assert_close(tangent, -1j * ts.derivative(evolved, 2 * math.pi, 2),
                                   rtol=1e-11, atol=2e-12)
        torch.testing.assert_close(ts.linear_flow(evolved, -.13, 2 * math.pi),
                                   ts.linear_flow(u, s - .13, 2 * math.pi),
                                   rtol=1e-12, atol=3e-14)

    def test_resample_preserves_centered_grid_fourier_amplitudes(self):
        def field(n):
            tau = grid(n, 11.)
            return .4 + .2j * torch.exp(-6j * math.pi * tau / 11.) + .1 * torch.exp(4j * math.pi * tau / 11.)
        for n in (33, 96):
            torch.testing.assert_close(ts.resample(field(64), n), field(n), rtol=1e-12, atol=2e-14)

    def test_resample_matches_scipy_including_nyquist_and_batches(self):
        rng = np.random.default_rng(53)
        for old_n, n in ((8, 15), (9, 16), (8, 16), (9, 5), (10, 6), (1, 4), (4, 1)):
            u = rng.normal(size=(2, 3, old_n)) + 1j * rng.normal(size=(2, 3, old_n))
            actual = ts.resample(torch.from_numpy(u), n)
            np.testing.assert_allclose(actual.numpy(), scipy_resample(u, n, axis=-1), atol=2e-14, rtol=1e-12)

    def test_rhs_and_resample_have_correct_autograd(self):
        generator = torch.Generator().manual_seed(7)
        u = (.1 * torch.randn(2, 8, dtype=torch.complex128, generator=generator)).requires_grad_()
        eps = torch.tensor([.025, .04], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda f, e: ts.rhs(f, e, 13.), (u, eps),
                                                 eps=1e-6, atol=2e-7, rtol=2e-5))
        self.assertTrue(torch.autograd.gradcheck(lambda f: ts.resample(ts.resample(f, 15), 6),
                                                 (u,), eps=1e-6, atol=2e-7, rtol=2e-5))

    def test_float_and_complex_precision_are_preserved(self):
        u = packet(grid(32)).to(torch.complex64)
        for result in (ts.rhs(u, .025, 7.), ts.linear_flow(u, .1, 7.),
                       ts.project(u, 5), ts.resample(u, 48), ts.derivative(u, 7.)):
            self.assertEqual(result.dtype, torch.complex64)
            self.assertEqual(result.device, u.device)
        self.assertEqual(ts.abs_derivative(u.real, 7.).dtype, torch.float32)

    def test_rejects_incompatible_parameters(self):
        u = packet(grid(32))
        for length in (0., -1., float("nan"), True):
            with self.assertRaises(ValueError):
                ts.rhs(u, .025, length)
        with self.assertRaises(TypeError):
            ts.rhs(u.real, .025, 7.)
        with self.assertRaises(ValueError):
            ts.rhs(u, -.025, 7.)
        with self.assertRaises(ValueError):
            ts.rhs(u, torch.tensor(.025j), 7.)
        with self.assertRaises(ValueError):
            ts.linear_flow(u.repeat(2, 1), torch.ones(3), 7.)
        for max_mode in (-1, True, 17):
            with self.assertRaises(ValueError):
                ts.project(u, max_mode)
        with self.assertRaises(ValueError):
            ts.resample(u, 0)


if __name__ == "__main__":
    unittest.main()

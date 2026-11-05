import unittest

import numpy as np

from dysthe_core.reference.spatial import SpatialDysthe, nls_profile, translate
from dysthe_core.reference.validation import (
    align_to_reference, full_rhs_defect, invariant_directional_rates,
    perturbation_response,
)


class ValidationTests(unittest.TestCase):
    def test_isolated_linear_symbol_and_integrating_factor(self):
        class LinearControl(SpatialDysthe):
            def nonlinear_hat(self, spectrum):
                return np.zeros_like(spectrum)

        m = LinearControl(128, 2*np.pi, .025)
        f = .3*np.exp(5j*m.tau)+.2j*np.exp(-7j*m.tau)
        rhs = np.fft.ifft(m.rhs_hat(np.fft.fft(f)))
        expected = 25j*.3*np.exp(5j*m.tau)-49*.2*np.exp(-7j*m.tau)
        np.testing.assert_allclose(rhs, expected, atol=3e-12)
        v = np.fft.fft(f); h = .37; step = m.stepper(h)
        for _ in range(10):
            v = step(v)
        exact = .3*np.exp(5j*m.tau+25j*10*h)+.2j*np.exp(-7j*m.tau+49j*10*h)
        np.testing.assert_allclose(np.fft.ifft(v), exact, atol=3e-14)

    def test_symmetry_alignment_reaches_roundoff_for_weak_signals(self):
        m = SpatialDysthe(1024, 80., .025)
        f = m.project(nls_profile(m.tau))
        for shift, phase in [(1.371928, .6342), (2.3e-8, -1.7e-8)]:
            u = translate(m, f, shift)*np.exp(1j*phase)
            aligned, fit = align_to_reference(m, u, f)
            self.assertLess(fit['shape_relative_l2'], 2e-14)
            self.assertLess(np.linalg.norm(aligned-f)/np.linalg.norm(f), 2e-14)
            self.assertAlmostEqual(fit['translation'], shift, delta=2e-14)
            self.assertAlmostEqual(fit['global_phase'], phase, delta=2e-14)

    def test_true_amplitude_deformation_is_not_fitted_away(self):
        m = SpatialDysthe(1024, 80., .025)
        f = m.project(nls_profile(m.tau))
        u = translate(m, 1.0001*f, .41)*np.exp(-.27j)
        aligned, fit = perturbation_response(m, u, f)
        self.assertAlmostEqual(fit['shape_relative_l2'], 1e-4, delta=2e-14)
        np.testing.assert_allclose(aligned, 1.0001*f, atol=2e-14)
        self.assertGreater(fit['raw_relative_l2'], .1)
        self.assertEqual(fit['amplitude_scaling'], 1.)

    def test_arbitrary_target_matches_traveling_profile_residual(self):
        m = SpatialDysthe(128, 2*np.pi, .03)
        f = m.project(.5+.2*np.exp(23j*m.tau)+.1j*np.exp(-17j*m.tau))
        mu, speed = 2., .7
        du = np.fft.ifft(1j*m.k*np.fft.fft(f))
        measured = full_rhs_defect(m, f, -1j*mu*f-speed*du)
        expected = m.profile_residual(f, mu, speed)
        for key in ['projected_relative_l2', 'full_periodic_relative_l2',
                    'discarded_rhs_relative_l2']:
            self.assertAlmostEqual(measured[key], expected[key], delta=5e-13)
        defect = full_rhs_defect(m, f)
        self.assertGreater(defect['full_periodic_relative_l2'], 1e-5)
        self.assertEqual(defect['projected_relative_l2'], 0.)
        self.assertEqual(defect['normalization'], 'full_rhs_l2')

    def test_continuum_rates_distinguish_projection_error(self):
        m = SpatialDysthe(128, 2*np.pi, .07)
        coeff = {-29: .08+.03j, -17: .2-.1j, 0: .35, 23: -.15+.06j}
        f = sum(a*np.exp(1j*k*m.tau) for k,a in coeff.items())
        full = invariant_directional_rates(m, f)
        projected = invariant_directional_rates(m, f, projected=True)
        self.assertLess(max(abs(v) for v in full.values()), 1e-8)
        self.assertLess(abs(projected['action']), 1e-10)
        self.assertGreater(abs(projected['energy']), 1e-5)


if __name__ == '__main__':
    unittest.main()

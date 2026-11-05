import unittest
import numpy as np
from scipy.signal import resample
from dysthe_core.reference.spatial import SpatialDysthe, nls_profile, orbital_fit, translate


class ReferenceTests(unittest.TestCase):
    def test_exact_spatial_nls_traveling_control(self):
        m = SpatialDysthe(2048, 80., 0.)
        mu, speed = 2., 1.
        f = m.project(nls_profile(m.tau, mu, speed))
        self.assertLess(m.profile_residual(f, mu, speed)['full_periodic_relative_l2'], 1e-10)
        errors = []
        for h in (.01, .005):
            v = np.fft.fft(f); step = m.stepper(h)
            for _ in range(round(.2 / h)):
                v = step(v)
            exact = translate(m, f, speed * .2) * np.exp(-1j * mu * .2)
            errors.append(np.linalg.norm(np.fft.ifft(v) - exact) / np.linalg.norm(f))
        self.assertLess(errors[1], errors[0] / 10)
        self.assertLess(errors[1], 1e-7)

    def test_nonzero_wavenumber_plane_wave(self):
        m = SpatialDysthe(128, 2 * np.pi, .04)
        amplitude, k = .6, 3
        f = amplitude * np.exp(1j * k * m.tau)
        omega = amplitude ** 2 - k ** 2 + 6 * m.epsilon * k * amplitude ** 2
        rhs = np.fft.ifft(m.rhs_hat(np.fft.fft(f)))
        np.testing.assert_allclose(rhs, -1j * omega * f, atol=2e-12)
        v = np.fft.fft(f); step = m.stepper(.001)
        for _ in range(100):
            v = step(v)
        np.testing.assert_allclose(np.fft.ifft(v), f * np.exp(-1j * omega * .1), atol=1e-10)

    def test_independent_multimode_mean_flow(self):
        m = SpatialDysthe(64, 2 * np.pi, .07)
        # Algebraic Fourier convolution, including the nonconstant mean flow.
        coeff = {-7: .08+.03j, -2: .2-.1j, 0: .35, 5: -.15+.06j}
        rho = {}
        for p, a in coeff.items():
            for q, b in coeff.items():
                rho[p-q] = rho.get(p-q, 0) + a*b.conjugate()
        out = {k: 1j*k*k*a for k,a in coeff.items()}
        for r, b in rho.items():
            for k,a in coeff.items():
                j = r+k
                out[j] = out.get(j,0) - 1j*b*a - m.epsilon*(8j*k*b*a + 2j*abs(r)*b*a)
        for p,a in coeff.items():
            for q,b in coeff.items():
                for r,c in coeff.items():
                    j = p+q-r
                    out[j] = out.get(j,0) + 2j*m.epsilon*r*a*b*c.conjugate()
        fine_tau = np.arange(8*m.n) * m.length/(8*m.n) - m.length/2
        full = sum(a*np.exp(1j*k*fine_tau) for k,a in out.items())
        # Projection is applied to output only, matching the specified Galerkin ODE.
        expected = m.project(resample(full,m.n))
        f = sum(a*np.exp(1j*k*m.tau) for k,a in coeff.items())
        np.testing.assert_allclose(np.fft.ifft(m.rhs_hat(np.fft.fft(f))), expected, atol=3e-13)

    def test_full_residual_detects_discarded_cubic_output(self):
        m = SpatialDysthe(64, 2*np.pi, .03)
        f = .5 + .2*np.exp(12j*m.tau)
        r = m.profile_residual(f,2.,0.)
        self.assertGreater(r['discarded_rhs_relative_l2'], 1e-3)
        self.assertGreater(r['full_periodic_relative_l2'], r['projected_relative_l2'])

    def test_fit_keeps_raw_phase_and_translation_distinct(self):
        m = SpatialDysthe(512,80.,0.)
        f = m.project(nls_profile(m.tau))
        u = translate(m,f,1.37)*np.exp(.6j)
        fit = orbital_fit(m,u,f)
        self.assertLess(fit['orbital_relative_l2'],1e-8)
        self.assertAlmostEqual(fit['translation'],1.37,places=7)
        self.assertAlmostEqual(fit['global_phase'],.6,places=7)
        self.assertGreater(np.linalg.norm(u-f)/np.linalg.norm(f),.5)

    def test_padded_quadrature_rejects_unretained_input(self):
        m = SpatialDysthe(64,2*np.pi,.04)
        with self.assertRaises(ValueError):
            m.invariants(np.exp(20j*m.tau))

    def test_analytic_plane_wave_invariants(self):
        m=SpatialDysthe(64,2*np.pi,.04)
        a,k=.6,3; rho=a*a
        d=m.invariants(a*np.exp(1j*k*m.tau))
        self.assertAlmostEqual(d['action'],m.length*rho,places=12)
        self.assertAlmostEqual(d['momentum'],m.length*(2*k*rho-2*m.epsilon*rho**2),places=12)
        self.assertAlmostEqual(d['energy'],m.length*(k*k*rho-.5*rho**2+(10/3)*m.epsilon**2*rho**3-5*m.epsilon*k*rho**2),places=12)


if __name__ == '__main__':
    unittest.main()

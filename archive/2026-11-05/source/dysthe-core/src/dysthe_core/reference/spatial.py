"""Periodic free-surface spatial Dysthe only; no canonical gauge is imposed.

Water-wave branch of legacy nonlinear/solver.py, with independent full-RHS
qualification and padded invariant quadrature. xi is distance; tau is retarded
time. Galerkin accuracy, whole-line accuracy and physical validity are separate.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import resample

def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return float(value)


@dataclass
class SpatialDysthe:
    n: int
    length: float
    epsilon: float

    def __post_init__(self):
        if type(self.n) is not int or self.n < 16 or self.n % 4:
            raise ValueError('n must be a multiple of four >= 16')
        self.length = _finite(self.length, 'length')
        self.epsilon = _finite(self.epsilon, 'epsilon')
        if self.length <= 0 or self.epsilon < 0:
            raise ValueError('length must be positive and epsilon nonnegative')
        self.dtau = self.length / self.n
        self.tau = np.arange(self.n) * self.dtau - self.length / 2
        self.k = 2 * np.pi * np.fft.fftfreq(self.n, d=self.dtau)
        self.mask = abs(np.fft.fftfreq(self.n) * self.n) < self.n / 4
        self.linear = 1j * self.k ** 2

    def field(self, u):
        u = np.asarray(u, dtype=np.complex128)
        if u.shape != (self.n,) or not np.isfinite(u).all():
            raise ValueError('Expected one finite complex profile on the declared grid')
        return u

    def project(self, u):
        return np.fft.ifft(np.fft.fft(self.field(u)) * self.mask)

    def nonlinear_hat(self, v):
        u = np.fft.ifft(v)
        rho = abs(u) ** 2
        du = np.fft.ifft(1j * self.k * v)
        mean = np.fft.ifft(abs(self.k) * np.fft.fft(rho)).real
        return np.fft.fft(-1j * rho * u - self.epsilon * (
            8 * rho * du + 2 * u ** 2 * du.conj() + 2j * u * mean)) * self.mask

    def rhs_hat(self, v):
        return self.linear * v + self.nonlinear_hat(v)

    def stepper(self, dxi):
        dxi = _finite(dxi, 'dxi')
        if dxi <= 0:
            raise ValueError('dxi must be positive')
        E = np.exp(dxi * self.linear)
        H = np.exp(dxi * self.linear / 2)

        def step(v):
            a = dxi * self.nonlinear_hat(v)
            b = dxi * H.conj() * self.nonlinear_hat(H * (v + a / 2))
            c = dxi * H.conj() * self.nonlinear_hat(H * (v + b / 2))
            d = dxi * E.conj() * self.nonlinear_hat(E * (v + c))
            return E * (v + (a + 2 * b + 2 * c + d) / 6) * self.mask
        return step

    def invariants(self, u):
        """Continuum functionals, exactly padded for retained polynomial degrees.

        These are not all exact invariants of the projected ODE or RK4 map.
        2*n > 6*K with K < n/4 integrates the sixth-degree term without aliasing.
        """
        u = self.field(u)
        v = np.fft.fft(u)
        if np.linalg.norm(v[~self.mask]) > 1e-12 * np.linalg.norm(v):
            raise ValueError('Padded invariant quadrature requires a retained-band field')
        u = resample(u, 2 * self.n)
        k = 2 * np.pi * np.fft.fftfreq(len(u), d=self.length / len(u))
        rho = abs(u) ** 2
        du = np.fft.ifft(1j * k * np.fft.fft(u))
        mean = np.fft.ifft(abs(k) * np.fft.fft(rho)).real
        current = np.imag(du * u.conj())
        e = self.epsilon
        cell = self.length / len(u)
        return {
            'action': float(cell * np.sum(rho)),
            'energy': float(cell * np.sum(abs(du) ** 2 - .5 * rho ** 2
                + (10 / 3) * e ** 2 * rho ** 3 - 5 * e * rho * current - e * rho * mean)),
            'momentum': float(cell * np.sum(2 * current - 2 * e * rho ** 2)),
        }

    def profile_residual(self, u, mu, speed):
        """Test the full periodic PDE on fourfold Fourier interpolation.

        Cubic output bandwidth is <=3*K; 4*n safely resolves it. This evaluates
        the trigonometric interpolant, not an independently proved whole-line wave.
        """
        u = self.field(u)
        mu = _finite(mu, 'mu')
        speed = _finite(speed, 'speed')
        fine = resample(u, 4 * self.n)
        k = 2 * np.pi * np.fft.fftfreq(len(fine), d=self.length / len(fine))
        v = np.fft.fft(fine)
        du = np.fft.ifft(1j * k * v)
        rho = abs(fine) ** 2
        mean = np.fft.ifft(abs(k) * np.fft.fft(rho)).real
        N = -1j * rho * fine - self.epsilon * (
            8 * rho * du + 2 * fine ** 2 * du.conj() + 2j * fine * mean)
        target = (-1j * mu - 1j * speed * k) * v
        denominator = np.linalg.norm(target)
        if denominator <= np.finfo(float).tiny:
            raise ValueError('Nonzero traveling reference required')
        defect = 1j * k ** 2 * v + np.fft.fft(N) - target
        retained = abs(np.fft.fftfreq(len(fine)) * len(fine)) < self.n / 4
        return {
            'projected_relative_l2': float(np.linalg.norm(defect[retained]) / denominator),
            'full_periodic_relative_l2': float(np.linalg.norm(defect) / denominator),
            'discarded_rhs_relative_l2': float(np.linalg.norm(defect[~retained]) / denominator),
            'oversampling_factor': 4,
        }


def nls_profile(tau, mu=2., speed=0.):
    nu = mu - speed ** 2 / 4
    if nu <= 0:
        raise ValueError('mu must exceed speed^2/4')
    return np.sqrt(2 * nu) / np.cosh(np.sqrt(nu) * tau) * np.exp(-.5j * speed * tau)


def solve_profile(model, mu, speed, tolerance=2e-11, max_iterations=12000):
    """Compute one conjugate-reflection-symmetric branch, not all waves.

    Petviashvili convergence is not dynamical stability. No symmetry is imposed
    during propagation. Requalification uses the full periodic RHS residual.
    """
    mu = _finite(mu, 'mu')
    speed = _finite(speed, 'speed')
    tolerance = _finite(tolerance, 'tolerance')
    if tolerance <= 0 or type(max_iterations) is not int or max_iterations < 1:
        raise ValueError('Positive tolerance and iteration count required')
    if mu <= speed ** 2 / 4:
        raise ValueError('Positive profile multiplier required')
    den = mu + model.k ** 2 + speed * model.k
    f = model.project(nls_profile(model.tau, mu, speed))
    rev = (-np.arange(model.n)) % model.n
    for iteration in range(max_iterations):
        v = np.fft.fft(f) * model.mask
        N = 1j * model.nonlinear_hat(v)
        residual = np.linalg.norm(den * v - N) / np.linalg.norm(den * v)
        if residual < tolerance:
            return f, {'iterations': iteration, 'symmetry_restricted': True,
                       **model.profile_residual(f, mu, speed)}
        ratio = np.vdot(v, den * v).real / np.vdot(v, N).real
        if not np.isfinite(ratio) or ratio <= 0:
            raise RuntimeError('Petviashvili ratio lost positivity')
        new = np.fft.ifft(ratio ** 1.5 * N / den)
        f = (new + new[rev].conj()) / 2
        if not np.isfinite(f).all():
            raise RuntimeError('Nonfinite profile')
    raise RuntimeError('Profile did not converge; this is not evidence of nonexistence')


def translate(model, u, shift):
    return np.fft.ifft(np.fft.fft(u) * np.exp(-1j * model.k * shift))


def orbital_fit(model, u, profile):
    """Search every discrete translation, then refine the best grid candidate.

    This is a practical periodic orbit fit, not a certified continuous global
    optimization. The fit preserves amplitude and branch parameters.
    """
    norm = np.linalg.norm(profile)
    if norm == 0:
        raise ValueError('Nonzero reference required')
    vf = np.fft.fft(profile)
    correlation = np.fft.ifft(np.fft.fft(u) * vf.conj())
    shift = int(np.argmax(abs(correlation))) * model.dtau
    if shift >= model.length / 2:
        shift -= model.length

    def evaluate(a):
        q = np.fft.ifft(vf * np.exp(-1j * model.k * a))
        phase = np.angle(np.vdot(q, u))
        return float(np.linalg.norm(u - q * np.exp(1j * phase)) / norm), float(phase)

    fit = minimize_scalar(lambda a: evaluate(a)[0], method='bounded',
        bounds=(shift - model.dtau, shift + model.dtau), options={'xatol': 1e-11})
    # Keep a grid-aligned exact solution if numerical refinement moves away.
    candidates = [(shift, *evaluate(shift)), (float(fit.x), *evaluate(fit.x))]
    a, error, phase = min(candidates, key=lambda x: x[1])
    a = (a + model.length / 2) % model.length - model.length / 2
    return {'orbital_relative_l2': error, 'translation': a, 'global_phase': phase}


def packet_diagnostics(model, u, core_radius=10.):
    rho = abs(u) ** 2
    total = float(np.sum(rho))
    if total == 0:
        raise ValueError('Nonzero packet required')
    peak_index = int(np.argmax(rho))
    center = model.tau[peak_index]
    distance = (model.tau - center + model.length / 2) % model.length - model.length / 2
    v = np.fft.fft(u)
    power = abs(v) ** 2
    modes = abs(np.fft.fftfreq(model.n) * model.n)
    rev = (-np.arange(model.n)) % model.n
    spectrum_total = float(np.sum(power))
    k_mean = float(np.sum(model.k * power) / spectrum_total)
    k_sigma = float(np.sqrt(np.sum((model.k - k_mean) ** 2 * power) / spectrum_total))
    return {
        **model.invariants(u),
        'peak': float(np.sqrt(rho[peak_index])),
        'peak_tau': float(center),
        'width_about_peak': float(np.sqrt(np.sum(distance ** 2 * rho) / total)),
        'core_tail_action_fraction': float(np.sum(rho[abs(distance) > core_radius]) / total),
        'boundary_action_fraction': float(np.sum(rho[abs(model.tau) > .45 * model.length]) / total),
        'retained_spectral_edge_action_fraction': float(np.sum(power[(modes >= .8 * model.n / 4) & model.mask]) / spectrum_total),
        'outside_retained_action_fraction': float(np.sum(power[~model.mask]) / spectrum_total),
        'raw_conjugate_reflection_defect': float(np.linalg.norm(u - u[rev].conj()) / np.linalg.norm(u)),
        'epsilon_peak': float(model.epsilon * np.sqrt(rho[peak_index])),
        'epsilon_spectral_sigma': model.epsilon * k_sigma,
        'epsilon_spectral_mean': model.epsilon * k_mean,
    }

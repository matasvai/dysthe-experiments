"""Independent periodic reference diagnostics for the approved water-wave model.

Orbit alignment removes only translation and constant phase. Its residual is a
distance in the symmetry quotient, not an instability criterion. All norms use
the declared periodic domain; no amplitude or branch-parameter fit is made.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import resample


def align_to_reference(model, field, reference):
    """Return the field aligned to reference and separate raw/shape metrics.

    Search all grid translations, refine the best candidate, and polish the
    correlation maximum using analytic Fourier derivatives. The last step
    avoids a scalar optimizer tolerance becoming a weak-perturbation signal.
    This is a practical periodic fit, not a certified continuous global search.
    Translation/phase describe field relative to reference, with the same sign
    convention as ``spatial.translate``. Phase is principal-valued in [-pi,pi].
    """
    field = model.field(field)
    reference = model.field(reference)
    norm = np.linalg.norm(reference)
    if norm == 0:
        raise ValueError('Nonzero reference required')
    vf, vu = np.fft.fft(reference), np.fft.fft(field)
    cross = vu * vf.conj()
    correlation = np.fft.ifft(cross)
    shift = int(np.argmax(abs(correlation))) * model.dtau
    if shift >= model.length / 2:
        shift -= model.length

    def coefficients(a):
        z = cross * np.exp(1j * model.k * a) / model.n
        return np.sum(z), np.sum(1j * model.k * z), np.sum(-model.k**2 * z)

    def score(a):
        return abs(coefficients(a)[0])**2

    bounds = (shift - model.dtau, shift + model.dtau)
    fit = minimize_scalar(lambda a: -score(a), method='bounded', bounds=bounds,
                          options={'xatol': 1e-13})
    candidates = [shift, float(fit.x)]
    polished = max(candidates, key=score)
    for _ in range(12):
        c, dc, ddc = coefficients(polished)
        first = 2 * np.real(dc * c.conj())
        second = 2 * np.real(ddc * c.conj() + dc * dc.conj())
        if not np.isfinite(second) or second >= 0:
            break
        increment = -first / second
        if abs(increment) > model.dtau:
            break
        proposal = polished + increment
        if not bounds[0] <= proposal <= bounds[1]:
            break
        polished = proposal
        if abs(increment) <= 4 * np.finfo(float).eps * max(1., abs(polished)):
            break
    candidates.append(polished)

    def evaluate(a):
        q = np.fft.ifft(vf * np.exp(-1j * model.k * a))
        phase = np.angle(np.vdot(q, field))
        error = np.linalg.norm(field - q * np.exp(1j * phase))
        return float(error), float(phase)

    # Compare actual residuals, whose small differences can be lost in |c|^2.
    a, (error, phase) = min(((a, evaluate(a)) for a in candidates),
                           key=lambda entry: entry[1][0])
    aligned = np.fft.ifft(vu * np.exp(1j * model.k * a)) * np.exp(-1j * phase)
    a = (a + model.length / 2) % model.length - model.length / 2
    return aligned, {
        'raw_relative_l2': float(np.linalg.norm(field-reference) / norm),
        'shape_relative_l2': error / norm,
        'translation': float(a),
        'global_phase': phase,
        'amplitude_scaling': 1.,
    }


def perturbation_response(model, perturbed, control):
    """Compare a perturbed trajectory with its matched propagated control.

    Returns aligned perturbed field and metrics. The shape L2 norm includes
    amplitude/width changes and radiation; translation and phase are reported
    separately. Signed response differences should be computed from the
    returned fields, never from differences of scalar norms.
    """
    perturbed, control = model.field(perturbed), model.field(control)
    aligned, metrics = align_to_reference(model, perturbed, control)
    metrics['raw_l2'] = float(np.sqrt(model.dtau) * np.linalg.norm(perturbed-control))
    metrics['shape_l2'] = float(np.sqrt(model.dtau) * np.linalg.norm(aligned-control))
    return aligned, metrics


def _fine_rhs(model, field):
    field = model.field(field)
    fine = resample(field, 4 * model.n)
    k = 2 * np.pi * np.fft.fftfreq(len(fine), d=model.length/len(fine))
    v = np.fft.fft(fine)
    du = np.fft.ifft(1j * k * v)
    rho = abs(fine)**2
    mean = np.fft.ifft(abs(k) * np.fft.fft(rho)).real
    rhs = np.fft.ifft(1j * k**2 * v) - 1j * rho * fine - model.epsilon * (
        8 * rho * du + 2 * fine**2 * du.conj() + 2j * fine * mean)
    retained = abs(np.fft.fftfreq(len(fine)) * len(fine)) < model.n/4
    return fine, k, rhs, retained


def full_rhs_defect(model, field, target=None):
    """Audit full periodic RHS and discarded modes on fourfold interpolation.

    Without target, defect means full RHS minus its retained-band projection;
    ratios are normalized by full RHS L2. With target (a time derivative on the
    original grid), defect is RHS-target and ratios use target L2, matching
    ``profile_residual`` when target=-i*mu*field-speed*field_tau. This is an
    instantaneous trigonometric-interpolant defect, not a propagation bound.
    """
    _, _, rhs, retained = _fine_rhs(model, field)
    vrhs = np.fft.fft(rhs)
    if target is None:
        target_hat = vrhs * retained
        denominator = np.linalg.norm(vrhs)
        normalization = 'full_rhs_l2'
    else:
        target_hat = np.fft.fft(resample(model.field(target), 4*model.n))
        denominator = np.linalg.norm(target_hat)
        normalization = 'target_l2'
        if denominator == 0 and np.linalg.norm(vrhs) != 0:
            raise ValueError('Nonzero target required for relative defect')
    defect = vrhs - target_hat
    denominator = denominator if denominator else 1.
    cell = model.length / (4*model.n)
    return {
        'projected_relative_l2': float(np.linalg.norm(defect[retained])/denominator),
        'full_periodic_relative_l2': float(np.linalg.norm(defect)/denominator),
        'discarded_rhs_relative_l2': float(np.linalg.norm(vrhs[~retained])/denominator),
        'full_rhs_l2': float(np.sqrt(cell)*np.linalg.norm(rhs)),
        'normalization': normalization,
        'oversampling_factor': 4,
    }


def invariant_directional_rates(model, field, projected=False):
    """Differentiate continuum A/M/E along full or Galerkin RHS.

    Require a retained-band field, as for model.invariants. Fourier padding
    resolves the variational integrands. This independent
    diagnostic distinguishes invariant-formula errors from the legitimate
    loss of continuum E/M conservation under a finite Galerkin projection.
    Rates are absolute per unit xi; near-zero invariant values need no division.
    """
    field = model.field(field)
    spectrum = np.fft.fft(field)
    if np.linalg.norm(spectrum[~model.mask]) > 1e-12*np.linalg.norm(spectrum):
        raise ValueError('Invariant rates require a retained-band field')
    fine, k, rhs, retained = _fine_rhs(model, field)
    if projected:
        rhs = np.fft.ifft(np.fft.fft(rhs) * retained)
    derivative = lambda z: np.fft.ifft(1j*k*np.fft.fft(z))
    mean_flow = lambda z: np.fft.ifft(abs(k)*np.fft.fft(z)).real
    du, drhs = derivative(fine), derivative(rhs)
    rho = abs(fine)**2
    drho = 2*np.real(fine.conj()*rhs)
    current = np.imag(du*fine.conj())
    dcurrent = np.imag(drhs*fine.conj()+du*rhs.conj())
    e = model.epsilon
    integrands = {
        'action': drho,
        'momentum': 2*dcurrent - 4*e*rho*drho,
        'energy': 2*np.real(du.conj()*drhs) - rho*drho + 10*e**2*rho**2*drho
                  - 5*e*(drho*current+rho*dcurrent)
                  - e*(drho*mean_flow(rho)+rho*mean_flow(drho)),
    }
    return {name: float(model.length*np.mean(value))
            for name, value in integrands.items()}

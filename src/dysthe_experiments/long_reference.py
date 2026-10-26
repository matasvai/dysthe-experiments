"""Refined finite-distance experiments for the approved spatial water-wave model.

Every comparison uses complex fields with their original phase. The periodic
reference is numerical evidence, never a global-existence certificate. Raw
trajectories are ignored run artifacts; JSON records their hashes and identities.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.signal import find_peaks, resample

from dysthe_core.contracts import require_water_wave_model
from dysthe_core.reference.spatial import SpatialDysthe
from .cpu_pilot import integrate
from .pilot_sensitivity import _direction, sobolev_norm


LIMITATION = ("Sampled, two-level numerical refinement in periodic boxes; not a "
              "rigorous error bound, proof of smoothness/global existence, or "
              "validation of Dysthe against the full water-wave equations.")


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_digest(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode()
                          + array.tobytes()).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def relative_error(approximate, reference, length, order=1):
    """Relative complex H^order error, without phase/translation alignment."""
    approximate, reference = np.asarray(approximate), np.asarray(reference)
    if approximate.shape[-1] != reference.shape[-1]:
        approximate = resample(approximate, reference.shape[-1], axis=-1)
    denominator = sobolev_norm(reference, length, order)
    if np.any(denominator <= np.finfo(float).tiny):
        raise ValueError('Relative errors require nonzero reference profiles')
    return sobolev_norm(approximate - reference, length, order) / denominator


def compare_fields(approximate, reference, length):
    return {f'h{order}': float(np.max(relative_error(approximate, reference, length, order)))
            for order in range(3)}


def _positions(c, horizon):
    step = float(c['sample_step'])
    count = horizon / step
    if horizon <= 0 or step <= 0 or not np.isclose(count, round(count), atol=1e-10, rtol=1e-10):
        raise ValueError('Positive horizon must be an integer multiple of sample_step')
    # Independent h and 2h comparisons must actually use the declared steps.
    for h in (c['reference_step'], 2 * c['reference_step']):
        ratio = step / h
        if not np.isclose(ratio, round(ratio), atol=1e-10, rtol=1e-10) or ratio < 1:
            raise ValueError('sample_step must divide into both h and 2h steps exactly')
    return np.arange(round(count) + 1, dtype=float) * step


def _case_root(run, stage, record, horizon):
    name = record['case_id']
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('case_id must be a safe directory name')
    root = Path(run).resolve() / stage / name / ('xi-' + format(horizon, '.12g'))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _identity(c, record, u0, horizon):
    require_water_wave_model(c)
    require_water_wave_model(record)
    return {'config': c, 'record': record, 'u0_sha256': _array_digest(u0),
            'horizon': float(horizon), 'module_sha256': _digest(__file__),
            'integrator_sha256': hashlib.sha256(inspect.getsource(integrate).encode()).hexdigest(),
            'reference_sha256': _digest(inspect.getfile(SpatialDysthe)),
            'sensitivity_source_sha256': _digest(inspect.getfile(_direction)),
            'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                        'scipy': scipy.__version__}}


def _trajectory(root, label, identity, u0, epsilon, length, n, h, xi):
    """Cache each independent trajectory; reject stale or corrupted artifacts."""
    path = root / (label + '.npz')
    meta_path = root / (label + '.json')
    expected = dict(identity, role=label, trajectory_u0_sha256=_array_digest(u0),
                    epsilon=float(epsilon), length=float(length), n=int(n), step=float(h))
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta['identity'] != expected:
            raise ValueError(f'Cache identity differs: {meta_path}')
        if meta['status'] == 'numerical_failure':
            return None, meta
        if not path.exists() or _digest(path) != meta['sha256']:
            raise ValueError(f'Trajectory checksum mismatch: {path}')
        with np.load(path, allow_pickle=False) as archive:
            fields, cached_xi = archive['fields'], archive['xi']
        if (fields.shape != (len(xi), n) or not np.array_equal(cached_xi, xi)
                or not np.isfinite(fields).all()):
            raise ValueError(f'Invalid cached trajectory: {path}')
        return fields, meta
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            fields, seconds = integrate(u0, epsilon, length, n, h, xi)
    except (FloatingPointError, RuntimeError) as error:
        meta = {'identity': expected, 'status': 'numerical_failure',
                'message': str(error), 'path': str(path),
                'limitation': 'Integration failure is not evidence of nonexistence.'}
        _write(meta_path, meta)
        return None, meta
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, fields=fields, xi=xi)
    temporary.replace(path)
    meta = {'identity': expected, 'status': 'complete', 'path': str(path),
            'sha256': _digest(path), 'seconds': seconds, 'n': n, 'step': h,
            'length': length, 'epsilon': epsilon}
    _write(meta_path, meta)
    return fields, meta


def field_diagnostics(fields, xi, length, epsilon, output_modes=None, tail_fraction=.8):
    """Diagnostics on complete profiles; thresholded events are added by caller.

    Spectral k uses fft kernel exp(-ik*tau); signed asymmetry includes the zero
    mode in total power. Recurrence removes a single global phase, never a shift.
    Invariant quadrature resamples to a grid supporting all supplied modes.
    """
    # Evaluate diagnostics in double precision even for float32 neural output.
    # A float32 interpolation introduces new roundoff modes above the supplied
    # profile's support and violates the invariant quadrature's strict cutoff.
    fields = np.asarray(fields, dtype=np.complex128)
    xi = np.asarray(xi)
    if fields.ndim != 2 or len(xi) != len(fields) or not np.isfinite(fields).all():
        raise ValueError('Expected finite (distance,profile) fields and matching distances')
    n = fields.shape[-1]
    k = 2 * np.pi * np.fft.fftfreq(n, d=length / n)
    modes = abs(np.fft.fftfreq(n) * n)
    cutoff = float(n / 4 if output_modes is None else output_modes)
    if cutoff <= 0 or cutoff >= n / 2:
        raise ValueError('Diagnostic cutoff must be between zero and Nyquist')
    if not 0 < tail_fraction < 1:
        raise ValueError('Spectral tail fraction must be between zero and one')
    coeff = np.fft.fft(fields, axis=-1) / n
    power = abs(coeff)**2
    total = np.sum(power, axis=-1)
    if np.any(total <= np.finfo(float).tiny):
        raise ValueError('Diagnostics require nonzero fields')
    centroid = np.sum(k * power, axis=-1) / total
    width = np.sqrt(np.sum((k - centroid[:, None])**2 * power, axis=-1) / total)
    raw_second = np.sqrt(np.sum(k**2 * power, axis=-1) / total)
    peak = np.max(abs(fields), axis=-1)
    initial_peak = float(peak[0])
    norms = {f'h{s}': sobolev_norm(fields, length, s) for s in range(3)}
    correlation = np.sum(fields * fields[0].conj(), axis=-1)
    phase = np.angle(correlation)
    aligned = fields * np.exp(-1j * phase[:, None])
    recurrence = np.linalg.norm(aligned - fields[0], axis=-1) / np.linalg.norm(fields[0])
    raw_recurrence = np.linalg.norm(fields - fields[0], axis=-1) / np.linalg.norm(fields[0])
    # Accommodate neural profiles whose retained K can exceed the solver's n/4.
    invariant_n = 4 * (n // 2 + 1)
    invariant_model = SpatialDysthe(invariant_n, length, epsilon)
    invariants = [invariant_model.invariants(resample(u, invariant_n)) for u in fields]
    invariant_series = {key: np.array([v[key] for v in invariants])
                        for key in ('action', 'energy', 'momentum')}
    initial = {key: float(value[0]) for key, value in invariant_series.items()}
    relative_drift = {key: (abs(value - value[0]) / abs(value[0])
                           if abs(value[0]) > 1e-12 * max(initial['action'], 1.) else None)
                      for key, value in invariant_series.items()}
    tail = np.sum(power[:, modes >= tail_fraction * cutoff], axis=-1) / total
    outside_mask = modes >= cutoff if output_modes is None else modes > cutoff
    outside = np.sum(power[:, outside_mask], axis=-1) / total
    boundary = np.sum(abs(fields[:, np.abs(np.arange(n)*length/n-length/2) >= .45*length])**2,
                      axis=-1) / np.sum(abs(fields)**2, axis=-1)
    signed = (np.sum(power[:, k > 0], axis=-1) - np.sum(power[:, k < 0], axis=-1)) / total
    skew = np.divide(np.sum((k-centroid[:, None])**3*power, axis=-1) / total,
                     width**3, out=np.zeros_like(width), where=width > 1e-14)
    stations = []
    for j, distance in enumerate(xi):
        stations.append({'xi': float(distance), 'nonlinear_lengths_initial_peak': float(initial_peak**2*distance),
                         'peak': float(peak[j]), 'peak_ratio': float(peak[j]/initial_peak),
                         'norms': {key: float(value[j]) for key, value in norms.items()},
                         'h2_ratio': float(norms['h2'][j]/norms['h2'][0]),
                         'epsilon_peak': float(epsilon*peak[j]),
                         'spectral_centroid': float(centroid[j]), 'spectral_width': float(width[j]),
                         'spectral_rms_wavenumber': float(raw_second[j]),
                         'epsilon_spectral_width': float(epsilon*width[j]),
                         'epsilon_spectral_rms_wavenumber': float(epsilon*raw_second[j]),
                         'spectral_signed_power_asymmetry': float(signed[j]),
                         'spectral_skewness': float(skew[j]),
                         'spectral_tail_action_fraction': float(tail[j]),
                         'outside_declared_cutoff_action_fraction': float(outside[j]),
                         'boundary_action_fraction': float(boundary[j]),
                         'raw_recurrence_relative_l2': float(raw_recurrence[j]),
                         'phase_aligned_recurrence_relative_l2': float(recurrence[j]),
                         'recurrence_global_phase': float(phase[j]),
                         'invariants': invariants[j],
                         'invariant_absolute_drift': {key: float(abs(value[j]-value[0]))
                                                      for key, value in invariant_series.items()},
                         'invariant_relative_drift': {key: None if value is None else float(value[j])
                                                      for key, value in relative_drift.items()}})
    maxima = {key: max(row[key] for row in stations)
              for key in ('peak', 'h2_ratio', 'epsilon_peak', 'epsilon_spectral_width',
                          'epsilon_spectral_rms_wavenumber', 'spectral_tail_action_fraction',
                          'boundary_action_fraction')}
    maxima['action_relative_drift'] = float(np.max(relative_drift['action']))
    return {'initial_peak': initial_peak, 'nonlinear_length_initial_peak': 1 / initial_peak**2,
            'initial_invariants': initial, 'maxima': maxima, 'stations': stations,
            'conventions': {
                'sobolev': 'H_s^2=L*sum_k (1+k^2)^s*|fft(u)/N|^2; H0 is L2',
                'spectral_tail': f'Fraction of action at |mode| >= {tail_fraction:g}*{cutoff:g}',
                'asymmetry': '(power at k>0 minus power at k<0)/total power including zero mode',
                'recurrence': 'Same tau origin, global phase fit only; raw complex distance also recorded',
                'invariants': 'SpatialDysthe.invariants: action, modified energy and modified momentum for the noncanonical free-surface envelope; not all exactly conserved by the projected ODE',
                'small_initial_invariant': 'Relative drift is null if initial invariant is nearly zero; absolute drift remains reported',
                'physical_indicators': 'epsilon*peak and epsilon*spectral width/RMS are formal weak-nonlinearity/narrowband indicators, not validated universal physical cutoffs'},
            'limitation': LIMITATION}


def sampled_events(diagnostics, c):
    stations = diagnostics['stations']
    peak = np.array([s['peak'] for s in stations])
    recurrence = np.array([s['phase_aligned_recurrence_relative_l2'] for s in stations])
    prominence = c.get('focusing_prominence', .02) * peak[0]
    indices = find_peaks(peak, prominence=prominence)[0]
    minima = find_peaks(-recurrence, prominence=c.get('recurrence_prominence', .02))[0]
    minima = [j for j in minima if recurrence[j] <= c.get('recurrence_tolerance', .05)]
    return {'focusing_peaks': [{'xi': stations[j]['xi'], 'peak': float(peak[j]),
                               'peak_ratio': stations[j]['peak_ratio']} for j in indices],
            'recurrence_candidates': [{'xi': stations[j]['xi'], 'relative_l2': float(recurrence[j])}
                                      for j in minima],
            'peak_prominence_absolute': float(prominence),
            'recurrence_tolerance': c.get('recurrence_tolerance', .05),
            'interpretation': 'Interior extrema of sampled traces; no promised recurrence, no continuous event-time accuracy, no translation alignment'}


def _qualification(errors, diagnostics, c):
    tolerance = {'h0': c['reference_l2_tolerance'], 'h1': c['reference_h1_tolerance'],
                 'h2': c.get('reference_h2_tolerance', 1e-3)}
    checks = {f'{role}_{norm}': value <= tolerance[norm]
              for role, values in errors.items() for norm, value in values.items()}
    checks['spectral_tail'] = diagnostics['maxima']['spectral_tail_action_fraction'] <= c.get('spectral_tail_tolerance', 1e-8)
    checks['action_drift'] = diagnostics['maxima']['action_relative_drift'] <= c.get('action_drift_tolerance', 1e-6)
    return {'qualified': all(checks.values()), 'checks': checks,
            'tolerances': dict(tolerance, spectral_tail=c.get('spectral_tail_tolerance', 1e-8),
                               action_relative_drift=c.get('action_drift_tolerance', 1e-6))}


def _bundle(c, root, identity, u0, epsilon, length, n, h, xi, prefix=''):
    fields, artifacts = {}, []
    for label, count, step in (('fine', n, h), ('grid', n//2, h), ('step', n, 2*h)):
        values, meta = _trajectory(root, prefix+label, identity, u0, epsilon, length, count, step, xi)
        artifacts.append(meta)
        if values is None:
            return {'qualified': False, 'status': 'numerical_failure', 'artifacts': artifacts,
                    'failed_configuration': label, 'message': meta['message'], 'limitation': LIMITATION}, None
        fields[label] = values
    errors = {key: compare_fields(fields[key], fields['fine'], length) for key in ('grid', 'step')}
    diagnostics = field_diagnostics(fields['fine'], xi, length, epsilon,
                                    tail_fraction=c.get('diagnostics_tail_fraction', .8))
    result = {'status': 'complete', 'max_relative_errors': errors, 'diagnostics': diagnostics,
              'events': sampled_events(diagnostics, c), 'artifact': artifacts[0], 'artifacts': artifacts,
              **_qualification(errors, diagnostics, c)}
    if not result['qualified']:
        result['status'] = 'numerically_unresolved'
    return result, fields['fine']


def reference_case(c, run, record, u0, horizon):
    """Independent space/step checks for Dysthe and the epsilon=0 NLS control."""
    root = _case_root(run, 'reference', record, horizon)
    xi = _positions(c, horizon)
    identity = _identity(c, record, u0, horizon)
    result, fine = _bundle(c, root, identity, u0, record['physics']['epsilon'], c['length'],
                           c['reference_grid'], c['reference_step'], xi)
    nls, nls_fine = _bundle(c, root, identity, u0, 0., c['length'], c['reference_grid'],
                           c['reference_step'], xi, prefix='nls-')
    nls['interpretation'] = 'Mathematical epsilon=0 control at the same scaled inlet/grid/distance, not a zero-steepness physical coordinate transformation'
    result.update(identity=identity, case_id=record['case_id'], horizon=float(horizon),
                  nls_control=nls, limitation=LIMITATION)
    if fine is not None and nls_fine is not None:
        result['model_comparison'] = {
            'qualified': result['qualified'] and nls['qualified'],
            'interpretation': 'Raw complex NLS minus Dysthe, relative to Dysthe; a model difference, not a numerical error estimate',
            'max_relative_difference': compare_fields(nls_fine, fine, c['length']),
            'stations': [{'xi': float(x), **{f'h{s}': float(relative_error(nls_fine[j], fine[j], c['length'], s))
                                           for s in range(3)}} for j, x in enumerate(xi)]}
    _write(root/'result.json', result)
    return result


def sensitivity_case(c, run, record, u0, horizon):
    """Centered directional response, delta halving, independent grid/step checks."""
    root = _case_root(run, 'sensitivity', record, horizon)
    xi = _positions(c, horizon)
    identity = _identity(c, record, u0, horizon)
    direction, non_symmetry = _direction(u0, c['length'])
    n, h, delta = c['reference_grid'], c['reference_step'], c.get('sensitivity_delta', .001)
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError('sensitivity_delta must be positive and finite')
    responses, artifacts = {}, []
    for label, count, step, amplitude in (('delta', n, h, delta), ('fine', n, h, delta/2),
                                         ('grid', n//2, h, delta/2), ('step', n, 2*h, delta/2)):
        pair = []
        for sign, sign_label in ((1, 'plus'), (-1, 'minus')):
            fields, meta = _trajectory(root, label+'-'+sign_label, identity, u0+sign*amplitude*direction,
                                       record['physics']['epsilon'], c['length'], count, step, xi)
            artifacts.append(meta)
            if fields is None:
                result = {'identity': identity, 'case_id': record['case_id'], 'horizon': horizon,
                          'qualified': False, 'status': 'numerical_failure', 'artifacts': artifacts,
                          'failed_configuration': label+'-'+sign_label, 'message': meta['message'],
                          'limitation': LIMITATION}
                _write(root/'result.json', result)
                return result
            pair.append(fields)
        responses[label] = (pair[0]-pair[1])/(2*amplitude)
    fine = responses['fine']
    errors = {label: {f'h{s}': relative_error(responses[label], fine, c['length'], s)
                      for s in range(3)} for label in ('delta', 'grid', 'step')}
    maxima = {label: {key: float(np.max(value)) for key, value in values.items()}
              for label, values in errors.items()}
    gain = sobolev_norm(fine, c['length'], 1) / sobolev_norm(direction, c['length'], 1)
    tolerance = c.get('sensitivity_tolerance', .01)
    initial_error = float(relative_error(fine[0], direction, c['length'], 1))
    qualified = max(values['h1'] for values in maxima.values()) <= tolerance and initial_error <= 1e-8
    result = {'identity': identity, 'case_id': record['case_id'], 'horizon': float(horizon),
              'qualified': qualified, 'status': 'complete' if qualified else 'numerically_unresolved',
              'max_relative_errors': maxima, 'max_h1_amplification': float(np.max(gain)),
              'initial_direction_relative_h1_error': initial_error,
              'direction_l2_fraction_outside_phase_translation_tangent_span': non_symmetry,
              'delta': delta, 'fine_delta': delta/2, 'h1_response_tolerance': tolerance,
              'definition': '(S_xi(u0+delta*h)-S_xi(u0-delta*h))/(2*delta), ||h||H1=1; fixed modes 2 and 5, no alignment',
              'stations': [{'xi': float(x), 'h1_amplification': float(gain[j]),
                            'relative_errors': {label: {key: float(value[j]) for key, value in values.items()}
                                                for label, values in errors.items()}}
                           for j, x in enumerate(xi)], 'artifacts': artifacts, 'limitation': LIMITATION}
    _write(root/'result.json', result)
    return result


def packet_initial(record, length, n, max_mode):
    """Reconstruct the declared analytic Gaussian packet, then apply its cutoff.

    The same physical packet parameters are periodized in each requested box.
    Doubling the box doubles max_mode to preserve the physical wavenumber cutoff.
    """
    if record['family'] != 'packet':
        raise ValueError('Only packet fields have an approved domain reconstruction')
    p = record['parameters']
    tau = np.arange(n)*length/n-length/2
    value = sum(p['amplitude_parameter'] * np.exp(-.5*((tau-p['center']+q*length)/p['width'])**2
                + 1j*p['chirp']*((tau-p['center']+q*length)/p['width'])**2) for q in (-1, 0, 1))
    modes = abs(np.fft.fftfreq(n)*n)
    return np.fft.ifft(np.fft.fft(value*np.exp(1j*p['phase']))*(modes <= max_mode))


def common_interval_errors(small, large, length):
    """H0/H1/H2 errors on [-L/2,L/2), differentiating BEFORE cropping.

    Full periodic derivatives avoid falsely imposing periodicity on the cropped
    large-box restriction. Both grids must have identical dtau.
    """
    small, large = np.asarray(small), np.asarray(large)
    n = small.shape[-1]
    if n % 2 or large.shape[-1] != 2*n or small.shape[:-1] != large.shape[:-1]:
        raise ValueError('Domain comparison requires doubled length/count and equal dtau')
    small_k = 2*np.pi*np.fft.fftfreq(n, d=length/n)
    large_k = 2*np.pi*np.fft.fftfreq(2*n, d=length/n)
    small_hat, large_hat = np.fft.fft(small, axis=-1), np.fft.fft(large, axis=-1)
    differences, denominators = [], []
    for order in range(3):
        a = np.fft.ifft((1j*small_k)**order*small_hat, axis=-1)
        b = np.fft.ifft((1j*large_k)**order*large_hat, axis=-1)[..., n//2:3*n//2]
        differences.append(np.sum(abs(a-b)**2, axis=-1))
        denominators.append(np.sum(abs(b)**2, axis=-1))
    out = {}
    for order, weights in enumerate(((1,), (1, 1), (1, 2, 1))):
        numerator = sum(w*differences[j] for j, w in enumerate(weights))
        denominator = sum(w*denominators[j] for j, w in enumerate(weights))
        if np.any(denominator <= np.finfo(float).tiny):
            raise ValueError('Domain reference must be nonzero on common interval')
        out[f'h{order}'] = np.sqrt(numerator/denominator)
    return out


def domain_case(c, run, record, u0, horizon):
    """Packet-only doubling, retaining dx and physical initial Fourier cutoff."""
    root = _case_root(run, 'domain', record, horizon)
    identity = _identity(c, record, u0, horizon)
    if record['family'] != 'packet':
        result = {'case_id': record['case_id'], 'horizon': horizon, 'status': 'not_applicable',
                  'qualified': None, 'reason': 'Periodic random/sideband inlets are not localized packets; no zero padding or relabeling is applied.'}
        _write(root/'result.json', result)
        return result
    xi = _positions(c, horizon)
    length, n, h = c['length'], c['reference_grid'], c['reference_step']
    rebuilt = packet_initial(record, length, len(u0), c['initial_modes'])
    reconstruction = float(relative_error(rebuilt, u0, length, 1))
    if reconstruction > 1e-10:
        raise ValueError('Recorded packet parameters do not reconstruct the supplied inlet')
    small, small_fields = _bundle(c, root, identity, u0, record['physics']['epsilon'], length, n, h, xi, prefix='small-')
    # Reconstruct on twice the original inlet grid as well, preserving its
    # sampling/filter choices instead of introducing a distinct analytic fit.
    large_u0 = packet_initial(record, 2*length, 2*len(u0), 2*c['initial_modes'])
    large, large_fields = _bundle(c, root, identity, large_u0, record['physics']['epsilon'], 2*length, 2*n, h, xi, prefix='large-')
    result = {'identity': identity, 'case_id': record['case_id'], 'horizon': float(horizon),
              'small_domain': small, 'large_domain': large, 'reconstruction_relative_h1_error': reconstruction,
              'qualified': False, 'status': 'numerical_failure', 'limitation': LIMITATION,
              'comparison': 'Same analytic packet, doubled period/grid count and initial mode cutoff; derivatives evaluated in full boxes before restricting to original interval; no translation or phase alignment'}
    if small_fields is not None and large_fields is not None:
        errors = common_interval_errors(small_fields, large_fields, length)
        maxima = {key: float(np.max(value)) for key, value in errors.items()}
        initial = {key: float(value[0]) for key, value in errors.items()}
        tolerance, initial_tolerance = c.get('domain_tolerance', .01), c.get('domain_initial_tolerance', .001)
        qualified = (small['qualified'] and large['qualified'] and maxima['h1'] <= tolerance
                     and initial['h1'] <= initial_tolerance)
        result.update(qualified=qualified, status='complete' if qualified else 'domain_sensitive_or_unresolved',
                      max_common_interval_relative_errors=maxima, initial_common_interval_relative_errors=initial,
                      h1_domain_tolerance=tolerance, h1_initial_tolerance=initial_tolerance,
                      stations=[{'xi': float(x), **{key: float(value[j]) for key, value in errors.items()}}
                                for j, x in enumerate(xi)])
    _write(root/'result.json', result)
    return result


def reference_paths(run, case_id, horizon):
    """Paths only; no directory creation or numerical work."""
    if Path(case_id).name != case_id or case_id in ('.', '..'):
        raise ValueError('case_id must be a safe directory name')
    root = Path(run).resolve()/'reference'/case_id/('xi-'+format(horizon, '.12g'))
    return root/'result.json', root/'fine.npz'


def load_reference(c, run, record, u0, horizon):
    """Read and verify a prepared reference; never run a CPU solve implicitly.

    Returns (result, fields, xi), with fields/xi None for an unqualified result.
    All Dysthe and NLS artifacts are checked, not
    only the trajectory later used as the neural evaluation target.
    """
    result_path, fine_path = reference_paths(run, record['case_id'], horizon)
    if not result_path.exists():
        raise FileNotFoundError(f'Prepare the numerical reference first: {result_path}')
    result = json.loads(result_path.read_text())
    expected = _identity(c, record, u0, horizon)
    if result.get('identity') != expected:
        raise ValueError('Prepared reference identity differs from this campaign')
    expected_xi = _positions(c, horizon)
    for meta in result.get('artifacts', []) + result.get('nls_control', {}).get('artifacts', []):
        if meta['status'] == 'numerical_failure':
            continue
        path = Path(meta['path'])
        if path.parent.resolve() != result_path.parent.resolve():
            raise ValueError('Reference artifact must belong to its declared case directory')
        if not path.exists() or _digest(path) != meta['sha256']:
            raise ValueError(f'Reference artifact checksum mismatch: {path}')
        with np.load(path, allow_pickle=False) as archive:
            if (archive['fields'].shape != (len(expected_xi), meta['n'])
                    or not np.array_equal(archive['xi'], expected_xi)
                    or not np.isfinite(archive['fields']).all()):
                raise ValueError(f'Invalid reference artifact: {path}')
    if not result.get('qualified'):
        return result, None, None
    if Path(result['artifact']['path']) != fine_path:
        raise ValueError('Prepared reference points to an unexpected fine trajectory')
    with np.load(fine_path, allow_pickle=False) as archive:
        return result, archive['fields'].copy(), archive['xi'].copy()

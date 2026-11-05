"""Reproducible, bounded postprocessing of stage-one reference trajectories.

Run with the stage-one core and experiment sources on PYTHONPATH. No evolution,
training, installation, or external service is invoked by this command.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.signal import resample

from dysthe_core import MODEL_ID, PHYSICAL_SYSTEM
from dysthe_core.reference import spatial, validation


PERTURBATIONS = ('control', 'odd_amplitude_plus')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def map_fields(coarse_tau, fine_tau, coarse, fine, kind):
    """Map fields to an asserted common grid; preserve the complex field.

    ``step`` requires identical coordinates, ``grid`` Fourier-interpolates the
    coarse periodic field at fixed length, and ``domain`` crops the central
    half of a domain doubled at exactly the same spacing. Last axis is tau.
    """
    coarse_tau, fine_tau = np.asarray(coarse_tau), np.asarray(fine_tau)
    coarse, fine = np.asarray(coarse), np.asarray(fine)
    if coarse.shape[-1] != coarse_tau.size or fine.shape[-1] != fine_tau.size:
        raise ValueError('Field and coordinate lengths disagree')
    if coarse_tau.size < 2 or fine_tau.size < 2:
        raise ValueError('At least two coordinates required')
    dc, df = coarse_tau[1] - coarse_tau[0], fine_tau[1] - fine_tau[0]
    lc, lf = dc * coarse_tau.size, df * fine_tau.size
    close = lambda a, b: np.isclose(a, b, atol=1e-11, rtol=1e-12)
    if kind == 'step':
        if coarse_tau.shape != fine_tau.shape or not np.allclose(coarse_tau, fine_tau, atol=1e-12, rtol=0):
            raise ValueError('Step comparison requires identical coordinates')
        return coarse, fine, fine_tau
    if kind == 'grid':
        if not close(lc, lf) or not close(coarse_tau[0], fine_tau[0]) or fine_tau.size <= coarse_tau.size:
            raise ValueError('Grid comparison requires fixed length and a finer grid')
        expected = fine_tau[0] + np.arange(fine_tau.size) * lf / fine_tau.size
        np.testing.assert_allclose(fine_tau, expected, atol=1e-12, rtol=0)
        return resample(coarse, fine_tau.size, axis=-1), fine, fine_tau
    if kind == 'domain':
        if fine_tau.size != 2 * coarse_tau.size or not close(df, dc) or not close(lf, 2 * lc):
            raise ValueError('Domain comparison requires a doubling at fixed spacing')
        cropped_tau = fine_tau[fine_tau.size // 4:3 * fine_tau.size // 4]
        np.testing.assert_allclose(coarse_tau, cropped_tau, atol=1e-12, rtol=0)
        return coarse, fine[..., fine_tau.size // 4:3 * fine_tau.size // 4], cropped_tau
    raise ValueError('Unknown comparison kind')


def _maxima(rows, key):
    names = set().union(*(row[key] for row in rows))
    return {name: max((abs(row[key][name]) for row in rows if row[key].get(name) is not None), default=None) for name in sorted(names)}


def matched_control_response(model, perturbed, control, profile):
    """Return signed shape in the matched control's initial profile frame."""
    aligned_perturbed, response = validation.perturbation_response(model, perturbed, control)
    aligned_control, control_fit = validation.align_to_reference(model, control, profile)
    signed_response = spatial.translate(model, aligned_perturbed-control,
                                        -control_fit['translation']) * np.exp(-1j*control_fit['global_phase'])
    return signed_response, aligned_control-profile, response, control_fit


def fold_doubled_domain(field):
    """Periodize a doubled-domain field onto its central half, along tau."""
    field = np.asarray(field)
    n = field.shape[-1]
    if n < 4 or n % 4:
        raise ValueError('A doubled domain requires a last axis divisible by four')
    central = field[..., n//4:3*n//4]
    # Right exterior shifts left by the small-domain length; left shifts right.
    exterior = np.concatenate((field[..., 3*n//4:], field[..., :n//4]), axis=-1)
    return central + exterior


def load_case(label, directory):
    directory = Path(directory)
    report = json.loads((directory / 'report.json').read_text())
    if report['model_id'] != MODEL_ID or report['physical_system'] != PHYSICAL_SYSTEM:
        raise ValueError(f'Incompatible physical model in {directory}')
    if digest(spatial.__file__) != report['core_source_sha256']:
        raise ValueError(f'Current spatial source differs from recorded propagation source in {directory}')
    if digest(directory / 'fields.npz') != report['fields_sha256']:
        raise ValueError(f'Field checksum mismatch in {directory}')
    case = report['case']
    if (case['epsilon'], case['mu'], case['speed'], case['xi_end']) != (.025, 2., 0., 20.):
        raise ValueError(f'Expected the declared epsilon=.025, mu=2, s=0, xi=20 case in {directory}')
    if not all(name in report['trajectories'] for name in PERTURBATIONS):
        raise ValueError(f'Missing matched control or positive odd perturbation in {directory}')
    declarations = {p['id']: p for p in case['perturbations']}
    if declarations['control']['kind'] != 'control' or declarations['control']['delta'] != 0 or declarations['odd_amplitude_plus']['kind'] != 'odd_amplitude' or declarations['odd_amplitude_plus']['delta'] != .001:
        raise ValueError(f'Perturbation definitions disagree in {directory}')
    with np.load(directory / 'fields.npz', allow_pickle=False) as archive:
        fields = {name: archive[name].copy() for name in ('tau', 'xi', 'profile') + PERTURBATIONS}
    model = spatial.SpatialDysthe(case['n'], case['length'], case['epsilon'])
    np.testing.assert_allclose(fields['tau'], model.tau, atol=1e-12, rtol=0)
    if not all(np.isfinite(array).all() for array in fields.values()):
        raise ValueError(f'Nonfinite saved fields in {directory}')
    count = len(fields['xi'])
    for name in PERTURBATIONS:
        if fields[name].shape != (count, model.n):
            raise ValueError('Unexpected trajectory array shape')
        np.testing.assert_allclose(fields['xi'], [row['xi'] for row in report['trajectories'][name]], atol=1e-12, rtol=0)
    shape, metrics, control_metrics, control_residual = [], [], [], []
    for perturbed, control in zip(fields['odd_amplitude_plus'], fields['control']):
        # The signed response is brought to the *control's* profile frame.
        # Independent fits of perturbed and control profiles would remove an
        # extra, different neutral component from the weak response.
        signed_response, residual, response, control_fit = matched_control_response(
            model, perturbed, control, fields['profile'])
        shape.append(signed_response)
        control_residual.append(residual)
        metrics.append(response)
        control_metrics.append(control_fit)
    shape = np.asarray(shape)
    norm_profile = np.linalg.norm(fields['profile'])
    shape_norm = np.linalg.norm(shape, axis=-1)
    shape_relative = shape_norm / norm_profile
    checks = {}
    trajectory_summaries = {}
    for name in PERTURBATIONS:
        history = report['trajectories'][name]
        flags = sorted(key for key in history[0] if key.endswith('_flag'))
        checks[name] = {}
        for endpoint, index in (('initial', 0), ('final', -1)):
            field = fields[name][index]
            checks[name][endpoint] = {'discarded_rhs': validation.full_rhs_defect(model, field)}
            if name == 'control':
                derivative = np.fft.ifft(1j * model.k * np.fft.fft(field))
                target = -1j * case['mu'] * field - case['speed'] * derivative
                checks[name][endpoint]['traveling_profile_defect'] = validation.full_rhs_defect(model, field, target=target)
        trajectory_summaries[name] = {
            'invariant_max_absolute_drifts': _maxima(history, 'invariant_absolute_drifts'),
            'invariant_max_relative_drifts': _maxima(history, 'invariant_relative_drifts'),
            'flags': {key: any(row[key] for row in history) for key in flags},
            'maximum_boundary_action_fraction': max(row['boundary_action_fraction'] for row in history),
            'maximum_retained_spectral_edge_action_fraction': max(row['retained_spectral_edge_action_fraction'] for row in history),
            'maximum_epsilon_peak': max(row['epsilon_peak'] for row in history),
            'maximum_epsilon_spectral_sigma': max(row['epsilon_spectral_sigma'] for row in history),
            'maximum_absolute_epsilon_spectral_mean': max(abs(row['epsilon_spectral_mean']) for row in history),
            'initial_invariants': {key: history[0][key] for key in ('action', 'momentum', 'energy')},
            'final_invariants': {key: history[-1][key] for key in ('action', 'momentum', 'energy')},
        }
    provenance = {'directory': str(directory.resolve()), 'report_sha256': digest(directory / 'report.json'), 'fields_sha256': report['fields_sha256'], 'recorded_core_source_sha256': report['core_source_sha256'], 'recorded_campaign_source_sha256': report['campaign_source_sha256'], 'current_spatial_source_matches_record': digest(spatial.__file__) == report['core_source_sha256']}
    summary = {'case': case, 'actual_dxi': report['actual_dxi'], 'provenance': provenance, 'resources': report['resources'], 'environment': report['environment'], 'profile_residual': model.profile_residual(fields['profile'], case['mu'], case['speed']), 'saved_profile_residual': report['profile_residual'], 'endpoint_rhs_diagnostics': checks, 'trajectories': trajectory_summaries, 'response': {'initial_shape_relative_l2': float(shape_relative[0]), 'final_shape_relative_l2': float(shape_relative[-1]), 'maximum_shape_relative_l2': float(max(shape_relative)), 'final_shape_l2': float(np.sqrt(model.dtau) * shape_norm[-1]), 'final_raw_relative_l2': metrics[-1]['raw_relative_l2'], 'final_relative_translation': metrics[-1]['translation'], 'final_relative_global_phase': metrics[-1]['global_phase'], 'maximum_control_shape_relative_l2': max(row['shape_relative_l2'] for row in control_metrics)}}
    history_rows = []
    for i, xi in enumerate(fields['xi']):
        history_rows.append({'case': label, 'xi': float(xi), 'shape_relative_l2': float(shape_relative[i]), 'shape_l2': float(np.sqrt(model.dtau) * shape_norm[i]), 'raw_relative_l2': metrics[i]['raw_relative_l2'], 'relative_translation': metrics[i]['translation'], 'relative_global_phase': metrics[i]['global_phase'], 'control_translation': control_metrics[i]['translation'], 'control_global_phase': control_metrics[i]['global_phase'], 'control_shape_relative_l2': control_metrics[i]['shape_relative_l2']})
    return {'label': label, 'summary': summary, 'fields': fields, 'shape': shape, 'shape_relative': shape_relative, 'metrics': metrics, 'control_metrics': control_metrics, 'control_residual': np.asarray(control_residual), 'history_rows': history_rows}


def compare_cases(label, coarse, fine, kind):
    a, b = coarse['fields'], fine['fields']
    np.testing.assert_allclose(a['xi'], b['xi'], atol=1e-12, rtol=0)
    profile_a, profile_b, tau = map_fields(a['tau'], b['tau'], a['profile'], b['profile'], kind)
    shape_a, shape_b, _ = map_fields(a['tau'], b['tau'], coarse['shape'], fine['shape'], kind)
    control_a, control_b, _ = map_fields(a['tau'], b['tau'], a['control'], b['control'], kind)
    pert_a, pert_b, _ = map_fields(a['tau'], b['tau'], a['odd_amplitude_plus'], b['odd_amplitude_plus'], kind)
    control_shape_a, control_shape_b, _ = map_fields(a['tau'], b['tau'], coarse['control_residual'], fine['control_residual'], kind)
    denominator = np.linalg.norm(profile_b)
    core = abs(tau) < 20
    norm = lambda value: float(np.linalg.norm(value) / denominator)
    rows = []
    for i, xi in enumerate(a['xi']):
        difference = shape_a[i] - shape_b[i]
        response_norm = np.linalg.norm(shape_b[i])
        if response_norm == 0:
            raise ValueError('Nonzero weak shape response required for a relative comparison')
        row = {'comparison': label, 'kind': kind, 'coarse': coarse['label'], 'fine': fine['label'], 'xi': float(xi), 'raw_control_field_relative_l2': norm(control_a[i] - control_b[i]), 'raw_odd_field_relative_l2': norm(pert_a[i] - pert_b[i]), 'control_shape_field_difference_relative_l2': norm(control_shape_a[i] - control_shape_b[i]), 'signed_shape_field_difference_relative_l2': norm(difference), 'signed_shape_core_difference_relative_l2': norm(difference[core]), 'signed_shape_exterior_difference_relative_l2': norm(difference[~core]), 'signed_shape_difference_over_response': float(np.linalg.norm(difference) / response_norm), 'signed_shape_core_difference_over_response': float(np.linalg.norm(difference[core]) / response_norm), 'fine_shape_relative_l2_on_common_window': norm(shape_b[i]), 'shape_norm_scalar_difference': abs(norm(shape_a[i]) - norm(shape_b[i]))}
        if kind == 'domain':
            n = b['tau'].size
            outside = np.concatenate((fine['shape'][i, :n // 4], fine['shape'][i, 3 * n // 4:]))
            folded = fold_doubled_domain(fine['shape'][i])
            whole_response_norm = np.linalg.norm(fine['shape'][i])
            row.update({'fine_shape_relative_l2_on_full_domain': norm(fine['shape'][i]), 'signed_shape_difference_over_full_domain_response': float(np.linalg.norm(difference) / whole_response_norm), 'signed_shape_core_difference_over_full_domain_response': float(np.linalg.norm(difference[core]) / whole_response_norm), 'large_domain_shape_outside_common_window_relative_l2': norm(outside), 'large_domain_exterior_over_whole_shape': float(np.linalg.norm(outside) / whole_response_norm), 'shape_difference_after_periodizing_large_domain_relative_l2': norm(shape_a[i] - folded)})
        rows.append(row)
    numeric_keys = [key for key, value in rows[0].items() if isinstance(value, float) and key != 'xi']
    summary = {'kind': kind, 'coarse': coarse['label'], 'fine': fine['label'], 'common_window': [float(tau[0]), float(tau[-1] + tau[1] - tau[0])], 'common_n': len(tau), 'profile_relative_l2': norm(profile_a - profile_b), 'final': rows[-1], 'maxima': {key: max(row[key] for row in rows) for key in numeric_keys}}
    return summary, rows


def _write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze(historical, campaign, extended, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    definitions = [
        ('historical-base', Path(historical) / 'e0.025-base', True),
        ('historical-step', Path(historical) / 'e0.025-step', True),
        ('historical-grid', Path(historical) / 'e0.025-grid', True),
        ('historical-domain', Path(historical) / 'e0.025-domain', True),
        ('step-h005', Path(campaign) / 'step-h005', False),
        ('step-h0025', Path(campaign) / 'step-h0025', False),
        ('domain-L320', Path(campaign) / 'domain-L320', False),
        ('domain-L640', Path(campaign) / 'domain-L640', False),
        ('domain-L1280', Path(extended) / 'domain-L1280', False),
    ]
    cases, pending = {}, []
    for label, directory, required in definitions:
        if not (directory / 'report.json').exists():
            if required:
                raise FileNotFoundError(f'Required historical report missing: {directory}')
            pending.append({'case': label, 'directory': str(directory.resolve())})
            continue
        cases[label] = load_case(label, directory)
    pairs = [
        ('step-005-0025', 'step-h005', 'step-h0025', 'step'),
        ('step-0025-00125', 'step-h0025', 'historical-base', 'step'),
        ('step-00125-000625', 'historical-base', 'historical-step', 'step'),
        ('grid-L160', 'historical-step', 'historical-grid', 'grid'),
        ('grid-L320', 'domain-L320', 'historical-domain', 'grid'),
        ('domain-L160-L320', 'historical-step', 'domain-L320', 'domain'),
        ('domain-L320-L640', 'domain-L320', 'domain-L640', 'domain'),
        ('domain-L640-L1280', 'domain-L640', 'domain-L1280', 'domain'),
        ('historical-domain-L160-L320', 'historical-grid', 'historical-domain', 'domain'),
    ]
    comparisons, comparison_rows = {}, []
    for label, a, b, kind in pairs:
        if a in cases and b in cases:
            comparisons[label], rows = compare_cases(label, cases[a], cases[b], kind)
            comparison_rows.extend(rows)
    orders = []
    step_names = [entry[0] for entry in pairs if entry[3] == 'step']
    for first, second in zip(step_names[:-1], step_names[1:]):
        if first not in comparisons or second not in comparisons:
            continue
        for aggregation in ('final', 'maxima'):
            for metric in ('raw_control_field_relative_l2', 'raw_odd_field_relative_l2', 'signed_shape_field_difference_relative_l2'):
                e1 = comparisons[first][aggregation][metric]
                e2 = comparisons[second][aggregation][metric]
                orders.append({'coarse_pair': first, 'fine_pair': second, 'aggregation': aggregation, 'metric': metric, 'coarse_difference': e1, 'fine_difference': e2, 'observed_order': float(np.log2(e1 / e2)) if e1 > 0 and e2 > 0 else None})
    data = {}
    for label, case in cases.items():
        prefix = label + '__'
        data[prefix + 'tau'] = case['fields']['tau']
        data[prefix + 'xi'] = case['fields']['xi']
        data[prefix + 'profile'] = case['fields']['profile']
        data[prefix + 'signed_matched_control_shape'] = case['shape']
        for key in ('shape_relative_l2', 'shape_l2', 'raw_relative_l2', 'relative_translation', 'relative_global_phase', 'control_translation', 'control_global_phase', 'control_shape_relative_l2'):
            data[prefix + key] = np.array([row[key] for row in case['history_rows']])
        data[prefix + 'relative_global_phase_unwrapped'] = np.unwrap(data[prefix + 'relative_global_phase'])
        data[prefix + 'control_phase_offset_from_exact_unwrapped'] = np.unwrap(data[prefix + 'control_global_phase'] + case['summary']['case']['mu'] * case['fields']['xi'])
    for label in comparisons:
        selected = [row for row in comparison_rows if row['comparison'] == label]
        for key, value in selected[0].items():
            if isinstance(value, float):
                data[label + '__' + key] = np.asarray([row[key] for row in selected])
    np.savez_compressed(output / 'figure-data.npz', **data)
    result = {'schema_version': 1, 'model_id': MODEL_ID, 'physical_system': PHYSICAL_SYSTEM, 'analysis_environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__}, 'analysis_sources': {'analysis_sha256': digest(__file__), 'spatial_sha256': digest(spatial.__file__), 'validation_sha256': digest(validation.__file__)}, 'cases': {label: case['summary'] for label, case in cases.items()}, 'pending_cases': pending, 'comparisons': comparisons, 'step_orders': orders, 'figure_data_sha256': digest(output / 'figure-data.npz'), 'interpretation': [
        'All comparisons are measured discrepancies of saved periodic trajectories, not rigorous error bounds or proofs of stability.',
        'Signed matched-control response is aligned perturbed minus control, then transformed by the matched control translation and phase into its own initial profile frame. Amplitude and branch parameters are fixed.',
        'Field convergence uses norms of complex field differences. shape_norm_scalar_difference is retained solely as a warning that nearly equal scalar norms do not imply equal fields.',
        'Absolute shape discrepancies are normalized by the fine profile norm on the common window; relative weak-response discrepancies use the fine shape norm on that same window. Core means |tau|<20 and uses the same whole-window normalization.',
        'Domain comparisons also retain explicitly named full-domain-response denominators for continuity with the historical 67 percent figure. Those denominators include radiation outside the common comparison window; do not mix them with common-window response ratios.',
        'Step comparisons use identical grids; grid comparisons Fourier-interpolate at fixed length; domain comparisons assert exact coordinate agreement after cropping a doubled domain at fixed spacing.',
        'Exterior norms and periodized-residual differences diagnose possible radiation wrapping. Folding is not a corrected nonlinear solution. Periodic nonlocal mean-flow effects and later radiation interactions are not excluded.',
        'full_rhs_defect without a target measures discarded RHS modes relative to full RHS. With a traveling target it measures instantaneous traveling-profile residual; neither is a propagation error bound.',
        'Continuum energy/momentum diagnostics are not automatically exact invariants of the finite Galerkin system. Small recorded drift alone does not establish accuracy.',
    ]}
    (output / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    _write_csv(output / 'comparison-history.csv', comparison_rows)
    _write_csv(output / 'response-history.csv', [row for case in cases.values() for row in case['history_rows']])
    _write_csv(output / 'step-orders.csv', orders)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--extended', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.historical, args.campaign, args.extended, args.output)
    print(json.dumps({'output': str(args.output.resolve()), 'completed_cases': list(result['cases']), 'pending_cases': result['pending_cases'], 'domain_final_relative_shape_differences': {label: comparison['final']['signed_shape_difference_over_response'] for label, comparison in result['comparisons'].items() if comparison['kind'] == 'domain'}}, indent=2))


if __name__ == '__main__':
    main()

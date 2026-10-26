"""Scientific and provenance checks for the staged finite-distance reference."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dysthe_experiments.cpu_pilot import initial_fields
from dysthe_experiments.long_reference import (
    common_interval_errors, compare_fields, domain_case, field_diagnostics,
    load_reference, packet_initial, reference_case, reference_paths,
    sensitivity_case, sobolev_norm,
)


def configuration():
    return {'model_id': 'water-wave-dysthe-spatial-v1',
            'physical_system': 'deep-water-gravity-waves', 'seed': 7,
            'grid': 64, 'reference_grid': 64, 'length': 2*np.pi,
            'reference_step': .0025, 'sample_step': .02,
            'horizon': .04, 'initial_modes': 6, 'groups_per_family': 3,
            'train_per_family': 1, 'validation_per_family': 1,
            'reference_l2_tolerance': 1e-5, 'reference_h1_tolerance': 1e-4,
            'reference_h2_tolerance': 1e-3, 'spectral_tail_tolerance': 1e-8,
            'action_drift_tolerance': 1e-6}


def inlet(c):
    record = initial_fields(c)[0][3]  # Periodic sideband metadata; use a plane wave below.
    record = copy.deepcopy(record)
    record['case_id'] = 'analytic-plane'
    record['parameters'] = {'amplitude': .5, 'wavenumber': 2}
    record['physics']['epsilon'] = .04
    tau = np.arange(c['grid'])*c['length']/c['grid']-c['length']/2
    return record, .5*np.exp(2j*tau)


class LongReferenceTests(unittest.TestCase):
    def test_single_precision_neural_profile_invariant_quadrature(self):
        c = configuration()
        _, u = inlet(c)
        fields = np.stack([u, u * np.exp(.2j)]).astype(np.complex64)
        result = field_diagnostics(fields, [0., .1], c['length'], .04, output_modes=6)
        expected_action = c['length'] * np.mean(abs(fields.astype(np.complex128))**2, axis=-1)
        for row, expected in zip(result['stations'], expected_action):
            self.assertAlmostEqual(row['invariants']['action'], expected, places=12)
        self.assertTrue(np.isfinite(result['maxima']['h2_ratio']))

    def test_exact_plane_wave_phase_spectrum_and_modified_invariants(self):
        c = configuration()
        record, u0 = inlet(c)
        epsilon, amplitude, k = .04, .5, 2.
        xi = np.array([0., .1, .2])
        omega = amplitude**2-k*k+6*epsilon*k*amplitude**2
        exact = np.exp(-1j*omega*xi[:, None])*u0
        diagnostics = field_diagnostics(exact, xi, c['length'], epsilon)
        end = diagnostics['stations'][-1]
        self.assertAlmostEqual(end['spectral_centroid'], k, places=13)
        self.assertLess(end['spectral_width'], 1e-12)
        self.assertAlmostEqual(end['spectral_signed_power_asymmetry'], 1., places=13)
        self.assertAlmostEqual(end['h2_ratio'], 1., places=13)
        self.assertAlmostEqual(end['nonlinear_lengths_initial_peak'], .05, places=13)
        self.assertLess(end['phase_aligned_recurrence_relative_l2'], 1e-14)
        self.assertGreater(end['raw_recurrence_relative_l2'], .1)
        length = c['length']
        self.assertAlmostEqual(end['invariants']['action'], length*amplitude**2, places=12)
        momentum = length*(2*k*amplitude**2-2*epsilon*amplitude**4)
        energy = length*(k*k*amplitude**2-.5*amplitude**4+(10/3)*epsilon**2*amplitude**6
                         -5*epsilon*k*amplitude**4)
        self.assertAlmostEqual(end['invariants']['momentum'], momentum, places=12)
        self.assertAlmostEqual(end['invariants']['energy'], energy, places=12)

    def test_complex_h2_errors_detect_phase_without_false_alignment(self):
        c = configuration()
        _, u = inlet(c)
        measured = compare_fields(u*np.exp(.3j), u, c['length'])
        for error in measured.values():
            self.assertAlmostEqual(error, 2*np.sin(.15), places=13)
        self.assertAlmostEqual(float(sobolev_norm(u, c['length'], 2)),
                               np.sqrt(c['length'])*.5*(1+4), places=12)

    def test_reference_nls_control_cache_and_tamper_rejection(self):
        c = configuration()
        record, u = inlet(c)
        with tempfile.TemporaryDirectory() as directory:
            result = reference_case(c, directory, record, u, .04)
            self.assertTrue(result['qualified'])
            self.assertTrue(result['nls_control']['qualified'])
            self.assertGreater(result['model_comparison']['max_relative_difference']['h0'], .0001)
            self.assertEqual(len(result['artifacts']), 3)
            self.assertEqual(result['artifacts'][1]['n'], 32)
            self.assertEqual(result['artifacts'][1]['step'], .0025)
            self.assertEqual(result['artifacts'][2]['n'], 64)
            self.assertEqual(result['artifacts'][2]['step'], .005)
            with patch('dysthe_core.reference.spatial.SpatialDysthe.stepper', side_effect=AssertionError('No solving allowed')):
                loaded, fields, xi = load_reference(c, directory, record, u, .04)
                repeated = reference_case(c, directory, record, u, .04)
            self.assertEqual(result, loaded)
            self.assertEqual(result, repeated)
            self.assertEqual(fields.shape, (3, 64))
            np.testing.assert_array_equal(xi, [0., .02, .04])
            self.assertEqual(Path(result['artifact']['path']), reference_paths(directory, record['case_id'], .04)[1])
            changed = dict(c, reference_h2_tolerance=2e-3)
            with self.assertRaisesRegex(ValueError, 'identity'):
                load_reference(changed, directory, record, u, .04)
            # A corrupted NLS refinement must be detected even though the PINN
            # reads only the Dysthe trajectory as its evaluation reference.
            nls_coarse = Path(result['nls_control']['artifacts'][1]['path'])
            nls_coarse.write_bytes(nls_coarse.read_bytes()+b'corruption')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_reference(c, directory, record, u, .04)

    def test_nonfinite_failure_is_saved_and_not_called_nonexistence(self):
        c = configuration()
        record, u = inlet(c)
        with tempfile.TemporaryDirectory() as directory:
            with patch('dysthe_core.reference.spatial.SpatialDysthe.stepper', side_effect=FloatingPointError('synthetic overflow')):
                result = reference_case(c, directory, record, u, .04)
            self.assertFalse(result['qualified'])
            self.assertEqual(result['status'], 'numerical_failure')
            self.assertTrue(reference_paths(directory, record['case_id'], .04)[0].exists())
            loaded, fields, xi = load_reference(c, directory, record, u, .04)
            self.assertFalse(loaded['qualified'])
            self.assertIsNone(fields)
            self.assertIsNone(xi)

    def test_refinement_gate_does_not_hide_failure(self):
        c = dict(configuration(), reference_l2_tolerance=1e-30)
        record, u = inlet(c)
        with tempfile.TemporaryDirectory() as directory:
            result = reference_case(c, directory, record, u, .04)
            self.assertFalse(result['qualified'])
            self.assertEqual(result['status'], 'numerically_unresolved')
            self.assertTrue(Path(result['artifact']['path']).exists())

    def test_sensitivity_checks_signed_response_and_delta_halving(self):
        c = configuration()
        record, u = inlet(c)
        with tempfile.TemporaryDirectory() as directory:
            result = sensitivity_case(c, directory, record, u, .04)
            self.assertTrue(result['qualified'])
            self.assertEqual(len(result['artifacts']), 8)
            self.assertLess(result['initial_direction_relative_h1_error'], 1e-10)
            self.assertAlmostEqual(result['stations'][0]['h1_amplification'], 1., places=10)
            self.assertLess(result['max_relative_errors']['delta']['h1'], 1e-6)

    def test_common_interval_derivatives_are_computed_before_cropping(self):
        n, length = 64, 2*np.pi
        small_tau = np.arange(n)*length/n-length/2
        large_tau = np.arange(2*n)*length/n-length
        small = np.exp(2j*small_tau)[None, :]
        large = np.exp(2j*large_tau)[None, :]
        for value in common_interval_errors(small, large, length).values():
            self.assertLess(float(value[0]), 1e-13)
        # A half-frequency wave is periodic only on the large box; an analytic
        # derivative restricted to the common interval stays smooth at its edges.
        big = np.exp(.5j*large_tau)[None, :]
        errors = common_interval_errors(np.ones_like(small), big, length)
        for value in errors.values():
            self.assertTrue(np.isfinite(value).all())
        with self.assertRaisesRegex(ValueError, 'doubled'):
            common_interval_errors(small, large[:, :-1], length)

    def test_packet_reconstruction_and_nonpacket_domain_exclusion(self):
        c = dict(configuration(), length=40., reference_grid=128)
        records, fields = initial_fields(c)
        rebuilt = packet_initial(records[0], c['length'], c['grid'], c['initial_modes'])
        np.testing.assert_array_equal(rebuilt, fields[0])
        with tempfile.TemporaryDirectory() as directory:
            result = domain_case(c, directory, records[0], fields[0], .04)
            self.assertEqual(result['reconstruction_relative_h1_error'], 0.)
            self.assertEqual(result['large_domain']['artifact']['n'], 256)
            self.assertEqual(result['large_domain']['artifact']['length'], 80.)
            self.assertEqual(set(result['max_common_interval_relative_errors']), {'h0', 'h1', 'h2'})
            skipped = domain_case(c, directory, records[3], fields[3], .04)
            self.assertEqual(skipped['status'], 'not_applicable')
            self.assertIsNone(skipped['qualified'])

    def test_horizon_and_scope_cannot_be_silently_reinterpreted(self):
        c = configuration()
        record, u = inlet(c)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'integer multiple'):
                reference_case(c, directory, record, u, .035)
            with self.assertRaisesRegex(ValueError, 'Scope violation'):
                reference_case(dict(c, model_id='different-model'), directory, record, u, .04)
            with self.assertRaisesRegex(ValueError, 'both h and 2h'):
                reference_case(dict(c, reference_step=.003), directory, record, u, .04)


if __name__ == '__main__':
    unittest.main()

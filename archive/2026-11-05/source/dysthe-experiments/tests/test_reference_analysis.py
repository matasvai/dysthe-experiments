"""Protect the common-coordinate mapping used by scientific comparisons."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dysthe_core.reference.spatial import SpatialDysthe, nls_profile, translate
from dysthe_experiments.reference_analysis import (
    fold_doubled_domain, load_case, map_fields, matched_control_response,
)
from dysthe_experiments.reference_controls import load_audited_legacy_profile


class ReferenceMappingTests(unittest.TestCase):
    def test_fixed_domain_fourier_interpolation_preserves_complex_mode(self):
        coarse_tau = np.arange(32) / 4 - 4
        fine_tau = np.arange(64) / 8 - 4
        coarse = np.exp(1j * 2 * np.pi * coarse_tau / 8)
        fine = np.exp(1j * 2 * np.pi * fine_tau / 8)
        mapped, reference, tau = map_fields(coarse_tau, fine_tau, coarse, fine, 'grid')
        np.testing.assert_allclose(mapped, reference, atol=1e-14)
        np.testing.assert_array_equal(tau, fine_tau)

    def test_domain_crop_preserves_physical_coordinates_and_complex_values(self):
        coarse_tau = np.arange(32) / 4 - 4
        large_tau = np.arange(64) / 4 - 8
        coarse = (1 + 2j) * np.exp(-coarse_tau**2)
        large = (1 + 2j) * np.exp(-large_tau**2)
        a, b, tau = map_fields(coarse_tau, large_tau, coarse, large, 'domain')
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(tau, coarse_tau)

    def test_equal_n_is_not_a_fixed_spacing_domain_comparison(self):
        coarse_tau = np.arange(32) / 4 - 4
        wrong_tau = np.arange(32) / 2 - 8
        with self.assertRaises(ValueError):
            map_fields(coarse_tau, wrong_tau, coarse_tau, wrong_tau, 'domain')

    def test_step_rejects_shifted_coordinates(self):
        tau = np.arange(32) / 4 - 4
        with self.assertRaises(ValueError):
            map_fields(tau, tau + .01, tau, tau, 'step')


    def test_matched_control_response_removes_shared_frame_and_relative_symmetry(self):
        model = SpatialDysthe(1024, 80., .025)
        profile = model.project(nls_profile(model.tau))
        control_shift, control_phase = 1.371928, .6342
        relative_shift, relative_phase = -.2135, -.137
        delta = 1e-4
        control = translate(model, profile, control_shift)*np.exp(1j*control_phase)
        perturbed = translate(model, (1+delta)*profile,
                              control_shift+relative_shift)*np.exp(1j*(control_phase+relative_phase))
        response, residual, metrics, control_fit = matched_control_response(
            model, perturbed, control, profile)
        np.testing.assert_allclose(response, delta*profile, atol=2e-14, rtol=0)
        np.testing.assert_allclose(residual, 0., atol=2e-14, rtol=0)
        self.assertAlmostEqual(metrics['translation'], relative_shift, delta=2e-14)
        self.assertAlmostEqual(metrics['global_phase'], relative_phase, delta=2e-14)
        self.assertAlmostEqual(control_fit['translation'], control_shift, delta=2e-14)
        self.assertAlmostEqual(control_fit['global_phase'], control_phase, delta=2e-14)

    def test_exterior_folding_preserves_complex_values_at_correct_coordinates(self):
        # Large domain [-8,8), small domain [-4,4), both delta_tau=.25.
        full = np.zeros(64, dtype=complex)
        full[58] = 2+3j  # tau=6.5 folds to tau=-1.5, small index 10.
        full[7] = -4+5j  # tau=-6.25 folds to tau=1.75, small index 23.
        full[32] = 7-2j  # tau=0 remains at small index 16.
        expected = np.zeros(32, dtype=complex)
        expected[10], expected[23], expected[16] = 2+3j, -4+5j, 7-2j
        np.testing.assert_array_equal(fold_doubled_domain(full), expected)
        np.testing.assert_array_equal(fold_doubled_domain(np.stack([full, -full])),
                                      np.stack([expected, -expected]))


class ReferenceProvenanceTests(unittest.TestCase):
    def test_profile_identity_coordinates_and_both_hashes_are_checked(self):
        model = SpatialDysthe(64, 80., .025)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'profile_mu2.npz'
            sidecar = path.with_suffix('.json')
            np.savez(path, u=np.ones((1, model.n), dtype=complex), t=np.array([0]), x0=model.tau)
            metadata = dict(kind='dysthe', epsilon1=1., epsilon2=1., nonlinear=True,
                            shape=[model.n], lengths=[model.length], epsilon=.025)
            sidecar.write_text(json.dumps(metadata))
            _, _, _, provenance = load_audited_legacy_profile(path)
            self.assertEqual(provenance['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(provenance['sidecar_sha256'], hashlib.sha256(sidecar.read_bytes()).hexdigest())
            metadata['kind'] = 'other'
            sidecar.write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                load_audited_legacy_profile(path)
            metadata['kind'] = 'dysthe'
            sidecar.write_text(json.dumps(metadata))
            np.savez(path, u=np.ones((1, model.n), dtype=complex), t=np.array([0]), x0=model.tau+.01)
            with self.assertRaises(AssertionError):
                load_audited_legacy_profile(path)

    def test_analysis_refuses_a_different_propagation_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = dict(model_id='water-wave-dysthe-spatial-v1',
                          physical_system='deep-water-gravity-waves',
                          core_source_sha256='0'*64)
            (directory/'report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'spatial source differs'):
                load_case('mismatched-source', directory)


if __name__ == '__main__':
    unittest.main()

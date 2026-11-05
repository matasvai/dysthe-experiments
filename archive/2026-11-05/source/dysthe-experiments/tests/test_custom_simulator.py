"""Custom-envelope contract, including two tiny real propagation smoke tests.

Preview-only rejection tests must not queue work. Numerical refinements are
deliberately outside this bounded UI/parser regression suite.
"""
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from dysthe_core.reference import spatial
from dysthe_experiments import simulator
from dysthe_experiments.simulator_server import Workbench, make_handler


FORMULA = r"A \exp(-\tau^2/(2 w^2)) \exp(i(k \tau+c \tau^2))"
SYMBOLS = dict(A=.8, w=2., k=.3, c=.01)
TINY = dict(n=256, length=40., dxi=.0025, xi_end=.005, samples=3)


def parameters(formula=FORMULA, symbols=None):
    return dict(formula=formula, symbols=SYMBOLS.copy() if symbols is None else symbols)


class CustomSimulatorTests(unittest.TestCase):
    def assert_preview_blocked(self, p, settings=None):
        """A semantic screen may reject explicitly or return a disabled preview."""
        try:
            result = simulator.preview_custom(p, TINY if settings is None else settings)
        except ValueError:
            return
        self.assertFalse(result['can_run'], result.get('warnings'))

    def test_gaussian_preview_agrees_with_existing_complex_initializer(self):
        p = parameters()
        preview = simulator.preview_custom(p, TINY)
        self.assertTrue(preview['can_run'], preview.get('warnings'))
        tau = np.asarray(preview['tau'])
        actual = np.asarray(preview['real'])+1j*np.asarray(preview['imag'])
        expected = SYMBOLS['A']*np.exp(-tau**2/(2*SYMBOLS['w']**2))*np.exp(
            1j*(SYMBOLS['k']*tau+SYMBOLS['c']*tau**2))
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(preview['amplitude'], abs(expected), atol=2e-14)
        self.assertGreater(np.max(abs(actual.imag)), .1)
        model = spatial.SpatialDysthe(TINY['n'], TINY['length'], .025)
        gaussian, _, _ = simulator.initial_field('gaussian', model, dict(
            amplitude=SYMBOLS['A'], width=SYMBOLS['w'],
            wavenumber=SYMBOLS['k'], chirp=SYMBOLS['c']))
        custom, reference, source = simulator.initial_field('custom', model, p)
        self.assertIsNone(reference)
        np.testing.assert_allclose(custom, gaussian, rtol=2e-14, atol=2e-14)
        self.assertEqual(preview['provenance']['formula'], FORMULA)
        self.assertEqual(preview['provenance']['symbols'], SYMBOLS)
        self.assertEqual(source['formula'], FORMULA)
        json.dumps(preview, allow_nan=False)

    def test_sech_and_scalar_broadcast_preview(self):
        preview = simulator.preview_custom(parameters(
            r"A \sech(\tau/w) \exp(i k \tau)", dict(A=.5, w=1., k=.2)),
            {**TINY, 'n': 512})
        self.assertTrue(preview['can_run'], preview.get('warnings'))
        tau = np.asarray(preview['tau'])
        expected = .5/np.cosh(tau)*np.exp(.2j*tau)
        np.testing.assert_allclose(np.asarray(preview['real'])+1j*np.asarray(preview['imag']),
                                   expected, rtol=2e-14, atol=2e-14)
        constant = simulator.preview_custom(parameters('0.5', {}), TINY)
        self.assertTrue(constant['can_run'], constant.get('warnings'))
        np.testing.assert_allclose(constant['real'], .5)
        np.testing.assert_allclose(constant['imag'], 0.)
        self.assertTrue(constant['warnings'], 'Nondecay/broad-field warning required')
        warning_text = json.dumps(constant['warnings']).lower()
        self.assertTrue(any(word in warning_text for word in ('boundary', 'decay', 'periodic')))
        self.assertTrue(any(word in warning_text for word in ('circular', 'center', 'centroid')))

    def test_zero_extreme_and_aliased_samples_cannot_run(self):
        for p in (parameters('0', {}), parameters('1e-300', {}),
                  parameters('1e300', {}), parameters('5', {}),
                  parameters(r'\exp(1000)', {})):
            with self.subTest(parameters=p):
                self.assert_preview_blocked(p)
        # A whole-grid carrier aliases exactly to a smooth Gaussian at N points.
        # An independently sampled 2N grid must reveal the missing carrier.
        aliased = parameters(r'\exp(-\tau^2/8) \exp(i k \tau)',
                             dict(k=2*np.pi*TINY['n']/TINY['length']))
        self.assert_preview_blocked(aliased)

    def test_nonlocalized_field_does_not_invent_packet_center_metrics(self):
        model = spatial.SpatialDysthe(TINY['n'], TINY['length'], .025)
        field = np.full(model.n, .5, dtype=np.complex128)
        row = simulator.diagnostics(model, field, field, None, model.invariants(field))
        self.assertLess(row['circular_concentration'], 1e-12)
        for metric in ('circular_center', 'rms_width', 'tail_action_fraction'):
            self.assertIsNone(row[metric], metric)
        json.dumps(row, allow_nan=False)

    def test_custom_settings_and_symbol_types_rejected_before_work(self):
        invalid = [
            (parameters(), {'n': True}), (parameters(), {'n': 16384}),
            (parameters(), {'length': float('nan')}),
            (parameters(), {'xi_end': float('inf')}),
            (parameters(), {'epsilon': .03}),
            (parameters(), {'samples': False}),
            (parameters(), {'n': 8192, 'xi_end': 4., 'dxi': .0003125}),
            (parameters(symbols=dict(A=True, w=2., k=.3, c=0.)), {}),
            (parameters(symbols=dict(A=float('nan'), w=2., k=.3, c=0.)), {}),
            (parameters(symbols=dict(A='1', w=2., k=.3, c=0.)), {}),
            (parameters(symbols=dict(A=[1], w=2., k=.3, c=0.)), {}),
            ({'formula': FORMULA, 'symbols': SYMBOLS, 'path': '/tmp/no'}, {}),
        ]
        for p, settings in invalid:
            with self.subTest(parameters=p, settings=settings):
                with self.assertRaises(ValueError):
                    simulator.validate_request('custom', p, settings)

    def test_fingerprint_tracks_formula_parameters_and_every_run_setting(self):
        preview = simulator.preview_custom(parameters(), TINY)
        fingerprint = preview['fingerprint']
        self.assertEqual(fingerprint, simulator.preview_custom(parameters(), TINY)['fingerprint'])
        variants = [(parameters(FORMULA+' + 0'), TINY),
                    (parameters(symbols={**SYMBOLS, 'A': .7}), TINY)]
        for name, value in dict(n=512, length=80., dxi=.00125,
                                xi_end=.01, samples=2).items():
            variants.append((parameters(), {**TINY, name: value}))
        for p, settings in variants:
            with self.subTest(parameters=p, settings=settings):
                self.assertNotEqual(fingerprint,
                                    simulator.preview_custom(p, settings)['fingerprint'])

    def test_tiny_complex_run_keeps_raw_field_and_reproducible_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'custom'
            report = simulator.simulate('custom', output, parameters(), TINY)
            self.assertEqual(report['steps'], 2)
            self.assertEqual(report['source']['formula'], FORMULA)
            self.assertEqual(report['source']['symbols'], SYMBOLS)
            self.assertFalse(report['qualification']['checks_available'])
            self.assertIn('unqualified', report['qualification']['status'])
            with np.load(output/'fields.npz', allow_pickle=False) as archive:
                self.assertEqual(archive['u'].dtype, np.dtype('complex128'))
                self.assertEqual(archive['u'].shape, (3, TINY['n']))
                self.assertEqual(archive['raw_initial'].dtype, np.dtype('complex128'))
                self.assertEqual(archive['provenance_json'].dtype.kind, 'U')
                provenance = json.loads(archive['provenance_json'].item())
                self.assertEqual(provenance['formula'], FORMULA)
                self.assertEqual(provenance['symbols'], SYMBOLS)
                self.assertEqual(provenance['preview_fingerprint'],
                                 report['source']['preview_fingerprint'])
                np.testing.assert_allclose(archive['u'][0], archive['initial'], atol=1e-15)
                np.testing.assert_allclose(archive['raw_initial'], archive['initial'], atol=1e-10)
                self.assertGreater(np.max(abs(archive['u'].imag)), .1)
                viewer = json.loads((output/'viewer.json').read_text())
                restored = np.asarray(viewer['real'])+1j*np.asarray(viewer['imag'])
                np.testing.assert_array_equal(restored, archive['u'])
            self.assertEqual(viewer['initial_provenance']['formula'], FORMULA)
            self.assertEqual(viewer['initial_provenance']['symbols'], SYMBOLS)
            json.dumps(report, allow_nan=False)
            json.dumps(viewer, allow_nan=False)


class CustomSimulatorHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workbench = Workbench(Path(self.temp.name))
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(self.workbench))
        self.base = f'http://127.0.0.1:{self.server.server_port}'
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.workbench.executor.shutdown(wait=True)
        self.temp.cleanup()

    def request(self, path, payload=None, body=None, headers=None):
        request = Request(self.base+path,
                          data=body if body is not None else (
                              None if payload is None else json.dumps(payload).encode()),
                          headers={'Content-Type': 'application/json', **(headers or {})})
        with urlopen(request, timeout=30) as response:
            return response.status, response.read()

    def test_preview_then_two_step_run_download_and_stale_token_rejection(self):
        payload = dict(case_id='custom', parameters=parameters(), settings=TINY)
        status, body = self.request('/api/preview', payload)
        self.assertEqual(status, 200)
        preview = json.loads(body)
        self.assertTrue(preview['can_run'])
        run = {**payload, 'preview_fingerprint': preview['fingerprint']}
        for changed in ({**run, 'parameters': parameters(symbols={**SYMBOLS, 'A': .7})},
                        {**run, 'settings': {**TINY, 'xi_end': .01}},
                        payload, {**run, 'preview_fingerprint': 'bogus'}):
            with self.subTest(payload=changed):
                with self.assertRaises(HTTPError) as context:
                    self.request('/api/run', changed)
                self.assertIn(context.exception.code, (400, 409))
                context.exception.close()
                self.assertEqual(self.workbench.jobs, {})
        status, body = self.request('/api/run', run)
        self.assertEqual(status, 202)
        job = json.loads(body)
        deadline = time.monotonic()+30
        while time.monotonic() < deadline:
            _, body = self.request('/api/jobs/'+job['job_id'])
            job = json.loads(body)
            if job['status'] in ('complete', 'error'):
                break
            threading.Event().wait(.01)
        self.assertEqual(job['status'], 'complete', job)
        _, body = self.request(job['result_url'])
        viewer = json.loads(body)
        self.assertEqual(viewer['initial_provenance']['formula'], FORMULA)
        self.assertEqual(viewer['source'], 'live')
        _, body = self.request(job['download_url'])
        with np.load(io.BytesIO(body), allow_pickle=False) as archive:
            self.assertEqual(archive['u'].dtype, np.dtype('complex128'))
            self.assertEqual(archive['u'].shape, (3, 256))
            self.assertIn('raw_initial', archive.files)
            self.assertEqual(json.loads(archive['provenance_json'].item())['formula'], FORMULA)
            self.assertGreater(np.max(abs(archive['u'].imag)), .1)
        # A service restart loses its in-memory dictionary but must retain the
        # user's completed result/download links from this feature installation.
        self.workbench.jobs.clear()
        _, body = self.request('/api/jobs/'+job['job_id'])
        restored = json.loads(body)
        self.assertEqual(restored['status'], 'complete')
        self.assertTrue(restored['restored_from_disk'])
        _, body = self.request(restored['result_url'])
        self.assertEqual(json.loads(body)['case_id'], 'custom')

    def test_malformed_preview_and_nonfinite_json_create_no_jobs(self):
        payloads = [[], dict(case_id=[]), dict(case_id='custom', parameters=[]),
                    dict(case_id='custom', parameters=parameters(), surprise=True),
                    dict(case_id='custom', parameters=parameters('0', {})),
                    dict(case_id='custom', parameters=parameters(r'\unknown(\tau)', {}))]
        for payload in payloads:
            with self.subTest(payload=payload):
                try:
                    status, body = self.request('/api/preview', payload)
                except HTTPError as error:
                    self.assertEqual(error.code, 400)
                    error.close()
                else:
                    self.assertEqual(status, 200)
                    self.assertFalse(json.loads(body)['can_run'])
                self.assertEqual(self.workbench.jobs, {})
        for token in ('NaN', 'Infinity', '-Infinity'):
            body = ('{"case_id":"custom","parameters":{"formula":"A",'
                    '"symbols":{"A":'+token+'}}}').encode()
            with self.subTest(token=token):
                with self.assertRaises(HTTPError) as context:
                    self.request('/api/preview', body=body)
                self.assertEqual(context.exception.code, 400)
                context.exception.close()
                self.assertEqual(self.workbench.jobs, {})
        self.assertFalse((Path(self.temp.name)/'live').exists())


if __name__ == '__main__':
    unittest.main()

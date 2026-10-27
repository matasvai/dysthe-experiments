"""Diagnostic exports must preserve failures without shipping research arrays."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from dysthe_experiments.export_evidence import export_run, sha256, verify_export


class EvidenceExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.run = self.base / 'cluster-run'
        self.run.mkdir()
        self.destination = self.base / 'export'
        self.write('config.json', {'model_id': 'water-wave-dysthe-spatial-v1', 'seeds': [11]})
        self.write('cases.json', [{'case_id': 'random-001', 'split': 'test'}])
        self.write('identity.json', {'source_sha256': {'dysthe_experiments.long_distance': 'a' * 64}})
        self.write('summary.json', {'complete': False, 'all_requested_gates_passed': False})
        self.write('tasks/pinn-000.json', {
            'stage': 'pinn', 'case_id': 'random-001', 'seed': 11, 'status': 'execution_error',
            'error': 'ValueError: persisted inlet changed',
            'traceback': 'Traceback (most recent call last):\n  persisted inlet changed\n',
            'results': [{'horizon': 4., 'qualified': False, 'status': 'accuracy_gate_failed'}],
        })
        raw = self.run / 'pinn/random-001/seed-11/xi-4/best-0.pt'
        raw.parent.mkdir(parents=True)
        raw.write_bytes(b'opaque checkpoint bytes; never deserialize')
        self.raw = raw
        self.write('pinn/random-001/seed-11/xi-4/result.json', {
            'errors': {'l2': .003, 'h1': .02, 'h2': .18},
            'stations': [{'xi': .5, 'h2': .17}], 'qualified': False,
            'checkpoints': {'best-0.pt': sha256(raw)},
        })
        self.write('launcher/pinn-0/a100-runtime-lock.json', {
            'source_sha256': {'dysthe_learning/pilot_models.py': 'b' * 64},
            'runtime': {'gpu': 'NVIDIA A100', 'torch': '2.10.0'},
        })
        self.write('inlets.json', {'artifact': {'path': str(self.run / 'inlets.npz'), 'sha256': hashlib.sha256(b'inlets').hexdigest()}})
        (self.run / 'inlets.npz').write_bytes(b'inlets')
        self.write('logs/slurm-test.out', 'ValueError: persisted inlet changed\n', json_value=False)

    def write(self, name, value, *, json_value=True):
        path = self.run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + '\n' if json_value else value)

    def test_all_task_horizon_runtime_and_failure_metadata_are_preserved(self):
        manifest = export_run(self.run, self.destination)
        self.assertEqual(verify_export(self.destination), manifest)
        self.assertEqual(manifest['raw_artifact_count'], 2)
        for relative in ('tasks/pinn-000.json', 'pinn/random-001/seed-11/xi-4/result.json',
                         'launcher/pinn-0/a100-runtime-lock.json', 'identity.json',
                         'inlets.json', 'logs/slurm-test.out'):
            self.assertEqual((self.destination / 'metadata' / relative).read_bytes(),
                             (self.run / relative).read_bytes())
        inventory = json.loads((self.destination / 'artifact-inventory.json').read_text())
        raws = [row for row in inventory['files'] if row['kind'] == 'raw_artifact']
        self.assertEqual(len(raws), 2)
        self.assertTrue(all(not row['copied'] and len(row['sha256']) == 64 for row in raws))
        self.assertFalse(list(self.destination.rglob('*.npz')))
        self.assertFalse(list(self.destination.rglob('*.pt')))
        self.assertEqual(manifest['source_integrity_findings'], [])
        self.assertIn('tasks/reference-000.json', manifest['missing_expected_evidence'])
        self.assertNotIn('pinn/random-001/seed-11/xi-4/result.json', manifest['missing_expected_evidence'])
        sources = json.loads((self.destination / 'source-manifests.json').read_text())
        self.assertIn('metadata/identity.json', [item['path'] for item in sources['records']])

    def test_missing_horizon_result_and_malformed_json_are_not_silently_lost(self):
        (self.run / 'pinn/random-001/seed-11/xi-4/result.json').unlink()
        self.write('tasks/reference-000.json', '{"status": "running",', json_value=False)
        manifest = export_run(self.run, self.destination)
        self.assertIn('pinn/random-001/seed-11/xi-4/result.json', manifest['missing_expected_evidence'])
        self.assertEqual(manifest['malformed_json_retained'][0]['path'], 'tasks/reference-000.json')
        self.assertEqual((self.destination / 'metadata/tasks/reference-000.json').read_bytes(),
                         (self.run / 'tasks/reference-000.json').read_bytes())

    def test_interrupted_metadata_and_unrecognized_artifacts_remain_inventoried(self):
        self.write('tasks/reference-000.json.tmp', '{"status": ', json_value=False)
        (self.run / 'inlets.npz.tmp').write_bytes(b'interrupted inlet archive')
        (self.run / 'last.tmp').write_bytes(b'interrupted opaque checkpoint')
        manifest = export_run(self.run, self.destination)
        self.assertTrue((self.destination / 'metadata/tasks/reference-000.json.tmp').is_file())
        self.assertEqual(manifest['raw_artifact_count'], 3)
        inventory = json.loads((self.destination / 'artifact-inventory.json').read_text())
        item = next(row for row in inventory['files'] if row['path'] == 'last.tmp')
        self.assertEqual(item['sha256'], sha256(self.run / 'last.tmp'))
        self.assertFalse(item['copied'])

    def test_changed_raw_artifact_is_reported_and_failed_evidence_still_exports(self):
        self.raw.write_bytes(b'changed checkpoint')
        manifest = export_run(self.run, self.destination)
        findings = manifest['source_integrity_findings']
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]['status'], 'declared_checksum_mismatch')
        self.assertEqual(findings[0]['observed_sha256'], sha256(self.raw))
        verify_export(self.destination)

    def test_embedded_task_checkpoints_and_canonical_manifests_use_run_paths(self):
        prediction = self.raw.parent / 'predictions.npz'
        prediction.write_bytes(b'predictions')
        result = json.loads((self.raw.parent / 'result.json').read_text())
        result.update(horizon=4., artifact={'path': str(prediction), 'sha256': sha256(prediction)})
        task_path = self.run / 'tasks/pinn-000.json'
        task = json.loads(task_path.read_text())
        task['results'] = [result]
        task_path.write_text(json.dumps(task))
        self.write('launcher/pinn-0/a100-job-101.json', {
            'canonical_inlets': {'manifest': 'inlets.json', 'sha256': sha256(self.run / 'inlets.json')},
        })
        manifest = export_run(self.run, self.destination)
        self.assertEqual(manifest['source_integrity_findings'], [])

    def test_aggregate_pinn_summary_uses_exact_method_seed_case_checkpoint_root(self):
        root = self.raw.parent
        checkpoint = root / 'pinn-slabs-11-random-001/best-0.pt'
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b'actual case checkpoint')
        item = {'method': 'pinn_slabs', 'seed': 11, 'case_id': 'random-001',
                'checkpoints': {'best-0.pt': sha256(checkpoint)}}
        (root / 'pinn-summary.json').write_text(json.dumps([item]))
        (checkpoint.parent / 'summary.json').write_text(json.dumps(item))
        manifest = export_run(self.run, self.destination)
        self.assertEqual(manifest['source_integrity_findings'], [])
        # A different best-0.pt already exists in xi-H. Its presence must not
        # hide removal of the exact per-case checkpoint named by the aggregate.
        checkpoint.unlink()
        second = export_run(self.run, self.base / 'export-missing')
        findings = second['source_integrity_findings']
        self.assertEqual(len(findings), 2)
        self.assertTrue(all(x['status'] == 'missing_declared_artifact' for x in findings))
        self.assertTrue(all(x['artifact'].endswith('pinn-slabs-11-random-001/best-0.pt') for x in findings))

    def test_tampering_with_exported_metadata_or_manifest_is_detected(self):
        export_run(self.run, self.destination)
        path = self.destination / 'metadata/tasks/pinn-000.json'
        original = path.read_bytes()
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'checksum/size mismatch'):
            verify_export(self.destination)
        path.write_bytes(original)
        (self.destination / 'evidence-manifest.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'manifest checksum mismatch'):
            verify_export(self.destination)

    def test_size_limits_fail_before_creating_an_incomplete_export(self):
        for options in ({'max_file_bytes': 10}, {'max_total_bytes': 10}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, 'limit'):
                export_run(self.run, self.destination, **options)
            self.assertFalse(self.destination.exists())
        self.write('large.log', 'x' * 10000, json_value=False)
        with self.assertRaisesRegex(ValueError, 'large.log'):
            export_run(self.run, self.destination, max_file_bytes=9000)

    def test_existing_destination_inside_run_and_symlinks_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            export_run(self.run, self.run / 'export')
        self.destination.mkdir()
        with self.assertRaisesRegex(ValueError, 'fresh'):
            export_run(self.run, self.destination)
        self.destination.rmdir()
        (self.run / 'outside.json').symlink_to(self.base / 'unrelated.json')
        with self.assertRaisesRegex(ValueError, 'regular files'):
            export_run(self.run, self.destination)

    def test_run_modified_while_hashing_is_rejected(self):
        original_hash = sha256

        def mutate_after_hash(path):
            result = original_hash(path)
            if Path(path) == self.raw:
                self.write('identity.json', {'source_sha256': {'changed': 'c' * 64}})
            return result

        with patch('dysthe_experiments.export_evidence.sha256', side_effect=mutate_after_hash):
            with self.assertRaisesRegex(ValueError, 'changed during export'):
                export_run(self.run, self.destination)
        self.assertFalse(self.destination.exists())

    def test_cli_exports_and_verifies_without_loading_arrays(self):
        for arguments in (['--run', str(self.run), '--destination', str(self.destination)],
                          ['--verify', str(self.destination)]):
            result = subprocess.run([sys.executable, '-m', 'dysthe_experiments.export_evidence', *arguments],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('passed', result.stdout)


if __name__ == '__main__':
    unittest.main()

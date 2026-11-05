import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('scope_guard', ROOT / 'scripts/check_scope.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)

class Scope(unittest.TestCase):
    def test_repository_scope(self):
        guard.check(ROOT)

    def test_rejects_different_physical_setting_and_dimension(self):
        for key, value in [('physical_system', 'optical-pulses'), ('model_id', 'vlasov-poisson'),
                           ('profile_dimensions', 3), ('axis_order', ['t', 'x', 'y', 'z'])]:
            scope = dict(guard.APPROVED)
            scope[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Scope violation'):
                guard.validate_scope(scope)

    def test_rejects_mixed_physics_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'scope.json').write_text(json.dumps(guard.APPROVED))
            (root / 'configs').mkdir()
            (root / 'configs/pilot.json').write_text(json.dumps({
                'model_id': 'optical-dysthe-127-v1', 'physical_system': 'optical-pulses'}))
            with self.assertRaisesRegex(ValueError, 'Scope violation'):
                guard.check(root)

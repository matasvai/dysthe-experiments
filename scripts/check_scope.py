"""Reject incompatible research scope before building or publishing artifacts."""
import json
from pathlib import Path

APPROVED = {
    'schema_version': 1,
    'model_id': 'water-wave-dysthe-spatial-v1',
    'physical_system': 'deep-water-gravity-waves',
    'formulation': 'spatial-free-surface-envelope',
    'profile_dimensions': 1,
    'axis_order': ['xi', 'tau'],
    'case_schema_version': 2,
}

def validate_scope(scope):
    if scope != APPROVED:
        raise ValueError('Scope violation: only the approved water-wave Dysthe contract is allowed; direct human authorization is required to change it')

def check(root):
    validate_scope(json.loads((root / 'scope.json').read_text()))
    for name in ('examples/case.json', 'configs/pilot.json', 'artifacts/manifest.example.json', 'evidence/claims.json', 'family.json'):
        path = root / name
        if not path.exists():
            continue
        value = json.loads(path.read_text())
        key = 'scope' if name == 'family.json' else 'model_id'
        if value.get(key) != APPROVED['model_id'] or value.get('physical_system') != APPROVED['physical_system']:
            raise ValueError(f'Scope violation in {name}: water-wave identity is required')
        for claim in value.get('claims', []):
            if claim.get('model_id') != APPROVED['model_id']:
                raise ValueError(f'Scope violation in {name}: claim uses a different equation')

if __name__ == '__main__':
    check(Path(__file__).resolve().parents[1])
    print('Water-wave scope checks passed; numerical provenance still requires review.')

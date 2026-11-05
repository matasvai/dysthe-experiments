from dysthe_core import MODEL_ID, PHYSICAL_SYSTEM

if MODEL_ID != 'water-wave-dysthe-spatial-v1' or PHYSICAL_SYSTEM != 'deep-water-gravity-waves':
    raise RuntimeError('Scope violation: learning requires the approved water-wave core')

METHODS = {
    'hybrid': {'target': 'field', 'branch': 'codex/hybrid-solver', 'implemented': False},
    'reduced': {'target': 'field', 'branch': 'codex/reduced-model', 'implemented': False},
    'deeponet': {'target': 'field', 'branch': 'codex/deeponet', 'implemented': False},
    'fno': {'target': 'field', 'branch': 'codex/fno-baseline', 'implemented': False},
    'active-gp': {'target': 'observable', 'branch': 'codex/active-learning', 'implemented': False},
}

def require_implemented(name):
    if name not in METHODS:
        raise ValueError(f'Unknown method: {name}')
    if not METHODS[name]['implemented']:
        raise NotImplementedError(f'{name} is planned for {MODEL_ID}; implement and validate its adapter first')
    return METHODS[name]

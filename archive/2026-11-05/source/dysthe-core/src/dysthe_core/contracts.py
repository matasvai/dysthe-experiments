"""Water-wave metadata checks; these do not certify array provenance or accuracy."""
import math

MODEL_ID = 'water-wave-dysthe-spatial-v1'
PHYSICAL_SYSTEM = 'deep-water-gravity-waves'

def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

def require_water_wave_model(record):
    if record.get('model_id') != MODEL_ID or record.get('physical_system') != PHYSICAL_SYSTEM:
        raise ValueError('Scope violation: approved water-wave model and physical system are required')

def validate_case(case):
    require_water_wave_model(case)
    if type(case.get('schema_version')) is not int or case['schema_version'] != 2:
        raise ValueError('Water-wave cases require schema version 2; legacy arrays cannot be relabeled')
    if case.get('boundary') != 'periodic' or case.get('axis_order') != ['xi', 'tau']:
        raise ValueError('Expected periodic water-wave envelope layout [xi,tau]')
    if case.get('field_encoding') != 'complex128':
        raise ValueError('Reference fields must declare complex128')
    for key in ('case_id', 'initial_condition_id', 'family'):
        if not isinstance(case.get(key), str) or not case[key].strip():
            raise ValueError(f'Missing {key}')
    shape, lengths = case.get('shape'), case.get('lengths')
    if not isinstance(shape, list) or len(shape) != 1 or type(shape[0]) is not int or shape[0] < 2:
        raise ValueError('Expected one profile grid size >= 2')
    if not isinstance(lengths, list) or len(lengths) != 1 or not finite_number(lengths[0]) or lengths[0] <= 0:
        raise ValueError('Expected one finite positive retarded-time period')
    positions = case.get('propagation_positions')
    if not isinstance(positions, list) or len(positions) < 2 or any(not finite_number(x) for x in positions):
        raise ValueError('Expected at least two finite propagation positions')
    if positions[0] != 0 or any(b <= a for a, b in zip(positions, positions[1:])):
        raise ValueError('Propagation positions must start at zero and increase strictly')
    if 'times' in case:
        raise ValueError('Use propagation_positions for xi, not the legacy times field')
    physics = case.get('physics')
    if not isinstance(physics, dict) or set(physics) != {'epsilon'}:
        raise ValueError('Only the water-wave steepness coefficient epsilon is supported')
    if not finite_number(physics['epsilon']) or physics['epsilon'] < 0:
        raise ValueError('epsilon must be finite and nonnegative; zero is the NLS control')

def validate_splits(records):
    """Keep every initial condition and its refinements in one data split.

    The producer must assign the same initial_condition_id to all variants of
    the same physical initial field. This check cannot discover mislabeled data.
    """
    if not records:
        raise ValueError('A split manifest cannot be empty')
    seen, groups = set(), {}
    for record in records:
        case_id, group, split = (record.get(k) for k in ('case_id', 'initial_condition_id', 'split'))
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError('Case IDs must be nonempty and unique')
        if not isinstance(group, str) or not group or split not in {'train', 'validation', 'test'}:
            raise ValueError('Invalid initial-condition group or split')
        if group in groups and groups[group] != split:
            raise ValueError('Initial-condition leakage across splits')
        seen.add(case_id)
        groups[group] = split

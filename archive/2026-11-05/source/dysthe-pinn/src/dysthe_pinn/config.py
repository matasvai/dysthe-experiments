"""JSON configuration with explicit validation; no hidden experiment defaults."""
import json
import math


def read_config(path):
    with open(path) as handle:
        config = json.load(handle)
    return validate(config)


def validate(c):
    if c.get('equation') != 'Dysthe3D-1.27':
        raise ValueError('Only full Dysthe3D equation (1.27) is supported')
    if c.get('dtype') not in ('float64', 'float32'):
        raise ValueError('dtype must be float64 or float32')
    if c['domain']['T'] <= 0 or len(c['domain']['lengths']) != 3 or min(c['domain']['lengths']) <= 0:
        raise ValueError('Positive T and three positive domain lengths required')
    if c['physics']['epsilon1'] <= 0:
        raise ValueError('Equation (1.27) assumes epsilon1 > 0')
    if not all(math.isfinite(c['physics'][k]) for k in ('epsilon1', 'epsilon2')):
        raise ValueError('Finite coefficients required')
    power = c['initial']['power']
    if type(power) is not int or power < 1 or c['initial']['amplitude'] <= 0:
        raise ValueError('Initial power must be a positive integer and amplitude positive')
    if any(type(n) is not int or n % 2 or n <= 4*power for n in c['reference']['shape']) or len(c['reference']['shape']) != 3:
        raise ValueError('Three even reference sizes strictly >4*initial.power required')
    for key in ('steps', 'batch_size', 'log_every', 'checkpoint_every', 'validation_points'):
        if c['training'][key] <= 0:
            raise ValueError(f'training.{key} must be positive')
    if c['reference']['dt'] <= 0 or c['reference']['snapshots'] < 2:
        raise ValueError('Positive reference dt and at least two snapshots required')
    for key in ('width', 'depth', 'harmonics'):
        if c['network'][key] < 1:
            raise ValueError(f'network.{key} must be positive')
    return c

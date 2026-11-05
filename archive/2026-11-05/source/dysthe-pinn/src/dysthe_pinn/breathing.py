"""Periodic localization diagnostics; a moving maximum alone is not a breather."""
import numpy as np


def localization(model, u):
    rho = abs(u)**2
    total = float(rho.sum())
    dv = model.volume_element
    result = {'effective_volume': float(dv*total**2/np.sum(rho*rho))}
    peak_index = np.unravel_index(np.argmax(rho), rho.shape)
    for j, (axis, length, name) in enumerate(zip(model.axes, model.lengths, 'xyz')):
        marginal = rho.sum(axis=tuple(k for k in range(3) if k != j))
        moment = np.sum(marginal*np.exp(2j*np.pi*axis/length))/total
        center = float(np.angle(moment)*length/(2*np.pi))
        distance = (axis-center+length/2) % length-length/2
        result[f'center_{name}'] = center
        result[f'center_strength_{name}'] = float(abs(moment))
        result[f'width_{name}'] = float(np.sqrt(np.sum(marginal*distance**2)/total))
        result[f'peak_{name}'] = float(axis[peak_index[j]])
    result['centers_well_defined'] = all(result[f'center_strength_{q}'] >= .05 for q in 'xyz')
    return result

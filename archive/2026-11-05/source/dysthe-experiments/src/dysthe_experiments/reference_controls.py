"""Independent analytic controls and audit of every stored profile."""
import hashlib
import json
import argparse
from pathlib import Path
import time

import numpy as np
from dysthe_core.reference.spatial import SpatialDysthe, nls_profile, translate, solve_profile
from dysthe_core.reference.validation import invariant_directional_rates


def propagate(m, f, h, T):
    v = np.fft.fft(f); step = m.stepper(h)
    for _ in range(round(T/h)):
        v = step(v)
    return np.fft.ifft(v)


def load_audited_legacy_profile(path):
    """Require the established legacy free-surface identity and xi=0 grid."""
    path = Path(path)
    sidecar = path.with_suffix('.json')
    metadata = json.loads(sidecar.read_text())
    required = dict(kind='dysthe', epsilon1=1., epsilon2=1., nonlinear=True)
    if any(metadata.get(key) != value for key, value in required.items()):
        raise ValueError(f'Incompatible legacy water-wave profile identity: {sidecar}')
    if len(metadata['shape']) != 1 or len(metadata['lengths']) != 1:
        raise ValueError('Expected one retarded-time profile coordinate')
    model = SpatialDysthe(metadata['shape'][0], metadata['lengths'][0], metadata['epsilon'])
    with np.load(path, allow_pickle=False) as archive:
        if archive['u'].shape != (1, model.n) or not np.array_equal(archive['t'], [0]):
            raise ValueError('Expected one legacy profile at xi=0')
        np.testing.assert_allclose(archive['x0'], model.tau, atol=1e-12, rtol=0)
        profile = model.field(archive['u'][0]).copy()
    return model, profile, metadata, {
        'profile_path': str(path.resolve()),
        'sidecar_path': str(sidecar.resolve()),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'sidecar_sha256': hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        'legacy_metadata_identity_verified': True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    result = {'nls': [], 'plane_wave': [], 'profiles': [], 'moving_dysthe': {}}
    result['invariant_directional_audit'] = {}
    for label, coeff in (
        ('low_modes', {-7: .08+.03j, -2: .2-.1j, 0: .35, 5: -.15+.06j}),
        ('near_cutoff', {-29: .08+.03j, -17: .2-.1j, 0: .35, 23: -.15+.06j})):
        m = SpatialDysthe(128, 2*np.pi, .07)
        f = sum(a*np.exp(1j*k*m.tau) for k,a in coeff.items())
        result['invariant_directional_audit'][label] = dict(
            coefficients={str(k):[a.real,a.imag] for k,a in coeff.items()},
            n=m.n,length=m.length,epsilon=m.epsilon,
            full=invariant_directional_rates(m,f),
            projected=invariant_directional_rates(m,f,projected=True))
    m = SpatialDysthe(2048, 80., 0.)
    f = m.project(nls_profile(m.tau, 2., 1.))
    for h in (.04, .02, .01, .005):
        u = propagate(m, f, h, 2.)
        exact = translate(m, f, 2.)*np.exp(-4j)
        result['nls'].append(dict(dxi=h, xi_end=2.,
            raw_relative_l2=float(np.linalg.norm(u-exact)/np.linalg.norm(f)),
            invariant_drifts={k:float((m.invariants(u)[k]-b)/abs(b))
                for k,b in m.invariants(f).items()}))
    m = SpatialDysthe(128, 2*np.pi, .025)
    for amplitude,k in ((.6,0),(.6,3),(.6,-3)):
        f = amplitude*np.exp(1j*k*m.tau)
        omega = amplitude**2-k*k+6*m.epsilon*k*amplitude**2
        rhs = np.fft.ifft(m.rhs_hat(np.fft.fft(f)))
        u = propagate(m,f,.001,2.)
        result['plane_wave'].append(dict(amplitude=amplitude,k=k,omega=omega,
            rhs_relative_l2=float(np.linalg.norm(rhs+1j*omega*f)/np.linalg.norm(f)),
            propagated_relative_l2=float(np.linalg.norm(u-f*np.exp(-2j*omega))/np.linalg.norm(f))))
    root = args.profiles
    for path in sorted(root.glob('profile_mu*.npz')):
        m, f, metadata, provenance = load_audited_legacy_profile(path)
        result['profiles'].append(dict(name=path.name,
            **provenance,
            n=m.n,length=m.length,epsilon=m.epsilon,mu=metadata['mu'],speed=metadata['speed'],
            epsilon_peak=float(m.epsilon*max(abs(f))),
            **m.profile_residual(f,metadata['mu'],metadata['speed'])))
    m = SpatialDysthe(4096,160.,.025)
    f,info = solve_profile(m,2.,1.,max_iterations=200)
    result['moving_dysthe']['profile'] = info
    result['moving_dysthe']['propagation'] = []
    for h in (.00125,.000625):
        u = propagate(m,f,h,2.)
        exact = translate(m,f,2.)*np.exp(-4j)
        result['moving_dysthe']['propagation'].append(dict(dxi=h,xi_end=2.,
            raw_relative_l2=float(np.linalg.norm(u-exact)/np.linalg.norm(f))))
    result['wall_seconds'] = time.perf_counter()-start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='profiles'},indent=2))


if __name__ == '__main__':
    main()

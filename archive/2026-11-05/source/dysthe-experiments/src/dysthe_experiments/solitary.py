"""Bounded single-case solitary-wave pilot; no ML or scheduler submission."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import re
import time

import numpy as np
import scipy
from dysthe_core import MODEL_ID, PHYSICAL_SYSTEM
from dysthe_core.reference import spatial


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(config):
    if config.get('model_id') != MODEL_ID or config.get('physical_system') != PHYSICAL_SYSTEM:
        raise ValueError('Only the approved free-surface water-wave equation is accepted')
    if config.get('profile_coordinate') != 'tau' or config.get('evolution_coordinate') != 'xi':
        raise ValueError('Require tau profile coordinate and xi propagation distance')
    cases = config['cases']
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Unique nonempty case IDs required')
    for c in cases:
        spatial.SpatialDysthe(c['n'], c['length'], c['epsilon'])
        if c['n'] > 32768:
            raise ValueError('Pilot grid budget exceeded')
        if c.get('profile_mode') not in ('saved','recompute'):
            raise ValueError('Explicit saved or recompute profile mode required')
        if type(c.get('profile_max_iterations')) is not int or not 1 <= c['profile_max_iterations'] <= 500:
            raise ValueError('Explicit profile iteration budget from 1 through 500 required')
        for key in ('mu', 'speed', 'dxi', 'xi_end'):
            spatial._finite(c[key], key)
        if c['mu'] <= c['speed']**2/4 or c['dxi'] <= 0 or c['xi_end'] <= 0:
            raise ValueError('Positive profile multiplier, step and horizon required')
        if type(c['samples']) is not int or not 2 <= c['samples'] <= 101:
            raise ValueError('samples must be an integer from 2 through 101')
        if not c['perturbations'] or len({p['id'] for p in c['perturbations']}) != len(c['perturbations']):
            raise ValueError('Unique perturbation IDs required')
        for p in c['perturbations']:
            if not re.fullmatch(r'[a-zA-Z0-9_-]+',p['id']) or p['id'] in ('tau','xi','profile'):
                raise ValueError('Perturbation ID must be an unreserved archive identifier')
            if p['kind'] not in ('control','amplitude','phase','odd_amplitude'):
                raise ValueError('Unsupported perturbation')
            spatial._finite(p['delta'], 'delta')
            if abs(p['delta']) > .1 or (p['kind']=='control' and p['delta'] != 0):
                raise ValueError('Require small perturbations and an exact control')
    for key in ('full_residual_max','projection_relative_l2_max','epsilon_peak_screen',
                'epsilon_spectral_sigma_screen','epsilon_spectral_mean_screen','boundary_action_max','spectral_edge_action_max'):
        if spatial._finite(config['qualification'][key],key) <= 0:
            raise ValueError('Positive qualification thresholds required')
    return config


def load_profile(case, config_path):
    path = (config_path.parent / case['profile']).resolve()
    sidecar = path.with_suffix('.json')
    metadata = json.loads(sidecar.read_text())
    required = {'kind':'dysthe','epsilon':case['epsilon'],'mu':case['mu'],
                'speed':case['speed'],'epsilon1':1.,'epsilon2':1.,'nonlinear':True}
    if any(metadata.get(k) != v for k,v in required.items()):
        raise ValueError('Legacy metadata disagrees with the approved water-wave case')
    with np.load(path,allow_pickle=False) as data:
        if data['u'].shape != (1, metadata['shape'][0]) or not np.array_equal(data['t'],[0]):
            raise ValueError('Expected one saved profile at xi=0')
        profile = data['u'][0].copy()
        tau = data['x0'].copy()
    old = spatial.SpatialDysthe(metadata['shape'][0],metadata['lengths'][0],case['epsilon'])
    np.testing.assert_allclose(tau,old.tau,atol=1e-12,rtol=0)
    old.field(profile)
    model = spatial.SpatialDysthe(case['n'],case['length'],case['epsilon'])
    provenance = {'profile_path':str(path),'profile_sha256':digest(path),
                  'sidecar_sha256':digest(sidecar),'legacy_kind':'dysthe',
                  'source_grid':{'n':old.n,'length':old.length}}
    if case['profile_mode']=='recompute':
        profile, info = spatial.solve_profile(model,case['mu'],case['speed'],
            max_iterations=case['profile_max_iterations'])
        provenance['regenerated_profile'] = info
    else:
        if (old.n,old.length) != (model.n,model.length):
            raise ValueError('Saved mode requires the original grid; explicitly select recompute')
        provenance['regenerated_profile'] = False
    projection_error = np.linalg.norm(profile-model.project(profile))/np.linalg.norm(profile)
    return model,model.project(profile),float(projection_error),provenance


def perturb(model,profile,p):
    # Shifted localized factors and odd real amplitude break conjugate reflection.
    gaussian = np.exp(-((model.tau-1.3)/2.)**2)
    if p['kind']=='control':
        u = profile.copy()
    elif p['kind']=='phase':
        u = profile*np.exp(1j*p['delta']*gaussian)
    elif p['kind']=='amplitude':
        u = profile*(1+p['delta']*gaussian)
    else:
        u = profile*(1+p['delta']*(model.tau/2.)*np.exp(-(model.tau/2.)**2))
    projected=model.project(u)
    loss=float(np.linalg.norm(u-projected)/np.linalg.norm(u))
    return projected,loss


def run(config_path,case_id,output,max_steps=1000):
    config_path = Path(config_path).resolve()
    config = validate(json.loads(config_path.read_text()))
    matching = [c for c in config['cases'] if c['id']==case_id]
    if len(matching)!=1:
        raise ValueError('Select one declared case; this command never runs a sweep')
    c = matching[0]; q = config['qualification']
    steps = math.ceil(c['xi_end']/c['dxi']); h = c['xi_end']/steps
    if type(max_steps) is not int or max_steps<1 or steps*len(c['perturbations'])>max_steps:
        raise ValueError('Total step budget exceeded; inspect --describe before raising --max-steps')
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Output directory already exists; choose a fresh directory')
    started = time.perf_counter()
    m,f,projection_error,provenance = load_profile(c,config_path)
    residual = m.profile_residual(f,c['mu'],c['speed'])
    if not math.isfinite(projection_error) or any(not math.isfinite(v) for v in residual.values()):
        raise ValueError('Nonfinite profile qualification')
    if projection_error>q['projection_relative_l2_max'] or residual['full_periodic_relative_l2']>q['full_residual_max']:
        raise ValueError(f'Profile qualification failed: projection={projection_error}, residual={residual}')
    profile_diagnostics = spatial.packet_diagnostics(m,f)
    if profile_diagnostics['epsilon_peak']>q['epsilon_peak_screen'] or profile_diagnostics['epsilon_spectral_sigma']>q['epsilon_spectral_sigma_screen'] or abs(profile_diagnostics['epsilon_spectral_mean'])>q['epsilon_spectral_mean_screen']:
        raise ValueError('Profile exceeds declared small-amplitude/narrowband design screens')
    indices = np.unique(np.round(np.linspace(0,steps,c['samples'])).astype(int))
    positions = indices*h
    histories = {}; arrays = {}; initial_projection_losses={}; step=m.stepper(h)
    for p in c['perturbations']:
        u0,loss=perturb(m,f,p)
        if not math.isfinite(loss) or loss>q['projection_relative_l2_max']:
            raise ValueError('Perturbation projection loss exceeds qualification threshold')
        initial_projection_losses[p['id']]=loss
        v=np.fft.fft(u0)
        baseline=m.invariants(u0); history=[]; snapshots=[]
        for j in range(steps+1):
            if j in indices:
                u=np.fft.ifft(v)
                exact=spatial.translate(m,f,c['speed']*j*h)*np.exp(-1j*c['mu']*j*h)
                error=u-exact; diag=spatial.packet_diagnostics(m,u)
                fit=spatial.orbital_fit(m,u,f)
                aligned=spatial.translate(m,u,-fit['translation'])*np.exp(-1j*fit['global_phase'])
                rev=(-np.arange(m.n))%m.n
                amplitude_weight=abs(exact)**2
                support=(abs(exact)>.01*max(abs(exact))) & (abs(u)>.01*max(abs(u)))
                phase_error=np.angle(u[support]*exact[support].conj())
                history.append({'xi':j*h,**diag,**fit,
                    'raw_complex_relative_l2':float(np.linalg.norm(error)/np.linalg.norm(f)),
                    'raw_complex_l1':float(m.dtau*np.sum(abs(error))),
                    'raw_complex_l2':float(np.sqrt(m.dtau*np.sum(abs(error)**2))),
                    'raw_complex_l4':float((m.dtau*np.sum(abs(error)**4))**.25),
                    'raw_complex_linf':float(max(abs(error))),
                    'aligned_conjugate_reflection_defect':float(np.linalg.norm(aligned-aligned[rev].conj())/np.linalg.norm(aligned)),
                    'weighted_phase_rmse':float(np.sqrt(np.sum(amplitude_weight[support]*phase_error**2)/np.sum(amplitude_weight[support]))) if support.any() else None,
                    'phase_excluded_reference_action_fraction':float(np.sum(amplitude_weight[~support])/np.sum(amplitude_weight)),
                    'invariant_absolute_drifts':{k:diag[k]-b for k,b in baseline.items()},
                    'invariant_relative_drifts':{k:(diag[k]-b)/abs(b) if abs(b)>1e-10*baseline['action'] else None for k,b in baseline.items()},
                    'boundary_flag':diag['boundary_action_fraction']>q['boundary_action_max'],
                    'spectral_edge_flag':diag['retained_spectral_edge_action_fraction']>q['spectral_edge_action_max'],
                    'physical_screen_flag':diag['epsilon_peak']>q['epsilon_peak_screen'] or diag['epsilon_spectral_sigma']>q['epsilon_spectral_sigma_screen'] or abs(diag['epsilon_spectral_mean'])>q['epsilon_spectral_mean_screen'],
                    'reference_wrap_flag':abs(c['speed']*j*h)+10>.45*m.length})
                snapshots.append(u.copy())
            if j<steps:
                v=step(v)
                if not np.isfinite(v).all():
                    raise RuntimeError('Propagation became nonfinite')
        histories[p['id']]=history; arrays[p['id']]=np.asarray(snapshots)
    output.mkdir(parents=True)
    np.savez_compressed(output/'fields.npz',tau=m.tau,xi=positions,profile=f,**arrays)
    report={'schema_version':1,'model_id':MODEL_ID,'physical_system':PHYSICAL_SYSTEM,
        'case':c,'config_sha256':digest(config_path),'qualification':q,
        'profile_projection_relative_l2':projection_error,'profile_residual':residual,
        'perturbation_projection_relative_l2':initial_projection_losses,
        'profile_diagnostics':profile_diagnostics,'source':provenance,
        'core_source_sha256':digest(spatial.__file__),'campaign_source_sha256':digest(__file__),
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform()},
        'actual_dxi':h,'steps_per_trajectory':steps,'trajectories':histories,
        'resources':{'wall_seconds':time.perf_counter()-started,'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        'interpretation':'Finite-window periodic persistence only; no spectral/nonlinear stability theorem or physical validation. Screens are numerical design choices. Padded continuum E/M need not be invariants of the Galerkin ODE. Tail action is not a radiation error bar.'}
    report['fields_sha256']=digest(output/'fields.npz')
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path); parser.add_argument('--case')
    parser.add_argument('--output',type=Path); parser.add_argument('--max-steps',type=int,default=1000)
    parser.add_argument('--describe',action='store_true')
    args=parser.parse_args()
    if args.describe:
        config=validate(json.loads(args.config.read_text()))
        for c in config['cases']:
            count=len(c['perturbations']); steps=math.ceil(c['xi_end']/c['dxi'])
            print(json.dumps({'id':c['id'],'n':c['n'],'length':c['length'],'total_steps':count*steps,
                'profile_mode':c['profile_mode'],'profile_iteration_budget':c['profile_max_iterations'],
                'snapshot_bytes_upper_bound':16*c['n']*(1+count*c['samples']),
                'fft_work_scale':count*steps*c['n']*math.log2(c['n'])}))
        return
    if not args.case or not args.output:
        parser.error('--case and fresh --output required')
    report=run(args.config,args.case,args.output,args.max_steps)
    print(json.dumps({'output':str(args.output.resolve()),'resources':report['resources'],
        'profile_residual':report['profile_residual']}))


if __name__=='__main__':
    main()

"""Small, bounded initial-envelope simulator for the validated water-wave solver.

No training, dataset sweep, external service or scheduler is invoked. A saved
viewer is precomputed; calling ``simulate`` performs a fresh local simulation.
The demonstration defaults are qualified only by ``run_checks`` at those exact
parameters/settings, never by the mere completion of propagation.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import time

import numpy as np
import scipy
from scipy.signal import resample

from dysthe_core import MODEL_ID, PHYSICAL_SYSTEM
from dysthe_core.reference import spatial
from .reference_analysis import map_fields
from . import envelope_formula as formula_language
from .envelope_formula import compile_formula, FORMULA_DEFAULT, SYMBOL_DEFAULTS

CASE_IDS = ['profile', 'perturbed', 'gaussian', 'custom']
DEFAULT_SETTINGS = dict(n=4096, length=160., epsilon=.025, dxi=.00125,
                        xi_end=4., samples=81)
CUSTOM_SETTINGS = dict(n=1024, length=160., epsilon=.025, dxi=.00125,
                       xi_end=.1, samples=21)
DEFAULT_PARAMETERS = {
    'profile': {},
    'perturbed': dict(delta=.02),
    'gaussian': dict(amplitude=1., width=2., wavenumber=.3, chirp=0.),
    'custom': dict(formula=FORMULA_DEFAULT, symbols=SYMBOL_DEFAULTS.copy()),
}
PARAMETER_FIELDS = {
    'profile': [],
    'perturbed': [dict(name='delta', minimum=0., maximum=.04, default=.02)],
    'gaussian': [dict(name='amplitude', minimum=.25, maximum=1.5, default=1.),
                 dict(name='width', minimum=1., maximum=4., default=2.),
                 dict(name='wavenumber', minimum=-1., maximum=1., default=.3),
                 dict(name='chirp', minimum=-.05, maximum=.05, default=0.)],
    'custom': [],
}
TITLES = {'profile':'Validated solitary profile',
          'perturbed':'Two-percent localized perturbation',
          'gaussian':'Gaussian complex packet', 'custom':'Custom formula envelope'}
LEGACY_SHA256 = '3d76bbc8ed8e2c4d164028282f9b1469ed6f6eae3f50f01400d89365c57c3ec2'
LEGACY_SIDECAR_SHA256 = 'c8e284e9b6b3b02160dc5d985e2882866997975e0bb42a90ec776e7b58f579cd'
DEFINITIONS = {
    'peak_amplitude':'max_tau |u| at each saved xi; envelope amplitude, not reconstructed free-surface elevation.',
    'rms_width':'sqrt(integral d(tau,c)^2 |u|^2 / integral |u|^2); d is signed shortest periodic distance about circular action centroid c. Broad/wrapped packets make this statistic ambiguous.',
    'displacement':'Unwrapped circular action-centroid displacement from xi=0. Defined only while the first circular moment is well separated from zero; not a group-velocity fit.',
    'tail_action_fraction':'Action outside periodic distance 8 from the moving circular action centroid divided by total action. Exterior action includes packet tails and spreading: it is not automatically radiation.',
    'boundary_action_fraction':'Fraction of action in fixed outer ten percent of the tau domain, |tau| > 0.45 L.',
    'mass_relative_error':'(action(xi)-action(0))/action(0), action=integral |u|^2 d tau. Conservation is a diagnostic, not an accuracy proof.',
    'phase_shift':'Unwrapped global phase from least-squares translation/global-phase fit against this case\'s initial complex envelope. Translation is fitted separately.',
    'orbital_shape_error':'Relative L2 distance to this case\'s initial complex envelope after removing its best translation and global phase. No amplitude rescaling is fitted. Genuine evolution of a Gaussian is expected.',
    'profile_exterior_residual_action':'For profile-based cases only: action of u minus its best translated/phased unperturbed solitary reference outside reference radius 8, divided by packet action. A radiation proxy, not a spectral radiation decomposition or error bar.',
    'retained_spectral_edge_action_fraction':'Fourier action fraction in outer twenty percent of retained |mode|<N/4 band.',
    'epsilon_peak':'epsilon max |u|; <=0.1 is a chosen weak-steepness screen, not a physical validity proof.',
    'epsilon_spectral_sigma':'epsilon times Fourier wavenumber standard deviation; <=0.1 is a chosen narrowband screen.',
    'circular_concentration':'Modulus of the normalized first circular action moment. Values near zero make center/width/displacement unreliable.',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def case_info():
    return [dict(case_id=k, id=k, title=TITLES[k], parameters=copy.deepcopy(DEFAULT_PARAMETERS[k]),
        parameter_fields=PARAMETER_FIELDS[k],
        settings=(CUSTOM_SETTINGS if k=='custom' else DEFAULT_SETTINGS).copy(), description={
            'profile':'Recomputed epsilon=.025, mu=2, speed=0 periodic traveling branch; full-PDE residual checked before propagation.',
            'perturbed':'Same branch multiplied by 1 + delta exp(-((tau-1.3)/2)^2), delta=.02 by default; no imposed symmetry during propagation.',
            'gaussian':'A exp(-tau^2/(2 width^2)) exp(i (wavenumber tau + chirp tau^2)); a complex initial envelope.',
            'custom':'Define u(0,tau) using a restricted LaTeX expression and declared real parameters. Preview first; custom runs are exploratory and have no refinement qualification.'}[k]) for k in CASE_IDS]


def configuration(case_id, parameters=None, settings=None):
    if not isinstance(case_id, str) or case_id not in CASE_IDS:
        raise ValueError('Unknown case; select profile, perturbed, gaussian or custom')
    if any(value is not None and not isinstance(value, dict) for value in (parameters, settings)):
        raise ValueError('Parameters and settings must be JSON objects')
    p = copy.deepcopy(DEFAULT_PARAMETERS[case_id])
    s = (CUSTOM_SETTINGS if case_id=='custom' else DEFAULT_SETTINGS).copy()
    for supplied, defaults, name in ((parameters or {}, p, 'parameter'), (settings or {}, s, 'setting')):
        if not isinstance(supplied, dict) or set(supplied)-set(defaults):
            raise ValueError(f'Unknown {name} names')
        defaults.update(supplied)
    for name, value in {**(p if case_id!='custom' else {}), **s}.items():
        try:
            spatial._finite(value, name)
        except (OverflowError, TypeError) as exc:
            raise ValueError(f'{name} must be a bounded finite real number') from exc
    if type(s['n']) is not int or s['n'] < 128 or s['n'] > 16384 or s['n'] % 4:
        raise ValueError('n must be a multiple of four from 128 to 16384')
    if type(s['samples']) is not int or not 2 <= s['samples'] <= 101:
        raise ValueError('samples must be an integer from 2 through 101')
    if not 40 <= s['length'] <= 640 or s['epsilon'] != .025:
        raise ValueError('This demo fixes epsilon=.025 and bounds length to [40,640]')
    if not .0003125 <= s['dxi'] <= .005 or not 0 < s['xi_end'] <= 4:
        raise ValueError('Require dxi in [.0003125,.005] and 0<xi_end<=4')
    steps = math.ceil(s['xi_end']/s['dxi'])
    if steps > 12800 or steps*s['n'] > 110000000:
        raise ValueError('Bounded local work budget exceeded')
    if case_id == 'perturbed' and not 0 <= p['delta'] <= .04:
        raise ValueError('Perturbation delta must lie in [0,.04]')
    if case_id == 'gaussian':
        if not .25 <= p['amplitude'] <= 1.5 or not 1 <= p['width'] <= 4:
            raise ValueError('Gaussian amplitude in [.25,1.5], width in [1,4] required')
        if abs(p['wavenumber']) > 1 or abs(p['chirp']) > .05:
            raise ValueError('Require |wavenumber|<=1 and |chirp|<=.05')
    if case_id == 'custom':
        if s['n'] > 8192:
            raise ValueError('Custom previews require n<=8192 for the independent 2N sampling check')
        if steps*s['n'] > 8000000:
            raise ValueError('Custom local work budget is 8,000,000 grid-steps; shorten the propagation interval or reduce N')
        compiled = compile_formula(p['formula'], p['symbols'])
        p['symbols'] = compiled.provenance()['symbols']
    return p, s


def validate_request(case_id, parameters=None, settings=None):
    """Validate a server request before queuing work; return normalized dictionaries."""
    return configuration(case_id, parameters, settings)


def custom_fingerprint(parameters, settings):
    """Bind a preview to its exact prescription, parser and numerical settings."""
    record=dict(parameters=parameters, settings=settings, model_id=MODEL_ID,
                projector='strict abs(mode)<N/4',
                parser_source_sha256=digest(formula_language.__file__),
                core_source_sha256=digest(spatial.__file__),
                simulator_source_sha256=digest(__file__))
    return hashlib.sha256(json.dumps(record,sort_keys=True,allow_nan=False,
                                      separators=(',',':')).encode()).hexdigest()


def _custom_initial(model, parameters, settings):
    """Bounded sampling checks; none constitutes continuum/physical validation."""
    compiled=compile_formula(parameters['formula'],parameters['symbols'])
    raw=compiled.evaluate(model.tau)
    peak=float(np.max(abs(raw)))
    if not 1e-12 <= peak <= 4.:
        raise ValueError('Sampled peak |u| must be between 1e-12 and 4 for this local demo; change the amplitude parameters')
    u=model.project(raw)
    if float(np.max(abs(u))) < 1e-12:
        raise ValueError('No usable retained-band profile remains; refine the grid or broaden the envelope')
    projection=float(np.linalg.norm(u-raw)/np.linalg.norm(raw))
    # A second independently sampled grid detects many aliases invisible at N.
    # It is still a finite test, not a proof that an arbitrary formula is resolved.
    fine_tau=np.arange(2*model.n)*model.length/(2*model.n)-model.length/2
    fine=compiled.evaluate(fine_tau)
    fine_peak=float(np.max(abs(fine)))
    crossgrid=float(np.linalg.norm(resample(raw,2*model.n)-fine)/np.linalg.norm(fine))
    fv=np.fft.fft(fine); modes=abs(np.fft.fftfreq(2*model.n)*(2*model.n))
    fine_loss=float(np.linalg.norm(fv[modes>=model.n/4])/np.linalg.norm(fv))
    packet=spatial.packet_diagnostics(model,u)
    _,_,concentration,_=center_and_width(model,u)
    endpoints=compiled.evaluate(np.array([-model.length/2,model.length/2]))
    seam=float(abs(endpoints[1]-endpoints[0])/peak)
    exterior=float(np.max(abs(raw[abs(model.tau)>.45*model.length]))/peak)
    warnings=[]
    def note(code,message,blocked=False):
        warnings.append(dict(code=code,message=message,blocks_run=bool(blocked)))
    if exterior>1e-6 or packet['boundary_action_fraction']>1e-8:
        note('nondecay','Envelope is appreciable near the periodic boundary. The calculation repeats this finite window; it is not a whole-line packet.')
    if seam>1e-6:
        note('periodic_seam',f'Endpoint mismatch / peak is {seam:.3g}. Periodic continuation introduces a seam; widen the domain or change the formula.')
    if projection>1e-8 or fine_loss>1e-8:
        note('spectral_resolution',f'Insufficient retained-band resolution: projection relative L2={projection:.3g}, 2N discarded-band relative norm={fine_loss:.3g}. Increase N or broaden/smooth the formula.',True)
    if crossgrid>1e-6:
        note('sampling_alias',f'N versus independently sampled 2N discrepancy is {crossgrid:.3g}. The formula is not resolved on this grid.',True)
    if packet['retained_spectral_edge_action_fraction']>1e-8:
        note('spectral_edge','Appreciable Fourier action lies near the retained-band edge. A grid refinement is needed before relying on propagation.')
    if model.epsilon*max(peak,fine_peak,float(np.max(abs(u))))>.1:
        note('steepness','The chosen epsilon times peak exceeds 0.1 in the preview. Reduce amplitude; this demo refuses this regime.',True)
    if packet['epsilon_spectral_sigma']>.1 or abs(packet['epsilon_spectral_mean'])>.1:
        note('bandwidth','Initial field exceeds the chosen epsilon-scaled bandwidth/carrier screen of 0.1. Broaden the envelope or reduce phase gradients.',True)
    if concentration<.5:
        note('packet_center','The circular action moment is weak. Packet center, width and displacement can be ambiguous; singular values are shown as unavailable.')
    note('exploratory','Custom input is exploratory: these finite sampling and regime screens do not establish numerical accuracy or physical applicability. Separate grid, step and domain comparisons are still required.')
    summary=dict(peak_amplitude=peak, projected_peak_amplitude=float(np.max(abs(u))),
        projection_relative_l2=projection, cross_grid_relative_l2=crossgrid,
        fine_discarded_band_relative_l2=fine_loss,
        boundary_action_fraction=packet['boundary_action_fraction'],
        boundary_peak_relative=exterior, endpoint_mismatch_relative=seam,
        retained_spectral_edge_action_fraction=packet['retained_spectral_edge_action_fraction'],
        epsilon_peak=model.epsilon*max(peak,fine_peak,float(np.max(abs(u)))),
        epsilon_spectral_sigma=packet['epsilon_spectral_sigma'],
        epsilon_spectral_mean=packet['epsilon_spectral_mean'],
        circular_concentration=concentration,
        failed_run_checks=[w['code'] for w in warnings if w['blocks_run']])
    provenance=dict(**compiled.provenance(), settings=settings.copy(),
        model_id=MODEL_ID,physical_system=PHYSICAL_SYSTEM,
        parser_source_sha256=digest(formula_language.__file__),
        core_source_sha256=digest(spatial.__file__),simulator_source_sha256=digest(__file__),
        preview_fingerprint=custom_fingerprint(parameters,settings),
        raw_initial_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
        projected_initial_sha256=hashlib.sha256(u.tobytes()).hexdigest(),
        initial_array_dtype='complex128', initial_array_shape=[model.n],
        projection_relative_l2=projection, initial_checks=summary,
        warnings=warnings, prescription='Restricted LaTeX formula sampled in retarded time, then projected to strict |mode|<N/4.')
    return raw,u,provenance,summary,warnings


def preview_custom(parameters=None, settings=None):
    """Preview one bounded expression without taking any propagation steps."""
    p,s=configuration('custom',parameters,settings)
    model=spatial.SpatialDysthe(s['n'],s['length'],s['epsilon'])
    raw,_,provenance,summary,warnings=_custom_initial(model,p,s)
    stride=max(1,math.ceil(model.n/1024)); shown=raw[::stride]
    phases=np.angle(shown); good=abs(shown)>=.01*summary['peak_amplitude']
    return dict(case_id='custom', parameters=p,settings=s,tau=model.tau[::stride].tolist(),
        real=shown.real.tolist(),imag=shown.imag.tolist(),amplitude=abs(shown).tolist(),
        phase=[float(v) if mask else None for v,mask in zip(phases,good)],
        preview_field='sampled formula before the numerical projector',
        warnings=warnings,summary=summary,can_run=not summary['failed_run_checks'],
        provenance=provenance,fingerprint=provenance['preview_fingerprint'],
        qualification='Exploratory custom input; refinement checks pending.')


def initial_field(case_id, model, parameters=None, settings=None):
    """Return projected initial field, optional solitary reference, provenance.

    The branch is recomputed on each declared periodic domain. Reusing/interpolating
    a historical finite-box profile on a different domain would change the PDE
    being qualified. Historical checksums identify the original branch only.
    """
    p = copy.deepcopy(DEFAULT_PARAMETERS[case_id])
    if parameters:
        p.update(parameters)
    profile = None
    provenance = {'prescription': next(c['description'] for c in case_info() if c['case_id']==case_id)}
    if case_id=='custom':
        p,s=configuration('custom',p,{**(settings or {}), 'n':model.n,
                                     'length':model.length,'epsilon':model.epsilon})
        _,u,source,summary,_=_custom_initial(model,p,s)
        if summary['failed_run_checks']:
            raise ValueError('Custom input blocked: '+'; '.join(w['message'] for w in source['warnings'] if w['blocks_run']))
        return u,None,source
    if case_id in ('profile', 'perturbed'):
        profile, residual = spatial.solve_profile(model, 2., 0., max_iterations=200)
        residual = model.profile_residual(profile, 2., 0.)
        if residual['full_periodic_relative_l2'] > 1e-8:
            raise ValueError('Full periodic profile residual exceeds 1e-8; refine this grid')
        provenance.update(profile_residual=residual, profile_mode='recompute-per-periodic-domain',
            branch=dict(mu=2., speed=0., epsilon=.025),
            historical_profile_sha256=LEGACY_SHA256,
            historical_sidecar_sha256=LEGACY_SIDECAR_SHA256)
        raw = profile.copy()
        if case_id == 'perturbed':
            raw *= 1+p['delta']*np.exp(-((model.tau-1.3)/2.)**2)
    else:
        raw = p['amplitude'] * np.exp(-model.tau**2/(2*p['width']**2)) * np.exp(
            1j*(p['wavenumber']*model.tau+p['chirp']*model.tau**2))
    u = model.project(raw)
    loss = float(np.linalg.norm(u-raw)/np.linalg.norm(raw))
    if not np.isfinite(u).all() or loss > 1e-8:
        raise ValueError(f'Initial projection loss {loss:.3g} exceeds 1e-8; refine this grid')
    provenance['projection_relative_l2'] = loss
    return u, profile, provenance


def center_and_width(model, field):
    rho = abs(field)**2
    total = float(np.sum(rho))
    moment = np.sum(rho*np.exp(2j*np.pi*model.tau/model.length))/total
    center = float(np.angle(moment)*model.length/(2*np.pi))
    distance = (model.tau-center+model.length/2)%model.length-model.length/2
    return center, float(np.sqrt(np.sum(distance**2*rho)/total)), float(abs(moment)), distance


def diagnostics(model, field, initial, profile, baseline):
    packet = spatial.packet_diagnostics(model, field, core_radius=8.)
    center, width, concentration, distance = center_and_width(model, field)
    fit = spatial.orbital_fit(model, field, initial)
    row = dict(peak_amplitude=packet['peak'], rms_width=width,
        circular_center=center, circular_concentration=concentration,
        tail_action_fraction=float(np.sum(abs(field[abs(distance)>8])**2)/np.sum(abs(field)**2)),
        boundary_action_fraction=packet['boundary_action_fraction'],
        retained_spectral_edge_action_fraction=packet['retained_spectral_edge_action_fraction'],
        epsilon_peak=packet['epsilon_peak'], epsilon_spectral_sigma=packet['epsilon_spectral_sigma'],
        epsilon_spectral_mean=packet['epsilon_spectral_mean'],
        mass_relative_error=(packet['action']-baseline['action'])/baseline['action'],
        action=packet['action'], energy=packet['energy'], momentum=packet['momentum'],
        energy_absolute_drift=packet['energy']-baseline['energy'],
        momentum_absolute_drift=packet['momentum']-baseline['momentum'],
        orbital_shape_error=fit['orbital_relative_l2'],
        fitted_translation=fit['translation'], phase_shift=fit['global_phase'])
    if concentration < 1e-8:
        # A uniform or nearly uniform periodic field has no identifiable packet
        # center. Do not turn roundoff in its circular moment into displacement.
        for name in ('rms_width','circular_center','tail_action_fraction',
                     'fitted_translation','phase_shift'):
            row[name]=None
    if profile is not None:
        branch_fit = spatial.orbital_fit(model, field, profile)
        aligned = spatial.translate(model, field, -branch_fit['translation'])*np.exp(-1j*branch_fit['global_phase'])
        residual = aligned-profile
        row['profile_exterior_residual_action'] = float(np.sum(abs(residual[abs(model.tau)>8])**2)/np.sum(abs(field)**2))
    return row


def _screens(history):
    limits = {'epsilon_peak':.1,'epsilon_spectral_sigma':.1,
              'boundary_action_fraction':1e-8,
              'retained_spectral_edge_action_fraction':1e-8}
    screen = {k:dict(maximum=max(row[k] for row in history), limit=v,
                    passed=max(row[k] for row in history)<=v) for k,v in limits.items()}
    mean = max(abs(row['epsilon_spectral_mean']) for row in history)
    screen['absolute_epsilon_spectral_mean'] = dict(maximum=mean,limit=.1,passed=mean<=.1)
    concentration = min(row['circular_concentration'] for row in history)
    screen['circular_concentration'] = dict(minimum=concentration,limit=.5,passed=concentration>=.5)
    return screen


def _write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def simulate(case_id, output, parameters=None, settings=None, progress=None):
    """Run one bounded local case and return its report; output must be fresh."""
    p,s = configuration(case_id, parameters, settings)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Choose a fresh output directory; completed evidence is never overwritten')
    start = time.perf_counter()
    if progress:
        progress(dict(phase='initializing',completed_steps=0,total_steps=math.ceil(s['xi_end']/s['dxi'])))
    model = spatial.SpatialDysthe(s['n'],s['length'],s['epsilon'])
    initial,profile,source = initial_field(case_id,model,p,settings=s)
    first = spatial.packet_diagnostics(model, initial)
    if first['epsilon_peak']>.1 or first['epsilon_spectral_sigma']>.1 or abs(first['epsilon_spectral_mean'])>.1:
        raise ValueError('Initial field fails declared weak-steepness/narrowband screens')
    steps = math.ceil(s['xi_end']/s['dxi']); h=s['xi_end']/steps
    if case_id=='custom':
        source.update(actual_dxi=h,steps=steps)
    indices = np.unique(np.round(np.linspace(0,steps,s['samples'])).astype(int))
    selected=set(indices.tolist()); frames=[]; rows=[]; xi=[]
    baseline=model.invariants(initial); step=model.stepper(h); v=np.fft.fft(initial)
    for j in range(steps+1):
        if j in selected:
            u=np.fft.ifft(v)
            frames.append(u.copy()); xi.append(j*h)
            rows.append(dict(xi=j*h,**diagnostics(model,u,initial,profile,baseline)))
            if progress:
                progress(dict(phase='propagating',completed_steps=j,total_steps=steps))
        if j<steps:
            v=step(v)
            if not np.isfinite(v).all():
                raise RuntimeError('Propagation became nonfinite; no qualified output produced')
    centers=[row['circular_center'] for row in rows]
    if all(value is not None for value in centers):
        displacement=np.unwrap(np.asarray(centers)*2*np.pi/model.length)*model.length/(2*np.pi)
        displacement-=displacement[0]
    else:
        displacement=[None]*len(rows)
    phases=[row['phase_shift'] for row in rows]
    phase=np.unwrap(phases) if all(value is not None for value in phases) else [None]*len(rows)
    for j,row in enumerate(rows):
        row['displacement']=float(displacement[j]) if displacement[j] is not None else None
        row['phase_shift']=float(phase[j]) if phase[j] is not None else None
    fields=np.asarray(frames,dtype=np.complex128);xi=np.asarray(xi)
    screens=_screens(rows)
    qualification=dict(status='unqualified: numerical refinements pending',checks_available=False,
        design_screens=screens,all_design_screens_pass=all(v['passed'] for v in screens.values()),
        notes=['Design screens do not establish physical applicability.',
               'Qualification belongs to these exact parameters, domain, grid, step and horizon.',
               'Finite periodic propagation is not a soliton discovery or stability result.'])
    if case_id=='custom':
        qualification['notes'].extend(w['message'] for w in source['warnings'])
    report=dict(schema_version=1,case_id=case_id,title=TITLES[case_id],model_id=MODEL_ID,
        physical_system=PHYSICAL_SYSTEM,parameters=p,settings=s,actual_dxi=h,steps=steps,
        source=source,qualification=qualification,diagnostic_definitions=DEFINITIONS,
        diagnostics=rows,core_source_sha256=digest(spatial.__file__),
        simulator_source_sha256=digest(__file__),
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        resources=dict(wall_seconds=time.perf_counter()-start,
                       process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        interpretation='One-coordinate spatial free-surface-envelope Dysthe; xi is propagation distance and tau is retarded time. Periodic boundary conditions and |D_tau| zero mode=0. Galerkin projector retains |mode|<N/4. Integrating-factor RK4. Padded continuum energy/momentum need not be invariants of the projected ODE or RK map. Exterior action is not automatically radiation.')
    output.mkdir(parents=True)
    arrays=dict(tau=model.tau,xi=xi,u=fields,initial=initial)
    if case_id=='custom':
        arrays['raw_initial']=compile_formula(p['formula'],p['symbols']).evaluate(model.tau)
        # Scalar Unicode, not a pickle/object array: the downloaded complex data
        # carries the exact input and checks even when separated from report.json.
        arrays['provenance_json']=np.asarray(json.dumps(source,sort_keys=True,allow_nan=False))
    if profile is not None:
        arrays['profile']=profile
    np.savez_compressed(output/'fields.npz',**arrays)
    report['fields_sha256']=digest(output/'fields.npz')
    _write_json(output/'report.json',report)
    write_viewer(output,report,arrays)
    if progress:
        progress(dict(phase='complete',completed_steps=steps,total_steps=steps))
    return report


def write_viewer(output, report=None, arrays=None):
    output=Path(output)
    if report is None:
        report=json.loads((output/'report.json').read_text())
    if arrays is None:
        with np.load(output/'fields.npz',allow_pickle=False) as data:
            arrays={k:data[k].copy() for k in data.files}
    stride=max(1,math.ceil(len(arrays['tau'])/1024))
    # Samples of the original complex field, not a projected or interpolated new solution.
    field=arrays['u'][:,::stride]
    keys=[k for k in report['diagnostics'][0] if k!='xi']
    viewer=dict(schema_version=1,case_id=report['case_id'],title=report['title'],source='precomputed',
        parameters=report['parameters'],settings=report['settings'],tau=arrays['tau'][::stride].tolist(),
        xi=arrays['xi'].tolist(),real=field.real.tolist(),imag=field.imag.tolist(),
        diagnostics={k:[r[k] for r in report['diagnostics']] for k in keys},
        diagnostic_definitions=DEFINITIONS,qualification=report['qualification'],
        full_data=dict(path='fields.npz',sha256=report['fields_sha256'],dtype='complex128',
                       shape=list(arrays['u'].shape),axis_order=['xi','tau'],viewer_tau_stride=stride),
        formulation=report['interpretation'])
    if report['case_id']=='custom':
        viewer['initial_provenance']=report['source']
    _write_json(output/'viewer.json',viewer)
    return viewer


def benchmark(output):
    """100 actual solver steps, three snapshots, on the heaviest default case."""
    result=simulate('perturbed',output,settings=dict(xi_end=.125,samples=3))
    result['benchmark']=dict(steps=result['steps'],approx_default_case_seconds=result['resources']['wall_seconds']*32,
        caution='Linear extrapolation only; disk compression and diagnostics add fixed cost.')
    _write_json(Path(output)/'report.json',result)
    return result


def _load_run(path):
    path=Path(path);report=json.loads((path/'report.json').read_text())
    if report['core_source_sha256']!=digest(spatial.__file__) or report['simulator_source_sha256']!=digest(__file__):
        raise ValueError('Existing run source hash disagrees with the current numerical source')
    if report['fields_sha256']!=digest(path/'fields.npz'):
        raise ValueError('Existing fields checksum differs')
    with np.load(path/'fields.npz',allow_pickle=False) as data:
        arrays={k:data[k].copy() for k in data.files}
    return report,arrays


def compare_runs(base_path, fine_path, kind):
    """Preserve complex phase; compare common physical coordinates over all frames."""
    base,a=_load_run(base_path);fine,b=_load_run(fine_path)
    np.testing.assert_allclose(a['xi'],b['xi'],atol=1e-12,rtol=0)
    u,v,tau=map_fields(a['tau'],b['tau'],a['u'],b['u'],kind)
    m=spatial.SpatialDysthe(len(tau),len(tau)*(tau[1]-tau[0]),.025)
    raw=[];orbital=[];shapes_a=[];shapes_b=[]
    # Fits use each run's own initial field; compare signed deformations after
    # removing translation/phase, with the initial prescription discrepancy explicit.
    for x,y in zip(u,v):
        raw.append(float(np.linalg.norm(x-y)/np.linalg.norm(y)))
        orbital.append(spatial.orbital_fit(m,x,y)['orbital_relative_l2'])
        for z,initial,container in ((x,u[0],shapes_a),(y,v[0],shapes_b)):
            fit=spatial.orbital_fit(m,z,initial)
            aligned=spatial.translate(m,z,-fit['translation'])*np.exp(-1j*fit['global_phase'])
            container.append(aligned-initial)
    da,db=np.asarray(shapes_a),np.asarray(shapes_b)
    norm=np.linalg.norm(v,axis=1)
    shape_diff=np.linalg.norm(da-db,axis=1)/norm
    response=np.linalg.norm(db,axis=1)/norm
    meaningful=response>1e-7
    response_ratios=np.divide(shape_diff,response,out=np.zeros_like(response),where=meaningful)
    limits=dict(raw_complex_relative_l2=2e-4,orbital_relative_l2=1e-4,
                shape_change_discrepancy_relative_l2=1e-4,
                shape_discrepancy_over_response=.02)
    metrics=dict(raw_complex_relative_l2=max(raw),orbital_relative_l2=max(orbital),
        shape_change_discrepancy_relative_l2=float(max(shape_diff)),
        shape_discrepancy_over_response=float(max(response_ratios)) if meaningful.any() else None)
    passed=all(metrics[k] is None or metrics[k]<=limit for k,limit in limits.items())
    return dict(kind=kind,comparison=f'{Path(base_path).name} → {Path(fine_path).name}',
        metrics=metrics,thresholds=limits,passed=bool(passed),error=metrics['raw_complex_relative_l2'],
        initial_complex_relative_l2=raw[0],final_complex_relative_l2=raw[-1],
        final_shape_discrepancy_over_response=float(response_ratios[-1]) if meaningful[-1] else None,
        final_response_relative_l2=float(response[-1]),
        note='Raw and orbital fields use common coordinates. Signed shape changes compare each solution with its own initial profile after fitted translation/phase. Response ratios are omitted below 1e-7 of field norm; no amplitude rescaling.',
        history=dict(xi=a['xi'].tolist(),raw_complex_relative_l2=raw,orbital_relative_l2=orbital,
                     shape_change_discrepancy_relative_l2=shape_diff.tolist(),
                     response_relative_l2=response.tolist(),shape_discrepancy_over_response=[float(r) if good else None for r,good in zip(response_ratios,meaningful)]))


def run_checks(case_id, root):
    """One default case plus independent grid, step and domain doublings."""
    if case_id=='custom':
        raise ValueError('The saved reference campaign does not qualify arbitrary formulas; custom inputs need a separately designed refinement comparison')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    base_path=root/case_id
    check_root=root/'checks'/case_id
    variations=[('base',{}),('grid',dict(n=8192)),('step',dict(dxi=.000625)),
                ('domain',dict(n=8192,length=320.))]
    for name,settings in variations:
        path=base_path if name=='base' else check_root/name
        if path.exists():
            r,_=_load_run(path)
            p,s=configuration(case_id,settings=settings)
            if (r['case_id'],r['parameters'],r['settings'])!=(case_id,p,s):
                raise ValueError('Existing numerical run configuration disagrees')
            print(json.dumps(dict(case=case_id,run=name,status='verified-existing')),flush=True)
        else:
            print(json.dumps(dict(case=case_id,run=name,status='starting')),flush=True)
            r=simulate(case_id,path,settings=settings)
            print(json.dumps(dict(case=case_id,run=name,status='complete',resources=r['resources'])),flush=True)
    checks=[compare_runs(base_path,check_root/k,k) for k in ('grid','step','domain')]
    reports=[json.loads(((base_path if name=='base' else check_root/name)/'report.json').read_text()) for name,_ in variations]
    passed=all(c['passed'] for c in checks) and all(r['qualification']['all_design_screens_pass'] for r in reports)
    qualification=dict(status='qualified for this bounded demonstration' if passed else 'refinement or design screen failed',
        checks_available=True,passed=passed,checks=checks,
        exact_default_parameters=DEFAULT_PARAMETERS[case_id],exact_default_settings=DEFAULT_SETTINGS,
        design_screens=reports[0]['qualification']['design_screens'],
        all_design_screens_pass=reports[0]['qualification']['all_design_screens_pass'],
        notes=['Two levels per refinement demonstrate agreement, not an observed convergence order or rigorous error bound.',
            'Domain profiles are recomputed at mu=2,speed=0; domain comparison includes periodic mean-flow dependence.',
            'All three refinements are independent of one another and compare all 81 saved snapshots.',
            'Custom parameters/settings remain unqualified until their own checks are run.',
            'This interval is xi in [0,4], not a long-time stability test.'])
    _write_json(check_root/'qualification.json',qualification)
    base=reports[0];base['qualification']=qualification
    _write_json(base_path/'report.json',base)
    write_viewer(base_path,base)
    return qualification


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('benchmark');p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('run');p.add_argument('--case',choices=CASE_IDS,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--parameters',default='{}');p.add_argument('--settings',default='{}')
    p=sub.add_parser('check');p.add_argument('--case',choices=[k for k in CASE_IDS if k!='custom'],required=True);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='benchmark':
        r=benchmark(args.output);print(json.dumps(dict(resources=r['resources'],benchmark=r['benchmark'])))
    elif args.command=='run':
        r=simulate(args.case,args.output,json.loads(args.parameters),json.loads(args.settings));print(json.dumps(dict(resources=r['resources'],qualification=r['qualification']['status'])))
    else:
        r=run_checks(args.case,args.output);print(json.dumps(dict(qualification=r['status'],checks=[{k:c[k] for k in ('kind','metrics','passed')} for c in r['checks']])))


if __name__=='__main__':
    main()

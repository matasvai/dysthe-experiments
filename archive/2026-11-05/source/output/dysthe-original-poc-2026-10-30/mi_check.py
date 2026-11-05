#!/usr/bin/env python3
"""Independent small-perturbation check of the temporal potential Dysthe solver.

Tests the *complex* modulational-instability eigenvalue: growth and phase drift.
This is a linear-regime implementation control, not packet refinement, a
continuum stability theorem, or validation of the water-wave approximation.

Run: python mi_check.py --solver run.py --output-dir mi-check
Optional: --figure /path/to/temporal-mi-check.pdf
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_solver(path):
    spec = importlib.util.spec_from_file_location('temporal_solver_under_test', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def exact_linearization(a, ell, epsilon, c):
    """Derived independently in real amplitude p and phase q, u=a(1+p+iq)e^-ia²T/2.

    At positive Fourier frequency ell, the vector (p_hat,q_hat) obeys y'=M y.
    The exact retained-model eigenvalue is lambda=gamma-i*omega, not the
    formally truncated 1979 growth expression which drops the epsilon² term.
    """
    P = ell**2/8
    omega = epsilon*(ell**3/16+c*a*a*ell)
    h = epsilon*a*a*ell/4
    gamma_squared = P*(a*a*(1-epsilon*abs(ell))-P)-h*h
    if gamma_squared <= 0:
        raise ValueError('This control requires an unstable discrete sideband')
    gamma = np.sqrt(gamma_squared)
    matrix = np.array([[-1j*(omega-h), -P],
                       [P-a*a*(1-epsilon*abs(ell)), -1j*(omega+h)]])
    # Setting p_hat=1 gives q_hat=(-gamma+i*h)/P for the growing branch.
    vectors = np.column_stack(([1, (-gamma+1j*h)/P],
                               [1, (gamma+1j*h)/P])).astype(complex)
    vectors /= np.linalg.norm(vectors, axis=0)[None, :]
    eigenvalues = np.array([gamma-1j*omega, -gamma-1j*omega])
    defect = np.max(abs(matrix@vectors-vectors*eigenvalues[None, :]))
    if defect > 1e-13:
        raise AssertionError('Independent matrix and closed-form eigenpair disagree')
    return eigenvalues[0], vectors, matrix, float(defect)


def one_trajectory(solver, a, ell, mode, length, epsilon, c, delta, n, step, times):
    kind = 'literal' if c == 1 else 'stability'
    eigenvalue, vectors, matrix, eigenpair_defect = exact_linearization(a, ell, epsilon, c)
    s = np.arange(n)*length/n-length/2
    carrier = np.exp(1j*ell*s)
    p = delta*np.real(vectors[0, 0]*carrier)
    q = delta*np.real(vectors[1, 0]*carrier)
    # Real p,q contain conjugate +/-ell components; this produces a realizable
    # complex-envelope perturbation rather than injecting one unphysical branch.
    initial = a*(1+p+1j*q)
    fields, elapsed = solver.solve(initial, length, epsilon, kind, n, step, times)
    disturbance = fields*np.exp(.5j*a*a*times[:, None])/a-1
    p_hat = np.mean(disturbance.real*carrier.conj()[None, :], axis=1)
    q_hat = np.mean(disturbance.imag*carrier.conj()[None, :], axis=1)
    branch = np.linalg.solve(vectors, np.vstack((p_hat, q_hat)))
    ratio = branch[0]/branch[0, 0]
    exact = np.exp(eigenvalue*times)
    log_gain = np.log(abs(ratio))
    phase = np.unwrap(np.angle(ratio))
    measured_growth = float(np.polyfit(times, log_gain, 1)[0])
    measured_phase_rate = float(np.polyfit(times, phase, 1)[0])
    metrics = {
        'n': n, 'step': step, 'seconds': elapsed,
        'measured_growth_rate': measured_growth,
        'measured_phase_rate': measured_phase_rate,
        'growth_rate_absolute_error': abs(measured_growth-eigenvalue.real),
        'phase_rate_absolute_error': abs(measured_phase_rate-eigenvalue.imag),
        'maximum_relative_complex_branch_error': float(np.max(abs(ratio/exact-1))),
        'maximum_phase_error_radians': float(np.max(abs(phase-eigenvalue.imag*times))),
        'initial_growing_coefficient': [float(branch[0, 0].real), float(branch[0, 0].imag)],
        'initial_stable_to_growing_coefficient_ratio': float(abs(branch[1, 0]/branch[0, 0])),
        'independent_matrix_eigenpair_defect': eigenpair_defect,
    }
    rows = [{'epsilon': epsilon, 'c': c, 'n': n, 'step': step, 'T': float(T),
             'measured_gain': float(abs(value)), 'exact_gain': float(abs(target)),
             'measured_phase': float(ph), 'exact_phase': float(eigenvalue.imag*T),
             'relative_complex_error': float(abs(value/target-1))}
            for T, value, target, ph in zip(times, ratio, exact, phase)]
    return metrics, rows, ratio


def independent_dealias_audit(solver):
    """Compare N-grid RHS to an independently constructed 4N-grid convolution.

    Exact projection of the cubic nonlinearities is supported because 4K<N,
    where every stage has |m|<=K. The quadratic density has |m|<=2K<N/2,
    so its untruncated |D| derivative is also resolved before multiplication.
    """
    n, length, epsilon = 64, 40., .1
    integer_modes = np.rint(np.fft.fftfreq(n)*n).astype(int)
    keep = abs(integer_modes) < n/4
    rng = np.random.default_rng(19791030)
    coeff = (rng.normal(size=n)+1j*rng.normal(size=n))*.02*keep
    u = np.fft.ifft(n*coeff)
    fine_n = 4*n
    fine_coeff = np.zeros(fine_n, complex)
    fine_coeff[integer_modes % fine_n] = coeff
    kfine = 2*np.pi*np.fft.fftfreq(fine_n, d=length/fine_n)
    ufine = np.fft.ifft(fine_n*fine_coeff)
    derivative = np.fft.ifft(1j*kfine*fine_n*fine_coeff)
    rho = abs(ufine)**2
    mean_flow = np.fft.ifft(abs(kfine)*np.fft.fft(rho)).real
    results = []
    for kind, c in [('literal', 1.), ('stability', 1.5)]:
        model = solver.Model(n, length, epsilon, kind)
        independently_padded = (-.5j*rho*ufine
            +epsilon*(-c*rho*derivative+.25*ufine**2*derivative.conj()
                       +.5j*ufine*mean_flow))
        fine_rhs = np.fft.fft(independently_padded)/fine_n
        target = fine_rhs[integer_modes % fine_n]*keep
        actual = model.nonlinear_hat(n*coeff)/n
        defect = float(np.linalg.norm(actual-target)/np.linalg.norm(target))
        rhs = np.fft.ifft(model.rhs_hat(n*coeff))
        # A scale-free rate defect, not time-integrator conservation.
        mass_defect = float(abs(2*np.real(np.vdot(u, rhs)))/(2*np.linalg.norm(u)*np.linalg.norm(rhs)))
        results.append({'kind': kind, 'c': c, 'relative_padded_nonlinear_rhs_defect': defect,
                        'normalized_semidiscrete_mass_derivative_defect': mass_defect})
    return {'n': n, 'padded_n': fine_n, 'maximum_retained_mode': int(max(abs(integer_modes[keep]))),
            'mask': '|m|<N/4', 'results': results}


def figure(rows, destination):
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10,
        'mathtext.fontset': 'dejavuserif', 'pdf.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(6.55, 4.7), sharex=True)
    for col, epsilon in enumerate((.05, .1)):
        for c, color, marker in [(1., '.5', 'o'), (1.5, 'black', 'x')]:
            values = [r for r in rows if r['epsilon'] == epsilon and r['c'] == c
                      and r['n'] == 128 and r['step'] == .01]
            T = np.array([r['T'] for r in values])
            exact_gain = np.array([r['exact_gain'] for r in values])
            measured_gain = np.array([r['measured_gain'] for r in values])
            exact_phase = np.array([r['exact_phase'] for r in values])
            measured_phase = np.array([r['measured_phase'] for r in values])
            # Growth is c-independent; plot one analytical curve to avoid an
            # obscured gray line being mistaken for missing evidence.
            if c == 1.5:
                axes[0, col].plot(T, exact_gain, color='black', lw=1.2, label='Exact retained model')
            axes[0, col].plot(T[::8], measured_gain[::8], linestyle='none', marker=marker,
                color=color, markersize=3.5, markerfacecolor='white', label=f'Solver, c={c:g}')
            axes[1, col].plot(T, exact_phase, color=color, lw=1.2, linestyle='-' if c==1.5 else '--')
            axes[1, col].plot(T[::8], measured_phase[::8], linestyle='none', marker=marker,
                color=color, markersize=3.5, markerfacecolor='white')
        axes[0, col].set_title(r'$\epsilon='+str(epsilon)+'$')
        axes[1, col].set_xlabel(r'Scaled time $T$')
        for ax in axes[:, col]:
            ax.set_xlim(0,4); ax.set_xticks([0,2,4]); ax.grid(axis='y', color='.9', lw=.5)
    axes[0,0].set_ylabel('Unstable-mode gain')
    axes[1,0].set_ylabel('Unstable-mode phase (rad)')
    axes[0,0].legend(frameon=False, fontsize=8, loc='upper left')
    fig.suptitle('Temporal potential Dysthe: linear MI control', fontsize=11, y=1.01)
    fig.text(.5,-.025,r'$a=0.8$, $\delta=10^{-6}$, $\ell=2\pi(5)/40$; lines: exact; symbols: nonlinear solver.',
             ha='center',va='top',fontsize=8)
    fig.tight_layout(h_pad=.9,w_pad=1.2)
    destination.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination,bbox_inches='tight',pad_inches=.06)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver',type=Path,default=Path(__file__).with_name('run.py'))
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).parent/'mi-check')
    parser.add_argument('--figure',type=Path)
    args=parser.parse_args()
    source_sha=sha(args.solver)
    solver=load_solver(args.solver)
    a, delta, length, mode = .8, 1e-6, 40., 5
    ell=2*np.pi*mode/length
    times=np.arange(81)*.05
    tolerance={'growth_rate_absolute':2e-6,'phase_rate_absolute':2e-6,
               'complex_branch_relative':2e-5,'padded_rhs_relative':2e-12,
               'semidiscrete_mass_derivative':2e-12}
    cases,rows=[],[]
    started=time.perf_counter()
    for epsilon in (.05,.1):
        for c in (1.,1.5):
            exact,_,_,_=exact_linearization(a,ell,epsilon,c)
            runs,curves=[],[]
            for n,step in [(128,.01),(64,.01),(128,.005)]:
                measured,data,ratio=one_trajectory(solver,a,ell,mode,length,epsilon,c,delta,n,step,times)
                measured['passed']=(measured['growth_rate_absolute_error']<=tolerance['growth_rate_absolute']
                    and measured['phase_rate_absolute_error']<=tolerance['phase_rate_absolute']
                    and measured['maximum_relative_complex_branch_error']<=tolerance['complex_branch_relative'])
                runs.append(measured);rows.extend(data);curves.append(ratio)
            cases.append({'epsilon':epsilon,'c':c,'model_id':solver.MODELS['literal' if c==1 else 'stability'],
                'exact_growth_rate':float(exact.real),'exact_phase_rate':float(exact.imag),
                'exact_modulation_frequency':float(-exact.imag),'trajectories':runs,
                'grid_change_relative_complex_branch':float(max(abs(curves[1]/curves[0]-1))),
                'step_change_relative_complex_branch':float(max(abs(curves[2]/curves[0]-1)))})
    padded=independent_dealias_audit(solver)
    wrong_variant_controls=[]
    for epsilon in (.05,.1):
        literal=next(case for case in cases if case['epsilon']==epsilon and case['c']==1.)
        standard=next(case for case in cases if case['epsilon']==epsilon and case['c']==1.5)
        wrong_drift_error=abs(literal['trajectories'][0]['measured_phase_rate']-standard['exact_phase_rate'])
        wrong_variant_controls.append({'epsilon':epsilon,
            'literal_solver_against_c1p5_phase_rate_error':wrong_drift_error,
            'phase_test_rejects_wrong_variant':wrong_drift_error>tolerance['phase_rate_absolute'],
            'exact_growth_rates_identical':literal['exact_growth_rate']==standard['exact_growth_rate']})
    passed=all(r['passed'] for case in cases for r in case['trajectories']) and all(
        r['relative_padded_nonlinear_rhs_defect']<=tolerance['padded_rhs_relative'] and
        r['normalized_semidiscrete_mass_derivative_defect']<=tolerance['semidiscrete_mass_derivative']
        for r in padded['results']) and all(r['phase_test_rejects_wrong_variant'] for r in wrong_variant_controls)
    if sha(args.solver)!=source_sha: raise RuntimeError('Solver source changed during validation')
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    csvpath=out/'temporal-mi-check.csv'
    with csvpath.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    figpath=args.figure or out/'temporal-mi-check.pdf'
    figure(rows,figpath)
    summary={'test':'independent_complex_eigenvalue_temporal_MI','passed':passed,
        'solver_sha256':source_sha,'check_script_sha256':sha(__file__),
        'parameters':{'a':a,'delta':delta,'length':length,'discrete_mode':mode,'ell':ell,'T_end':4.,'samples':len(times)},
        'tolerances':tolerance,'runtime':{'python':platform.python_version(),'numpy':np.__version__,
          'scipy':solver.scipy.__version__,'elapsed_seconds':time.perf_counter()-started},
        'independent_formula':{'P':'ell^2/8','h':'epsilon*a^2*ell/4',
          'omega':'epsilon*(ell^3/16+c*a^2*ell)',
          'gamma_squared':'P*(a^2*(1-epsilon*abs(ell))-P)-h^2',
          'unstable_eigenvalue':'gamma-i*omega',
          'matrix':'[[-i*(omega-h), -P], [P-a^2*(1-epsilon*abs(ell)), -i*(omega+h)]]',
          'eigenvector':'(1,(-gamma+i*h)/P), Euclidean-normalized; p=delta*Re(v0*exp(i*ell*s)), q=delta*Re(v1*exp(i*ell*s))',
          'measurement':'Remove exact uniform carrier phase, extract positive-frequency real p/q coefficients, project into growing/stable eigenbasis; fit log modulus and unwrapped phase over all saved times.'},
        'cases':cases,'dealias_audit':padded,'wrong_variant_negative_controls':wrong_variant_controls,
        'artifacts':{'csv_sha256':sha(csvpath),'figure_sha256':sha(figpath)},
        'limitations':['Tiny perturbation about one uniform wavetrain; T<=4 in scaled temporal coordinates.',
          'This independent MI test does not replace nonlinear-packet grid and time refinement.',
          'At this perturbation size the measured errors are near floating-point cancellation limits; these numbers do not measure the asymptotic temporal convergence order.',
          'No continuum stability/existence theorem or water-wave approximation error bound is established.',
          'The real drift distinguishes c=1 from c=1.5; growth alone cannot.']}
    (out/'temporal-mi-check.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':passed,'cases':len(cases),'trajectories':12,
        'max_growth_error':max(r['growth_rate_absolute_error'] for c in cases for r in c['trajectories']),
        'max_phase_rate_error':max(r['phase_rate_absolute_error'] for c in cases for r in c['trajectories']),
        'max_complex_branch_error':max(r['maximum_relative_complex_branch_error'] for c in cases for r in c['trajectories']),
        'summary':str(out/'temporal-mi-check.json')},indent=2))
    if not passed: raise SystemExit(1)


if __name__=='__main__':main()

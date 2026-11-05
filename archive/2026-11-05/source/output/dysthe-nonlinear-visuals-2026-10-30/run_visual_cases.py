#!/usr/bin/env python3
"""Checked small-time temporal potential-envelope data, not a 3D spatial wave model.

Three initial profiles and three equations: full c=3/2 temporal Dysthe, cubic NLS,
and the exactly evaluated free linear part of temporal Dysthe. All arrays retain
complex phase. Reference solver snapshot is unchanged from the October30 POC.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import platform
import time
import numpy as np
import scipy
from scipy.signal import resample
from vendor_temporal_solver import Model, solve, inlets, diagnostics, errors, norm, sha, write_json, MODELS

ROOT=Path(__file__).resolve().parent
LINEAR_ID='dysthe-1979-temporal-potential-1d-free-linear-control-v1'
CONFIG={'n':1024,'length':40.,'epsilon':.05,'step':.0025,'initial_modes':12,'seed':19791030,'max_time':4.,'tolerances':{'h0':1e-5,'h1':1e-4,'h2':1e-3,'tail':1e-8,'action':1e-6},'primary_model_id':MODELS['stability'],'coordinates':'A=epsilon*u; s=epsilon*(x-t/2); T=epsilon**2*t; original x,t carrier-dimensionless','interpretation':'Periodic 1D longitudinal temporal potential envelope; c=3/2 later-standard/stability-consistent variant; not literal printed c=1, not active spatial free-surface baseline, not full2D original.'}

def saved_times():
 # Integer ticks of .005 ensure every save respects both fine and coarse steps.
 ticks=np.unique(np.concatenate([np.arange(0,21),np.arange(25,201,5),np.arange(210,801,10)]))
 return ticks*.005

def initial_profiles(config):
 data=inlets(config)
 values=[('gaussian',data[0][1],data[0][2]),('random',data[2][1],data[2][2])]
 n,L=config['n'],config['length'];s=np.arange(n)*L/n-L/2
 pars={'amplitude_each':.6,'width_each':2.2,'centres':[-5.,5.],'phase_left':0.,'phase_right':float(np.pi/3),'chirp':0.}
 u=sum(.6*np.exp(-.5*((s-center+j*L)/2.2)**2+1j*phase) for center,phase in [(-5.,0.),(5.,np.pi/3)] for j in [-1,0,1])
 mask=abs(np.fft.fftfreq(n)*n)<=config['initial_modes']
 u=np.fft.ifft(np.fft.fft(u)*mask)
 values.append(('two_packet',u,pars))
 return values

def free_linear(u0,config,T,n=None):
 n=n or config['n'];m=Model(n,config['length'],config['epsilon'],'stability')
 v0=np.fft.fft(m.project(resample(u0,n)))
 return np.fft.ifft(np.exp(T[:,None]*m.linear[None,:])*v0[None,:],axis=-1)

def near_zero(u,linear,u0,config,T):
 model=Model(config['n'],config['length'],config['epsilon'],'stability')
 n0=np.fft.ifft(model.nonlinear_hat(np.fft.fft(u0)))
 rows=[]
 for j,t in enumerate(T):
  if 0<t<=.1+1e-12:
   delta=u[j]-linear[j]
   rows.append({'T':float(t),'nonlinear_minus_linear_h0':float(norm(delta,config['length'],0)),'difference_quotient_h0':float(norm(delta/t,config['length'],0)),'quotient_minus_initial_nonlinearity_relative_h0':float(norm(delta/t-n0,config['length'],0)/norm(n0,config['length'],0)),'pointwise_max_complex_difference':float(abs(delta).max()),'pointwise_max_magnitude_difference':float(abs(abs(u[j])-abs(linear[j])).max())})
 return {'initial_nonlinearity_h0':float(norm(n0,config['length'],0)),'definition':'For positive T, (u_Dysthe-u_linear)/T tends to nonlinear RHS at u0. Quotient differences diagnose early nonlinear change; they are not a continuum error certificate.','rows':rows}

def run(output):
 output=Path(output)
 if output.exists() and any(output.iterdir()):raise ValueError('Output must be new or empty; preserve previous numerical evidence')
 output.mkdir(parents=True,exist_ok=True)
 config=dict(CONFIG);T=saved_times();s=np.arange(config['n'])*config['length']/config['n']-config['length']/2;k=2*np.pi*np.fft.fftfreq(config['n'],d=config['length']/config['n'])
 ident={'config':config,'solver_sha256':sha(ROOT/'vendor_temporal_solver.py'),'driver_sha256':sha(__file__),'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform()},'model_ids':{'dysthe':MODELS['stability'],'nls':MODELS['nls'],'linear':LINEAR_ID}}
 summary={'identity':ident,'saved_time_count':len(T),'saved_times':T.tolist(),'cases':[],'limitations':['No new neural training or GPU execution.','No nonlinear MI claim from isolated packet snapshots.','Periodic-domain evidence throughT4; no whole-line decay, global existence, physical free-surface validation or measured NOAA comparison.','u is a potential envelope with amplitude and phase, not actual surface elevation.','All max|u| observations are sampled grid maxima, with independent spatial refinement.']}
 write_json(output/'identity.json',ident)
 for family,u0,params in initial_profiles(config):
  arrays={'T':T,'s':s,'k':k,'u0':u0};permodel={};cost={};qualified=True
  for name,kind in [('dysthe','stability'),('nls','nls')]:
   fields={};seconds={}
   for role,n,h in [('fine',config['n'],config['step']),('grid',config['n']//2,config['step']),('step',config['n'],2*config['step'])]:
    fields[role],seconds[role]=solve(u0,config['length'],config['epsilon'],kind,n,h,T)
   refs={role:{key:float(val.max()) for key,val in errors(fields[role],fields['fine'],config['length']).items()} for role in ['grid','step']}
   diag=diagnostics(fields['fine'],T,config['length'],config['epsilon']);tol=config['tolerances']
   passed=all(v<=tol[key] for d in refs.values() for key,v in d.items()) and diag['max_spectral_tail']<=tol['tail'] and diag['max_action_drift']<=tol['action'];qualified &= passed
   permodel[name]={'model_id':ident['model_ids'][name],'qualified':bool(passed),'refinement':refs,'diagnostics':diag,'seconds':seconds}
   arrays[name]=fields['fine'];arrays[name+'_power']=abs(np.fft.fft(fields['fine'],axis=-1)/config['n'])**2
   cost[name]=sum(seconds.values())
  start=time.perf_counter();linear=free_linear(u0,config,T);cost['linear']=time.perf_counter()-start
  power=abs(np.fft.fft(linear,axis=-1)/config['n'])**2
  power_error=float(np.max(np.sum(abs(power-power[0]),axis=-1)/np.sum(power[0])))
  linear_grid=free_linear(u0,config,T,config['n']//2)
  linref={key:float(val.max()) for key,val in errors(linear_grid,linear,config['length']).items()}
  diag=diagnostics(linear,T,config['length'],config['epsilon'])
  linpass=power_error<1e-12 and diag['max_action_drift']<1e-12 and all(v<config['tolerances'][key] for key,v in linref.items())
  qualified &= linpass
  permodel['linear']={'model_id':LINEAR_ID,'qualified':bool(linpass),'evaluation':'u(T)=IFFT(exp(T*(i*k²/8-i*epsilon*k³/16))*FFT(u0)), independently for every saved T; no time stepping','grid_refinement':linref,'relative_l1_spectral_power_change':power_error,'diagnostics':diag,'seconds':cost['linear']}
  arrays['linear']=linear;arrays['linear_power']=power
  path=output/f'{family}.npz';np.savez_compressed(path,**arrays)
  parity={name:float(abs(arrays[name][0]-u0).max()) for name in ['dysthe','nls','linear']};qualified &= all(v<1e-12 for v in parity.values())
  comparisons={name:{key:float(val.max()) for key,val in errors(arrays['dysthe'],arrays[name],config['length']).items()} for name in ['nls','linear']}
  case={'family':family,'parameters':params,'initial_projection_modes':config['initial_modes'],'model':'dysthe c=3/2 with NLS and exact free-linear controls','qualified':bool(qualified),'status':'complete' if qualified else 'numerically_unresolved','artifact':{'path':path.name,'sha256':sha(path),'dtype':'complex128','grid':config['n'],'array_axes':['saved_time','s_grid']},'initial_parity_max_absolute':parity,'models':permodel,'comparisons_max_relative':comparisons,'near_zero':near_zero(arrays['dysthe'],linear,u0,config,T),'integration_seconds':cost}
  summary['cases'].append(case);write_json(output/(family+'.json'),case);write_json(output/'summary.json',summary)
  print(family,'qualified=',qualified,'maxpeak=',permodel['dysthe']['diagnostics']['max_peak'],'gridH2=',permodel['dysthe']['refinement']['grid']['h2'],'stepH2=',permodel['dysthe']['refinement']['step']['h2'],flush=True)
 write_json(output/'summary.json',summary)
 return summary

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'data');a=p.parse_args();run(a.output)
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Independently refined, very early nonlinear envelope evolution.

Historical T4 arrays remain unchanged. This driver reuses their saved initial
complex profiles and compares Dysthe/NLS with their respective free linear parts.
"""
from pathlib import Path
import argparse
import platform
import time
import numpy as np
import scipy
from scipy.signal import resample
from vendor_temporal_solver import Model, solve, errors, diagnostics, norm, sha, write_json, MODELS
from run_visual_cases import ROOT, LINEAR_ID

TIMES=np.array([0.,1e-4,1e-3,1e-2])


def exact_linear(u0,c,T,kind='stability',n=None):
 n=n or c['n'];m=Model(n,c['length'],c['epsilon'],kind)
 u=m.project(resample(u0,n));v=np.fft.fft(u)
 return np.fft.ifft(np.exp(T[:,None]*m.linear[None,:])*v[None,:],axis=-1)


def initial_sources(data):
 import json
 s=json.loads((data/'summary.json').read_text());c=s['identity']['config'];profiles=[]
 for r in s['cases']:
  path=data/r['artifact']['path']
  if sha(path)!=r['artifact']['sha256']:raise ValueError('Saved initial-profile source checksum differs')
  with np.load(path,allow_pickle=False) as z:u0=z['u0'].copy()
  profiles.append((r['family'],u0,{'source_array':str(path.name),'source_sha256':sha(path),'initial_projection_modes':12,'parameters':r['parameters']}))
 tail=data/'gaussian-tail-audit.npz';j=json.loads((data/'gaussian-tail-audit.json').read_text())
 if sha(tail)!=j['artifact']['sha256']:raise ValueError('Gaussian high-cutoff source checksum differs')
 with np.load(tail,allow_pickle=False) as z:u24=z['p24_u0'].copy()
 profiles.append(('gaussian_p24',u24,{'source_array':tail.name,'source_sha256':sha(tail),'initial_projection_modes':24,'purpose':'Initial spectral-cutoff control, distinct from the P12 Gaussian data.'}))
 return c,profiles


def quotient_rows(fields,linear,u0,c,T,kind):
 m=Model(c['n'],c['length'],c['epsilon'],kind)
 n0=np.fft.ifft(m.nonlinear_hat(np.fft.fft(u0)));den=float(norm(n0,c['length'],0));rows=[]
 for j,t in enumerate(T):
  if t==0:continue
  delta=fields[j]-linear[j];q=delta/t
  rows.append({'T':float(t),'difference_quotient_h0':float(norm(q,c['length'],0)),'initial_nonlinearity_h0':den,'relative_quotient_defect_h0':float(norm(q-n0,c['length'],0)/den),'max_complex_nonlinear_minus_linear':float(abs(delta).max()),'max_magnitude_nonlinear_minus_linear':float(abs(abs(fields[j])-abs(linear[j])).max())})
 return rows


def run(data,output):
 data,output=Path(data),Path(output)
 if output.exists() and any(output.iterdir()):raise ValueError('Choose a new or empty output directory; existing evidence is preserved')
 c,profiles=initial_sources(data);c=dict(c,step=2.5e-5,max_time=.01);output.mkdir(parents=True,exist_ok=True)
 s=np.arange(c['n'])*c['length']/c['n']-c['length']/2;k=2*np.pi*np.fft.fftfreq(c['n'],d=c['length']/c['n'])
 summary={'identity':{'driver_sha256':sha(__file__),'solver_sha256':sha(ROOT/'vendor_temporal_solver.py'),'initial_case_driver_sha256':sha(ROOT/'run_visual_cases.py'),'config':c,'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__}},'times':TIMES.tolist(),'cases':[],'interpretation':'Very early periodic1Dtemporal potential-envelope evolution. c3/2 primary. Ordinary max|u|; separate P24Gaussian controls the input truncation. No neural or long nonlinear training/evolution claim.','quotient_definition':'For each nonlinear equation with matched linear operator L, (u_NL(T)-exp(TL)u0)/T tends to its nonlinear RHS N(u0) asT approaches0. The NLS-matched free linear field omits Dysthe cubic dispersion.'}
 for name,u0,source in profiles:
  arrays={'T':TIMES,'s':s,'k':k,'u0':u0};models={};passes=[]
  for key,kind in [('dysthe','stability'),('nls','nls')]:
   vals={};seconds={}
   for role,n,h in [('fine',c['n'],c['step']),('grid',c['n']//2,c['step']),('step',c['n'],2*c['step'])]:vals[role],seconds[role]=solve(u0,c['length'],c['epsilon'],kind,n,h,TIMES)
   ref={role:{norm_name:float(v.max()) for norm_name,v in errors(vals[role],vals['fine'],c['length']).items()} for role in ['grid','step']};d=diagnostics(vals['fine'],TIMES,c['length'],c['epsilon'])
   passed=all(v<=c['tolerances'][n] for r in ref.values() for n,v in r.items()) and d['max_action_drift']<=c['tolerances']['action'] and d['max_spectral_tail']<=c['tolerances']['tail'];passes.append(passed)
   linear=exact_linear(u0,c,TIMES,kind);linear_key='linear' if key=='dysthe' else 'linear_nls';arrays[linear_key]=linear
   arrays[key]=vals['fine'];arrays[key+'_power']=abs(np.fft.fft(vals['fine'],axis=-1)/c['n'])**2
   models[key]={'model_id':MODELS[kind],'qualified':bool(passed),'refinement':ref,'diagnostics':d,'seconds':seconds,'matched_linear_quotient':quotient_rows(vals['fine'],linear,u0,c,TIMES,kind)}
  arrays['linear_power']=abs(np.fft.fft(arrays['linear'],axis=-1)/c['n'])**2
  power=arrays['linear_power'];power_change=float(np.max(np.sum(abs(power-power[0]),axis=-1)/power[0].sum()))
  linear_ref={n:float(v.max()) for n,v in errors(exact_linear(u0,c,TIMES,'stability',c['n']//2),arrays['linear'],c['length']).items()}
  models['linear']={'model_id':LINEAR_ID,'evaluation':'Exact Fourier phase evaluated at each saved time','relative_spectral_power_change':power_change,'grid_refinement':linear_ref,'diagnostics':diagnostics(arrays['linear'],TIMES,c['length'],c['epsilon']),'qualified':bool(power_change<1e-12 and all(v<c['tolerances'][n] for n,v in linear_ref.items()))};passes.append(models['linear']['qualified'])
  parity={key:float(abs(arrays[key][0]-u0).max()) for key in ['dysthe','nls','linear','linear_nls']};passes.append(all(v<1e-12 for v in parity.values()))
  path=output/(name+'.npz');np.savez_compressed(path,**arrays)
  case={'family':name,'source_initial_data':source,'models':models,'qualified':bool(all(passes)),'initial_parity':parity,'artifact':{'path':path.name,'sha256':sha(path),'dtype':'complex128','axes':['saved_T','s_grid']}}
  summary['cases'].append(case);write_json(output/(name+'.json'),case);write_json(output/'summary.json',summary)
  print(name,'qualified=',case['qualified'],'quotient defects=',[r['relative_quotient_defect_h0'] for r in models['dysthe']['matched_linear_quotient']],flush=True)
 return summary


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=ROOT/'data');p.add_argument('--output',type=Path,default=ROOT/'data/near-zero');a=p.parse_args();run(a.data,a.output)
if __name__=='__main__':main()

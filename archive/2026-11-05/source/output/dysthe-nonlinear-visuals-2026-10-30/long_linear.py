#!/usr/bin/env python3
"""Direct free-linear Fourier evaluation throughT1024 on the fixed periodic box.

This experiment investigates periodic dispersive/interference behaviour. It is
NOT a whole-line decay calculation and does not evolve nonlinear Dysthe to1024.
"""
from pathlib import Path
import argparse
import json
import platform
import time
import numpy as np
import scipy
from scipy.signal import resample
from vendor_temporal_solver import diagnostics, errors, sha, write_json
from run_visual_cases import ROOT, LINEAR_ID
from near_zero import exact_linear

TIMES=np.array([0.,1e-4,1e-3,1e-2,.1,1.,2.,4.,8.,16.,32.,64.,128.,256.,512.,1024.])


def run(data,output):
 data,output=Path(data),Path(output)
 if output.exists() and any(output.iterdir()):raise ValueError('Choose a fresh output directory; do not overwrite archived calculations')
 source_summary=json.loads((data/'summary.json').read_text());c=source_summary['identity']['config'];output.mkdir(parents=True,exist_ok=True)
 n,L=c['n'],c['length'];s=np.arange(n)*L/n-L/2;k=2*np.pi*np.fft.fftfreq(n,d=L/n)
 summary={'identity':{'driver_sha256':sha(__file__),'exact_linear_helper_sha256':sha(ROOT/'near_zero.py'),'solver_sha256':sha(ROOT/'vendor_temporal_solver.py'),'source_initial_config':c,'evaluation_config':{'n':n,'length':L,'epsilon':c['epsilon'],'max_time':float(TIMES[-1]),'time_step':None},'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__}},'model_id':LINEAR_ID,'times':TIMES.tolist(),'domain':'Periodic interval [-20,20), L40','evaluation':'u(T)=IFFT(exp(T*(i*k²/8-i*epsilon*k³/16))*FFT(projected u0)); independently evaluated output times, not large integration steps','limitations':['Not nonlinear evolution throughT1024.','Not a whole-line decay experiment: nonzero conserved mass imposes max|u|>=sqrt(M/L).','Sparse long-time output may miss intervening maxima and does not establish a recurrence period.','Phase-aligned L2 distance is supplementary; the primary observable is ordinary envelope maximum.','Envelope-coordinate spectrum is not a NOAA surface-elevation frequency spectrum.'],'cases':[]}
 for old in source_summary['cases']:
  path=data/old['artifact']['path'];assert sha(path)==old['artifact']['sha256']
  with np.load(path,allow_pickle=False) as z:u0=z['u0'].copy()
  start=time.perf_counter();fields=exact_linear(u0,c,TIMES);elapsed=time.perf_counter()-start
  coarse=exact_linear(u0,c,TIMES,n=n//2);ref={name:float(val.max()) for name,val in errors(coarse,fields,L).items()}
  power=abs(np.fft.fft(fields,axis=-1)/n)**2;power_change=float(np.max(np.sum(abs(power-power[0]),axis=-1)/power[0].sum()));diag=diagnostics(fields,TIMES,L,c['epsilon'])
  # Peak interpolation checks sampling of the continuum trigonometric polynomial.
  dense=resample(fields,4*n,axis=-1);peak=abs(fields).max(axis=-1);peak_dense=abs(dense).max(axis=-1);peak_change=abs(peak-peak_dense)/peak_dense
  action0=diag['rows'][0]['action'];lower=float(np.sqrt(action0/L));passed=all(v<=c['tolerances'][name] for name,v in ref.items()) and power_change<=1e-12 and diag['max_action_drift']<=1e-12 and float(peak_change.max())<=1e-3
  name=old['family'];dest=output/(name+'.npz');np.savez_compressed(dest,T=TIMES,s=s,k=k,u0=u0,linear=fields,linear_power=power,peak=peak,peak_refined=peak_dense)
  case={'family':name,'qualified':bool(passed),'source_initial_data':{'array':path.name,'sha256':sha(path),'projection_modes':old['initial_projection_modes']},'artifact':{'path':dest.name,'sha256':sha(dest),'dtype':'complex128','axes':['saved_T','s_grid']},'diagnostics':diag,'grid_refinement':ref,'relative_l1_spectral_power_change':power_change,'periodic_peak_lower_bound':lower,'peak_refinement':{'fine_spatial_points':4*n,'max_relative_difference':float(peak_change.max()),'threshold':1e-3,'note':'Dense Fourier interpolation evaluates the same trigonometric polynomial; this is a sampling check, not continuum PDE convergence.'},'rows':[{'T':float(t),'ordinary_grid_linf':float(peak[j]),'dense_grid_linf':float(peak_dense[j]),'peak_ratio':float(peak[j]/peak[0]),'relative_peak_grid_change':float(peak_change[j]),'periodic_mass_lower_bound':lower} for j,t in enumerate(TIMES)],'seconds':elapsed}
  summary['cases'].append(case);write_json(output/(name+'.json'),case);write_json(output/'summary.json',summary)
  print(name,'qualified=',passed,'P1024=',peak[-1],'initial=',peak[0],'lower=',lower,'peak samplingchange=',peak_change.max(),flush=True)
 return summary


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=ROOT/'data');p.add_argument('--output',type=Path,default=ROOT/'data/long-linear');a=p.parse_args();run(a.data,a.output)
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Quantify Gaussian initial-projection ringing before interpreting tiny tail ripples."""
from pathlib import Path
import argparse
import json
import numpy as np
from vendor_temporal_solver import solve, norm, errors, sha, write_json
from run_visual_cases import ROOT, CONFIG, free_linear


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=ROOT/'data');a=p.parse_args()
 data=a.data;out=data/'gaussian-tail-audit.npz';meta=data/'gaussian-tail-audit.json'
 if out.exists() or meta.exists():raise ValueError('Tail audit already exists; preserve evidence and choose a fresh data directory')
 summary=json.loads((data/'summary.json').read_text());c=summary['identity']['config'];case=next(r for r in summary['cases'] if r['family']=='gaussian')
 source=data/case['artifact']['path'];assert sha(source)==case['artifact']['sha256']
 with np.load(source,allow_pickle=False) as z:
  s=z['s'];T=z['T'][z['T']<=.1+1e-12];base=z['dysthe'][:len(T)];linear12=z['linear'][:len(T)]
 pars=case['parameters'];L=c['length'];n=c['n']
 analytic=sum(pars['amplitude']*np.exp((-.5+1j*pars['chirp'])*((s-pars['center']+j*L)/pars['width'])**2+1j*pars['phase']) for j in [-1,0,1])
 modes=abs(np.fft.fftfreq(n)*n);project=lambda cutoff:np.fft.ifft(np.fft.fft(analytic)*(modes<=cutoff))
 u12,u24=project(12),project(24);tail=abs(analytic)<.01*abs(analytic).max()
 initial=[]
 for cutoff,u in [(12,u12),(24,u24)]:
  initial.append({'cutoff':cutoff,'max_absolute_complex_difference':float(abs(u-analytic).max()),'relative_h0':float(norm(u-analytic,L,0)/norm(analytic,L,0)),'relative_h2':float(norm(u-analytic,L,2)/norm(analytic,L,2)),'tail_max_absolute_complex_difference':float(abs(u-analytic)[tail].max()),'tail_max_absolute_magnitude_difference':float(abs(abs(u)-abs(analytic))[tail].max())})
 high,secs=solve(u24,L,c['epsilon'],'stability',n,c['step'],T)
 coarse,coarse_secs=solve(u24,L,c['epsilon'],'stability',n,2*c['step'],T)
 lin24=free_linear(u24,c,T)
 refine={key:float(val.max()) for key,val in errors(coarse,high,L).items()}
 np.savez_compressed(out,T=T,s=s,analytic_u0=analytic,p12_u0=u12,p24_u0=u24,tail_mask=tail,dysthe_p12=base,dysthe_p24=high,linear_p12=linear12,linear_p24=lin24)
 result={'config':{'T_max':.1,'n':n,'length':L,'epsilon':c['epsilon'],'fine_step':c['step'],'comparison_step':2*c['step'],'cutoffs':[12,24]},'model_id':c['primary_model_id'],'driver_sha256':sha(__file__),'solver_sha256':sha(ROOT/'vendor_temporal_solver.py'),'original_gaussian_array_sha256':sha(source),'artifact':{'path':out.name,'sha256':sha(out)},'interpretation':'P12 and P24 define distinct initial profiles. Their differences measure sensitivity to initial spectral truncation, not solver discretization error. Original analytic profile is a periodized Gaussian. Tail region is fixed from analytic initial |u0|<1% initial peak; it is not a peak-aligned moving mask.','initial_projection':initial,'high_cutoff_time_step_refinement':refine,'high_cutoff_step_qualified':all(v<=c['tolerances'][key] for key,v in refine.items()),'baseline_initial_matches_projection_max_abs':float(abs(base[0]-u12).max()),'seconds':{'fine':secs,'comparison':coarse_secs},'rows':[{'T':float(t),'tail_max_p12_p24_complex_difference':float(abs(base[j]-high[j])[tail].max()),'tail_max_p12_p24_magnitude_difference':float(abs(abs(base[j])-abs(high[j]))[tail].max()),'tail_max_p12_nonlinear_linear_complex_difference':float(abs(base[j]-linear12[j])[tail].max()),'tail_max_p24_nonlinear_linear_complex_difference':float(abs(high[j]-lin24[j])[tail].max()),'tail_max_p12_nonlinear_linear_magnitude_difference':float(abs(abs(base[j])-abs(linear12[j]))[tail].max()),'tail_max_p24_nonlinear_linear_magnitude_difference':float(abs(abs(high[j])-abs(lin24[j]))[tail].max())} for j,t in enumerate(T)]}
 write_json(meta,result)
 print(json.dumps({'initial_projection':initial,'high_cutoff_time_step_refinement':refine,'qualified':result['high_cutoff_step_qualified']},indent=2))
if __name__=='__main__':main()

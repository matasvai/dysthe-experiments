#!/usr/bin/env python3
"""Packet domain checks and portable CSV/plot-data exports for the temporal POC."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.signal import resample
from run import solve, errors, diagnostics, inlets, write_json, sha


def common_errors(small,large,length):
 n=small.shape[-1]
 if large.shape[-1]!=2*n:raise ValueError('Double length/count required')
 k1=2*np.pi*np.fft.fftfreq(n,d=length/n);k2=2*np.pi*np.fft.fftfreq(2*n,d=length/n)
 v1=np.fft.fft(small,axis=-1);v2=np.fft.fft(large,axis=-1)
 diff=[];den=[]
 for p in range(3):
  a=np.fft.ifft((1j*k1)**p*v1,axis=-1)
  b=np.fft.ifft((1j*k2)**p*v2,axis=-1)[:,n//2:3*n//2]
  diff.append(np.sum(abs(a-b)**2,axis=-1));den.append(np.sum(abs(b)**2,axis=-1))
 return {f'h{s}':np.sqrt(sum(w*diff[j] for j,w in enumerate(weights))/sum(w*den[j] for j,w in enumerate(weights))) for s,weights in enumerate(((1,),(1,1),(1,2,1)))}


def domain_checks(results,horizon=16.):
 root=Path(results);summary=json.loads((root/'summary.json').read_text());c=summary['identity']['config'];out=[]
 for record in summary['cases']:
  if record['family']!='packet' or record['model'] not in ('literal','stability') or record['T']!=horizon:continue
  row={'key':record['key'],'T':horizon,'epsilon':record['epsilon'],'model_id':record['model_id'],'small_reference_qualified':record['qualified'],'qualified':False}
  if not record['qualified']:
   row['status']='blocked_by_small_box_refinement';out.append(row);continue
  with np.load(root/record['artifact']['path'],allow_pickle=False) as z:small=z['fields'];T=z['T']
  bigger=dict(c,length=2*c['length'],n=2*c['n'],initial_modes=2*c['initial_modes'])
  u0=inlets(bigger)[0][1];arrays={};cost={}
  try:
   for role,n,h in [('fine',bigger['n'],c['step']),('grid',c['n'],c['step']),('step',bigger['n'],2*c['step'])]:
    arrays[role],cost[role]=solve(u0,bigger['length'],record['epsilon'],record['model'],n,h,T)
  except FloatingPointError as error:
   row.update(status='nonfinite_large_domain_failure',message=str(error));out.append(row);continue
  refinement={role:{key:float(value.max()) for key,value in errors(arrays[role],arrays['fine'],bigger['length']).items()} for role in ('grid','step')}
  domain=common_errors(small,arrays['fine'],c['length']);diag=diagnostics(arrays['fine'],T,bigger['length'],record['epsilon'])
  qualified_ref=all(value<=c['tolerances'][key] for vals in refinement.values() for key,value in vals.items()) and diag['max_spectral_tail']<=c['tolerances']['tail'] and diag['max_action_drift']<=c['tolerances']['action']
  row.update(status='complete',qualified=qualified_ref and float(domain['h1'].max())<=.01 and float(domain['h1'][0])<=.001,large_reference_qualified=qualified_ref,refinement=refinement,initial_common_interval_difference={key:float(value[0]) for key,value in domain.items()},max_common_interval_difference={key:float(value.max()) for key,value in domain.items()},max_large_boundary_action_fraction=max(x['boundary_action_fraction'] for x in diag['rows']),seconds=cost,thresholds={'common_interval_h1':.01,'initial_common_interval_h1':.001},rows=[{'T':float(t),**{key:float(value[j]) for key,value in domain.items()}} for j,t in enumerate(T)])
  if not row['qualified']:row['status']='domain_sensitive_or_unresolved'
  out.append(row);print('domain',row['key'],row['status'],row['max_common_interval_difference'],flush=True)
 result={'source_sha256':sha(__file__),'solver_sha256':sha(Path(__file__).with_name('run.py')),'reference_summary_sha256':sha(root/'summary.json'),'definition':'Same analytic Gaussian packet and matched physical initial Fourier cutoff, period/grid both doubled; full periodic derivatives computed before restricting to common original interval. No zero padding, phase fit, or translation alignment.','limitation':'Two periodic domains do not establish a whole-line limit; periodic nonlocal mean flow also changes with period.','cases':out}
 write_json(root/'domain-checks.json',result)
 return result


def export(results,destination):
 root=Path(results);dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
 summary=json.loads((root/'summary.json').read_text());rows=[]
 for r in summary['cases']:
  diag=r.get('diagnostics',{})
  rows.append({'family':r['family'],'epsilon':r['epsilon'],'model':r['model'],'T':r['T'],'status':r['status'],'qualified':r['qualified'],'max_peak':diag.get('max_peak'),'max_h2_ratio':diag.get('max_h2_ratio'),'max_action_drift':diag.get('max_action_drift'),'max_tail':diag.get('max_spectral_tail'),'max_grid_h2':r.get('refinement',{}).get('grid',{}).get('h2'),'max_step_h2':r.get('refinement',{}).get('step',{}).get('h2'),'integration_seconds':sum(r.get('seconds',{}).values())})
 with (dest/'case-summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 write_json(dest/'summary.json',summary)
 # Only the longest qualified run for each input/model is copied for plotting.
 # These downsampled fields are NOT the arrays used for refinement metrics.
 artifacts=[]
 for family in ('packet','sideband','random'):
  for eps in summary['identity']['config']['epsilons']:
   for kind in summary['identity']['config']['models']:
    candidates=[r for r in summary['cases'] if r['family']==family and r['epsilon']==eps and r['model']==kind and r['qualified']]
    if not candidates:continue
    r=max(candidates,key=lambda x:x['T'])
    with np.load(root/r['artifact']['path'],allow_pickle=False) as z:
     full=z['fields'];times=z['T'];L=summary['identity']['config']['length'];n=256
     reduced=resample(full,n,axis=-1).astype(np.complex64)
     cutoff_power=abs(np.fft.fft(full,axis=-1))**2
     indices=abs(np.fft.fftfreq(full.shape[-1])*full.shape[-1])>=n/2
     omitted=float(np.max(cutoff_power[:,indices].sum(axis=-1)/cutoff_power.sum(axis=-1)))
     path=dest/(r['key']+'-plot.npz')
     np.savez_compressed(path,fields=reduced,T=times,s=np.arange(n)*L/n-L/2)
     artifacts.append({'path':path.name,'sha256':sha(path),'key':r['key'],'model_id':r['model_id'],'reference_source_sha256':r['artifact']['sha256'],'maximum_action_fraction_omitted_for_plotting':omitted,'usage':'Visualization only; complex64, Fourier resampled 256 points; full-grid refinement results are in JSON'})
 write_json(dest/'plot-manifest.json',{'artifacts':artifacts,'source_summary_sha256':sha(root/'summary.json')})


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,default=Path(__file__).parent/'results-refined');p.add_argument('--domain',action='store_true');p.add_argument('--export',type=Path);a=p.parse_args()
 if a.domain:domain_checks(a.results)
 if a.export:export(a.results,a.export)
if __name__=='__main__':main()

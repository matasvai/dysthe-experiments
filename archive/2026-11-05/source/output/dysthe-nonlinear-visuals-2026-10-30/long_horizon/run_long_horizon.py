#!/usr/bin/env python3
"""CPU temporal Dysthe horizon ladder: periodic recurrence and isolated-packet checks.

Only c=3/2 temporal potential Dysthe is accepted. Every trajectory and refinement
stream advances from its own original data, with no reference resets. On failure,
later requested horizons are recorded as blocked. No training is involved.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import scipy
from scipy.signal import resample
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from vendor_temporal_solver import Model, MODELS, norm, sha, write_json
from run_visual_cases import initial_profiles

ROOT=Path(__file__).resolve().parent
MODEL_ID=MODELS['stability']


def validate(c):
 if c.get('model_id')!=MODEL_ID:raise ValueError('Only the explicitly versioned c=3/2 temporal potential model is accepted')
 if c['track'] not in ('periodic','isolated'):raise ValueError('Unknown physical-domain track')
 if not c['horizons'] or any(not np.isfinite(x) or x<=0 for x in c['horizons']) or any(b<=a for a,b in zip(c['horizons'],c['horizons'][1:])):raise ValueError('Positive increasing horizons required')
 if not np.isfinite(c['step']) or c['step']<=0:raise ValueError('Positive step required')
 if c['n']%4 or c['n']<128:raise ValueError('Fine grid must be a multiple of four >=128')
 if c['length']<=0 or c['epsilon']<0 or c['initial_kmax']<=0:raise ValueError('Invalid physical parameters')
 if c['max_snapshots_per_stage']<2:raise ValueError('Need at least endpoint snapshots')
 if c['monitor_interval']<2*c['step'] or not np.isclose(c['monitor_interval']/(2*c['step']),round(c['monitor_interval']/(2*c['step']))):raise ValueError('Monitoring must align with the coarse time step')
 for t in c['horizons']:
  if not np.isclose(t/(2*c['step']),round(t/(2*c['step']))):raise ValueError('Horizon must align with coarse time step')
 if set(c['tolerances'])!={'h0','h1','h2','action','tail'} or any(not np.isfinite(v) or v<=0 for v in c['tolerances'].values()):raise ValueError('All numerical tolerances must be finite and positive')
 if c['track']=='isolated' and set(c['families'])-{'gaussian','two_packet'}:raise ValueError('Isolated runs require re-evaluable analytic packets; periodic random profiles cannot be zero-padded')
 if set(c['families'])-{'gaussian','two_packet','random'}:raise ValueError('Unknown initial family')
 if c['initial_kmax'] >= 2*np.pi*(c['n']//8)/c['length']:raise ValueError('Initial spectrum exceeds the comparison grid retained band')


def packet_profile(family,n,L,c):
 s=np.arange(n)*L/n-L/2
 if family=='gaussian':
  p=c['gaussian'];terms=[(p['amplitude'],p['center'],p['width'],p['phase'],p['chirp'])]
 elif family=='two_packet':
  p=c['two_packet'];terms=[(p['amplitude_each'],p['centres'][j],p['width_each'],p['phases'][j],0.) for j in range(2)]
 else:
  if c['track']!='periodic':raise ValueError('Random field cannot be an isolated packet')
  ic={'n':n,'length':L,'epsilon':c['epsilon'],'initial_modes':int(np.floor(c['initial_kmax']*L/(2*np.pi)+1e-10)),'seed':c['seed']}
  return initial_profiles(ic)[1][1]
 u=sum(a*np.exp((-.5+1j*chirp)*((s-centre+j*L)/width)**2+1j*phase) for a,centre,width,phase,chirp in terms for j in range(-3,4))
 k=2*np.pi*np.fft.fftfreq(n,d=L/n)
 return np.fft.ifft(np.fft.fft(u)*(abs(k)<=c['initial_kmax']+1e-12))


def point_diag(u,L,epsilon):
 n=len(u);k=2*np.pi*np.fft.fftfreq(n,d=L/n);p=abs(np.fft.fft(u)/n)**2
 s=np.arange(n)*L/n-L/2
 return {'peak':float(abs(u).max()),'action':float(L*p.sum()),'h2':float(norm(u,L,2)), 'tail':float(p[abs(np.fft.fftfreq(n)*n)>=.8*n/4].sum()/p.sum()),'boundary_action_fraction':float(np.sum(abs(u[abs(s)>.45*L])**2)/np.sum(abs(u)**2)),'steepness_indicator':float(epsilon*abs(u).max()),'scaled_rms_bandwidth':float(epsilon*np.sqrt(np.sum(k*k*p)/np.sum(p)))}


def discrepancy(a,b,L):
 if len(a)!=len(b):a=resample(a,len(b))
 return {f'h{j}':float(norm(a-b,L,j)/norm(b,L,j)) for j in range(3)}


def common_box_discrepancy(small,big,L):
 n=len(small)
 if len(big)!=2*n:raise ValueError('Matched spacing and doubled grid required')
 k=2*np.pi*np.fft.fftfreq(n,d=L/n);kb=2*np.pi*np.fft.fftfreq(2*n,d=L/n)
 fs=np.fft.fft(small);fb=np.fft.fft(big);diff=[];den=[]
 for j in range(3):
  a=np.fft.ifft((1j*k)**j*fs);b=np.fft.ifft((1j*kb)**j*fb)[n//2:3*n//2]
  diff.append(np.sum(abs(a-b)**2));den.append(np.sum(abs(b)**2))
 result={f'h{j}':float(np.sqrt(sum(w*diff[i] for i,w in enumerate(weights))/sum(w*den[i] for i,w in enumerate(weights)))) for j,weights in enumerate([(1,),(1,1),(1,2,1)])}
 result['global_peak_relative_difference']=float(abs(abs(small).max()-abs(big).max())/abs(big).max())
 return result


class Stream:
 def __init__(self,n,L,h,u0,epsilon):
  self.model=Model(n,L,epsilon,'stability');self.h=h;self.u0=self.model.project(u0);self.v=np.fft.fft(self.u0);self.T=0.;self.advance=self.model.stepper(h);self.seconds=0.
 def to(self,T):
  count=round((T-self.T)/self.h)
  if count<0 or not np.isclose(count*self.h,T-self.T,rtol=1e-8,atol=1e-10):raise ValueError('Output time does not align with integration step')
  start=time.perf_counter();start_T=self.T;completed=0
  try:
   with np.errstate(over='raise',invalid='raise',divide='raise'):
    for _ in range(count):
     self.v=self.advance(self.v);completed+=1
  finally:
   # A failed step leaves v at the last successfully completed step.
   self.seconds+=time.perf_counter()-start
   self.T=T if completed==count else start_T+completed*self.h
  u=np.fft.ifft(self.v)
  if not np.isfinite(u).all():raise FloatingPointError('Nonfinite integration; not evidence of PDE nonexistence')
  return u


def streams_for(family,c):
 out={}
 for box,factor in [('base',1)]+([('double',2)] if c['track']=='isolated' else []):
  L=c['length']*factor;n=c['n']*factor;canonical=packet_profile(family,n,L,c)
  for role,N,h in [('fine',n,c['step']),('grid',n//2,c['step']),('step',n,2*c['step'])]:
   out[f'{box}-{role}']=Stream(N,L,h,resample(canonical,N),c['epsilon'])
 return out


def calibration(c,steps):
 streams=streams_for(c['families'][0],c);rows=[]
 for name,s in streams.items():
  s.to(steps*s.h)
  rows.append({'stream':name,'n':s.model.n,'length':s.model.length,'step':s.h,'measured_steps':steps,'measured_seconds':s.seconds,'seconds_per_step':s.seconds/steps})
 return {'status':'calibration_only','target_horizons':c['horizons'],'measurement':rows,'fixed_grid_estimated_seconds_all_families':{str(T):len(c['families'])*sum(r['seconds_per_step']*T/r['step'] for r in rows) for T in c['horizons']},'limitations':['Extrapolation only; no target horizon was executed.','Assumes fixed grid, period and time step; excludes setup, diagnostics, output, retries and adaptive growth.','Does not establish affordability or scientific qualification at distant times.']}


def run(c,output,calibration_steps=0):
 validate(c);output=Path(output)
 if output.exists():raise ValueError('Use a fresh output directory; previous evidence will not be overwritten')
 output.mkdir(parents=True)
 identity={'config':c,'runner_sha256':sha(__file__),'solver_sha256':sha(ROOT.parent/'vendor_temporal_solver.py'),'initial_helper_sha256':sha(ROOT.parent/'run_visual_cases.py'),'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform()}}
 write_json(output/'identity.json',identity)
 if calibration_steps:
  if calibration_steps<2:raise ValueError('At least two calibration steps required')
  result=calibration(c,calibration_steps);result['identity']=identity;write_json(output/'summary.json',result);return result
 summary={'identity':identity,'status':'running','requested_horizons':c['horizons'],'cases':[],'limits':['Periodic track measures recurrence and bounded-box evolution, not whole-line decay.','Isolated track is a two-period approximation with declared gates; not a certified whole-line limit.','Sampled diagnostics can miss between-monitor-time events.','No new neural training, GPU use, or cluster execution is implied.','No automatic grid, domain or step adaptation; change configuration and rerun after a failed gate.','Checkpoint endpoints are saved, but automatic restart/resume is not implemented.']}
 for family in c['families']:
  streams=streams_for(family,c);initial={key:point_diag(s.u0,s.model.length,c['epsilon']) for key,s in streams.items()};rows=[];start_T=0.;blocked=False;stages=[]
  initial_domain=None
  if c['track']=='isolated':initial_domain=common_box_discrepancy(streams['base-fine'].u0,streams['double-fine'].u0,c['length'])
  for horizon in c['horizons']:
   if blocked:
    stages.append({'T':horizon,'qualified':False,'status':'blocked_by_previous_gate'});continue
   coarse=2*c['step'];begin=round(start_T/coarse);end=round(horizon/coarse);jump=round(c['monitor_interval']/coarse)
   ticks=np.unique(np.r_[begin,np.arange(begin+jump,end,jump),end]).astype(int)
   select=set(np.unique(np.round(np.linspace(0,len(ticks)-1,min(c['max_snapshots_per_stage'],len(ticks)))).astype(int)))
   snapshots={name:[] for name in streams if name.endswith('fine')};snapshot_T=[];failure=None;early_gate_time=None
   try:
    for count,tick in enumerate(ticks):
     T=float(tick*coarse);fields={name:s.to(T) for name,s in streams.items()};metrics={}
     for box in ['base']+(['double'] if c['track']=='isolated' else []):
      f=fields[f'{box}-fine'];L=streams[f'{box}-fine'].model.length;d=point_diag(f,L,c['epsilon']);d['action_relative_drift']=abs(d['action']/initial[f'{box}-fine']['action']-1)
      metrics[box]={'diagnostics':d,'grid':discrepancy(fields[f'{box}-grid'],f,L),'step':discrepancy(fields[f'{box}-step'],f,L)}
     row={'T':T,'boxes':metrics}
     if c['track']=='isolated':row['domain']=common_box_discrepancy(fields['base-fine'],fields['double-fine'],c['length'])
     if not rows or T!=rows[-1]['T']:rows.append(row)
     if count in select:
      snapshot_T.append(T)
      for name in snapshots:snapshots[name].append(fields[name])
     tol=c['tolerances']
     point_numeric=all(all(value<=tol[key] for role in ['grid','step'] for key,value in box[role].items()) and box['diagnostics']['action_relative_drift']<=tol['action'] and box['diagnostics']['tail']<=tol['tail'] for box in metrics.values())
     point_domain=True
     if c['track']=='isolated':
      dtol=c['domain_tolerances'];point_domain=initial_domain['h1']<=dtol['initial_h1'] and all(row['domain'][key]<=dtol[key] for key in ['h0','h1','h2','global_peak_relative_difference']) and all(box['diagnostics']['boundary_action_fraction']<=dtol['boundary_action_fraction'] for box in metrics.values())
     if not point_numeric or not point_domain:
      early_gate_time=T
      if count not in select:
       snapshot_T.append(T)
       for name in snapshots:snapshots[name].append(fields[name])
      break
   except FloatingPointError as error:failure=str(error)
   tol=c['tolerances'];numeric_ok=not failure and all(all(val<=tol[key] for role in ['grid','step'] for key,val in box[role].items()) and box['diagnostics']['action_relative_drift']<=tol['action'] and box['diagnostics']['tail']<=tol['tail'] for row in rows for box in row['boxes'].values())
   domain_ok=True
   if c['track']=='isolated':
    dtol=c['domain_tolerances'];domain_ok=initial_domain['h1']<=dtol['initial_h1'] and all(all(row['domain'][key]<=dtol[key] for key in ['h0','h1','h2','global_peak_relative_difference']) and all(box['diagnostics']['boundary_action_fraction']<=dtol['boundary_action_fraction'] for box in row['boxes'].values()) for row in rows)
   qualified=bool(numeric_ok and domain_ok);status='qualified' if qualified else ('nonfinite_numerical_failure' if failure else ('numerically_unresolved' if not numeric_ok else 'domain_sensitive_or_boundary_contaminated'))
   artifact=output/f'{family}-T{horizon:g}.npz'
   arrays={'T':np.asarray(snapshot_T),**{name:np.asarray(value) for name,value in snapshots.items()}}
   for name,s in streams.items():arrays['checkpoint_'+name]=s.v;arrays['initial_'+name]=s.u0
   np.savez_compressed(artifact,**arrays)
   stage={'T':horizon,'requested_horizon':horizon,'last_monitored_T':rows[-1]['T'] if rows else None,'first_failed_monitor_T':early_gate_time,'checkpoint_stream_times':{name:s.T for name,s in streams.items()},'qualified':qualified,'status':status,'numerical_checks_passed':bool(numeric_ok),'domain_checks_passed':bool(domain_ok) if c['track']=='isolated' else None,'initial_domain_comparison':initial_domain,'failure':failure,'max_checks':{box:{role:{key:max(row['boxes'][box][role][key] for row in rows) for key in ['h0','h1','h2']} for role in ['grid','step']} for box in rows[0]['boxes']} if rows else {},'monitor_count_total':len(rows),'snapshot_count_this_stage':len(snapshot_T),'artifact':{'path':artifact.name,'sha256':sha(artifact),'contains':'Bounded snapshots and per-stream endpoint states; not every internal step.'},'stream_total_seconds':{name:s.seconds for name,s in streams.items()}}
   stages.append(stage);write_json(output/f'{family}-T{horizon:g}.json',stage);write_json(output/f'{family}-diagnostics.json',{'track':c['track'],'family':family,'rows':rows})
   print(c['track'],family,'T=',horizon,status,flush=True);blocked=not qualified;start_T=horizon
  case={'family':family,'target_horizons':c['horizons'],'qualified_horizon':max([0.]+[s['T'] for s in stages if s['qualified']]),'stages':stages}
  summary['cases'].append(case);write_json(output/'summary.json',summary)
 summary['status']='all_requested_horizons_qualified' if all(case['qualified_horizon']==c['horizons'][-1] for case in summary['cases']) else 'stopped_at_failed_gate';write_json(output/'summary.json',summary);return summary


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--calibrate-steps',type=int,default=0);a=p.parse_args()
 c=json.loads(a.config.read_text());s=run(c,a.output,a.calibrate_steps)
 return 0 if s['status'] in ('all_requested_horizons_qualified','calibration_only') else 2
if __name__=='__main__':raise SystemExit(main())

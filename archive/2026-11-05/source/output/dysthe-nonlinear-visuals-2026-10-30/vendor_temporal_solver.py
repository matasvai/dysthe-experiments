#!/usr/bin/env python3
"""Standalone 1D temporal-potential Dysthe POC, separately versioned from spatial research.

Original 1979 longitudinal restriction in slow moving-frame coordinates:
 A=epsilon*u, s=epsilon*(x-t/2), T=epsilon**2*t.
 u_T=-i/8*u_ss-i/2*|u|^2*u + epsilon*(u_sss/16
       -c*|u|^2*u_s + u^2*conj(u_s)/4 + i*u*|D_s|(|u|^2)/2).
 c=1: literal printed RHS sign; c=3/2: stability-consistent sign variant.
This is not the active spatial free-surface-envelope baseline or the full 2D
1979 system. The sign ambiguity is retained, never silently corrected.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.signal import find_peaks, resample

MODELS = {
 'literal': 'dysthe-1979-temporal-potential-1d-literal-v1',
 'stability': 'dysthe-1979-temporal-potential-1d-stability-sign-v1',
 'nls': 'dysthe-1979-temporal-potential-1d-nls-control-v1',
 'meanflow': 'dysthe-1979-temporal-potential-1d-meanflow-only-v1',
}
LIMITATION = 'Finite-distance periodic numerical evidence only; no continuum existence, full 2D stability, or water-wave validity certificate.'


def sha(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


@dataclass
class Model:
 n:int
 length:float
 epsilon:float
 kind:str='literal'
 def __post_init__(self):
  if type(self.n) is not int or self.n<32 or self.n%4: raise ValueError('Grid must be a multiple of four >=32')
  if not np.isfinite(self.length) or self.length<=0: raise ValueError('Length must be positive')
  if not np.isfinite(self.epsilon) or self.epsilon<0: raise ValueError('Epsilon must be nonnegative')
  if self.kind not in MODELS: raise ValueError('Unknown explicitly versioned temporal model')
  self.s=np.arange(self.n)*self.length/self.n-self.length/2
  self.k=2*np.pi*np.fft.fftfreq(self.n,d=self.length/self.n)
  self.mask=abs(np.fft.fftfreq(self.n)*self.n)<self.n/4
  self.linear=1j*self.k**2/8
  if self.kind in ('literal','stability'):self.linear-=1j*self.epsilon*self.k**3/16
 def project(self,u):
  u=np.asarray(u,dtype=np.complex128)
  if u.shape!=(self.n,) or not np.isfinite(u).all():raise ValueError('Expected one finite complex profile')
  return np.fft.ifft(np.fft.fft(u)*self.mask)
 def abs_derivative(self,rho):
  return np.fft.ifft(abs(self.k)*np.fft.fft(rho)).real
 def nonlinear_hat(self,v):
  u=np.fft.ifft(v);rho=abs(u)**2
  value=-.5j*rho*u
  if self.kind!='nls':
   value+=.5j*self.epsilon*u*self.abs_derivative(rho)
  if self.kind in ('literal','stability'):
   du=np.fft.ifft(1j*self.k*v)
   c=1. if self.kind=='literal' else 1.5
   value+=self.epsilon*(-c*rho*du+.25*u*u*du.conj())
  return np.fft.fft(value)*self.mask
 def rhs_hat(self,v):
  return self.linear*v+self.nonlinear_hat(v)
 def stepper(self,h):
  if not np.isfinite(h) or h<=0:raise ValueError('Positive step required')
  E=np.exp(h*self.linear);H=np.exp(.5*h*self.linear)
  def step(v):
   a=h*self.nonlinear_hat(v)
   b=h*H.conj()*self.nonlinear_hat(H*(v+a/2))
   c=h*H.conj()*self.nonlinear_hat(H*(v+b/2))
   d=h*E.conj()*self.nonlinear_hat(E*(v+c))
   return E*(v+(a+2*b+2*c+d)/6)*self.mask
  return step


def solve(u0,length,epsilon,kind,n,step,positions):
 model=Model(n,length,epsilon,kind)
 u=model.project(resample(u0,n));v=np.fft.fft(u)
 positions=np.asarray(positions,dtype=float)
 if positions.ndim!=1 or positions[0]!=0 or np.any(np.diff(positions)<=0):raise ValueError('Saved times must begin at zero and increase')
 differences=np.diff(positions);steps=np.rint(differences/step).astype(int)
 if np.any(steps<1) or not np.allclose(steps*step,differences,rtol=1e-10,atol=1e-12):raise ValueError('Saved times must respect the declared step')
 fields=[u];advance=model.stepper(step);start=time.perf_counter()
 with np.errstate(over='raise',invalid='raise',divide='raise'):
  for count in steps:
   for _ in range(count):v=advance(v)
   value=np.fft.ifft(v)
   if not np.isfinite(value).all():raise FloatingPointError('Nonfinite field; no inference of nonexistence')
   fields.append(value)
 return np.asarray(fields),time.perf_counter()-start


def norm(u,length,order=0):
 n=u.shape[-1];k=2*np.pi*np.fft.fftfreq(n,d=length/n)
 return np.sqrt(length*np.sum((1+k*k)**order*abs(np.fft.fft(u,axis=-1)/n)**2,axis=-1))


def errors(a,b,length):
 if a.shape[-1]!=b.shape[-1]:a=resample(a,b.shape[-1],axis=-1)
 return {f'h{s}':norm(a-b,length,s)/norm(b,length,s) for s in range(3)}


def inlets(c):
 n,L=c['n'],c['length'];s=np.arange(n)*L/n-L/2
 rng=np.random.default_rng(c['seed']);modes=abs(np.fft.fftfreq(n)*n)
 phase=.3
 params={'amplitude':.85,'width':3.,'center':-2.,'chirp':.08,'phase':phase}
 def packet(length,grid):
  x=np.arange(grid)*length/grid-length/2
  return sum(params['amplitude']*np.exp((-.5+1j*params['chirp'])*((x-params['center']+j*length)/params['width'])**2+1j*phase) for j in (-1,0,1))
 packet_u=packet(L,n)
 q=2*np.pi*5/L
 sideband=.8*(1+.05*np.cos(q*s)+.025j*np.sin(q*s))*np.exp(-.2j)
 indices=np.arange(-5,6);coeff=(rng.normal(size=len(indices))+1j*rng.normal(size=len(indices)))*np.exp(-.16*indices**2)
 random=np.exp(2j*np.pi*s[:,None]*indices/L)@coeff
 random*=.85/np.max(abs(random))
 data=[('packet',packet_u,params),('sideband',sideband,{'carrier_amplitude':.8,'modulation_mode':5,'modulation_wavenumber':q,'cosine_depth':.05,'imaginary_sine_depth':.025,'phase':-.2}),('random',random,{'seed':c['seed'],'coefficients':[[float(z.real),float(z.imag)] for z in coeff],'peak_before_projection':.85})]
 return [(family,np.fft.ifft(np.fft.fft(u)*(modes<=c['initial_modes'])),params) for family,u,params in data]


def diagnostics(u,T,length,epsilon):
 n=u.shape[-1];k=2*np.pi*np.fft.fftfreq(n,d=length/n);modes=abs(np.fft.fftfreq(n)*n)
 f=np.fft.fft(u,axis=-1)/n;power=abs(f)**2;total=power.sum(axis=-1)
 peak=abs(u).max(axis=-1);action=length*total;A0=peak[0]
 centroid=(k*power).sum(axis=-1)/total
 width=np.sqrt(((k-centroid[:,None])**2*power).sum(axis=-1)/total)
 norms={f'h{s}':norm(u,length,s) for s in range(3)}
 tail=power[:,modes>=.8*(n/4)].sum(axis=-1)/total
 phase=np.angle((u*u[0].conj()).sum(axis=-1))
 recurrence=np.linalg.norm(u*np.exp(-1j*phase[:,None])-u[0],axis=-1)/np.linalg.norm(u[0])
 boundary=abs(u[:,abs(np.arange(n)*length/n-length/2)>.45*length])**2
 boundary_fraction=boundary.sum(axis=-1)/np.sum(abs(u)**2,axis=-1)
 peaks=find_peaks(peak,prominence=.02*A0)[0]
 returns=find_peaks(-recurrence,prominence=.02)[0]
 rows=[{'T':float(t),'nonlinear_lengths':float(.5*A0*A0*t),'peak':float(peak[j]),'peak_ratio':float(peak[j]/A0),'physical_envelope_peak':float(epsilon*peak[j]),'action':float(action[j]),'action_relative_drift':float(abs(action[j]-action[0])/action[0]),'norms':{key:float(value[j]) for key,value in norms.items()},'h2_ratio':float(norms['h2'][j]/norms['h2'][0]),'spectral_centroid':float(centroid[j]),'spectral_width':float(width[j]),'scaled_bandwidth_indicator':float(epsilon*width[j]),'spectral_tail_fraction':float(tail[j]),'boundary_action_fraction':float(boundary_fraction[j]),'phase_aligned_recurrence_l2':float(recurrence[j])} for j,t in enumerate(T)]
 return {'rows':rows,'max_peak':float(peak.max()),'max_h2_ratio':float((norms['h2']/norms['h2'][0]).max()),'max_spectral_tail':float(tail.max()),'max_action_drift':float(np.max(abs(action-action[0])/action[0])),'initial_peak':float(A0),'nonlinear_length':float(2/A0**2),'focusing_peaks':[{'T':float(T[j]),'peak':float(peak[j])} for j in peaks],'recurrence_candidates':[{'T':float(T[j]),'relative_l2':float(recurrence[j])} for j in returns if recurrence[j]<.05],'definitions':{'norm':'Hs²=L sum (1+k²)^s |fft(u)/N|²','nonlinear_length':'2/max|u(0)|², because cubic coefficient is 1/2','tail':'action fraction at |mode|>=0.8*N/4','recurrence':'Global phase only, no translation alignment; sampled local minima <0.05','steepness_bandwidth':'epsilon*peak and epsilon*spectral width are formal indicators, no universal physical-validity threshold'}}


def default_config():
 return {'seed':19791030,'n':512,'length':40.,'initial_modes':12,'step':.01,'sample_step':.2,'horizons':[4.,8.,16.,32.],'epsilons':[.05,.1],'models':['literal','stability','nls','meanflow'],'tolerances':{'h0':1e-5,'h1':1e-4,'h2':1e-3,'tail':1e-8,'action':1e-6},'equation_scope':'1D longitudinal restriction of the original temporal potential-envelope formulation, not the full 2D equation or active spatial free-surface baseline','model_ids':MODELS,'sign_ambiguity':'Printed-sign c=1 and stability-consistent c=1.5 are separate model hypotheses','coordinates':'A=epsilon*u; s=epsilon*(x-t/2); T=epsilon^2*t. epsilon=0 is an NLS mathematical control, not a coordinate transformation.'}


def run(config,root):
 root=Path(root);root.mkdir(parents=True,exist_ok=True)
 identity={'config':config,'source_sha256':sha(__file__),'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__}}
 identity_file=root/'identity.json'
 if identity_file.exists() and json.loads(identity_file.read_text())!=identity:raise ValueError('Run identity changed; use a new output directory')
 write_json(identity_file,identity)
 write_json(root/'config.json',config)
 summary={'identity':identity,'limitation':LIMITATION,'cases':[],'comparisons':[]}
 all_initial=inlets(config)
 for family,u0,params in all_initial:
  for epsilon in config['epsilons']:
   outputs={}
   for kind in config['models']:
    previous_qualified=True
    for horizon in config['horizons']:
     key=f'{family}-eps{epsilon:g}-{kind}-T{horizon:g}'
     result_path=root/(key+'.json');array_path=root/(key+'.npz')
     if not previous_qualified:
      summary['cases'].append({'key':key,'family':family,'epsilon':epsilon,'model':kind,'model_id':MODELS[kind],'T':horizon,'status':'blocked_by_previous_refinement','qualified':False});continue
     if result_path.exists():
      result=json.loads(result_path.read_text())
      if result['source_sha256']!=identity['source_sha256'] or result['config_sha256']!=sha(root/'config.json'):raise ValueError('Cached result identity differs')
      if result.get('artifact'):
       if sha(array_path)!=result['artifact']['sha256']:raise ValueError('Trajectory checksum differs')
       with np.load(array_path,allow_pickle=False) as z:outputs[(kind,horizon)]=z['fields']
     else:
      T=np.arange(round(horizon/config['sample_step'])+1)*config['sample_step']
      fields={};cost={};failure=None
      for role,n,step in [('fine',config['n'],config['step']),('grid',config['n']//2,config['step']),('step',config['n'],2*config['step'])]:
       try:fields[role],cost[role]=solve(u0,config['length'],epsilon,kind,n,step,T)
       except FloatingPointError as error:failure={'configuration':role,'message':str(error)};break
      result={'key':key,'family':family,'epsilon':epsilon,'model':kind,'model_id':MODELS[kind],'T':horizon,'parameters':params,'source_sha256':identity['source_sha256'],'config_sha256':sha(root/'config.json'),'seconds':cost,'status':'nonfinite_failure' if failure else 'complete','qualified':False}
      if failure:result['failure']=failure
      else:
       diagnostics_result=diagnostics(fields['fine'],T,config['length'],epsilon)
       refinement={role:{norm:float(values.max()) for norm,values in errors(fields[role],fields['fine'],config['length']).items()} for role in ('grid','step')}
       tol=config['tolerances'];qualified=all(value<=tol[norm] for values in refinement.values() for norm,value in values.items()) and diagnostics_result['max_spectral_tail']<=tol['tail'] and diagnostics_result['max_action_drift']<=tol['action']
       np.savez_compressed(array_path,fields=fields['fine'],T=T,s=np.arange(config['n'])*config['length']/config['n']-config['length']/2,u0=u0)
       result.update(qualified=qualified,status='complete' if qualified else 'numerically_unresolved',refinement=refinement,diagnostics=diagnostics_result,artifact={'path':array_path.name,'sha256':sha(array_path)})
       outputs[(kind,horizon)]=fields['fine']
      write_json(result_path,result)
     summary['cases'].append(result);previous_qualified=result['qualified']
     print(key,result['status'],result.get('refinement'),flush=True)
   for horizon in config['horizons']:
    for label,left,right in [('literal_vs_nls','literal','nls'),('stability_vs_nls','stability','nls'),('literal_vs_stability','literal','stability'),('meanflow_vs_literal','meanflow','literal')]:
     if (left,horizon) in outputs and (right,horizon) in outputs:
      vals=errors(outputs[left,horizon],outputs[right,horizon],config['length'])
      relevant=[r for r in summary['cases'] if r['family']==family and r['epsilon']==epsilon and r['model'] in (left,right) and r['T']==horizon]
      summary['comparisons'].append({'family':family,'epsilon':epsilon,'T':horizon,'comparison':label,'reference_model':right,'both_numerically_qualified':len(relevant)==2 and all(r['qualified'] for r in relevant),'max_relative_complex_difference':{key:float(value.max()) for key,value in vals.items()},'endpoint_relative_complex_difference':{key:float(value[-1]) for key,value in vals.items()}})
   write_json(root/'summary.json',summary)
 write_json(root/'summary.json',summary)
 return summary


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--config',type=Path);parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results')
 parser.add_argument('--max-time',type=float,help='Maximum scaled time, retaining configured smaller gates')
 parser.add_argument('--families',nargs='+',choices=['packet','sideband','random'])
 args=parser.parse_args();config=json.loads(args.config.read_text()) if args.config else default_config()
 if args.max_time is not None:config['horizons']=[x for x in config['horizons'] if x<=args.max_time]
 if args.families:
  raise ValueError('Use all three input families for the declared POC comparison')
 run(config,args.output)

if __name__=='__main__':main()

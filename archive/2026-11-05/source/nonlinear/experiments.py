"""Run saved, reproducible benchmarks, branches, perturbations and collisions.
Usage: python -m nonlinear.experiments [all|benchmark|profiles|dynamics|transverse]
"""
import os,json,time,sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import numpy as np
from pathlib import Path
from scipy.signal import resample,find_peaks
from .solver import FourierModel,profile,nls_profile,evolve,translate,align_error
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/nonlinear';OUT.mkdir(exist_ok=True)
def save(name,model,snaps,rows,extra=None):
 meta=dict(name=name,shape=model.shape,lengths=model.lengths,kind=model.kind,epsilon=model.epsilon,epsilon1=model.epsilon1,epsilon2=model.epsilon2,nonlinear=model.nonlinear,diagnostics=rows)| (extra or {})
 (OUT/f'{name}.json').write_text(json.dumps(meta,indent=2));np.savez_compressed(OUT/f'{name}.npz',u=snaps,t=np.array([a['t'] for a in rows]),**{f'x{i}':a for i,a in enumerate(model.axes)})
 print('SAVED',name,rows[-1].get('complex_error'),flush=True)
 return meta
def benchmark():
 rows=[]
 for n,L,dt in [(1024,80,.04),(1024,80,.02),(1024,80,.01),(1024,80,.005),(256,80,.0025),(512,80,.0025),(1024,80,.0025),(512,40,.0025),(2048,160,.0025)]:
  m=FourierModel((n,),(L,),kind='optical');f=nls_profile(m.axes[0]);u,d=evolve(m,f,20,dt,41,f,0,1)
  name=f'nls_N{n}_L{L}_h{dt}';save(name,m,u,d,dict(dt=dt,T=20));rows.append(dict(n=n,L=L,dt=dt,final_error=d[-1]['complex_error'],aligned_error=d[-1]['aligned_error'],mass_drift=max(abs(a['mass']/d[0]['mass']-1) for a in d)))
 m=FourierModel((1024,),(80,),kind='optical',nonlinear=False);f=nls_profile(m.axes[0]);u,d=evolve(m,f,20,.02,101);save('linear_control',m,u,d,dict(dt=.02,T=20))
 (OUT/'benchmark_summary.json').write_text(json.dumps(rows,indent=2))
def profiles():
 rows=[]
 for mu,speeds in [(2.,[0.,-2.,2.]),(3.,[0.])]:
  for speed in speeds:
   old=None
   for e in [0,.025,.05,.075,.1,.125,.15]:
    m=FourierModel((2048,),(160.,),epsilon=e);f,d=profile(m,mu,speed,old);old=f
    tag=f'profile_mu{mu:g}_s{speed:g}_e{e:g}';save(tag,m,np.array([f]),[dict(t=0,**m.invariants(f))],d);rows.append(dict(name=tag,**d))
 checks=[]
 for n,L in [(512,80),(1024,80),(2048,80),(512,40),(2048,160),(4096,320)]:
  m=FourierModel((n,),(L,),epsilon=.15);f,d=profile(m,2,2);save(f'profile_check_N{n}_L{L}',m,np.array([f]),[dict(t=0,**m.invariants(f))],d);checks.append(dict(n=n,L=L,**d))
 (OUT/'profile_summary.json').write_text(json.dumps(rows,indent=2));(OUT/'profile_checks.json').write_text(json.dumps(checks,indent=2))
def collision_summary(m,u0,uend,profiles,speeds,T,offset):
 x=m.axes[0];peaks,_=find_peaks(abs(uend),distance=int(5/(m.lengths[0]/m.shape[0])))
 ids=sorted(peaks,key=lambda j:abs(uend[j]),reverse=True)[:2];centers=sorted(float(x[j]) for j in ids)
 mask=np.ones_like(x,dtype=bool)
 for c in centers:mask &= abs(x-c)>6
 initialmask=(abs(x+offset)>6)&(abs(x-offset)>6)
 tail=float(np.sum(abs(uend[mask])**2)/np.sum(abs(uend)**2));tail0=float(np.sum(abs(u0[initialmask])**2)/np.sum(abs(u0)**2))
 # Compare original incoming profiles to outgoing pulses in disjoint half-domains.
 fits=[]
 for f,s in zip(profiles,speeds):
  expected=(-offset if s>0 else offset)+s*T
  region=x>0 if expected>0 else x<0
  masked=np.where(region,uend,0);err,shift,phase=align_error(m,masked,f,expected,8.)
  fits.append(dict(speed=s,aligned_error=err,translation=shift,phase=phase,peak=float(abs(masked).max())))
 return dict(peak_centers=centers,tail_outside_two_radius6_cores=tail,initial_tail_fraction=tail0,excess_tail_fraction=tail-tail0,outgoing_fits=fits)
def dynamics():
 summary=[]
 for e in [0,.05,.1,.15]:
  for s in [0.,2.,-2.]:
   m=FourierModel((2048,),(160.,),epsilon=e);f,info=profile(m,2,s)
   u,d=evolve(m,f,20,.0025,101,f,s,-2)
   meta=save(f'prop_e{e:g}_s{s:g}',m,u,d,dict(dt=.0025,T=20,profile=info));summary.append({k:v for k,v in meta.items() if k!='diagnostics'}|{'final':d[-1]})
 for delta,mode in [(.01,'amplitude'),(-.01,'amplitude'),(.01,'phase')]:
  m=FourierModel((2048,),(160.,),epsilon=.15);f,info=profile(m,2,0);x=m.axes[0]
  u0=f*(1+delta*np.cos(.7*x)) if mode=='amplitude' else f*np.exp(1j*delta*np.exp(-x*x/16))
  u,d=evolve(m,u0,40,.0025,161,f,0,-2);save(f'perturb_{mode}_{delta:g}',m,u,d,dict(dt=.0025,T=40,delta=delta,mode=mode))
 # Integrator refinement for the most strongly asymmetric traveling wave.
 for dt in [.01,.005,.00125]:
  m=FourierModel((2048,),(160.,),epsilon=.15);f,info=profile(m,2,2);u,d=evolve(m,f,20,dt,41,f,2,-2);save(f'timestep_dysthe_{dt:g}',m,u,d,dict(dt=dt,T=20))
 # Collision controls, time-step/grid/domain changes; all use independently solved profiles.
 cases=[(0,2048,160,.0025),(0,2048,160,.00125),(.15,2048,160,.0025),(.15,2048,160,.00125),(.15,4096,160,.00125),(.15,4096,320,.00125)]
 coll=[]
 for e,n,L,dt in cases:
  m=FourierModel((n,),(L,),epsilon=e);fp,ip=profile(m,2,2);fm,im=profile(m,2,-2)
  u0=translate(m,fp,-12)+translate(m,fm,12)
  u,d=evolve(m,u0,14,dt,141)
  result=collision_summary(m,u0,u[-1],[fp,fm],[2,-2],14,12)
  name=f'collision_e{e:g}_N{n}_L{L}_h{dt:g}';save(name,m,u,d,dict(dt=dt,T=14,collision=result));coll.append(dict(name=name,epsilon=e,n=n,L=L,dt=dt,**result))
 (OUT/'propagation_summary.json').write_text(json.dumps(summary,indent=2));(OUT/'collision_summary.json').write_text(json.dumps(coll,indent=2))
def transverse():
 # Full optical equation. z-independent data also give an exact two-coordinate reduction.
 cases=[('stripe',128,16,16,.02,False),('mixed',128,16,16,.02,True),('mixed_time',128,16,16,.01,True),('mixed_grid',256,24,24,.01,True),('mixed_domain',256,32,32,.01,True)]
 summary=[]
 for name,nx,ny,nz,dt,perturb in cases:
  Lx=80. if name=='mixed_domain' else 40.;Ly=Lz=8*np.pi if name=='mixed_domain' else 4*np.pi
  m=FourierModel((nx,ny,nz),(Lx,Ly,Lz),kind='optical',epsilon1=1,epsilon2=1,workers=2)
  x,y,z=np.meshgrid(*m.axes,indexing='ij',sparse=True);base=np.sqrt(2)/np.cosh(x)*np.ones(m.shape)
  u0=base*(1+.01*np.cos(.5*y)*np.cos(.5*z)) if perturb else base.astype(complex)
  # Record slices and diagnostic rows, avoiding hundreds of full 3D snapshots on disk.
  u,d=evolve(m,u0,8,dt,33,base,0,1)
  idx=(slice(None),slice(None),ny//2,nz//2)
  meta=dict(dt=dt,T=8,perturbation='.01 cos(y/2) cos(z/2)' if perturb else 'none',full_equation=True)
  # Save full fields at five times; separate diagnostic histories retain 33 times.
  chosen=np.array([0,8,16,24,32]);save('optical3d_'+name,m,u[chosen],[d[i] for i in chosen],meta)
  np.savez_compressed(OUT/f'optical3d_{name}_slices.npz',t=[a['t'] for a in d],xy=u[:,:,:,nz//2],xz=u[:,:,ny//2,:])
  (OUT/f'optical3d_{name}_history.json').write_text(json.dumps(d,indent=2))
  summary.append(dict(name=name,shape=m.shape,lengths=m.lengths,dt=dt,final=d[-1],initial=d[0]))
 (OUT/'transverse_summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':
 start=time.time();arg=sys.argv[1] if len(sys.argv)>1 else 'all'
 for name,fn in [('benchmark',benchmark),('profiles',profiles),('dynamics',dynamics),('transverse',transverse)]:
  if arg in ['all',name]:print('RUN',name,flush=True);fn()
 print('WALL',time.time()-start,flush=True)

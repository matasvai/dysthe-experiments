"""Refinements prompted by first-batch diagnostics; retain all original runs."""
import sys
import numpy as np
from .solver import FourierModel,profile,evolve,translate
from .experiments import save,collision_summary,OUT
import json
arg=sys.argv[1] if len(sys.argv)>1 else 'all'
if arg in ['all','collision']:
 for n,dt in [(4096,.000625),(8192,.000625)]:
  m=FourierModel((n,),(160.,),epsilon=.15);fp,_=profile(m,2,2);fm,_=profile(m,2,-2);u0=translate(m,fp,-12)+translate(m,fm,12)
  u,d=evolve(m,u0,14,dt,141);c=collision_summary(m,u0,u[-1],[fp,fm],[2,-2],14,12)
  save(f'collision_refined_N{n}_h{dt:g}',m,u,d,dict(dt=dt,T=14,collision=c))
if arg in ['all','perturb']:
 for delta in [0.,.01]:
  m=FourierModel((2048,),(160.,),epsilon=.15);f,_=profile(m,2,0);u0=f*(1+delta*np.cos(.7*m.axes[0]));u,d=evolve(m,u0,40,.00125,161,f,0,-2);save(f'perturb_refined_{delta:g}',m,u,d,dict(dt=.00125,T=40,delta=delta))
if arg in ['all','transverse']:
 for name,nx,ny,nz,Lx,Ly,dt in [('fine',256,32,32,40.,4*np.pi,.01),('time',256,32,32,40.,4*np.pi,.005),('finer',256,48,48,40.,4*np.pi,.005),('length',512,32,32,80.,4*np.pi,.005)]:
  m=FourierModel((nx,ny,nz),(Lx,Ly,Ly),kind='optical',workers=2)
  x,y,z=np.meshgrid(*m.axes,indexing='ij',sparse=True);base=np.sqrt(2)/np.cosh(x)*np.ones(m.shape);u0=base*(1+.01*np.cos(.5*y)*np.cos(.5*z))
  u,d=evolve(m,u0,4,dt,17,base,0,1);chosen=[0,4,8,12,16];save('transverse_refined_'+name,m,u[chosen],[d[i] for i in chosen],dict(dt=dt,T=4))
  np.savez_compressed(OUT/f'transverse_refined_{name}_slices.npz',t=[a['t'] for a in d],xy=u[:,:,:,nz//2],xz=u[:,:,ny//2,:])
  (OUT/f'transverse_refined_{name}_history.json').write_text(json.dumps(d,indent=2))

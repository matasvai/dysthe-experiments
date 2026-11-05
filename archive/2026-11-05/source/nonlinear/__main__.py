"""Configurable nonlinear envelope simulation, with JSON metadata and NPZ snapshots."""
import argparse,json
from pathlib import Path
import numpy as np
from .solver import FourierModel,profile,nls_profile,evolve
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',choices=['dysthe','optical'],default='dysthe')
p.add_argument('--n',default='1024',help='comma-separated grid sizes')
p.add_argument('--lengths',default='160',help='comma-separated periodic lengths')
p.add_argument('--epsilon',type=float,default=.15)
p.add_argument('--epsilon1',type=float,default=1.)
p.add_argument('--epsilon2',type=float,default=1.)
p.add_argument('--mu',type=float,default=2.)
p.add_argument('--speed',type=float,default=0.)
p.add_argument('--initial',choices=['solitary','stripe','gaussian'],default='solitary')
p.add_argument('--amplitude',type=float,default=1.)
p.add_argument('--width',type=float,default=3.)
p.add_argument('--perturbation',type=float,default=0.)
p.add_argument('--linear',action='store_true')
p.add_argument('--T',type=float,default=20.)
p.add_argument('--dt',type=float,default=.0025)
p.add_argument('--samples',type=int,default=101)
p.add_argument('--out',default='results/nonlinear/custom')
a=p.parse_args();shape=tuple(map(int,a.n.split(',')));lengths=tuple(map(float,a.lengths.split(',')))
if len(shape)!=len(lengths) or len(shape) not in [1,2,3] or any(n<16 or n%4 for n in shape):p.error('matching 1–3 dimensions; grid sizes must be multiples of four, at least 16')
if min(lengths)<=0 or a.T<=0 or a.dt<=0 or a.width<=0:p.error('lengths, T, dt and width must be positive')
m=FourierModel(shape,lengths,kind=a.model,epsilon=a.epsilon,epsilon1=a.epsilon1,epsilon2=a.epsilon2,nonlinear=not a.linear)
axes=np.meshgrid(*m.axes,indexing='ij',sparse=True);info=None
if a.initial=='solitary':
 if a.model!='dysthe' or len(shape)!=1:p.error('solitary solves the one-dimensional Dysthe profile; use stripe for an optical benchmark')
 # Compute a nonlinear profile even if it will subsequently be run as a linear control.
 pm=FourierModel(shape,lengths,epsilon=a.epsilon);u,info=profile(pm,a.mu,a.speed)
elif a.initial=='stripe':
 if a.model!='optical' or a.speed!=0:p.error('stripe is the stationary optical NLS reduction; set speed=0')
 u=np.sqrt(2*a.mu)/np.cosh(np.sqrt(a.mu)*axes[0])*np.ones(shape)
else:u=a.amplitude*np.exp(-sum(x*x for x in axes)/(2*a.width*a.width))
u=np.asarray(u,dtype=complex)
if a.perturbation:
 modulation=np.ones(shape)
 for j in (range(1,len(shape)) if len(shape)>1 else range(1)):
  modulation*=np.cos(2*np.pi*axes[j]/lengths[j])
 u*=1+a.perturbation*modulation
snaps,rows=evolve(m,u,a.T,a.dt,a.samples)
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
np.savez_compressed(out/'snapshots.npz',u=snaps,t=[d['t'] for d in rows],**{f'x{i}':x for i,x in enumerate(m.axes)})
(out/'run.json').write_text(json.dumps(dict(config=vars(a),profile=info,diagnostics=rows),indent=2))
print('Saved',out,'final peak',rows[-1]['peak'])

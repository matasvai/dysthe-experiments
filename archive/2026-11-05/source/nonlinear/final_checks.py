"""Cross-check finest practical collision domain and longitudinal optical grid."""
import numpy as np,json
from .solver import FourierModel,profile,translate,evolve
from .experiments import save,collision_summary,OUT
m=FourierModel((8192,),(320.,),epsilon=.15);fp,_=profile(m,2,2);fm,_=profile(m,2,-2);u0=translate(m,fp,-12)+translate(m,fm,12)
u,d=evolve(m,u0,14,.000625,141);c=collision_summary(m,u0,u[-1],[fp,fm],[2,-2],14,12)
save('collision_refined_domain',m,u,d,dict(dt=.000625,T=14,collision=c))
m=FourierModel((512,32,32),(40.,4*np.pi,4*np.pi),kind='optical',workers=2);x,y,z=np.meshgrid(*m.axes,indexing='ij',sparse=True);base=np.sqrt(2)/np.cosh(x)*np.ones(m.shape);u0=base*(1+.01*np.cos(.5*y)*np.cos(.5*z))
u,d=evolve(m,u0,4,.005,17,base,0,1);idx=[0,4,8,12,16];save('transverse_refined_xfine',m,u[idx],[d[i] for i in idx],dict(dt=.005,T=4))
np.savez_compressed(OUT/'transverse_refined_xfine_slices.npz',t=[a['t'] for a in d],xy=u[:,:,:,16],xz=u[:,:,16,:]);(OUT/'transverse_refined_xfine_history.json').write_text(json.dumps(d,indent=2))

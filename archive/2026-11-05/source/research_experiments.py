"""Research-motivated experiments. All peaks are numerical candidates, not certified suprema."""
import os,json,time,warnings
os.environ.setdefault('MPLCONFIGDIR','/tmp/dysthe-mpl')
from pathlib import Path
import numpy as np
import torch
from scipy.integrate import quad,IntegrationWarning
from scipy.special import roots_legendre
from functools import lru_cache
from scipy.optimize import minimize
from decay_dataset import FAMILIES,input_field,setup,weighted
from long_time_decay import scan,continuous_sum,params
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/research_experiments';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4);torch.set_default_dtype(torch.float64)
def save(name,obj):
 (OUT/(name+'.json')).write_text(json.dumps(obj,indent=2))
def scaling():
 rows=[]
 for family in FAMILIES:
  x,f,fh,w,K=setup(family,-1,0.,96,144)
  ref=weighted(fh,w,K,0.,8,.5)
  for lam in [.5,1.,2.]:
   xx=x/lam;h=1.5/lam
   ff=lam**3*input_field(family,lam*xx,0.);ff=ff/(ff.abs().sum()*h**3)
   k=2*np.pi*torch.fft.fftfreq(96,d=h);kk=k[:,None,None];pp=k[None,:,None]**2+k[None,None,:]**2
   ww=kk**3-3*kk*pp;t=8/lam**3
   val=weighted(torch.fft.fftn(ff),ww,kk,0.,t,.5)
   rows.append(dict(family=family,scale=lam,t=t,length=144/lam,spacing=h,
    relative_field_error=float(torch.linalg.vector_norm(val/lam**3.5-ref)/torch.linalg.vector_norm(ref)),
    ratio=float(t**(7/6)*val.abs().max()),reference_ratio=float(8**(7/6)*ref.abs().max())))
 save('scaling',rows);print('Scaling complete',flush=True)
def cases():
 cc=[]
 for reg,s,b in [('homogeneous',-1,0.),('inhomogeneous',1,1.)]:
  for sx in [3.,6.]:
   for delta in [-.75,-.25,0.,.25,.75]:
    cc.append(dict(name=f'{reg}_sx{sx:g}_offset{delta:g}',regime=reg,sign=s,beta=b,sx=sx,sy=3.,qx=-b+delta,qy=0.,kind='gaussian',group='axial sweep',offset=delta))
  cc.append(dict(name=reg+'_cancel',regime=reg,sign=s,beta=b,sx=3.,sy=3.,qx=0.,qy=0.,kind='cancel',group='zero integral',offset=0.))
 cc.append(dict(name='inhomogeneous_unshifted_control',regime='inhomogeneous',sign=1,beta=1.,sx=3.,sy=3.,qx=0.,qy=0.,kind='gaussian',group='control',offset=1.))
 # On and either side of the second curvature-degenerate set, qx(qx+1)=qy².
 for d in [-.2,0.,.2]:
  cc.append(dict(name=f'inhomogeneous_cone{d:g}',regime='inhomogeneous',sign=1,beta=1.,sx=6.,sy=6.,qx=.5,qy=np.sqrt(.75)+d,kind='gaussian',group='curvature surface',offset=d))
 return cc
def solve_setup(c,n,L):
 x=torch.arange(n)*(L/n)-L/2;X,Y,Z=x[:,None,None],x[None,:,None],x[None,None,:]
 f=torch.exp(-X**2/(2*c['sx']**2)-(Y**2+Z**2)/(2*c['sy']**2))*torch.exp(1j*(c['qx']*X+c['qy']*Y))
 if c['kind']=='cancel':f=f*X/c['sx']
 norm=f.abs().sum()*(L/n)**3;f=f/norm
 k=2*np.pi*torch.fft.fftfreq(n,d=L/n);K=k[:,None,None];P=k[None,:,None]**2+k[None,None,:]**2
 w=K**3+3*c['sign']*K*P+3*c['beta']*P
 return x,torch.fft.fftn(f),w,K,float(abs(f.sum()*(L/n)**3))
def sweep():
 cc=cases();save('sweep_config',dict(cases=cc,times=[4,8,16],levels=[['small',96,144],['large',128,192],['fine',192,192]],screen=.02,normalization='discrete L1=1 on each mesh; same continuum recipe',scope='separate grid/domain screens; not joint convergence or certified maxima'))
 rows=[]
 for c in cc:
  path=OUT/(c['name']+'.json')
  if path.exists():rows.extend(json.loads(path.read_text()));continue
  results={}
  for label,n,L in [('small',96,144),('large',128,192),('fine',192,192)]:
   x,fh,w,K,mean=solve_setup(c,n,L)
   records=[]
   for t in [4,8,16]:
    a=weighted(fh,w,K,c['beta'],t,.5).abs();idx=np.unravel_index(int(a.argmax()),a.shape)
    peak=float(a.max());edge=max(float(a[0,:,:].max()),float(a[-1,:,:].max()),float(a[:,0,:].max()),float(a[:,-1,:].max()),float(a[:,:,0].max()),float(a[:,:,-1].max()))
    records.append(dict(t=t,peak=peak,ratio=t**(7/6)*peak,location=[float(x[i]) for i in idx],boundary_to_peak=edge/peak,absolute_integral=mean))
   results[label]=records
   del x,fh,w,K,a
  rr=[]
  for i,t in enumerate([4,8,16]):
   dom=abs(results['small'][i]['peak']-results['large'][i]['peak'])/results['large'][i]['peak']
   grd=abs(results['large'][i]['peak']-results['fine'][i]['peak'])/results['fine'][i]['peak']
   rr.append(dict(**c,**results['fine'][i],domain_change=dom,grid_change=grd,passes_screen=dom<.02 and grd<.02,levels={k:v[i] for k,v in results.items()}))
  save(c['name'],rr);rows.extend(rr);print('Sweep',c['name'],flush=True)
 save('sweep',rows)
@lru_cache(None)
def gauss_nodes(n):return roots_legendre(n)
def integrand(k,point,t,s,b,sx,sy,q):
 x,r=point;d=sy*sy-6j*t*(s*k+b)
 return np.exp(-sx*sx*(k-q)**2/2+1j*(t*k**3+k*x)-r*r/(2*d))*np.abs(k+b)**.5/(4*np.pi**2*d)
def gaussian_quad(point,t,s,b,sx,sy,q,order=256,tail=12,kind="gaussian"):
 cuts=np.unique(np.r_[np.linspace(q-tail/sx,q+tail/sx,65),[-b] if q-tail/sx<-b<q+tail/sx else []])
 lo=np.minimum(np.sqrt(abs(cuts[:-1]+b)),np.sqrt(abs(cuts[1:]+b)))[:,None]
 hi=np.maximum(np.sqrt(abs(cuts[:-1]+b)),np.sqrt(abs(cuts[1:]+b)))[:,None]
 side=np.where((cuts[:-1]+cuts[1:])/2<-b,-1,1)[:,None];z,wt=gauss_nodes(order);u=(hi+lo)/2+(hi-lo)*z/2
 kk=-b+side*u*u
 factor=(-1j*np.sqrt(np.pi/2)*sx*kk) if kind=="cancel" else 1
 return np.sum(integrand(kk,point,t,s,b,sx,sy,q)*factor*2*u*wt*(hi-lo)/2)
def phase_partition_quad(point,t,s,b,sx,sy,q,tail=12,tolerance=2e-13):
 # Separate adaptive Gauss-Kronrod implementation: principal-phase paneling, then cusp substitution.
 # Stationary points of tk^3+xk are inserted explicitly; residual complex amplitude is integrated adaptively.
 x,r=point;a=q-tail/sx;end=q+tail/sx;cuts=[a]
 while a<end:
  # Upper bound on phase derivative throughout a trial step keeps oscillation per panel small.
  step=min(.2,np.pi/(2*(3*t*(abs(a)+.2)**2+abs(x)+1)))
  a=min(end,a+step);cuts.append(a)
 stationary=[-np.sqrt(-x/(3*t)),np.sqrt(-x/(3*t))] if x<0 else []
 cuts=np.unique(np.r_[cuts,[v for v in [-b]+stationary if cuts[0]<v<end]])
 value=0j;estimated=0.;caught=[]
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always',IntegrationWarning)
  for a,z in zip(cuts[:-1],cuts[1:]):
   side=-1 if (a+z)/2<-b else 1
   u0,u1=sorted([np.sqrt(abs(a+b)),np.sqrt(abs(z+b))])
   def fn(u):return integrand(-b+side*u*u,point,t,s,b,sx,sy,q)*2*u
   re,e1=quad(lambda u:float(fn(u).real),u0,u1,epsabs=tolerance/(len(cuts)-1),epsrel=1e-10,limit=100)
   im,e2=quad(lambda u:float(fn(u).imag),u0,u1,epsabs=tolerance/(len(cuts)-1),epsrel=1e-10,limit=100)
   value+=re+1j*im;estimated+=e1+e2
 return value,dict(error_estimate=estimated,panels=len(cuts)-1,warnings=[str(w.message) for w in caught])
def quadrature():
 rr=[]
 for c in json.loads((ROOT/'results/spatial_reliability/summary.json').read_text()):
  s,b=(-1,0.) if c['regime']=='homogeneous_minus' else (1,1.);sx,sy,q=params(c['family'],b)
  point=[c['continuum_candidate']['x'],c['continuum_candidate']['r']]
  for t in [256,512,1024]:
   # t>256 keeps the same coordinate: pointwise method stress test, not a spatial maximum.
   g=gaussian_quad(point,t,s,b,sx,sy,q,512);adaptive,info=phase_partition_quad(point,t,s,b,sx,sy,q)
   wider=gaussian_quad(point,t,s,b,sx,sy,q,512,14);lower=gaussian_quad(point,t,s,b,sx,sy,q,256)
   rr.append(dict(regime=c['regime'],family=c['family'],t=t,point=point,amplitude=float(abs(g)),relative_disagreement=float(abs(g-adaptive)/abs(g)),order_change=float(abs(g-lower)/abs(g)),truncation_change=float(abs(g-wider)/abs(wider)),**info))
   save('quadrature',rr);print('Quadrature',c['family'],c['regime'],t,flush=True)
def identity():
 rows=[]
 for regime,s,b in [('homogeneous',-1,0.),('inhomogeneous',1,1.)]:
  for width in [.75,1.,1.5,2.,3.]:
   checkpoint=OUT/f'identity_{regime}_{width}.json'
   if checkpoint.exists():rows.append(json.loads(checkpoint.read_text()));continue
   t=1.;v1=scan(t,s,b,width,width,0.,2048,8192,128,1)
   v2=scan(t,s,b,width,width,0.,4096,16384,256,1)
   v3=scan(t,s,b,width,width,0.,4096,16384,256,.5)
   loc=[v3['x_peak'],v3['r_peak']]
   fit=minimize(lambda p:-abs(gaussian_quad(p,t,s,b,width,width,0.))/v3['peak'],loc,method='Nelder-Mead',bounds=[(loc[0]-1,loc[0]+1),(max(0,loc[1]-1),loc[1]+1)],options={'xatol':1e-6,'fatol':1e-10,'maxiter':150})
   peak=abs(gaussian_quad(fit.x,t,s,b,width,width,0.,512));spectral=continuous_sum(fit.x,t,s,b,width,width,0.,4096,16384)
   rec=dict(regime=regime,width=width,t=1.,ratio=float(peak),location=fit.x.tolist(),optimizer_success=bool(fit.success),domain_change=abs(v1['peak']-v2['peak'])/v2['peak'],radial_change=abs(v2['peak']-v3['peak'])/v3['peak'],spectral_integral_discrepancy=float(abs(spectral-gaussian_quad(fit.x,t,s,b,width,width,0.,512))/peak),levels=[v1,v2,v3])
   rec['passes_screen']=rec['optimizer_success'] and max(rec[k] for k in ['domain_change','radial_change','spectral_integral_discrepancy'])<.02
   save(f'identity_{regime}_{width}',rec);rows.append(rec);save('identity',rows);print('Identity',regime,width,flush=True)
 save('identity',rows)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('experiment',choices=['scaling','sweep','quadrature','identity']);args=p.parse_args()
 globals()[args.experiment]()

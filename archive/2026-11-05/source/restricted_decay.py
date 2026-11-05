"""Cancellation pilot for the homogeneous 3-coordinate linear symbol.
Exact continuum L1 normalization; analytic transverse Gaussian integration.
Run from project root: python restricted_decay.py
"""
from pathlib import Path
import json, os, time
os.environ.setdefault('MPLCONFIGDIR','/tmp/dysthe-mpl')
import numpy as np
from scipy.fft import fftfreq,ifft
from scipy.integrate import quad
from scipy.optimize import minimize
OUT=Path(__file__).resolve().parent/'results/restricted_decay';OUT.mkdir(parents=True,exist_ok=True)
C=[1.,np.sqrt(2/np.pi),4*np.exp(-.5)/np.sqrt(2*np.pi)]
TIMES=[16,32,64,128,256]
LEVELS=[('base',8192,.5,512,4),('length',16384,.5,512,4),('fine',16384,.25,1024,2)]
def factor(k,s,m):return (-1j*s*k)**m/C[m]
def spectrum(k,s,m):
 if m<3:return np.exp(-s*s*k*k/2)*factor(k,s,m)
 q=.75 if m==3 else 0.;z=(k-q)/.25;out=np.zeros_like(k,dtype=float);mask=abs(z)<1
 out[mask]=np.exp(1-1/(1-z[mask]**2))
 return out/BUMP_L1
# L1 norm is invariant under shifting the spectrum (modulation in x).
def bump_norm(L,dx):
 k=2*np.pi*fftfreq(round(L/dx),dx);z=k/.25;f=np.zeros_like(k);mask=abs(z)<1;f[mask]=np.exp(1-1/(1-z[mask]**2))
 return float(np.abs(ifft(f)/dx).sum()*dx)
BUMP_L1=bump_norm(2048,.03125)
def scan(t,s,m,L,dx,rmax,dr):
 n=round(L/dx);k=2*np.pi*fftfreq(n,dx);d=9+6j*t*k
 coeff=np.exp(1j*t*k**3)*np.abs(k)**.5*spectrum(k,s,m)/(2*np.pi*d)
 best=(-1,0,0);edge=0.;seeds=[]
 for rr in np.array_split(np.arange(0,rmax+dr/2,dr),max(1,int((rmax/dr+1)/8))):
  vals=np.abs(ifft(coeff[None,:]*np.exp(-rr[:,None]**2/(2*d)),axis=1,workers=4)*(n/L))
  for i,r in enumerate(rr):
   j=int(vals[i].argmax());v=float(vals[i,j]);x=(j if j<n//2 else j-n)*dx
   seeds.append((v,x,float(r)))
   if v>best[0]:best=(v,x,float(r))
  edge=max(edge,float(vals[:,n//2].max()))
  if rr[-1]==rmax:edge=max(edge,float(vals[-1].max()))
 return dict(peak=best[0],x=best[1],r=best[2],edge_fraction=edge/best[0],seeds=sorted(seeds,reverse=True)[:4])
def value(point,t,s,m,L=16384,dx=.25):
 k=2*np.pi*fftfreq(round(L/dx),dx);d=9+6j*t*k;x,r=point
 return np.sum(np.exp(1j*(t*k**3+k*x)-r*r/(2*d))*np.abs(k)**.5*spectrum(k,s,m)/(2*np.pi*d))/L

def independent(point,t,s,m):
 x,r=point
 def f(k):
  d=9+6j*t*k
  return np.exp(1j*(t*k**3+k*x)-r*r/(2*d))*abs(k)**.5*spectrum(np.asarray(k),s,m)/(4*np.pi**2*d)
 re=e1=im=e2=0.
 cuts=[-12/s,0,12/s] if m<3 else ([.5,1.] if m==3 else [-.25,0,.25])
 for a,b in zip(cuts[:-1],cuts[1:]):
  v,e=quad(lambda k:f(k).real,a,b,epsabs=1e-13,epsrel=1e-9,limit=3000);re+=v;e1+=e
  v,e=quad(lambda k:f(k).imag,a,b,epsabs=1e-13,epsrel=1e-9,limit=3000);im+=v;e2+=e
 return complex(re,im),e1+e2

def main():
 start=time.time();rows=[]
 for s in [3.,6.]:
  for m in (range(5) if s==3 else range(3)):
   for t in TIMES:
    path=OUT/f's{s:g}_m{m}_t{t}.json'
    if path.exists():rows.append(json.loads(path.read_text()));continue
    levels={name:scan(t,s,m,L,dx,R,dr) for name,L,dx,R,dr in LEVELS}
    fine=levels['fine'];fits=[]
    for _,x,r in fine['seeds']:
     fit=minimize(lambda p:-abs(value(p,t,s,m))/fine['peak'],[x,r],method='Nelder-Mead',bounds=[(x-3,x+3),(max(0,r-5),r+5)],options={'xatol':1e-5,'fatol':1e-9,'maxiter':180})
     fits.append((abs(value(fit.x,t,s,m)),fit))
    pk,fit=max(fits,key=lambda z:z[0]);v=value(fit.x,t,s,m);ref,err=independent(fit.x,t,s,m)
    wider=value(fit.x,t,s,m,32768)
    checks=dict(length_change=abs(levels['base']['peak']/levels['length']['peak']-1),grid_and_radial_change=abs(levels['length']['peak']/fine['peak']-1),pointwise_period_change=float(abs(v-wider)/abs(wider)),independent_field_discrepancy=float(abs(wider-ref)/abs(ref)),quadrature_error_estimate=float(err),edge_fraction=fine['edge_fraction'])
    rec=dict(sx=s,sy=3,m=m,t=t,peak=float(abs(ref)),ratio=float(t**(7/6)*abs(ref)),x=float(fit.x[0]),r=float(fit.x[1]),optimizer_success=bool(fit.success),checks=checks,levels=levels)
    rec['passes_1pct_screen']=bool(fit.success and max(checks[k] for k in ['length_change','grid_and_radial_change','pointwise_period_change','independent_field_discrepancy','edge_fraction'])<.01)
    path.write_text(json.dumps(rec,indent=2));rows.append(rec)
    print(s,m,t,'R',rec['ratio'],'pass',rec['passes_1pct_screen'],flush=True)
 (OUT/'summary.json').write_text(json.dumps(rows,indent=2))
 (OUT/'config.json').write_text(json.dumps(dict(times=TIMES,levels=LEVELS,C=C,widths=[3,6],orders=[0,1,2],bump_cases={3:"gap support kx in (0.5,1)",4:"control support kx in (-0.25,0.25)"},bump_L1=BUMP_L1,bump_norm_relative_change=abs(BUMP_L1/bump_norm(1024,.0625)-1),symbol='kx^3-3*kx*(ky^2+kz^2)',alpha=.5,normalization='exact continuum L1=1',wall_seconds=time.time()-start),indent=2))
if __name__=='__main__':main()

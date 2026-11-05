"""New whole-domain candidate searches at 512 and 1024; preserves earlier batches."""
import os,json,time
os.environ.setdefault('MPLCONFIGDIR','/tmp/dysthe-mpl')
from pathlib import Path
import numpy as np
from scipy.fft import fftfreq,ifft
from scipy.optimize import minimize
from long_time_decay import params,REGIMES,FAMILIES,continuous_sum
from research_experiments import gaussian_quad,phase_partition_quad
OUT=Path(__file__).resolve().parent/'results/extended_decay';OUT.mkdir(exist_ok=True)
def scan(t,s,b,sx,sy,q,L,dx,rlo,rhi,dr):
 n=int(L/dx);k=2*np.pi*fftfreq(n,d=dx);d=sy*sy-6j*t*(s*k+b)
 coeff=np.exp(-sx*sx*(k-q)**2/2+1j*t*k**3)*abs(k+b)**.5/(2*np.pi*d)
 radii=np.arange(rlo,rhi+dr*.1,dr);seeds=[];edge=0.
 for off in range(0,len(radii),8):
  rr=radii[off:off+8];a=abs(ifft(coeff[None,:]*np.exp(-rr[:,None]**2/(2*d)),axis=1,workers=4)/dx)
  for i in range(len(rr)):
   j=int(a[i].argmax());seeds.append(dict(peak=float(a[i,j]),x=float((j if j<n//2 else j-n)*dx),r=float(rr[i])))
  edge=max(edge,float(a[:,n//2].max()))
  if off+len(rr)==len(radii):edge=max(edge,float(a[-1].max()))
 seeds=sorted(seeds,key=lambda a:a['peak'],reverse=True)[:12]
 return dict(L=L,dx=dx,rlo=rlo,rhi=rhi,dr=dr,best=seeds[0],seeds=seeds,edge=edge)
def main():
 start=time.time();rows=[]
 for t in [512,1024]:
  for name,s,b in REGIMES:
   for family in FAMILIES:
    sx,sy,q=params(family,b);p=OUT/f'{name}_{family}_{t}.json'
    st=json.loads(p.read_text()) if p.exists() else dict(regime=name,family=family,t=t,scans={})
    def save():p.write_text(json.dumps(st,indent=2))
    if st.get('complete'):rows.append(st);continue
    for label,L,dx,rlo,rhi,dr in [('base',32768,.5,0,4096,4),('length',65536,.5,0,4096,4),('annulus',65536,.5,4098,8192,2),('grid',65536,.25,0,4096,2)]:
     if label not in st['scans']:
      st['scans'][label]=scan(t,s,b,sx,sy,q,L,dx,rlo,rhi,dr);save()
     print(name,family,t,label,st['scans'][label]['best'],flush=True)
    candidates=[]
    for seed in st['scans']['grid']['seeds'][:8]+st['scans']['annulus']['seeds'][:2]:
     if any(np.hypot(seed['x']-a['x'],seed['r']-a['r'])<1 for a in candidates):continue
     fit=minimize(lambda z:-abs(gaussian_quad(z,t,s,b,sx,sy,q,512))/seed['peak'],[seed['x'],seed['r']],method='Nelder-Mead',bounds=[(seed['x']-4,seed['x']+4),(max(0,seed['r']-4),seed['r']+4)],options={'xatol':1e-5,'fatol':1e-10,'maxiter':240})
     candidates.append(dict(x=float(fit.x[0]),r=float(fit.x[1]),peak=float(abs(gaussian_quad(fit.x,t,s,b,sx,sy,q,512))),success=bool(fit.success)))
    best=max(candidates,key=lambda a:a['peak']);point=[best['x'],best['r']]
    value=gaussian_quad(point,t,s,b,sx,sy,q,1024);lower=gaussian_quad(point,t,s,b,sx,sy,q,512)
    adaptive,info=phase_partition_quad(point,t,s,b,sx,sy,q)
    wider=gaussian_quad(point,t,s,b,sx,sy,q,1024,14)
    spectra=[continuous_sum(point,t,s,b,sx,sy,q,L,int(L/.25)) for L in [32768,65536,131072]]
    a=st['scans'];peak=abs(value)
    checks=dict(length_change=abs(a['base']['best']['peak']-a['length']['best']['peak'])/a['length']['best']['peak'],grid_change=abs(a['grid']['best']['peak']-a['length']['best']['peak'])/a['grid']['best']['peak'],annulus_peak_fraction=a['annulus']['best']['peak']/peak,order_change=float(abs(value-lower)/peak),adaptive_disagreement=float(abs(value-adaptive)/peak),frequency_tail_change=float(abs(value-wider)/peak),pointwise_period_change=float(abs(spectra[-1]-spectra[-2])/peak),spectral_integral_disagreement=float(abs(spectra[-1]-value)/peak),outer_boundary_fraction=max(a['grid']['edge'],a['annulus']['edge'])/peak)
    st.update(candidates=candidates,peak=float(peak),ratio=float(t**(7/6)*peak),x=best['x'],r=best['r'],checks=checks,adaptive=info,optimizer_success=best['success'],complete=True)
    st['passes_screen']=all(checks[k]<.001 for k in ['length_change','grid_change','order_change','adaptive_disagreement','frequency_tail_change','pointwise_period_change','spectral_integral_disagreement']) and checks['annulus_peak_fraction']<1 and not info['warnings'] and best['success']
    save();rows.append(st);(OUT/'summary.json').write_text(json.dumps(rows,indent=2));print('RESULT',name,family,t,st['ratio'],'PASS',st['passes_screen'],flush=True)
 (OUT/'config.json').write_text(json.dumps(dict(times=[512,1024],alpha=.5,scope='four Gaussian-based cases; sampled whole-domain searches, not certified continuum suprema',screen=.001,radial_extent=8192,full_scan_max_L=65536,pointwise_period_max_L=131072,wall_seconds=time.time()-start),indent=2))
if __name__=='__main__':main()

"""Follow up flagged axisymmetric Gaussian cases with the whole-space transverse reduction."""
import json
import numpy as np
from scipy.optimize import minimize
from research_experiments import OUT,scan,gaussian_quad,save
from scipy.fft import ifft,fftfreq

def cancel_scan(t,sign,beta,sx,sy,q,length,n,rmax,dr):
 assert q==0
 k=2*np.pi*fftfreq(n,d=length/n);d=sy*sy-6j*t*(sign*k+beta)
 coeff=np.exp(-sx*sx*k*k/2+1j*t*k**3)*abs(k+beta)**.5/(2*np.pi*d)*(-1j*np.sqrt(np.pi/2)*sx*k)
 radii=np.arange(0,rmax+dr/2,dr);best=(-1,0,0);edge=0.
 for start in range(0,len(radii),16):
  rr=radii[start:start+16];mag=abs(ifft(coeff[None,:]*np.exp(-rr[:,None]**2/(2*d)),axis=1,workers=4)*n/length)
  i,j=np.unravel_index(mag.argmax(),mag.shape)
  if mag[i,j]>best[0]:best=(float(mag[i,j]),float((j if j<n//2 else j-n)*length/n),float(rr[i]))
  edge=max(edge,float(mag[:,n//2].max()))
  if start+len(rr)==len(radii):edge=max(edge,float(mag[-1].max()))
 return dict(peak=best[0],x_peak=best[1],r_peak=best[2],edge_to_peak=edge/best[0])

rows=json.loads((OUT/'sweep.json').read_text());out=[]
for c in rows:
 if c['passes_screen']:continue
 assert c['kind'] in ['gaussian','cancel'] and c['qy']==0
 scanner=cancel_scan if c['kind']=='cancel' else scan
 key=f"followup_{c['name']}_{c['t']}";path=OUT/(key+'.json')
 if path.exists():out.append(json.loads(path.read_text()));continue
 args=[c['t'],c['sign'],c['beta'],c['sx'],c['sy'],c['qx']]
 a=scanner(*args,8192,32768,256,1)
 b=scanner(*args,16384,65536,512,.5)
 start=[b['x_peak'],b['r_peak']]
 def val(p,n=256):return gaussian_quad(p,*args,n,kind=c["kind"])
 opt=minimize(lambda p:-abs(val(p))/b['peak'],start,method='Nelder-Mead',bounds=[(start[0]-1,start[0]+1),(max(0,start[1]-1),start[1]+1)],options={'xatol':1e-6,'fatol':1e-10,'maxiter':150})
 v=val(opt.x,512);peak=abs(v)
 d=dict(name=c['name'],t=c['t'],original_ratio=c['ratio'],refined_ratio=float(c['t']**(7/6)*peak),location=opt.x.tolist(),optimizer_success=bool(opt.success),domain_and_radial_change=abs(a['peak']-b['peak'])/b['peak'],original_to_refined=abs(c['peak']-peak)/peak,sampled_to_integral=abs(b['peak']-peak)/peak,quadrature_order_change=float(abs(v-val(opt.x))/peak),boundary_to_peak=b['edge_to_peak'],levels=[a,b])
 d['passes_followup']=bool(opt.success and max(d[k] for k in ['domain_and_radial_change','sampled_to_integral','quadrature_order_change'])<.001)
 save(key,d);out.append(d);save('followup',out);print(c['name'],c['t'],d['passes_followup'],flush=True)
save('followup',out)

"""One-variable-at-a-time spatial reliability audit at t=256, alpha=1/2."""
import os,json,csv,time
os.environ.setdefault('MPLCONFIGDIR','/tmp/dysthe-mpl')
from pathlib import Path
import numpy as np
from scipy.fft import fftfreq,ifft
from scipy.special import roots_legendre
from functools import lru_cache
from scipy.optimize import minimize
from long_time_decay import REGIMES,FAMILIES,params,continuous_sum

OUT=Path(__file__).resolve().parent/'results/spatial_reliability'
OUT.mkdir(parents=True,exist_ok=True)

def search(sign,beta,sx,sy,q,length,rstart,rmax):
    n=4*length;k=2*np.pi*fftfreq(n,d=.25);d=sy*sy-1536j*(sign*k+beta)
    coeff=np.exp(-sx*sx*(k-q)**2/2+256j*k**3)*np.abs(k+beta)**.5/(2*np.pi*d)
    radii=np.arange(rstart,rmax+1,2.);best=(-1,0,0);edge=0.
    for start in range(0,len(radii),8):
        rr=radii[start:start+8]
        a=np.abs(ifft(coeff[None,:]*np.exp(-rr[:,None]**2/(2*d)),axis=1,workers=4)*4)
        i,j=np.unravel_index(a.argmax(),a.shape)
        if a[i,j]>best[0]:best=(float(a[i,j]),float((j if j<n//2 else j-n)*.25),float(rr[i]))
        edge=max(edge,float(a[:,n//2].max()))
        if start+len(rr)==len(radii):edge=max(edge,float(a[-1].max()))
    return dict(peak=best[0],x=best[1],r=best[2],boundary_amplitude=edge)

@lru_cache(None)
def rule(order):return roots_legendre(order)

def integral(point,sign,beta,sx,sy,q,tol):
    x,r=point
    def f(k):
        d=sy*sy-1536j*(sign*k+beta)
        return np.exp(-sx*sx*(k-q)**2/2+1j*(256*k**3+k*x)-r*r/(2*d))*abs(k+beta)**.5/(4*np.pi**2*d)
    cuts=np.array(sorted(set(np.linspace(q-12/sx,q+12/sx,65).tolist()+([-beta] if q-12/sx<-beta<q+12/sx else []))))
    side=np.where((cuts[:-1]+cuts[1:])/2 < -beta,-1,1)[:,None]
    ends=np.sqrt(abs(cuts+beta));lo=np.minimum(ends[:-1],ends[1:])[:,None];hi=np.maximum(ends[:-1],ends[1:])[:,None]
    def evaluate(order):
        nodes,weights=rule(order);u=(hi+lo)/2+(hi-lo)*nodes/2
        return np.sum(f(-beta+side*u*u)*2*u*weights*(hi-lo)/2)
    order=256 if tol>=1e-11 else 512
    value=evaluate(order)
    return value,float(abs(value-evaluate(order//2)))

def main():
    begin=time.perf_counter();results=[]
    old=list(csv.DictReader(open(OUT.parent/'long_time/domain_scans.csv')))
    for regime,sign,beta in REGIMES:
        for family in FAMILIES:
            sx,sy,q=params(family,beta)
            oldrow=next(r for r in old if r['regime']==regime and r['family']==family and r['level']=='extra_domain')
            base=dict(peak=float(oldrow['peak']),x=float(oldrow['x_peak']),r=float(oldrow['r_peak']))
            path=OUT/f'{regime}_{family}.json'
            # Checkpoints permit resuming without repeating expensive domain scans.
            state=json.loads(path.read_text()) if path.exists() else dict(regime=regime,family=family,base=base)
            for name,L,rstart,rmax in [('longitudinal',65536,0,2048),('radial_annulus',65536,2050,4096),('second_longitudinal',131072,0,4096)]:
                if name not in state:
                    state[name]=search(sign,beta,sx,sy,q,L,rstart,rmax)
                    path.write_text(json.dumps(state,indent=2))
                print(regime,family,name,flush=True)
            middle=max([state['longitudinal'],state['radial_annulus']],key=lambda a:a['peak'])
            last=state['second_longitudinal']
            # Refine the two-dimensional peak using an independent, non-periodic integral.
            fit=minimize(lambda p:-abs(integral(p,sign,beta,sx,sy,q,1e-11)[0])/last['peak'],
                [last['x'],last['r']],method='Nelder-Mead',
                bounds=[(last['x']-2,last['x']+2),(max(0,last['r']-4),last['r']+4)],
                options={'xatol':1e-5,'fatol':1e-10,'maxiter':180})
            v,e=integral(fit.x,sign,beta,sx,sy,q,1e-11)
            fine,ef=integral(fit.x,sign,beta,sx,sy,q,1e-13)
            spectral=continuous_sum(fit.x,256,sign,beta,sx,sy,q,131072,524288)
            polished=[]
            for length in [32768,65536,131072]:
                ff=minimize(lambda p:-abs(continuous_sum(p,256,sign,beta,sx,sy,q,length,4*length))/last['peak'],
                    fit.x,method='Nelder-Mead',bounds=[(fit.x[0]-.5,fit.x[0]+.5),(max(0,fit.x[1]-.5),fit.x[1]+.5)],
                    options={'xatol':1e-5,'fatol':1e-10,'maxiter':120})
                polished.append(dict(length=length,peak=float(abs(continuous_sum(ff.x,256,sign,beta,sx,sy,q,length,4*length))),
                    x=float(ff.x[0]),r=float(ff.x[1]),success=bool(ff.success)))
            checks=dict(first_longitudinal_change=abs(base['peak']-state['longitudinal']['peak'])/state['longitudinal']['peak'],
                transverse_expansion_change=abs(middle['peak']-state['longitudinal']['peak'])/middle['peak'],
                second_longitudinal_change=abs(last['peak']-middle['peak'])/last['peak'],
                local_refinement_change=abs(abs(fine)-last['peak'])/abs(fine),
                quadrature_tightening_change=float(abs(v-fine)/abs(fine)),
                final_spectral_integral_discrepancy=float(abs(spectral-fine)/abs(fine)),
                final_polished_peak_drift=abs(polished[-1]['peak']-polished[-2]['peak'])/polished[-1]['peak'],
                final_polished_location_drift=float(np.hypot(polished[-1]['x']-polished[-2]['x'],polished[-1]['r']-polished[-2]['r'])),
                boundary_to_peak=last['boundary_amplitude']/abs(fine))
            state.update(checks=checks,polished=polished,continuum_candidate=dict(peak=float(abs(fine)),ratio=float(256**(7/6)*abs(fine)),
                x=float(fit.x[0]),r=float(fit.x[1]),quadrature_error_estimate=float(ef),success=bool(fit.success)))
            state['passes_peak_screen']=all(checks[k]<.001 for k in ['first_longitudinal_change','transverse_expansion_change','second_longitudinal_change','final_polished_peak_drift','final_spectral_integral_discrepancy'])
            path.write_text(json.dumps(state,indent=2));results.append(state)
    (OUT/'summary.json').write_text(json.dumps(results,indent=2))
    (OUT/'config.json').write_text(json.dumps(dict(t=256,alpha=.5,dx=.25,dr=2,lengths=[32768,65536,131072],radii=[2048,4096],
        peak_tolerance=.001,quadrature='64 k-panels plus cusp split; k=-beta +/- u^2; Gauss-Legendre orders 256 and 512',
        scope='same four Gaussian-family cases; finite spatial search, not a certified supremum',wall_seconds_current_invocation=time.perf_counter()-begin),indent=2))
    print('Completed spatial audit',flush=True)

if __name__=='__main__':main()

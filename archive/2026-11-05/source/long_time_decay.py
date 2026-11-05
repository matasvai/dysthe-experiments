"""Long-time Gaussian-family weighted decay via analytic transverse integration.

Canonical w=kx^3+3*sign*kx*(ky^2+kz^2)+3*beta*(ky^2+kz^2).
Initial f=exp(-x^2/(2*sx^2)-r^2/(2*sy^2))*exp(i*q*x)/L1.
Only x is periodic in the FFT quadrature; transverse variables are whole-space.
"""
from pathlib import Path
import os,csv,json,time
os.environ.setdefault('MPLCONFIGDIR','/tmp/dysthe-mpl')
import numpy as np
from scipy.fft import ifft,fftfreq
from scipy.integrate import quad
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent/'results/long_time'
OUT.mkdir(parents=True,exist_ok=True)
TIMES=[16,32,64,128,256]
REGIMES=[('homogeneous_minus',-1,0.),('inhomogeneous_plus',1,1.)]
FAMILIES=['gaussian','near_degeneracy']
LEVELS=[('base',4096,8192,512,4),('expanded',8192,16384,1024,4),
        ('refined',8192,32768,1024,2)]

def params(family,beta):return (3. if family=='gaussian' else 6.),3.,(0. if family=='gaussian' else -beta)

def scan(t,sign,beta,sx,sy,q,length,n,rmax,dr):
    k=2*np.pi*fftfreq(n,d=length/n)
    d=sy*sy-6j*t*(sign*k+beta)
    coeff=np.exp(-sx*sx*(k-q)**2/2+1j*t*k**3)*np.abs(k+beta)**.5/(2*np.pi*d)
    radii=np.arange(0,rmax+dr/2,dr);best=(-1,None,None); edge=0.
    # Batches bound memory independently of the transverse search extent.
    for offset in range(0,len(radii),16):
        rs=radii[offset:offset+16]
        values=ifft(coeff[None,:]*np.exp(-rs[:,None]**2/(2*d[None,:])),axis=1,workers=4)*(n/length)
        mag=np.abs(values);i,j=np.unravel_index(np.argmax(mag),mag.shape)
        if mag[i,j]>best[0]:best=(float(mag[i,j]),float((j if j<n//2 else j-n)*length/n),float(rs[i]))
        # Edge of radial search; longitudinal edge at +/-length/2.
        edge=max(edge,float(mag[:,n//2].max()))
        if offset+len(rs)==len(radii):edge=max(edge,float(mag[-1].max()))
    return dict(peak=best[0],x_peak=best[1],r_peak=best[2],edge_to_peak=edge/best[0])

def continuous_sum(point,t,sign,beta,sx,sy,q,length=8192,n=32768):
    # Continuous spatial evaluation of the same Fourier quadrature (no mesh snapping).
    x,r=point;k=2*np.pi*fftfreq(n,d=length/n);d=sy*sy-6j*t*(sign*k+beta)
    return np.sum(np.exp(-sx*sx*(k-q)**2/2+1j*(t*k**3+k*x)-r*r/(2*d))*
                  np.abs(k+beta)**.5/(2*np.pi*d))/length

def independent_quad(x,r,t,sign,beta,sx,sy,q):
    # Whole-space k quadrature, truncated at 12 Gaussian standard deviations.
    def f(k):
        d=sy*sy-6j*t*(sign*k+beta)
        return np.exp(-sx*sx*(k-q)**2/2+1j*(t*k**3+k*x)-r*r/(2*d))*np.abs(k+beta)**.5/d/(4*np.pi**2)
    lo,hi=q-12/sx,q+12/sx
    cuts=sorted(set([lo,hi]+([-beta] if lo<-beta<hi else [])))
    val=0j;err=0.
    for a,b in zip(cuts[:-1],cuts[1:]):
        re,e1=quad(lambda k:f(k).real,a,b,epsabs=2e-11,epsrel=1e-9,limit=2000)
        im,e2=quad(lambda k:f(k).imag,a,b,epsabs=2e-11,epsrel=1e-9,limit=2000)
        val+=re+1j*im;err+=e1+e2
    return val,err

def main():
    start=time.perf_counter();rows=[];summary=[]
    for regime,sign,beta in REGIMES:
        for family in FAMILIES:
            sx,sy,q=params(family,beta)
            for t in TIMES:
                vals=[]
                for label,length,n,rmax,dr in LEVELS:
                    res=scan(t,sign,beta,sx,sy,q,length,n,rmax,dr)
                    rows.append(dict(regime=regime,family=family,t=t,level=label,length=length,n=n,rmax=rmax,dr=dr,**res))
                    vals.append(res)
                best=vals[-1]
                extra_change=None
                if t==256:
                    # Same fine spatial spacing, quadrupled period and doubled radial range.
                    extra=scan(t,sign,beta,sx,sy,q,32768,131072,2048,2)
                    rows.append(dict(regime=regime,family=family,t=t,level='extra_domain',length=32768,n=131072,rmax=2048,dr=2,**extra))
                    extra_change=abs(best['peak']-extra['peak'])/extra['peak']
                    best=extra
                length,n=(32768,131072) if t==256 else (8192,32768)
                # Local polish after the broad x/r scan, keeping r>=0.
                fit=minimize(lambda p:-abs(continuous_sum(p,t,sign,beta,sx,sy,q,length,n))/best['peak'],
                    [best['x_peak'],best['r_peak']],method='Nelder-Mead',
                    bounds=[(best['x_peak']-2,best['x_peak']+2),(max(0,best['r_peak']-4),best['r_peak']+4)],
                    options={'xatol':1e-6,'fatol':1e-10,'maxiter':150})
                point=fit.x;field=continuous_sum(point,t,sign,beta,sx,sy,q,length,n);peak=abs(field)
                ref,err=independent_quad(*point,t,sign,beta,sx,sy,q)
                rec=dict(regime=regime,family=family,t=t,peak=float(peak),ratio=float(t**(7/6)*peak),
                    x_peak=float(point[0]),r_peak=float(point[1]),
                    domain_change=abs(vals[0]['peak']-vals[1]['peak'])/vals[1]['peak'],
                    grid_change=abs(vals[1]['peak']-vals[2]['peak'])/vals[2]['peak'],
                    local_peak_change=abs(peak-best['peak'])/peak,
                    extra_domain_change=extra_change,
                    quadrature_relative_error=float(abs(field-ref)/abs(ref)),
                    quadrature_error_estimate=float(err),edge_to_peak=best['edge_to_peak'],
                    optimizer_success=bool(fit.success))
                summary.append(rec)
                print(f"{regime} {family} t={t}: R={rec['ratio']:.7f}, quadrature error={rec['quadrature_relative_error']:.2e}",flush=True)
                (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    with open(OUT/'domain_scans.csv','w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    config=dict(times=TIMES,levels=LEVELS,extra_domain_at_256=dict(length=32768,n=131072,rmax=2048,dr=2),alpha=.5,normalization='exact continuum L1=(2*pi)^(3/2)*sx*sy^2',
        method='analytic transverse Gaussian integration; longitudinal FFT quadrature; radial search; local peak optimization; independent adaptive quadrature',
        initial_data='exp(-x^2/(2*sx^2)-r^2/(2*sy^2))*exp(i*q*x)/L1',
        families={'gaussian':{'sx':3,'sy':3,'q':0},'near_degeneracy':{'sx':6,'sy':3,'q':'-beta'}},
        scope='Two artificial families, two canonical regimes; no general-data or measured-data result',wall_seconds=time.perf_counter()-start)
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained',sharex=True,sharey=True)
    for i,(reg,_,_) in enumerate(REGIMES):
        for j,family in enumerate(FAMILIES):
            rr=[r for r in summary if r['regime']==reg and r['family']==family]
            ax=axs[i,j];ax.plot(TIMES,[r['ratio']*1000 for r in rr],'o-',color=['#226a9e','#c76b27'][i],lw=2)
            ax.set_xscale('log',base=2);ax.set_xticks(TIMES,labels=[str(t) for t in TIMES]);ax.grid(alpha=.2)
            ax.set_title(('Gaussian' if j==0 else 'Near-degeneracy packet')+'\n'+('Homogeneous' if i==0 else 'Inhomogeneous'))
            ax.annotate(f"{rr[-1]['ratio']:.5f}",(256,rr[-1]['ratio']*1000),xytext=(-5,10),textcoords='offset points',ha='right')
    fig.supxlabel('Evolution time t (each step doubles time)');fig.supylabel('Decay ratio R(t), in units of 0.001')
    fig.suptitle('Weighted decay through t = 256\nA plateau is consistent with the proposed power; continued growth needs further testing',fontsize=14)
    fig.savefig(OUT/'decay_256.png',dpi=170);fig.savefig(OUT/'decay_256.pdf');plt.close(fig)
    lines=['# Weighted decay through t=256','',
        'Two Gaussian-based families in both canonical regimes, alpha=1/2. This uses analytic transverse integration to avoid a prohibitively large full 3D grid.',
        'The plotted diagnostic remains R(t)=t^(7/6) || |Dx+beta|^(1/2) S(t)f ||infinity / ||f||1.',
        'The transverse profile is radial. The spatial maximum is searched over x and radial distance r, not only the origin.','',
        '| Regime | Input | t | R(t) | Domain change | Grid change | Independent integral error |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['regime']} | {r['family']} | {r['t']} | {r['ratio']:.7f} | {r['domain_change']:.3%} | {r['grid_change']:.3%} | {r['quadrature_relative_error']:.3%} |")
    lines+=['','## Numerical interpretation',
        'Domain check doubles longitudinal period 4096 to 8192 and radial search 512 to 1024. Grid check halves x spacing from 0.5 to 0.25 and radial spacing from 4 to 2.',
        'At t=256 an additional period-32768, radial-extent-2048 scan retains the fine spacings. Its peak is used in the final curve; its comparison is saved as extra_domain_change in summary.json.',
        'Local optimization polishes the sampled maximum. The adaptive integral independently checks the complex field at that candidate maximum, not global maximality.',
        'Finite search ranges, sampled maxima and finite quadrature prevent certification of the true supremum. Boundary amplitudes and optimizer status are saved.',
        'No theorem constant, stronger exponent, or counterexample follows solely from a finite set of ratios.',
        'Other artificial data families are not extended by this Gaussian-specific method. All earlier 3D FFT results remain intact.',
        'Run `python3 long_time_decay.py` to reproduce.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()

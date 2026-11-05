"""Generate continuum-defined artificial data and test weighted dispersive ratios.

Canonical coordinates of manuscript (6.2). No optical-coordinate conversion.
All input recipes can be resampled on new grids without changing the input.
"""
from pathlib import Path
import os, json, csv
os.environ.setdefault('MPLCONFIGDIR','/tmp/dysthe-mpl')
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/decay_dataset'; OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4); torch.set_default_dtype(torch.float64)
FAMILIES=['gaussian','anisotropic','smooth_bump','two_packets','chirped',
          'algebraic_tapered','near_degeneracy','away_degeneracy',
          'random_11','random_22','random_33']
TIMES=np.geomspace(.25,4,13); ALPHAS=[.25,.5,.75]
REGIMES=[('homogeneous_minus',-1,0.),('inhomogeneous_plus',1,1.)]

def input_field(name,x,beta):
    X,Y,Z=x[:,None,None],x[None,:,None],x[None,None,:]
    r2=X*X+Y*Y+Z*Z
    f=torch.exp(-r2/18).to(torch.complex128)
    if name=='anisotropic':f=torch.exp(-X*X/72-Y*Y/18-Z*Z/32).to(torch.complex128)
    if name=='smooth_bump':
        s=r2/100
        f=torch.where(s<1,torch.exp(1-1/torch.clamp(1-s,min=1e-12)),0).to(torch.complex128)
    if name=='two_packets':
        f=torch.exp(-((X+5)**2+Y*Y+Z*Z)/18)+.7j*torch.exp(-((X-5)**2+Y*Y+Z*Z)/18)
    if name=='chirped':f=f*torch.exp(.04j*X*X)
    if name=='algebraic_tapered':f=((1+r2/9)**-2*torch.exp(-(r2/400)**2)).to(torch.complex128)
    if name in ['near_degeneracy','away_degeneracy']:
        center=-beta if name=='near_degeneracy' else -beta+1
        f=torch.exp(-X*X/72-Y*Y/18-Z*Z/18)*torch.exp(1j*center*X)
    if name.startswith('random_'):
        rng=np.random.default_rng(int(name.split('_')[1]))
        modes=rng.normal(0,.25,(12,3)); phases=rng.uniform(0,2*np.pi,12)
        carrier=sum(torch.exp(1j*(q[0]*X+q[1]*Y+q[2]*Z+ph)) for q,ph in zip(modes,phases))/np.sqrt(12)
        f=f*carrier
    return f

def setup(name,sign,beta,n=96,length=96):
    x=torch.arange(n)*(length/n)-length/2; h=length/n
    f=input_field(name,x,beta); l1=float(f.abs().sum()*h**3)
    f=f/l1
    k=2*np.pi*torch.fft.fftfreq(n,d=h)
    K=k[:,None,None]; P=k[None,:,None]**2+k[None,None,:]**2
    w=K**3+sign*3*K*P+3*beta*P
    return x,f,torch.fft.fftn(f),w,K

def weighted(fh,w,K,beta,t,alpha):
    return torch.fft.ifftn(fh*torch.exp(1j*t*w)*torch.abs(K+beta)**alpha)

def main():
    rows=[];checks=[]; cards=[]; slices={}
    for regime,sign,beta in REGIMES:
        for name in FAMILIES:
            x,f,fh,w,K=setup(name,sign,beta)
            card=dict(family=name,regime=regime,beta=beta,sign=sign,
                      normalized_l1=float(f.abs().sum()),l2=float(torch.linalg.vector_norm(f)))
            cards.append(card); slices[f'{regime}_{name}']=f[:,48,:].numpy()
            np.savez_compressed(OUT/f'{regime}_{name}.npz',field=f.numpy(),coordinate=x.numpy())
            for t in TIMES:
                for alpha in ALPHAS:
                    u=weighted(fh,w,K,beta,float(t),alpha)
                    norm=float(u.abs().max())
                    rows.append(dict(regime=regime,family=name,t=float(t),alpha=alpha,
                        weighted_linf=norm,ratio=float(t)**(1+alpha/3)*norm))
            # End-point refinement and domain check for alpha=.5, all inputs.
            base=weighted(fh,w,K,beta,4,.5)
            for label,n,length in [('grid',192,96),('domain',144,144)]:
                _,_,ff,ww,kk=setup(name,sign,beta,n,length)
                ref=weighted(ff,ww,kk,beta,4,.5)
                view=ref[::2,::2,::2] if label=='grid' else ref[24:120,24:120,24:120]
                field_error=float(torch.linalg.vector_norm(base-view)/torch.linalg.vector_norm(view))
                # Grid uses the full fine maximum; domain uses the shared original box.
                refpeak=float((ref if label=='grid' else view).abs().max())
                checks.append(dict(regime=regime,family=name,check=label,
                    relative_field_error=field_error,
                    relative_peak_difference=abs(float(base.abs().max())-refpeak)/refpeak))
            print(regime,name,flush=True)
    with open(OUT/'weighted_decay.csv','w') as f:
        out=csv.DictWriter(f,fieldnames=list(rows[0]));out.writeheader();out.writerows(rows)
    manifest=dict(equation='S(t)f = inverse_FT(exp(+it*w)*FT(f))',
        w='kx^3 + sign*3*kx*(ky^2+kz^2) + 3*beta*(ky^2+kz^2)',
        diagnostic='t^(1+alpha/3) * || |Dx+beta|^alpha S(t)f ||_infinity / ||f||_1',
        interpretation='Exploratory finite periodic-box ratios, not certified whole-space bounds.',
        convention='Use +itw as in the proofs of Propositions 6.2/6.3; (6.1) has inconsistent printed signs.',
        times=TIMES.tolist(),alphas=ALPHAS,n=96,length=96,normalization='discrete L1=1 on each grid',
        torch_version=torch.__version__,data=cards,checks=checks)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for rr,(regime,_,_) in enumerate(REGIMES):
        for name in FAMILIES:
            data=[r for r in rows if r['family']==name and r['regime']==regime and r['alpha']==.5]
            axs[rr,0].loglog(TIMES,[r['weighted_linf'] for r in data],label=name)
            axs[rr,1].loglog(TIMES,[r['ratio'] for r in data],label=name)
        axs[rr,0].set(title=regime.replace('_',' '),ylabel='Weighted maximum',xlabel='t')
        axs[rr,1].set(title=r'Compensated ratio, $\alpha=1/2$',ylabel=r'$t^{7/6}$ times weighted maximum',xlabel='t')
    axs[0,0].legend(fontsize=6,ncol=2)
    for ax in axs.flat:ax.grid(alpha=.2)
    fig.savefig(OUT/'decay_comparison.png',dpi=170);fig.savefig(OUT/'decay_comparison.pdf');plt.close(fig)
    lines=['# First weighted-decay data batch','',
        '22 artificial initial fields: 11 recipes in each of two canonical regimes. All are L1-normalized.',
        '858 measurements: 13 times from 0.25 to 4, and alpha = 0.25, 0.5, 0.75.',
        'The production grid is 96^3 on [-48,48)^3. Field arrays are complex128; axes are (x,y,z).','',
        'The plots measure the actual fractional-derivative-weighted quantity, not ordinary amplitude.',
        'A growing compensated ratio on this short interval is not a counterexample: the unknown bound may still be finite.',
        'No asymptotic exponent is fitted before a reliable longer-time window is established.','',
        '| Regime | Input | Max ratio (alpha=.5) | Grid peak change at t=4 | Domain field change at t=4 |',
        '|---|---|---:|---:|---:|']
    for regime,_,_ in REGIMES:
        for name in FAMILIES:
            vals=[r['ratio'] for r in rows if r['family']==name and r['regime']==regime and r['alpha']==.5]
            cc=[c for c in checks if c['family']==name and c['regime']==regime]
            lines.append(f"| {regime} | {name} | {max(vals):.5g} | {cc[0]['relative_peak_difference']:.2%} | {cc[1]['relative_field_error']:.2%} |")
    lines+=['','## Limits and next step',
        'Checks currently cover t=4 and alpha=.5 only. They compare finite boxes; weighted multipliers can create long spatial tails.',
        'Large domain errors invalidate interpreting those cases as evidence for a whole-space estimate. Refine flagged cases before extending time.',
        'Each resolution uses the same continuum recipe and its own quadrature L1 normalization; errors include normalization differences.',
        'The algebraic profile is explicitly tapered; it is not an untapered power-law test.',
        'The degenerate-frequency recipes depend on beta, so their fields differ between regimes.',
        'Random fields use 12 Gaussian-distributed wavevectors (standard deviation .25) and random phases inside a Gaussian window; seeds are 11,22,33.',
        'Run `python3 decay_dataset.py` to regenerate all inputs, diagnostics and plots.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()

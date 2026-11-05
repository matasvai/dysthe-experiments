"""Larger-box, later-time checks and legible plots for the weighted decay study."""
import csv,json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/dysthe-mpl')
import numpy as np
import torch
import matplotlib.pyplot as plt
from decay_dataset import OUT,ROOT,FAMILIES,REGIMES,setup,weighted

SELECTED=['gaussian','smooth_bump','near_degeneracy','random_33']
TIMES=[.25,.5,1,2,4,6,8,12,16]
CHECK_TIMES=[4,8,16]
LABELS={'gaussian':'Gaussian','anisotropic':'Elongated Gaussian','smooth_bump':'Smooth compact bump',
        'two_packets':'Two packets','chirped':'Chirped Gaussian','algebraic_tapered':'Tapered algebraic profile',
        'near_degeneracy':'Spectrum near degeneracy','away_degeneracy':'Spectrum away from degeneracy',
        'random_11':'Random packet: seed 11','random_22':'Random packet: seed 22','random_33':'Random packet: seed 33'}
COLORS=['#226a9e','#c76b27']
REG_LABELS=['Homogeneous (minus, beta=0)','Inhomogeneous (plus, beta=1)']

def savefig(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=170)
    fig.savefig(OUT/(name+'.pdf'))
    plt.close(fig)

def broad_panels():
    rows=list(csv.DictReader(open(OUT/'weighted_decay.csv')))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(4,3,figsize=(11,11),layout='constrained',sharex=True,sharey=True)
    for ax,name in zip(axs.flat,FAMILIES):
        for (reg,_,_),label,color in zip(REGIMES,REG_LABELS,COLORS):
            rr=[r for r in rows if r['regime']==reg and r['family']==name and float(r['alpha'])==.5]
            ax.plot([float(r['t']) for r in rr],[1000*float(r['ratio']) for r in rr],label=label,color=color,lw=2)
        ax.set_title(LABELS[name],fontsize=10);ax.grid(alpha=.2);ax.set_ylim(0,5)
    axs[-1,-1].axis('off')
    axs[-1,-1].text(0,.95,'How to read these panels\n\nOne input family per panel.\nBlue and orange = two equations.\n\nVertical axis: decay-test ratio.\nFlat: matches the candidate power.\nFalling: faster over that interval.\nRising: slower over that interval.\n\nShort-time growth does not\ndisprove a uniform bound.',va='top',fontsize=9,transform=axs[-1,-1].transAxes)
    fig.supxlabel('Evolution time t');fig.supylabel('Decay-test ratio R(t), displayed in units of 0.001')
    fig.suptitle('First batch: all families, with two curves per panel\nalpha = 1/2; short-time, finite-box screening',fontsize=15)
    axs[0,0].legend(fontsize=7)
    savefig(fig,'families_clear')

def main():
    broad_panels();rows=[];checks=[]
    for reg,sign,beta in REGIMES:
        for name in SELECTED:
            results={};common={}
            for n,length in [(96,96),(192,192),(288,288),(288,192)]:
                x,f,fh,w,K=setup(name,sign,beta,n,length)
                del f
                times=TIMES if length==288 else CHECK_TIMES
                for t in times:
                    u=weighted(fh,w,K,beta,t,.5)
                    a=u.abs();peak=float(a.max());idx=np.unravel_index(int(a.argmax()),a.shape)
                    margin=min(float(length/2-abs(x[i])) for i in idx)
                    rec=dict(regime=reg,family=name,n=n,length=length,h=length/n,t=t,
                             weighted_linf=peak,ratio=t**(7/6)*peak,peak_boundary_distance=margin)
                    results[(n,length,t)]=rec;rows.append(rec)
                    if t in CHECK_TIMES:
                        # Common [-48,48)^3 nodes, spacing 2.
                        start=int((length-96)/2/(length/n)); stride=round(2/(length/n));count=48*stride
                        common[(n,length,t)]=u[start:start+count:stride,start:start+count:stride,start:start+count:stride].clone()
                del fh,w,K,u,a
            for t in CHECK_TIMES:
                for label,left,right in [('domain_96_to_192',(96,96),(192,192)),
                         ('domain_192_to_288',(192,192),(288,288)),
                         ('grid_h1_to_h2over3',(192,192),(288,192))]:
                    a=results[(*left,t)];b=results[(*right,t)]
                    ua=common[(*left,t)];ub=common[(*right,t)]
                    checks.append(dict(regime=reg,family=name,t=t,check=label,
                        relative_peak_difference=abs(a['weighted_linf']-b['weighted_linf'])/b['weighted_linf'],
                        common_cube_relative_l2=float(torch.linalg.vector_norm(ua-ub)/torch.linalg.vector_norm(ub))))
            print(f'Checked {reg}: {name}',flush=True)
    with open(OUT/'refined_decay.csv','w') as f:
        out=csv.DictWriter(f,fieldnames=list(rows[0]));out.writeheader();out.writerows(rows)
    (OUT/'refinement_checks.json').write_text(json.dumps(checks,indent=2))
    fig,axs=plt.subplots(2,4,figsize=(12,6),layout='constrained',sharex=True,sharey=True)
    for i,(reg,_,_) in enumerate(REGIMES):
        for ax,name in zip(axs[i],SELECTED):
            rr=[r for r in rows if r['regime']==reg and r['family']==name and r['length']==288]
            ax.plot([r['t'] for r in rr],[1000*r['ratio'] for r in rr],color=COLORS[i],lw=2)
            for t in CHECK_TIMES:
                cc=[c for c in checks if c['regime']==reg and c['family']==name and c['t']==t and c['check']!='domain_96_to_192']
                passed=all(c['relative_peak_difference']<.02 for c in cc)
                v=next(r['ratio'] for r in rr if r['t']==t)*1000
                ax.plot(t,v,'o' if passed else 'x',color=COLORS[i] if passed else '#b32636',ms=7,mew=1.5)
            ax.set_title(LABELS[name],fontsize=10);ax.set_xticks([0,4,8,12,16]);ax.grid(alpha=.2)
        axs[i,0].set_ylabel(('Homogeneous' if i==0 else 'Inhomogeneous')+'\nR(t) / 0.001')
    fig.supxlabel('Evolution time t')
    fig.suptitle('Extended decay test: one input and one equation per panel\nDots: peak changes <2% in both checks; crosses: at least one check exceeds 2%',fontsize=13)
    savefig(fig,'refined_clear')
    lines=['# Refined weighted-decay experiment','',
      'The original 11-family plots are now separated into individual panels. The larger-domain follow-up selects Gaussian, smooth bump, near-degeneracy, and random seed 33 in both regimes.',
      'Follow-up uses alpha=1/2 (candidate power 7/6), times 0.25 to 16, and lengths 96, 192, 288 at spacing 1. A separate length-192 grid with spacing 2/3 checks resolution.',
      'Domain and resolution checks occur at t=4,8,16. The displayed extended curves use length 288 at spacing 1.','',
      '## How to read the ratio',
      'R(t)=t^(7/6) || |Dx+beta|^(1/2) S(t)f ||infinity / ||f||1.',
      'Flat means the weighted maximum follows t^(-7/6) over that interval; rising means slower decay over that interval, falling means faster. A bounded rising curve is fully compatible with the claimed inequality.',
      'The theorem does not require R to decrease. No universal constant is inferred from this finite collection.',
      'Dots mean the two peak comparisons change by less than 2%; this is an exploratory numerical threshold, not a rigorous error estimate. Unmarked intermediate times have not been independently checked.','',
      '| Equation | Input | t | Ratio | Domain peak change: 192 to 288 | Grid peak change: h=1 to 2/3 |','|---|---|---:|---:|---:|---:|']
    for reg,_,_ in REGIMES:
        for name in SELECTED:
            for t in CHECK_TIMES:
                v=next(r['ratio'] for r in rows if r['regime']==reg and r['family']==name and r['length']==288 and r['t']==t)
                cc=[c for c in checks if c['regime']==reg and c['family']==name and c['t']==t]
                lines.append(f"| {reg} | {name} | {t} | {v:.5g} | {cc[1]['relative_peak_difference']:.2%} | {cc[2]['relative_peak_difference']:.2%} |")
    lines+=['','The JSON also records full complex-field differences on a common cube. Field L2 and peak sensitivity are different error measures; the latter directly targets the inequality.',
      'The grid and domain checks vary one axis of discretization at a time and do not establish a jointly converged infinite-domain supremum. Further box sizes may give nonmonotone errors near the fractional multiplier cusp.',
      'These runs remain exploratory. No asymptotic exponent, theorem strengthening, or counterexample is established.',
      'Reproduce with `python3 refine_decay.py` after `python3 decay_dataset.py`.']
    (OUT/'REFINEMENT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()

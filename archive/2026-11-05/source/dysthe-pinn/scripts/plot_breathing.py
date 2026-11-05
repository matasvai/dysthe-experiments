#!/usr/bin/env python3
"""Plot actual available diagnostics; never extrapolate into uncomputed times."""
import argparse,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/dysthe-pinn-mpl')
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def envelope(t, y, separation=12.):
    # One minimum and maximum per approximate oscillation cycle.
    groups = np.floor((t-t[0])/separation).astype(int)
    upper=[];lower=[]
    for group in np.unique(groups):
        ix=np.flatnonzero(groups==group)
        if len(ix)<2:continue
        upper.append(ix[np.argmax(y[ix])]);lower.append(ix[np.argmin(y[ix])])
    result=[]
    for candidates in [upper,lower]:
        ix=np.array(candidates,dtype=int)
        if len(ix)<2:
            result.append((np.array([]),np.array([]),ix));continue
        fine=np.linspace(t[ix[0]],t[ix[-1]],max(500,len(t)*3))
        result.append((fine,PchipInterpolator(t[ix],y[ix])(fine),ix))
    return result


def make_report(run, out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    s=json.loads((Path(run)/'summary.json').read_text());rows=s['rows']
    t=np.array([r['t'] for r in rows]);p=np.array([r['peak'] for r in rows])
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(2,1,figsize=(11,6),layout='constrained')
    coverage=[]
    for a,lo,hi in [(ax[0],max(0,float(t.min())),float(t.max())),(ax[1],1500,2500)]:
        m=(t>=lo)&(t<=hi);a.set(xlabel='Propagation coordinate t',ylabel='Domain peak magnitude')
        if m.sum()<3:
            a.text(.5,.5,'Window not yet reached',transform=a.transAxes,ha='center');continue
        x,y=t[m],p[m];a.plot(x,y,lw=.65,color='#527b9a',alpha=.8,label='Recorded peak')
        # Resample only to a uniformly spaced subset if the cadence changes across the full run.
        if not np.allclose(np.diff(x),np.diff(x)[0]):
            regular=np.isclose(x,np.round(x),atol=1e-9);xe,ye=x[regular],y[regular]
        else:xe,ye=x,y
        for (xx,yy,ix),name,color in zip(envelope(xe,ye),['Upper extrema trace','Lower extrema trace'],['#bd3d32','#258553']):
            a.plot(xx,yy,color=color,lw=1.8,label=name)
            coverage.append(dict(window=[lo,hi],trace=name,knots_t=xe[ix].tolist(),knots_y=ye[ix].tolist()))
        if hi>t.max():
            a.axvspan(t.max(),hi,color='gray',alpha=.12);a.text((t.max()+hi)/2,.9,'Not yet computed',transform=a.get_xaxis_transform(),ha='center',fontsize=9)
        a.set_xlim(lo,hi);a.grid(alpha=.2);a.legend(fontsize=8,ncol=3)
    ax[0].set_title(f"Available through t={t.max():g}; grid {s['config']['reference']['shape'][0]}³, dt={s['config']['reference']['dt']:g}");ax[1].set_title('Requested focus: 1500–2500')
    fig.savefig(out/'peak_envelopes.png',dpi=180);plt.close(fig)
    manifest=dict(status=s['status'],available_end=float(t.max()),method='PCHIP through the maximum and minimum in successive 12-unit bins; not a certified bounding curve or confidence interval.',traces=coverage)
    (out/'envelope-method.json').write_text(json.dumps(manifest,indent=2))
    if 'effective_volume' in rows[0]:
        m=(t>=1500)&(t<=2500)
        if m.sum()>2:
            fig,ax=plt.subplots(3,1,figsize=(10,7),layout='constrained',sharex=True)
            ax[0].plot(t[m],p[m]);ax[0].set(ylabel='Peak')
            for q in 'xyz':
                width=np.array([r[f'width_{q}'] if r[f'center_strength_{q}']>=.05 else np.nan for r in rows]);ax[1].plot(t[m],width[m],label=q)
            ax[1].set(ylabel='Wrapped RMS width');ax[1].legend()
            ax[2].plot(t[m],np.array([r['effective_volume'] for r in rows])[m]);ax[2].set(ylabel='Effective volume',xlabel='Propagation coordinate t')
            for a in ax:a.grid(alpha=.2)
            fig.suptitle('Breathing diagnostics: peak, width, and concentration')
            fig.savefig(out/'localization.png',dpi=180);plt.close(fig)
    centers=s.get('showcase_centers',[])
    if centers:
        fig,axes=plt.subplots(4,3,figsize=(12,10),layout='constrained')
        for a,c in zip(axes.flat,centers[:12]):
            m=(t>=c-24)&(t<=c+24)
            if m.sum():a.plot(t[m]-c,p[m],lw=1)
            else:a.text(.5,.5,'Pending',ha='center',transform=a.transAxes)
            a.set(title=f't ≈ {c}',xlabel='Offset from milestone',ylabel='Peak');a.grid(alpha=.2)
        fig.savefig(out/'thousand_showcases.png',dpi=160);plt.close(fig)
        # One reference slice at each reached milestone, using the frame file rather than loading the archive.
        settings=json.loads((Path(run)/'settings.json').read_text());dt=settings['dt']
        fig,axes=plt.subplots(4,3,figsize=(12,10),layout='constrained')
        for a,c in zip(axes.flat,centers[:12]):
            fp=Path(run)/'frames'/f'{round(c/dt):09d}.npy'
            if fp.exists():
                u=np.load(fp,mmap_mode='r');n=u.shape[1];im=a.imshow(abs(u[:,n//2,:]),origin='lower',extent=[-6,6,-6,6],vmin=0,vmax=.3,aspect='equal');a.set(title=f't = {c}',xlabel='z',ylabel='x')
            else:a.text(.5,.5,'Pending',ha='center',transform=a.transAxes);a.set_title(f't = {c}')
        fig.suptitle('Central y=0 magnitude slices; common scale 0–0.3')
        fig.savefig(out/'spatial_showcases.png',dpi=160);plt.close(fig)
    text=r'''\documentclass{article}
\usepackage[margin=1in]{geometry}\usepackage{graphicx,amsmath}
\title{Periodic Pulse: Peak Envelopes and Localization}\author{}\date{}
\begin{document}\maketitle
This report uses the available submitted diagnostics. Upper and lower curves are shape-preserving cubic interpolants through the maximum and minimum in successive 12-unit propagation intervals (approximately one fitted base oscillation period). They describe slow modulation and are not confidence intervals or guaranteed upper and lower bounds. Endpoint traces stop at the first and last selected extrema; no extrapolation is performed. Grey regions have no computed data.
\begin{center}\includegraphics[width=\linewidth]{peak_envelopes.png}\end{center}
'''
    if (out/'localization.png').exists():
        text+=r'''\clearpage
For $\rho=|E|^2$, the effective volume is $(\int\rho)^2/\int\rho^2$. A decrease indicates greater concentration. Widths are intensity-weighted RMS distances wrapped about each circular centroid. A centroid strength below $0.05$ makes that width ambiguous, and it is omitted from the plot. The $z$ width is a retarded-time width, not a third laboratory-space width. A growing peak accompanied by smaller widths and effective volume is compatible with squeezing, but does not establish a coherent breather.
\begin{center}\includegraphics[width=\linewidth]{localization.png}\end{center}
'''
    if centers:
        text+=r'''\clearpage\begin{center}\includegraphics[width=\linewidth]{thousand_showcases.png}\end{center}
\clearpage\begin{center}\includegraphics[width=\linewidth]{spatial_showcases.png}\end{center}
Slices show the center plane only and can miss a moving off-center pulse. Widths, centroids, peak locations, and the full saved fields must be interpreted together. No inference of a whole-space decay rate follows from these periodic trajectories.
'''
    text+=r'\end{document}'
    (out/'report.tex').write_text(text)
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();make_report(a.run,a.output)

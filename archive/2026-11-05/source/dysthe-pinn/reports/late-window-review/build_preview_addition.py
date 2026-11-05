from pathlib import Path
import json,os,shutil
import numpy as np
os.environ['MPLCONFIGDIR']='/tmp/dysthe-pinn-mpl'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
out=Path(__file__).resolve().parent;root=out.parents[1]
input_dir=out if (out/'preview-summary.json').exists() else root/'results/breathing-preview-12288'
s=json.loads((input_dir/('preview-summary.json' if input_dir==out else 'summary.json')).read_text());a=s['rows'];t=np.array([r['t'] for r in a]);p=np.array([r['peak'] for r in a]);v=np.array([r['effective_volume'] for r in a]);m=(t>=1500)&(t<=2500)
settings=json.loads((input_dir/('preview-settings.json' if input_dir==out else 'settings.json')).read_text());assert settings['config']==s['config'];assert np.all(np.diff(t)>0);assert np.isfinite([t,p,v]).all();assert abs(a[-1]['relative_mass_drift'])>.001 and abs(a[-2]['relative_mass_drift'])<=.001
stats=dict(status=s['status'],end=s['current_t'],runtime_seconds=s['invocation_seconds'],focus_count=int(m.sum()),focus_peak_min=float(p[m].min()),focus_peak_max=float(p[m].max()),focus_volume_min=float(v[m].min()),focus_volume_max=float(v[m].max()),focus_peak_volume_correlation=float(np.corrcoef(p[m],v[m])[0,1]),ambiguous_centers=sum(not r['centers_well_defined'] for r in a if 1500<=r['t']<=2500))
(out/'preview-statistics.json').write_text(json.dumps(stats,indent=2))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(2,1,figsize=(10,5.2),layout='constrained')
ax[0].plot(t,p,lw=.45,color='#66859b',label='Coarse preview peak')
# Use integer-time diagnostics for consistent full-history envelope binning.
integer=np.isclose(t,np.round(t));x=t[integer];y=p[integer];g=np.floor(x/12).astype(int)
for kind,color in [('upper','#b9382c'),('lower','#28804d')]:
 ix=[]
 for b in np.unique(g):
  q=np.flatnonzero(g==b)
  if len(q)>1:ix.append(q[np.argmax(y[q]) if kind=='upper' else np.argmin(y[q])])
 xx=np.linspace(x[ix[0]],x[ix[-1]],10000);ax[0].plot(xx,PchipInterpolator(x[ix],y[ix])(xx),lw=1,color=color,label=kind.title()+' trace')
ax[0].axvspan(1500,2500,color='gray',alpha=.15);ax[0].set(xlabel='Propagation coordinate t',ylabel='Peak',title='Preview stopped at t=10321 (target: 12288)');ax[0].legend(fontsize=8,ncol=3)
# A 120-unit detail makes the relationship readable; statistics use the entire requested window.
z=(t>=1580)&(t<=1700);ax[1].plot(t[z],p[z],lw=1,color='#245a81',label='Peak');right=ax[1].twinx();right.plot(t[z],v[z],lw=1,color='#b95d24',alpha=.8,label='Effective volume');ax[1].set(xlabel='Propagation coordinate t',ylabel='Peak',title='Detail inside 1500–2500: higher peaks accompany greater concentration');right.set_ylabel('Effective volume')
ax[1].legend(loc='upper left',fontsize=8);right.legend(loc='upper right',fontsize=8)
for q in ax:q.grid(alpha=.2)
fig.savefig(out/'preview_results.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(3,4,figsize=(10,6),layout='constrained',sharey=True)
for q,c in zip(axes.flat,range(1000,12001,1000)):
 z=(t>=c-24)&(t<=c+24)
 if z.any():q.plot(t[z]-c,p[z],lw=.8,color='#245a81')
 else:q.text(.5,.5,'Not reached',transform=q.transAxes,ha='center',color='#a63329',fontsize=9)
 q.set(title=f't ≈ {c}',xlabel='Local offset');q.grid(alpha=.2)
for q in axes[:,0]:q.set_ylabel('Peak')
fig.savefig(out/'preview_milestones.png',dpi=180);plt.close(fig)
text=r'''\clearpage
\section{Addition: results of the coarse breathing preview}
This addition reviews the newly submitted $32^3$, $\Delta t=0.5$ reference run, commit \texttt{b12b69c}. It uses the same equation, initial pulse, and periodic box as the main report. It is a separate calculation, not an extension of the previously reviewed $64^3$ trajectory, and no PINN was trained.
\begin{center}\small\begin{tabular}{lr}\toprule
Quantity & Recorded result\\\midrule
Target / reached propagation & $12288$ / $10321$\\
Elapsed execution time & $121.24$ seconds\\
Integration steps / reported saved frames & $20642$ / $726$\\
Maximum relative mass drift & $1.000073\times10^{-3}$\\
Maximum spectral-edge fraction & $2.65135\times10^{-6}$\\
Status & \textnormal{failed\_diagnostics}\\\bottomrule\end{tabular}\end{center}
The run stopped at the first recorded crossing of the $0.1\%$ mass-drift threshold. Mass increased numerically; this is a conservation error, not evidence of physical amplification. The spectral-edge threshold did not trigger. The complete $1500$--$2500$ observation interval was reached, as were the showcases through $10000$; $11000$ and $12000$ were not reached.
\begin{center}\includegraphics[width=\linewidth]{preview_results.png}\end{center}
\noindent\textbf{Preview Figure A.} The available long trajectory with upper and lower extrema traces, followed by a detail showing peak height and effective volume on separate vertical scales. Extrema traces interpolate maxima and minima in 12-unit bins and are descriptive, not rigorous bounds. The shaded interval is $1500$--$2500$.
\clearpage
\section{Concentration and thousand-point showcases}
Over $1500\leq t\leq2500$, the 2,001 diagnostic samples have peak magnitudes between $0.073723$ and $0.236918$. The effective volume
\[
V_{\mathrm{eff}}=\frac{\left(\int |E|^2\,dV\right)^2}{\int |E|^4\,dV}
\]
ranges from $164.37$ to $924.41$; smaller values indicate a more concentrated field. Peak and effective volume have Pearson correlation $-0.9134$ across this interval. This is a descriptive association between temporally dependent samples, not an independent-sample significance test or evidence of causation.

The highest recorded focus-window peak occurs at $t=1660.5$, with $V_{\mathrm{eff}}=165.17$. Thus this preview provides more evidence for repeated concentration and spreading than the peak-only data did. However, 338 of 2,001 focus samples ($16.9\%$) have at least one weak circular centroid, making the corresponding width difficult to interpret. A single coherent localized breather is not established. The effective-volume measure is independent of the centroid choice, but it too cannot distinguish all possible multi-pulse structures.
\begin{center}\includegraphics[width=\linewidth]{preview_milestones.png}\end{center}
\noindent\textbf{Preview Figure B.} Peak histories within $\pm24$ of each thousand. All panels share the vertical scale. These are scalar histories, not spatial pulse images; the last two requested windows were not reached.

On the common integer times $1500$--$2048$, the coarse preview differs from the earlier $64^3$, $\Delta t=0.01$ result in sampled peak by at most $1.04\%$, with mean relative discrepancy $0.223\%$. This is useful overlap agreement, but it combines grid and step changes and does not validate the complex field or the trajectory beyond $2048$. Focus-window mass drift reaches $0.0190\%$, below the operational stop threshold; passing that check alone does not establish field accuracy.

Only the summary and settings were supplied, so spatial shape, phase, and the reported field-frame files were not independently inspected. The result is a successful inexpensive qualitative preview that terminated on a numerical diagnostic, with breathing-like concentration modulation rather than a demonstrated breather or a decay law.
'''
(out/'preview_addition.tex').write_text(text)

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vendor_temporal_solver import Model
R=Path(__file__).resolve().parent;F=R/'report/figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
families=['gaussian','random','two_packet'];labels=['Gaussian','Random','Two packets'];colors=['#d47d29','#6858ac','#168f9a']
fig,ax=plt.subplots(1,3,figsize=(11,3.4),layout='constrained')
for f,label,c in zip(families,labels,colors):
 z=np.load(R/'data/near-zero'/f'{f}.npz');T=z['T'][1:];u=z['dysthe'];l=z['linear'];u0=z['u0'];p=abs(u).max(axis=-1);non=np.fft.ifft(Model(len(u0),40,.05,'stability').nonlinear_hat(np.fft.fft(u0)))
 ax[0].semilogx(T,1e6*(p[1:]/p[0]-1),'o-',c=c,label=label)
 ax[1].loglog(T,abs(u[1:]-l[1:]).max(axis=-1),'o-',c=c)
 q=(u[1:]-l[1:])/T[:,None];err=abs(q-non).max(axis=-1)/abs(non).max();ax[2].loglog(T,err,'o-',c=c)
for a in ax:a.set_xlabel('Scaled time $T$');a.set_xticks([1e-4,1e-3,1e-2],['0.0001','0.001','0.01'])
ax[0].set(title='Ordinary peak change',ylabel='Peak change (ppm)')
ax[1].set(title='Nonlinear deviation from free evolution',ylabel=r'$\|u_{\rm D}-u_{\rm lin}\|_\infty$')
ax[2].set(title='Near-zero consistency',ylabel=r'Relative defect of $(u_{\rm D}-u_{\rm lin})/T$')
fig.legend(*ax[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
fig.savefig(F/'08-near-zero-times.pdf',bbox_inches='tight');fig.savefig(F/'08-near-zero-times.png',bbox_inches='tight',dpi=170);plt.close(fig)
fig,ax=plt.subplots(1,3,figsize=(11,3.4),layout='constrained')
peak_ratios=[]
for family in families:
 with np.load(R/'data/long-linear'/f'{family}.npz') as data:
  peak_ratios.extend(data['peak_refined'][data['T']>=16]/data['peak_refined'][0])
shared_ymax=max(1.12,1.08*max(peak_ratios))
for j,(f,label,c) in enumerate(zip(families,labels,colors)):
 z=np.load(R/'data/long-linear'/f'{f}.npz');T=z['T'];mask=T>=16;p=z['peak_refined'];p0=p[0];floor=np.sqrt(np.mean(abs(z['u0'])**2))/p0
 ax[j].semilogx(T[mask],p[mask]/p0,'o-',c=c,lw=2,label='Exact periodic linear evolution');ax[j].axhline(floor,c='.35',ls='--',label='Conserved-action lower bound');ax[j].set(title=label,xlabel='$T$',ylabel='$P(T)/P(0)$' if j==0 else '',ylim=(0,shared_ymax),xticks=[16,64,256,1024],xticklabels=['16','64','256','1024'])
fig.legend(*ax[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
fig.suptitle('Distant-time linear snapshots on a fixed periodic interval · not nonlinear results',fontsize=12)
fig.savefig(F/'09-linear-to-1024.pdf',bbox_inches='tight');fig.savefig(F/'09-linear-to-1024.png',bbox_inches='tight',dpi=170)

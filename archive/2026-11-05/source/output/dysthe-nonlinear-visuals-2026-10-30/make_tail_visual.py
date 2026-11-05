from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
z=np.load(ROOT/'data/gaussian-tail-audit.npz',allow_pickle=False);s=z['s'];T=z['T'];tail=z['tail_mask'];v=z['dysthe_p24']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
fig,ax=plt.subplots(2,2,figsize=(10,6.5),layout='constrained')
for t,c,sty in [(0,'#236f9c','-'),(.025,'#6858ac','--'),(.1,'#d47d29',':')]:
 j=np.argmin(abs(T-t));ax[0,0].plot(s,abs(v[j]),c=c,ls=sty,label=f'T={T[j]:g}',lw=2);ax[0,1].semilogy(s,np.maximum(abs(v[j]),1e-16),c=c,ls=sty,lw=1.5)
ax[0,0].set(title='Whole envelope: almost unchanged near zero',ylabel='$|u|$',xlabel='$s$');ax[0,0].legend(frameon=False)
ax[0,1].set(title='Magnified tails on a logarithmic scale',ylabel='$|u|$',xlabel='$s$',ylim=(1e-12,1))
for cutoff,c in [(12,'#d47d29'),(24,'#168f9a')]:
 e=abs(z[f'p{cutoff}_u0']-z['analytic_u0']);ax[1,0].semilogy(s,np.maximum(e,1e-18),c=c,lw=1.2,label=f'Initial cutoff {cutoff}')
ax[1,0].set(title='Initial projection already produces tiny ripples',ylabel='Absolute initial discrepancy',xlabel='$s$',ylim=(1e-18,1e-7));ax[1,0].legend(frameon=False)
projection=np.max(abs(z['dysthe_p12'][:,tail]-z['dysthe_p24'][:,tail]),axis=1)
nonlinear=np.max(abs(z['dysthe_p24'][:,tail]-z['linear_p24'][:,tail]),axis=1)
ax[1,1].semilogy(T,np.maximum(projection,1e-18),c='#d47d29',label='Effect of initial cutoff');ax[1,1].semilogy(T,np.maximum(nonlinear,1e-18),c='#236f9c',label='Dysthe minus linear (cutoff 24)');ax[1,1].set(title=r'A fixed tail region: $|u(s,0)|<0.01\,P(0)$',xlabel='$T$',ylabel='Maximum absolute complex difference',ylim=(1e-12,1e-5));ax[1,1].legend(frameon=False,fontsize=8)
fig.suptitle('Gaussian near $T=0$ · small differences require a controlled initial spectrum',fontsize=13)
fig.savefig(ROOT/'report/figures/07-gaussian-near-zero.pdf');fig.savefig(ROOT/'report/figures/07-gaussian-near-zero.png',dpi=170)

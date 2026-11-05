from pathlib import Path
import json,os
import numpy as np
from scipy.optimize import minimize_scalar
os.environ.setdefault('MPLCONFIGDIR','/tmp/dysthe-pinn-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parent
s=json.loads((out/'summary.json').read_text());q=[r for r in s['rows'] if 1536<=r['t']<=1664]
t=np.array([r['t'] for r in q]);y=np.array([r['peak'] for r in q]);train=np.arange(len(t))%5!=0

def design(t,f,K):
 u=t-1600.;tau=u/64.;a=[np.ones_like(t),tau]
 for k in range(1,K+1):
  for v in [np.cos(2*np.pi*k*f*u),np.sin(2*np.pi*k*f*u)]:a.extend([v,tau*v])
 return np.array(a).T

def fit(K,mask):
 def objective(f):
  X=design(t,f,K);c=np.linalg.lstsq(X[mask],y[mask],rcond=None)[0]
  return np.mean((X[mask]@c-y[mask])**2)
 opt=minimize_scalar(objective,bounds=(.084,.091),method='bounded',options={'xatol':1e-13})
 X=design(t,opt.x,K);c=np.linalg.lstsq(X[mask],y[mask],rcond=None)[0]
 return opt.x,c,X@c
scores=[]
for K in range(1,5):
 f,c,pred=fit(K,train);scores.append(dict(harmonics=K,validation_rmse=float(np.sqrt(np.mean((pred[~train]-y[~train])**2)))))
K=min(scores,key=lambda r:r['validation_rmse'])['harmonics'];f,c,pred=fit(K,np.ones(len(t),dtype=bool))
rmse=float(np.sqrt(np.mean((pred-y)**2)));r2=float(1-np.sum((pred-y)**2)/np.sum((y-y.mean())**2))
result=dict(interval=[1536,1664],center=1600,scale=64,n=len(t),harmonics=K,frequency=f,period=1/f,rapid_second_harmonic_period=1/(2*f),coefficients=c.tolist(),rmse=rmse,r_squared=r2,validation_selection=scores,qualification='Descriptive local harmonic fit. Validation used to select harmonic count; not independent test or future-time forecast.')
(out/'oscillation-fit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
grid=np.linspace(1536,1664,3000);curve=design(grid,f,K)@c
fig,ax=plt.subplots(2,1,figsize=(10,5.6),layout='constrained',gridspec_kw={'height_ratios':[2.5,1]})
ax[0].plot(grid,curve,color='#c55326',lw=1.3,label='Four-harmonic descriptive fit');ax[0].scatter(t,y,s=13,facecolors='white',edgecolors='#245a81',linewidths=.9,zorder=3,label='Recorded peak (spacing 1)')
ax[0].set(ylabel='Peak magnitude',title=f'Local oscillation fit: R² = {r2:.4f}, RMSE = {rmse:.5f}');ax[0].legend(fontsize=9,ncol=2)
ax[1].axhline(0,color='gray',lw=.7);ax[1].plot(t,y-pred,'.-',ms=3,lw=.65,color='#245a81');ax[1].set(xlabel='Propagation coordinate t',ylabel='Data − fit')
for a in ax:a.set_xlim(1536,1664);a.grid(alpha=.2)
fig.savefig(out/'oscillation_fit.png',dpi=200);plt.close(fig)
text=r'''\clearpage
\section{A descriptive fit to the rapid oscillations}
The late peak contains repeated sharp rises and falls. A single sinusoid does not describe their shape well. To visualize this structure, the interval $1536\leq t\leq1664$ is fitted with a slowly varying baseline and four harmonics:
\[
\widehat P(t)=c_0+c_1\tau+
\sum_{k=1}^{4}\left[(a_k+b_k\tau)\cos(2\pi kfu)
+(d_k+e_k\tau)\sin(2\pi kfu)\right],
\qquad u=t-1600,\quad\tau=u/64.
\]
A harmonic is an oscillation at an integer multiple of a base frequency. Combining harmonics allows sharper peaks than one sine wave; the linear amplitude factors allow their height to change across this local interval. The 18 linear coefficients are fitted by least squares, while the base frequency is optimized in $0.084\leq f\leq0.091$, guided by the sampled spectrum. This frequency interval is a modelling choice, not a physical constraint.
\begin{center}\includegraphics[width=\linewidth]{oscillation_fit.png}\end{center}
\noindent\textbf{Figure 3.} Recorded peak values and the local oscillatory fit, with residuals beneath. The smooth line approximates the sampled data; it is not a new PDE solution or a prediction outside this interval.

The fitted base period is PERIOD, and its second harmonic has period RAPID. The refit on all 129 points has RMSE RMSE and $R^2=RSQ$. These quantify local descriptive agreement, not long-time numerical accuracy.

Harmonic counts one through four were compared by withholding every fifth point. Their validation RMSEs were $0.03474$, $0.01802$, $0.00870$, and $0.00523$, respectively. Four harmonics were selected and then refitted using all points. Because these withheld points selected the model, they are not an independent final test. The fit uses 19 adjustable quantities including frequency. Sampling at unit intervals cannot rule out aliased higher frequencies or establish the exact continuous peak between records. Neither the fitted periods nor the high local $R^2$ establish a soliton, a decay exponent, or a globally periodic orbit.
'''
text=text.replace('PERIOD',f'${1/f:.4f}$').replace('RAPID',f'${1/(2*f):.4f}$').replace('RMSE RMSE',f'RMSE ${rmse:.5f}$').replace('RSQ',f'{r2:.4f}')
(out/'oscillation_fit.tex').write_text(text)

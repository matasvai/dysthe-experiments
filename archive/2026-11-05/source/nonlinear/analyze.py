"""Tables and scientific figures from saved runs; does not rerun the solver."""
import os,json,platform
os.environ.setdefault('MPLCONFIGDIR','/tmp/dysthe-mpl')
from pathlib import Path
import numpy as np
import scipy
from scipy.signal import resample
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/nonlinear';FIG=ROOT/'report/nonlinear_figures';FIG.mkdir(exist_ok=True)
def meta(name):return json.loads((OUT/f'{name}.json').read_text())
def data(name):return np.load(OUT/f'{name}.npz')
def history(name):return meta(name)['diagnostics']
def drift(d,key):return max(abs(a[key]-d[0][key]) for a in d)/ (abs(d[0][key]) if key=='mass' else max(1,abs(d[0][key])))
def savefig(fig,name):
 for ext in ['pdf','png']:fig.savefig(FIG/f'{name}.{ext}',dpi=180)
 plt.close(fig)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'mathtext.fontset':'cm','font.family':'serif'})
summary={}
bench=meta('benchmark_summary');summary['benchmark']=bench
# Exact NLS and linear contrast.
a=data('nls_N1024_L80_h0.0025');b=data('linear_control')
fig,ax=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
for i in [0,10,40]:ax[0].plot(a['x0'],abs(a['u'][i]),label=f"NLS t={a['t'][i]:g}",lw=1.3)
ax[0].plot(b['x0'],abs(b['u'][-1]),'--',label='Linear t=20');ax[0].set(xlim=(-12,12),xlabel='x',ylabel='Envelope magnitude');ax[0].legend(fontsize=8)
for n in ['nls_N1024_L80_h0.0025','linear_control']:
 d=history(n);ax[1].plot([v['t'] for v in d],[v['peak']/d[0]['peak'] for v in d],label='NLS' if n.startswith('nls') else 'Linear')
ax[1].set(xlabel='Evolution time',ylabel='Peak / initial peak');ax[1].legend();savefig(fig,'nls_balance')
fig,ax=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
rr=[a for a in bench if a['n']==1024 and a['L']==80];rr=sorted(rr,key=lambda a:a['dt']);xx=np.array([a['dt'] for a in rr]);yy=np.array([a['final_error'] for a in rr]);ax[0].loglog(xx,yy,'o-',label='Complex-field error');ax[0].loglog(xx,yy[-2]*(xx/xx[-2])**4,'--',label='Fourth-order reference');ax[0].set(xlabel='Time step',ylabel='Relative error at t=20');ax[0].legend(fontsize=8)
for speed in [0,2,-2]:
 rr=[a for a in meta('profile_summary') if a['mu']==2 and a['speed']==speed];ax[1].plot([a['epsilon'] for a in rr],[a['mass'] for a in rr],'o-',label=f's={speed}')
ax[1].set(xlabel='Dysthe steepness parameter',ylabel='Profile action');ax[1].legend();savefig(fig,'convergence_branch')
fig,ax=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
for e in [0,.05,.1,.15]:
 z=data(f'profile_mu3_s0_e{e:g}');ax[0].plot(z['x0'],abs(z['u'][0]),label=f'epsilon={e:g}')
ax[0].set(xlim=(-5,5),xlabel='Profile coordinate',ylabel='Ground-state magnitude, mu=3');ax[0].legend(fontsize=8)
for s in [-2,0,2]:
 z=data(f'profile_mu2_s{s:g}_e0.15');ax[1].plot(z['x0'],abs(z['u'][0]),label=f's={s}')
ax[1].set(xlim=(-6,6),xlabel='Profile coordinate',ylabel='Magnitude, mu=2, epsilon=0.15');ax[1].legend();savefig(fig,'profiles')
fig,ax=plt.subplots(1,2,figsize=(10,3.7),layout='constrained');z=data('prop_e0.15_s2');sel=(z['x0']>-8)&(z['x0']<50);im=ax[0].pcolormesh(z['x0'][sel],z['t'],abs(z['u'][:,sel]),shading='auto',rasterized=True,cmap='viridis');fig.colorbar(im,ax=ax[0],label='Magnitude');ax[0].set(xlabel='Profile coordinate',ylabel='Propagation variable')
for dt,name in [(.01,'timestep_dysthe_0.01'),(.005,'timestep_dysthe_0.005'),(.0025,'prop_e0.15_s2'),(.00125,'timestep_dysthe_0.00125')]:
 d=history(name);ax[1].semilogy([a['t'] for a in d][1:],[a['complex_error'] for a in d][1:],label=f'h={dt:g}')
ax[1].set(xlabel='Propagation variable',ylabel='Error against computed traveling wave');ax[1].legend(fontsize=8);savefig(fig,'propagation')
fig,ax=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
for name,label in [('perturb_amplitude_0.01','+1% amplitude'),('perturb_amplitude_-0.01','-1% amplitude'),('perturb_phase_0.01','Phase perturbation')]:
 d=history(name);t=[a['t'] for a in d];ax[0].plot(t,[a['aligned_error'] for a in d],label=label);ax[1].plot(t,[a['peak']/d[0]['peak'] for a in d],label=label)
ax[0].set(xlabel='Propagation variable',ylabel='Error after phase/translation alignment');ax[1].set(xlabel='Propagation variable',ylabel='Peak / perturbed initial peak');ax[0].legend(fontsize=8);savefig(fig,'perturbations')
# Collision panels share display conventions and avoid hiding small radiation.
fig,ax=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for j,name in enumerate(['collision_e0_N2048_L160_h0.00125','collision_refined_N8192_h0.000625']):
 if not (OUT/f'{name}.npz').exists():name='collision_e0.15_N4096_L160_h0.00125' if j else name
 z=data(name);sel=abs(z['x0'])<32;x=z['x0'][sel];mag=abs(z['u'][:,sel]);im=ax[0,j].pcolormesh(x,z['t'],mag,shading='auto',rasterized=True,cmap='viridis',vmin=0,vmax=4);fig.colorbar(im,ax=ax[0,j],label='Magnitude')
 im=ax[1,j].pcolormesh(x,z['t'],np.log10(np.maximum(mag,1e-6)),shading='auto',rasterized=True,cmap='magma',vmin=-4,vmax=.6);fig.colorbar(im,ax=ax[1,j],label='log10 magnitude')
 for a in ax[:,j]:a.set(xlabel='Profile coordinate',ylabel='Propagation variable')
 ax[0,j].set_title('NLS collision' if j==0 else 'Dysthe collision')
savefig(fig,'collisions')
fig,ax=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
for j,name in enumerate(['collision_e0_N2048_L160_h0.00125','collision_refined_N8192_h0.000625']):
 if not (OUT/f'{name}.json').exists():name='collision_e0.15_N4096_L160_h0.00125'
 z=data(name);m=meta(name);e=m['epsilon'];fits=m['collision']['outgoing_fits']
 for a,ls in zip(fits,['-','--']):
  s=a['speed'];p0=data(f'profile_mu2_s{s:g}_e{e:g}');q=np.linspace(-5,5,501);initial=np.interp(q,p0['x0'],abs(p0['u'][0]));final=np.interp(q+a['translation'],z['x0'],abs(z['u'][-1]));ax[j].plot(q,initial,ls,color='black',alpha=.6,label=f'Incoming s={s:g}');ax[j].plot(q,final,ls,color='#b35806',label=f'Outgoing s={s:g}')
 ax[j].set(title='NLS' if j==0 else 'Dysthe',xlabel='Coordinate relative to fitted pulse center',ylabel='Magnitude');ax[j].legend(fontsize=7)
savefig(fig,'collision_profiles')
# Leading-order water illustration from a computed spatial Dysthe profile.
e=.05;z=data('profile_mu2_s0_e0.05');xp=np.linspace(365,435,1200);tp=800.;xi=e*e*xp;tau=e*(2*xp-tp);F=np.interp(tau,z['x0'],z['u'][0].real)+1j*np.interp(tau,z['x0'],z['u'][0].imag);B=e*F*np.exp(-2j*xi);eta=(B*np.exp(1j*(xp-tp))).real
fig=plt.figure(figsize=(10,6),layout='constrained');ax=fig.add_subplot(211,projection='3d');xx,yy=np.meshgrid(xp[::4]-400,np.linspace(-12,12,32));zz=np.broadcast_to(eta[::4],xx.shape);ax.plot_surface(xx,yy,zz,cmap='ocean',rstride=1,cstride=1,linewidth=0);ax.set(xlabel="k0 x - 400",ylabel="k0 y",zlabel="k0 eta (leading order)");ax.view_init(25,-65)
a=fig.add_subplot(212);a.plot(xp-400,eta,lw=.8,label='Carrier reconstruction');a.plot(xp-400,abs(B),'k--',label='Envelope bounds');a.plot(xp-400,-abs(B),'k--');a.set(xlabel='Dimensionless physical distance from packet center',ylabel='Dimensionless elevation');a.legend(fontsize=8);savefig(fig,'water_reconstruction')
# Tables, summaries, and comparisons.
summary['propagation']=[]
for e in [0,.05,.1,.15]:
 for s in [0,2,-2]:
  d=history(f'prop_e{e:g}_s{s:g}');summary['propagation'].append(dict(epsilon=e,speed=s,error=d[-1]['complex_error'],aligned=d[-1]['aligned_error'],mass_drift=drift(d,'mass'),energy_drift=drift(d,'energy'),momentum_drift=drift(d,'momentum')))
summary['collisions']=[]
for p in sorted(OUT.glob('collision*.json')):
 m=json.loads(p.read_text())
 if 'diagnostics' not in m:continue
 d=m['diagnostics'];summary['collisions'].append(dict(name=p.stem,n=m['shape'][0],L=m['lengths'][0],dt=m['dt'],epsilon=m['epsilon'],mass_drift=drift(d,'mass'),energy_drift=drift(d,'energy'),spectral_edge=max(a['spectral_edge_mass'] for a in d),**m['collision']))
summary['perturbations']=[]
for p in sorted(OUT.glob('perturb*.json')):
 m=json.loads(p.read_text());d=m['diagnostics'];summary['perturbations'].append(dict(name=p.stem,initial_aligned=d[0]['aligned_error'],final_aligned=d[-1]['aligned_error'],translation=d[-1]['translation'],peak_ratio=d[-1]['peak']/d[0]['peak'],mass_drift=drift(d,'mass')))
summary['profiles']=meta('profile_summary');summary['profile_checks']=meta('profile_checks')
if (OUT/'transverse_refined_finer_history.json').exists():
 fig,ax=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
 for name,label in [('optical3d_stripe','Unperturbed'),('optical3d_mixed','Coarse, exploratory'),('optical3d_mixed_grid','First refinement'),('transverse_refined_finer','Finer, validated window')]:
  d=meta(name+'_history');ax[0].semilogy([a['t'] for a in d],np.maximum([a['transverse_fraction'] for a in d],1e-15),label=label);ax[1].plot([a['t'] for a in d],[a['peak'] for a in d],label=label)
 for a in ax:a.axvline(4,color='black',ls=':',lw=1);a.set_xlabel('Optical propagation variable')
 ax[0].set_ylim(.003,1);ax[0].set_ylabel('Relative transverse variation');ax[1].set_ylabel('Peak magnitude');ax[0].legend(fontsize=7);savefig(fig,'transverse_growth')
 z=np.load(OUT/'transverse_refined_finer_slices.npz');coords=data('transverse_refined_finer');fig,axs=plt.subplots(2,3,figsize=(10,6),layout='constrained');sel=abs(coords['x0'])<6
 for j,i in enumerate([0,8,16]):
  for row,key,lab in [(0,'xy','y'),(1,'xz','z')]:
   mag=abs(z[key][i])[sel,:];im=axs[row,j].pcolormesh(coords['x0'][sel],coords['x1'],mag.T,shading='auto',rasterized=True,cmap='viridis',vmin=0,vmax=1.6);axs[row,j].set(xlabel='x',ylabel=lab,title=f"t={z['t'][i]:g}")
 fig.colorbar(im,ax=axs.ravel().tolist(),label='Envelope magnitude');savefig(fig,'transverse_slices')
 summary['transverse']=[]
 for name in ['optical3d_stripe','optical3d_mixed','optical3d_mixed_time','optical3d_mixed_grid','optical3d_mixed_domain','transverse_refined_fine','transverse_refined_time','transverse_refined_finer','transverse_refined_length','transverse_refined_xfine']:
  if (OUT/f'{name}_history.json').exists():
   d=meta(name+'_history');summary['transverse'].append(dict(name=name,final_t=d[-1]['t'],fraction=d[-1]['transverse_fraction'],peak=d[-1]['peak'],mass_drift=drift(d,'mass'),spectral_edge=max(a['spectral_edge_mass'] for a in d)))
 ref=data('transverse_refined_finer')['u'][-1]
 comparisons=[]
 for name in ['fine','time','length','xfine']:
  if not (OUT/f'transverse_refined_{name}.npz').exists():continue
  c=data('transverse_refined_'+name);u=c['u'][-1]
  if name=='length':u=u[128:384]
  q=resample(resample(resample(ref,u.shape[0],axis=0),u.shape[1],axis=1),u.shape[2],axis=2)
  comparisons.append(dict(name=name,complex_error=float(np.linalg.norm(u-q)/np.linalg.norm(q))))
 summary['transverse_comparisons']=comparisons
summary['environment']=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,date='2026-09-14')
(OUT/'analysis_summary.json').write_text(json.dumps(summary,indent=2));print('Figures and summary generated')

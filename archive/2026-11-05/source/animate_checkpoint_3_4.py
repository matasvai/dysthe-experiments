"""Whole-space Gaussian transverse reduction for formal checkpoint section 3.4.
Canonical LINEAR symbol kx^3-3*kx*(ky^2+kz^2), original pair and normalization.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/dysthe-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import json
import numpy as np
from scipy.fft import ifft,fftfreq,fftshift
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'output/checkpoint_3_4_animations';DATA=ROOT/'results/checkpoint_3_4_animations'
OUT.mkdir(exist_ok=True);DATA.mkdir(exist_ok=True)
xold=np.arange(192)*.75-72
g1=np.exp(-(xold+5)**2/18);g2=np.exp(-(xold-5)**2/18)
Z=float(np.sum(abs(g1+.7j*g2))*.75*(np.sum(np.exp(-xold*xold/18))*.75)**2)
C=(2*np.pi)**1.5*27/Z
initialpeak=float(abs(np.exp(-(-5+5)**2/18)+.7j*np.exp(-(-5-5)**2/18))/Z)
def field(t,x,r,L=4096.,N=16384):
 k=2*np.pi*fftfreq(N,d=L/N);d=9+6j*t*k
 coef=C*np.exp(-4.5*k*k+1j*t*k**3)*(np.exp(5j*k)+.7j*np.exp(-5j*k))/(2*np.pi*d)
 grid=(np.arange(N)-N//2)*L/N
 out=np.empty((len(r),len(x)),complex)
 for i in range(0,len(r),24):
  rr=r[i:i+24];v=fftshift(ifft(coef[None,:]*np.exp(-rr[:,None]**2/(2*d)),axis=1,workers=2),axes=1)*(N/L)
  for j,row in enumerate(v):out[i+j]=np.interp(x,grid,row.real)+1j*np.interp(x,grid,row.imag)
 return out
# All plotted x nodes lie on the integration FFT mesh; interpolation is then exact.
def extent(t):return np.ceil((24+4*t)/4)*4

def quadpoint(t,x,r):
 def fn(k):
  d=9+6j*t*k
  return C*np.exp(-4.5*k*k+1j*(t*k**3+k*x)-r*r/(2*d))*(np.exp(5j*k)+.7j*np.exp(-5j*k))/(4*np.pi*np.pi*d)
 return quad(lambda k:fn(k).real,-4,4,epsabs=1e-12,limit=1500)[0]+1j*quad(lambda k:fn(k).imag,-4,4,epsabs=1e-12,limit=1500)[0]

def audit():
 rows=[]
 for t in [0,4,8,32,64,128,256]:
  e=extent(t);dx=max(.25,np.ceil((2*e/512)/.25)*.25);x=np.arange(-e,e+dx/2,dx);r=np.linspace(0,e,193);u=field(t,x,r);a=abs(u);i,j=np.unravel_index(np.argmax(a),a.shape)
  ref=field(t,x,r,L=8192,N=32768);fine=field(t,x,r,L=4096,N=32768)
  q=quadpoint(t,x[j],r[i]);edge=max(a[:,0].max(),a[:,-1].max(),a[-1].max())/a.max()
  rows.append(dict(t=t,extent=e,sampled_peak=float(a.max()),x_peak=float(x[j]),r_peak=float(r[i]),box_relative_field=float(np.linalg.norm(ref-u)/np.linalg.norm(ref)),grid_relative_field=float(np.linalg.norm(fine-u)/np.linalg.norm(fine)),quadrature_relative_error=float(abs(q-u[i,j])/abs(q)),view_boundary_to_peak=float(edge)))
  print('AUDIT',rows[-1],flush=True)
 (DATA/'audit.json').write_text(json.dumps(rows,indent=2))
 return rows

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#293c50','text.color':'#20364d','axes.titleweight':'bold'})
def make(stage,t0,t1,count=65):
 times=np.linspace(t0,t1,count);frames=[];rows=[]
 # Carrier has the same illustrative factor exp(3ix) as checkpoint 3.4.
 detailx=np.arange(-12,12+.0625,.125)
 for n,t in enumerate(times):
  e=extent(t);dx=max(.25,np.ceil((2*e/1024)/.25)*.25);x=np.arange(-e,e+dx/2,dx);r=np.linspace(0,e,321);u=field(t,x,r);a=abs(u);peak=a.max();z=np.r_[-r[:0:-1],r];plane=np.vstack([a[:0:-1],a]);line=u[0];detail=field(t,detailx,np.array([0.]),N=32768)[0]
  edge=max(a[:,0].max(),a[:,-1].max(),a[-1].max())/peak
  fig=plt.figure(figsize=(12,8),dpi=100,facecolor='#f7f9fc');gs=fig.add_gridspec(2,2,height_ratios=[1.1,.8],left=.075,right=.955,bottom=.1,top=.83,wspace=.27,hspace=.38)
  ax=fig.add_subplot(gs[0,0]);im=ax.pcolormesh(x,z,plane/initialpeak,cmap='magma',norm=LogNorm(1e-5,1),shading='auto',rasterized=True)
  ax.set(xlabel='x',ylabel='z  (slice y = 0)',title='Spreading field |u| / initial peak');ax.set_aspect('equal',adjustable='box');cb=fig.colorbar(im,ax=ax,pad=.025,fraction=.04);cb.ax.tick_params(labelsize=8)
  ec=min(e,4*np.ceil((24+.5*t)/4));xc=np.arange(-ec,ec+.25,.5);rc=np.linspace(0,ec,81);ac=abs(field(t,xc,rc));zc=np.r_[-rc[:0:-1],rc];ac=np.vstack([ac[:0:-1],ac]);stride=max(1,len(xc)//140)
  ax=fig.add_subplot(gs[0,1],projection='3d');xx,zz=np.meshgrid(xc[::stride],zc[::2]);vv=ac[::2,::stride]/peak
  ax.plot_surface(xx,zz,vv,cmap='viridis',vmin=0,vmax=1,linewidth=0,antialiased=False,rcount=vv.shape[0],ccount=vv.shape[1]);ax.set(xlim=(-ec,ec),ylim=(-ec,ec),zlim=(0,1.05),xlabel='x',ylabel='z',title=f'Central shape / current peak (view ±{ec:g})');ax.view_init(25,-62);ax.set_box_aspect((1.5,1,.65));ax.tick_params(labelsize=8,pad=0)
  ax=fig.add_subplot(gs[1,0]);ax.semilogy(x,np.maximum(abs(line)/initialpeak,1e-7),color='#176f94',lw=1.6);ax.set(xlim=(-e,e),ylim=(1e-5,1.2),xlabel='x  (full widening view)',ylabel='|u(x,0,0)| / initial peak',title='Centerline amplitude: fixed vertical scale');ax.grid(alpha=.2)
  ax=fig.add_subplot(gs[1,1]);env=abs(detail)/initialpeak;wave=np.real(detail*np.exp(3j*detailx))/initialpeak
  ax.fill_between(detailx,-env,env,color='#b94a64',alpha=.13);ax.plot(detailx,wave,color='#263f54',lw=.9);ax.plot(detailx,env,color='#b94a64',lw=1.3);ax.plot(detailx,-env,color='#b94a64',lw=1.3)
  stageheight={ 'early':1.1,'middle':.65,'late':.18}[stage]
  ax.set(xlim=(-12,12),ylim=(-stageheight,stageheight),xlabel='x  (fixed close-up)',ylabel='Amplitude / initial peak',title='Illustrative carrier with envelope bounds');ax.grid(alpha=.2)
  fig.suptitle(f'Two phase-offset packets  |  {stage.upper()} STAGE',fontsize=18,fontweight='bold',y=.966)
  fig.text(.5,.905,f't = {t:6.2f}     •     viewing range ±{e:g}     •     sampled peak / initial = {peak/initialpeak:.4f}',ha='center',fontsize=12)
  fig.text(.5,.035,'Formal checkpoint §3.4  •  Linear canonical evolution  •  Carrier added for illustration; not fluid-particle motion',ha='center',fontsize=9,color='#53677b')
  fig.canvas.draw();im=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()).convert('P',palette=Image.Palette.ADAPTIVE,colors=192);frames.append(im);plt.close(fig)
  rows.append(dict(t=float(t),view_halfwidth=float(e),sampled_peak=float(peak),view_boundary_to_peak=float(edge)))
  if n in [0,count//2,count-1]:im.convert('RGB').save(OUT/f'{stage}_{n:03d}.png')
  if n%16==0:print(stage,n,flush=True)
 frames[0].save(OUT/f'{stage}_t{t0:g}_to_{t1:g}.gif',save_all=True,append_images=frames[1:],duration=[120]*(count-1)+[1600],loop=0,optimize=False,disposal=2)
 (DATA/f'{stage}_frames.json').write_text(json.dumps(rows,indent=2));print('SAVED',stage,flush=True)
if __name__=='__main__':
 audit()
 for spec in [('early',0,8),('middle',8,64),('late',64,256)]:make(*spec)
 (DATA/'config.json').write_text(json.dumps(dict(source='formal checkpoint section 3.4 / figure 8',symbol='kx^3 - 3*kx*(ky^2+kz^2)',normalization_divisor=Z,initial_peak_reference=initialpeak,method='analytic transverse Gaussian integration and longitudinal FFT quadrature',L=4096,N=16384,carrier_N=32768,carrier='Re(u exp(3ix))',frames_per_stage=65),indent=2))

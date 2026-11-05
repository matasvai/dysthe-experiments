import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/dysthe-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import json
import numpy as np
from scipy.fft import ifft,fftfreq,fftshift
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,TwoSlopeNorm
from PIL import Image
from animate_checkpoint_3_4 import Z,C,initialpeak,field
OUT=Path('output/packet_alternative_views');OUT.mkdir(exist_ok=True)
def parts(t,x,r):
 N=16384;L=4096;k=2*np.pi*fftfreq(N,L/N);d=9+6j*t*k;grid=(np.arange(N)-N//2)*L/N
 base=C*np.exp(-4.5*k*k+1j*t*k**3)/(2*np.pi*d);ans=[]
 for mod in [np.exp(5j*k),.7j*np.exp(-5j*k)]:
  out=np.empty((len(r),len(x)),complex)
  for j in range(0,len(r),24):
   v=fftshift(ifft((base*mod)[None,:]*np.exp(-r[j:j+24,None]**2/(2*d)),axis=1,workers=2),axes=1)*(N/L)
   for i,row in enumerate(v):out[j+i]=np.interp(x,grid,row.real)+1j*np.interp(x,grid,row.imag)
  ans.append(out)
 return ans
plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
# Space-time map on an actual time axis, with both full view and center detail.
times=np.r_[np.linspace(0,8,65),np.linspace(8,64,113)[1:],np.linspace(64,256,193)[1:]]
x=np.arange(-400,400.25,.5);u=np.array([field(t,x,np.array([0.]))[0] for t in times]);mag=abs(u)/initialpeak
fig,ax=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
for a,lim in zip(ax,[(-400,400),(-35,65)]):
 keep=(x>=lim[0])&(x<=lim[1]);im=a.pcolormesh(x[keep],times,mag[:,keep],shading='auto',norm=LogNorm(1e-4,1),cmap='magma',rasterized=True);a.set(xlabel='x (y = z = 0)',ylabel='Evolution time t',title='Full centerline history' if lim[0]==-400 else 'Central detail: same data and color scale')
fig.colorbar(im,ax=ax,label='|u| / initial peak');fig.suptitle('Where the packet travels and spreads',fontsize=17)
fig.savefig(OUT/'space_time.png',dpi=180);fig.savefig(OUT/'space_time.pdf');plt.close(fig)
print('SPACE TIME',flush=True)
# Separate interference from intensity: normalized local contrast in [-1,1].
fig,ax=plt.subplots(2,3,figsize=(13,8),layout='constrained');records=[];meshdata=[]
for j,t in enumerate([0,8,64]):
 e=24+.5*t;xx=np.arange(-e,e+.25,.5);rr=np.linspace(0,e,161);p,q=parts(t,xx,rr);v=p+q;pk=abs(v).max();den=abs(p)**2+abs(q)**2;cross=2*np.real(p*np.conj(q));contrast=np.divide(cross,den,out=np.zeros_like(cross),where=den>1e-30);contrast[abs(v)<.005*pk]=np.nan
 zz=np.r_[-rr[:0:-1],rr];vv=np.vstack([abs(v[:0:-1]),abs(v)]);cc=np.vstack([contrast[:0:-1],contrast]);im=ax[0,j].pcolormesh(xx,zz,vv/pk,shading='auto',cmap='viridis',vmin=0,vmax=1,rasterized=True);im2=ax[1,j].pcolormesh(xx,zz,cc,shading='auto',cmap='RdBu_r',vmin=-1,vmax=1,rasterized=True)
 for a in ax[:,j]:a.set(xlabel='x',ylabel='z (y = 0)',aspect='equal')
 ax[0,j].set_title(f't = {t} | peak / initial = {pk/initialpeak:.3f}')
 records.append(dict(t=t,max_identity_residual=float(np.max(abs(abs(v)**2-(den+cross))))))
fig.colorbar(im,ax=ax[0],label='Magnitude / current peak');fig.colorbar(im2,ax=ax[1],label='Interference contrast: blue cancels, red reinforces');fig.suptitle('Interference: what the combined magnitude hides',fontsize=17)
fig.savefig(OUT/'interference.png',dpi=180);fig.savefig(OUT/'interference.pdf');plt.close(fig)
# Phase-colored height surfaces; omit negligible-amplitude points.
fig=plt.figure(figsize=(13,4.6),layout='constrained')
for j,t in enumerate([0,8,64]):
 e=16+.15*t;xx=np.linspace(-e,e,121);zz=np.linspace(-e,e,101);v=field(t,xx,abs(zz));a=abs(v);pk=a.max();h=a/pk;phase=np.angle(v);colors=plt.cm.twilight((phase+np.pi)/(2*np.pi));colors[:,:,3]=(a>.01*pk).astype(float);X,Y=np.meshgrid(xx,zz);ax=fig.add_subplot(1,3,j+1,projection='3d');ax.plot_surface(X,Y,h,facecolors=colors,rcount=101,ccount=121,shade=False,linewidth=0);ax.set(xlabel='x',ylabel='z',zlim=(0,1.05),title=f't = {t}');ax.view_init(30,-60);ax.set_zticks([0,.5,1]);ax.tick_params(labelsize=8,pad=0)
sm=plt.cm.ScalarMappable(norm=plt.Normalize(-np.pi,np.pi),cmap='twilight');fig.colorbar(sm,ax=fig.axes,shrink=.6,label='Complex phase (radians)',ticks=[-np.pi,0,np.pi]);fig.suptitle('Height = envelope magnitude / current peak; color = phase',fontsize=15)
fig.savefig(OUT/'phase_surface.png',dpi=180);fig.savefig(OUT/'phase_surface.pdf');plt.close(fig)
print('STATIC VIEWS',flush=True)
# Peak-following movie, fixed local spatial window. Late times have wider spacing.
mt=np.r_[np.linspace(0,8,25),np.linspace(8,64,25)[1:],np.linspace(64,256,25)[1:]];frames=[];track=[]
for i,t in enumerate(mt):
 scanx=np.arange(-24,100.25,.25);scan=field(t,scanx,np.array([0.]))[0];center=scanx[np.argmax(abs(scan))];local=np.arange(-20,20.25,.25);rr=np.linspace(0,20,121);v=field(t,center+local,rr);pk=abs(v).max();z=np.r_[-rr[:0:-1],rr];aa=np.vstack([abs(v[:0:-1]),abs(v)])
 fig,axs=plt.subplots(1,2,figsize=(11,4.8),layout='constrained');im=axs[0].pcolormesh(local,z,aa/pk,cmap='viridis',vmin=0,vmax=1,shading='auto');axs[0].set(xlabel='x minus tracked peak position',ylabel='z (y = 0)',aspect='equal',title='Fixed-width view following the peak');fig.colorbar(im,ax=axs[0],label='Magnitude / current local peak')
 axs[1].plot(local,abs(v[0])/initialpeak,color='#1c718a',lw=2);axs[1].set(xlim=(-20,20),ylim=(0,1.05),xlabel='x minus tracked peak position',ylabel='Magnitude / initial peak',title='Actual amplitude: fixed vertical scale');axs[1].grid(alpha=.2)
 fig.suptitle(f't = {t:.2f}   |   peak position x = {center:.2f}   |   peak / initial = {pk/initialpeak:.4f}',fontsize=14)
 fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()).convert('P',palette=Image.Palette.ADAPTIVE,colors=192));plt.close(fig);track.append(dict(t=float(t),x_peak=float(center),peak_ratio=float(pk/initialpeak)))
 if i%24==0:print('TRACK',i,flush=True)
frames[0].save(OUT/'moving_closeup.gif',save_all=True,append_images=frames[1:],duration=[140]*(len(frames)-1)+[1600],loop=0,disposal=2)
frames[len(frames)//2].convert('RGB').save(OUT/'moving_preview.png')
# Radial isocontours, revolved about x in the interactive viewer; no invented geometry.
for t in [0,8,64,256]:
 e=24+4*t;xx=np.arange(-e,e+.25,.5);rr=np.unique(np.r_[np.linspace(0,min(e,128),257),np.linspace(min(e,128),e,201)]);v=field(t,xx,rr);a=abs(v);pk=a.max();fig,ax=plt.subplots();cont=ax.contour(xx,rr,a/pk,levels=[.1,.3,.6]);levels=[]
 for level,segs in zip(cont.levels,cont.allsegs):
  lines=[]
  for seg in segs:
   if len(seg)<3:continue
   # Arc-length resampling of the real computed level contour.
   d=np.r_[0,np.cumsum(np.linalg.norm(np.diff(seg,axis=0),axis=1))];target=np.linspace(0,d[-1],min(100,len(seg)));lines.append(np.round(np.c_[np.interp(target,d,seg[:,0]),np.interp(target,d,seg[:,1])],3).tolist())
  levels.append(dict(level=float(level),lines=lines))
 plt.close(fig);meshdata.append(dict(t=t,peak_ratio=float(pk/initialpeak),levels=levels))
(OUT/'interactive_data.json').write_text(json.dumps(meshdata,separators=(',',':')))
(OUT/'diagnostics.json').write_text(json.dumps(dict(interference_identity_checks=records,peak_track=track),indent=2))
print('DONE',flush=True)

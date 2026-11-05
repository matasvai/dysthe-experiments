#!/usr/bin/env python3
"""Recreate the checkpoint's scientific visual style with actual 1D temporal trajectories."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource,Normalize
from matplotlib.animation import FuncAnimation,FFMpegWriter
from scipy.signal import resample
ROOT=Path(__file__).resolve().parent; FIG=ROOT/'report'/'figures';FIG.mkdir(parents=True,exist_ok=True)
BLUE='#236f9c';ORANGE='#d47d29';PALE='#d4e8f0';PURPLE='#6858ac';TEAL='#168f9a';NAVY='#18384b'
COLORS=[ORANGE,PURPLE,TEAL];FAMILIES=['gaussian','random','two_packet'];LABELS=['Gaussian packet','Smooth random profile','Two phase-offset packets']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelcolor':NAVY,'text.color':NAVY,'axes.titlecolor':NAVY,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16,'pdf.fonttype':42,'ps.fonttype':42,'savefig.bbox':'tight'})
D={f:np.load(ROOT/'data'/f'{f}.npz',allow_pickle=False) for f in FAMILIES}

def save(fig,name):
 fig.savefig(FIG/(name+'.pdf'));fig.savefig(FIG/(name+'.png'),dpi=170);plt.close(fig)
def idx(t,T):return int(np.argmin(abs(T-t)))
def footer(fig,text):fig.text(.5,.012,text,ha='center',fontsize=8,color=NAVY)
def clean3d(ax):
 ax.view_init(elev=24,azim=-65);ax.set_box_aspect((3.5,1.7,1))
 for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.pane.fill=False
 ax.tick_params(labelsize=8,pad=1)

def main(animate=False):
 z=D['gaussian'];s=z['s'];T=z['T'];u=z['dysthe']; j=idx(4,T)
 # Direct recreation of opening envelope/carrier panel, using actual evolved complex field.
 fig,ax=plt.subplots(2,1,figsize=(10,5.2),sharex=True,gridspec_kw={'height_ratios':[1.4,1]},layout='constrained')
 carrier=np.real(u[j]*np.exp(3j*s));amp=abs(u[j])
 ax[0].fill_between(s,-amp,amp,color=PALE);ax[0].plot(s,carrier,color=BLUE,lw=1,label='Illustrative carrier');ax[0].plot(s,amp,color=ORANGE,lw=2,label=r'Envelope $\pm|u|$');ax[0].plot(s,-amp,color=ORANGE,lw=2)
 ax[0].set(title='Nonlinear Gaussian packet: the carrier sits inside the envelope',ylabel='Dimensionless amplitude');ax[0].legend(ncol=2,frameon=False,fontsize=9)
 scale=amp.max();ax[1].plot(s,amp/scale,color=ORANGE,lw=2,label='Magnitude / current peak');ax[1].plot(s,amp**2/scale**2,color=PURPLE,ls='--',lw=2,label='Squared magnitude / current peak²')
 ax[1].set(xlabel='Moving coordinate $s$',ylabel='Normalized shape',xlim=(-14,10));ax[1].legend(loc='upper right',ncol=1,frameon=False,fontsize=8.5)
 fig.suptitle(r'$T=4$, $\epsilon=0.05$ · temporal potential envelope · illustrative carrier wavenumber 3',fontsize=10)
 save(fig,'01-envelope-carrier')
 # Wide blue space-time surface plus companion snapshots, analogous to oldwater panel.
 fig=plt.figure(figsize=(10,8));gs=fig.add_gridspec(3,1,height_ratios=[2.6,1,1],hspace=.52)
 ax=fig.add_subplot(gs[0],projection='3d');rows=np.unique([idx(t,T) for t in np.linspace(0,4,61)]);cols=np.arange(0,len(s),4);S,TT=np.meshgrid(s[cols],T[rows]);A=abs(u[np.ix_(rows,cols)])
 ax.plot_surface(S,TT,A,cmap='Blues',vmin=0,vmax=A.max()*1.12,linewidth=0,antialiased=True,rcount=len(rows),ccount=len(cols),rasterized=False)
 for t in [0,2,4]:jj=idx(t,T);ax.plot(s,np.full_like(s,T[jj]),abs(u[jj]),color=ORANGE,lw=1.5,zorder=5)
 clean3d(ax);ax.set_box_aspect((3.5,1.7,1),zoom=1.3);ax.set(xlabel='Position $s$',ylabel='Time $T$',zlabel='$|u|$',title='A nonlinear packet evolving in time')
 for pos,t in enumerate([0,4],1):
  q=fig.add_subplot(gs[pos]);jj=idx(t,T);a=abs(u[jj]);q.fill_between(s,-a,a,color=PALE);q.plot(s,np.real(u[jj]*np.exp(3j*s)),color=BLUE,lw=1);q.plot(s,a,color=ORANGE,lw=1.7,label=r'Dysthe envelope $\pm|u|$');q.plot(s,-a,color=ORANGE,lw=1.7);q.plot(s,abs(z['linear'][jj]),color='.3',ls='--',lw=1.2,label='Exact linear envelope')
  q.set(xlim=(-15,12),ylim=(-1.1,1.1),ylabel='Amplitude',title=f'$T={t:g}$',xlabel='Moving coordinate $s$' if t==4 else None);q.legend(loc='upper right',ncol=2,fontsize=8,frameon=False)
 fig.subplots_adjust(left=.1,right=.95,top=.96,bottom=.09);footer(fig,'The surface axes are position, time and envelope magnitude. The oscillatory traces are illustrative, not calibrated surface elevation.')
 save(fig,'02-space-time-packet')
 # Three-family space-timecomparison withfixedphysicalamplitude scale.
 fig=plt.figure(figsize=(13,4.6));zmax=max(abs(D[f]['dysthe']).max() for f in FAMILIES)
 for j,f in enumerate(FAMILIES):
  z=D[f];s=z['s'];T=z['T'];rows=np.unique([idx(t,T) for t in np.linspace(0,4,61)]);cols=np.arange(0,len(s),8);S,TT=np.meshgrid(s[cols],T[rows]);a=abs(z['dysthe'][np.ix_(rows,cols)])
  ax=fig.add_subplot(1,3,j+1,projection='3d');ax.plot_surface(S,TT,a,color=COLORS[j],alpha=.94,linewidth=0,antialiased=True,rcount=len(rows),ccount=len(cols));clean3d(ax);ax.set(xlabel='$s$',ylabel='$T$',zlabel='',xticks=[-20,0,20],yticks=[0,2,4],zticks=[0,.5,1],zlim=(0,zmax*1.08),title=LABELS[j]);ax.text2D(.015,.55,'$|u|$',transform=ax.transAxes,rotation=90);ax.set_box_aspect((3.5,1.7,1),zoom=.88)
 fig.subplots_adjust(left=.025,right=.945,bottom=.1,top=.9,wspace=.12);footer(fig,'Same model, period and amplitude scale. Three-dimensional views of one spatial coordinate evolving in time.')
 save(fig,'03-three-inputs')
 # Matched carrier snapshots withfixedscaleacrosscasesandtimes.
 fig,ax=plt.subplots(3,3,figsize=(11,7.4),sharex=True,sharey=True,layout='constrained')
 for r,f in enumerate(FAMILIES):
  z=D[f];s=z['s'];T=z['T']
  for c,t in enumerate([0,2,4]):
   jj=idx(t,T);v=z['dysthe'][jj];a=abs(v);ax[r,c].fill_between(s,-a,a,color=COLORS[r],alpha=.1);ax[r,c].plot(s,np.real(v*np.exp(3j*s)),color=NAVY,lw=.65);ax[r,c].plot(s,a,color=COLORS[r],lw=1.7);ax[r,c].plot(s,-a,color=COLORS[r],lw=1.7);ax[r,c].plot(s,abs(z['linear'][jj]),ls='--',c='.45',lw=1)
   ax[r,c].set(xlim=(-20,20),ylim=(-1.1,1.1),title=f'$T={t:g}$; peak {a.max():.3f}')
   if c==0:ax[r,c].set_ylabel(LABELS[r]+'\nAmplitude')
   if r==2:ax[r,c].set_xlabel('Moving coordinate $s$')
 fig.suptitle('Matched snapshots · coloured bounds: Dysthe · grey dashed: exact linear',fontsize=12)
 save(fig,'04-matched-snapshots')
 # Space-time magnitude and spectralpower:fixedscalesshowrealchanges, nofake2ndspatialaxis.
 fig,ax=plt.subplots(2,3,figsize=(11,6.1),layout='constrained');powermax=0
 for j,f in enumerate(FAMILIES):
  z=D[f];s=z['s'];T=z['T'];u=z['dysthe'];k=np.fft.fftshift(z['k']);power=np.fft.fftshift(abs(np.fft.fft(u,axis=-1)/len(s))**2,axes=-1);power/=power[0].sum();mask=abs(k)<=3
  im=ax[0,j].pcolormesh(s[::4],T,abs(u[:,::4]),shading='auto',cmap='magma',vmin=0,vmax=zmax,rasterized=False)
  im2=ax[1,j].pcolormesh(k[mask],T,np.log10(np.maximum(power[:,mask],1e-12)),shading='auto',cmap='magma',vmin=-8,vmax=0,rasterized=False)
  ax[0,j].set(title=LABELS[j],xlabel='Position $s$',ylabel='$T$' if j==0 else '')
  ax[1,j].set(xlabel='Modulation wavenumber $k$',ylabel='$T$' if j==0 else '')
 for im,axes,label in [(im,ax[0,:],'$|u|$'),(im2,ax[1,:],'log₁₀ mode power / initial total')]:
  cb=fig.colorbar(im,ax=axes,shrink=.9,label=label);cb.solids.set_rasterized(False)
 fig.suptitle('Real-space envelope and its Fourier power · nonlinear Dysthe',fontsize=12)
 save(fig,'05-position-and-fourier')
 # OrdinaryLinf: noassumeddecaynor weightedexponent.
 fig,ax=plt.subplots(1,3,figsize=(11,3.1),layout='constrained')
 for j,f in enumerate(FAMILIES):
  z=D[f];T=z['T'];p0=max(abs(z['u0']))
  for kind,color,sty,label in [('linear','.45','--','Exact linear'),('nls',TEAL,':','Cubic NLS'),('dysthe',COLORS[j],'-','Dysthe (case colour)')]:ax[j].plot(T,abs(z[kind]).max(axis=-1)/p0,color=color,ls=sty,lw=2,label=label)
  ax[j].set(title=LABELS[j],xlabel='$T$',ylabel='$\\|u(T)\\|_\\infty/\\|u(0)\\|_\\infty$' if j==0 else '')
 fig.legend(*ax[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
 save(fig,'06-ordinary-peak')
 if animate:
  fig,ax=plt.subplots(1,1,figsize=(10,4.5),layout='constrained');z=D['gaussian'];s=z['s'];T=z['T'];u=z['dysthe'];a0=abs(u[0]);ax.set(xlim=(-15,12),ylim=(-1.15,1.15),xlabel='Moving coordinate $s$',ylabel='Illustrative amplitude')
  wave,=ax.plot(s,np.real(u[0]*np.exp(3j*s)),color=BLUE,lw=1);up,=ax.plot(s,a0,color=ORANGE,lw=2,label='Nonlinear envelope');down,=ax.plot(s,-a0,color=ORANGE,lw=2);lin,=ax.plot(s,a0,color='.4',ls='--',label='Exact linear envelope');ax.legend(loc='lower right',frameon=False)
  title=ax.set_title('');fig.suptitle('Temporal potential Dysthe · ε=0.05 · illustrative carrier k=3',fontsize=11)
  frames=[idx(t,T) for t in np.linspace(0,4,81)]
  def update(j):
   a=abs(u[j]);wave.set_ydata(np.real(u[j]*np.exp(3j*s)));up.set_ydata(a);down.set_ydata(-a);lin.set_ydata(abs(z['linear'][j]));title.set_text(f'T = {T[j]:.3f}   |   peak / initial = {a.max()/a0.max():.3f}');return wave,up,down,lin,title
  FuncAnimation(fig,update,frames=frames,interval=100,blit=False).save(ROOT/'gaussian-evolution.mp4',writer=FFMpegWriter(fps=10,metadata={'title':'Illustrative carrier around computed Dysthe envelope'},extra_args=['-pix_fmt','yuv420p']),dpi=110);plt.close(fig)
 print('Saved figures to',FIG)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--animate',action='store_true');a=p.parse_args();main(a.animate)

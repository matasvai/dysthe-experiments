import os,ast,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/dysthe-mpl')
import numpy as np
from scipy.fft import ifft,fftfreq,fftshift
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from animate_checkpoint_3_4 import C
node=next(n for n in ast.parse(Path('make_packet_alternative_views.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='parts');exec(compile(ast.Module(body=[node],type_ignores=[]),'parts','exec'))
OUT=Path('output/packet_alternative_views');fig,ax=plt.subplots(2,3,figsize=(12,7.4),layout='constrained');checks=[]
for j,t in enumerate([0,8,64]):
 e=24+.5*t;x=np.arange(-e,e+.25,.5);r=np.linspace(0,e,161);p,q=parts(t,x,r);den=abs(p)**2+abs(q)**2;mask=den<1e-5*den.max();z=np.r_[-r[:0:-1],r]
 for row,phase in enumerate([np.pi/2,0.]):
  b=q if row==0 else -1j*q;cross=2*np.real(p*np.conj(b));c=np.divide(cross,den,out=np.zeros_like(cross),where=den>1e-30);maxcontrast=float(abs(c[~mask]).max());c[mask]=np.nan;cc=np.vstack([c[:0:-1],c]);im=ax[row,j].pcolormesh(x,z,cc,cmap='RdBu_r',vmin=-1,vmax=1,shading='auto',rasterized=True);ax[row,j].set(aspect='equal',xlabel='x',ylabel='z (y = 0)',title=f't = {t}');
  if row==0:ax[row,j].text(.5,.5,'Zero cross-term',transform=ax[row,j].transAxes,ha='center',color='#23354b')
  checks.append(dict(t=t,phase=phase,max_absolute_contrast=maxcontrast))
fig.colorbar(im,ax=ax,label='Contrast: blue cancels; red reinforces',shrink=.85);fig.suptitle('Interference contrast\nTop: original 90° phase offset    |    Bottom: additional 0° phase control',fontsize=15)
fig.savefig(OUT/'interference_control.png',dpi=180);fig.savefig(OUT/'interference_control.pdf');plt.close(fig)
(OUT/'interference_control_checks.json').write_text(json.dumps(checks,indent=2));print(checks)

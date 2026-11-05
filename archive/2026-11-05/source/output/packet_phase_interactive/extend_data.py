import sys,json,base64
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from animate_checkpoint_3_4 import field,initialpeak,quadpoint
OUT=Path(__file__).resolve().parent
T=np.r_[np.arange(0,17),np.arange(18,65,2),np.arange(72,257,8),np.arange(288,1025,32),np.arange(1056,2049,32),2056]
rows=[];packed=[];checks=[]
for i,t in enumerate(T):
 t=float(t);e=(16+.15*min(t,256))*max(1,t/256)**(1/3);x=np.linspace(-e,e,65);r=np.linspace(0,e,33)
 L=4096*2**int(np.ceil(np.log2(max(1,t/256))));N=int(L*4)
 u=field(t,x,r,L=L,N=N);pk=float(abs(u).max())
 # Only the upper half is stored: the field is exactly radial in y,z.
 mag=np.round(abs(u)/pk*255).astype('uint8');phase=np.round((np.angle(u)+np.pi)/(2*np.pi)*255).astype('uint8')
 packed.append(np.stack([mag,phase],axis=-1).tobytes())
 rows.append(dict(t=t,extent=round(e,6),n=65,peak_ratio=pk/initialpeak,L=L,N=N))
 if t in [0,8,64,256,512,1024,2056]:
  # Same coordinates, denser integration quadrature and doubled period.
  ref=field(t,x,r,L=2*L,N=2*N);fine=field(t,x,r,L=L,N=2*N)
  checks.append(dict(t=t,L=L,N=N,period_relative_field=float(np.linalg.norm(u-ref)/np.linalg.norm(ref)),spacing_relative_field=float(np.linalg.norm(u-fine)/np.linalg.norm(fine)),peak_ratio=pk/initialpeak))
  print('CHECK',checks[-1],flush=True)
 if i%20==0:print('FRAME',i,len(T),flush=True)
(OUT/'extended_metadata.json').write_text(json.dumps(rows,separators=(',',':')))
(OUT/'extended_payload.txt').write_text(base64.b64encode(b''.join(packed)).decode())
(OUT/'extended_checks.json').write_text(json.dumps(checks,indent=2))
print('DONE',len(T),len(b''.join(packed)),flush=True)

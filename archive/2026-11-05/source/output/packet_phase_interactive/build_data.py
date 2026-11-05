import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from animate_checkpoint_3_4 import field,initialpeak
rows=[]
for t in [0,4,8,16,32,64,128,256]:
 e=16+.15*t;x=np.linspace(-e,e,65);z=x.copy();u=field(t,x,abs(z));pk=float(abs(u).max())
 rows.append(dict(t=t,extent=round(e,3),n=65,peak_ratio=pk/initialpeak,magnitude=np.round(abs(u)/pk*65535).astype(int).ravel().tolist(),phase=np.round(np.angle(u)/np.pi*32767).astype(int).ravel().tolist()))
p=Path(__file__).resolve().parent/'data.json';p.write_text(json.dumps(rows,separators=(',',':')));print(p.stat().st_size)

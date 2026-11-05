"""Retain initial pilot flags and independently refine their search grids."""
import json
import numpy as np
from restricted_decay import OUT,scan,value,independent
from scipy.optimize import minimize
for m,R,dr in [(3,4096,2),(4,2048,1)]:
 p=OUT/f's3_m{m}_t256.json';a=json.loads(p.read_text());s=scan(256,3,m,16384,.25,R,dr)
 candidates=[]
 for _,x,r in s['seeds']:
  fit=minimize(lambda v:-abs(value(v,256,3,m))/s['peak'],[x,r],method='Nelder-Mead',bounds=[(x-3,x+3),(max(0,r-5),r+5)],options={'xatol':1e-6,'fatol':1e-10,'maxiter':180})
  ref,err=independent(fit.x,256,3,m);candidates.append((abs(ref),fit,err))
 peak,fit,err=max(candidates,key=lambda v:v[0])
 follow=dict(rmax=R,dr=dr,scan=s,x=float(fit.x[0]),r=float(fit.x[1]),peak=float(peak),optimized_peak_change=abs(peak/a['peak']-1),sampled_peak_change=abs(s['peak']/a['levels']['fine']['peak']-1),optimizer_success=bool(fit.success))
 follow['passes_1pct_screen']=bool(fit.success and max(follow['optimized_peak_change'],follow['sampled_peak_change'],s['edge_fraction'])<.01)
 a['followup']=follow;p.write_text(json.dumps(a,indent=2));print(m,follow['passes_1pct_screen'],follow['sampled_peak_change'],s['edge_fraction'],flush=True)
rows=[json.loads(p.read_text()) for p in sorted(OUT.glob('s*_m*_t*.json'))]
(OUT/'summary.json').write_text(json.dumps(rows,indent=2))

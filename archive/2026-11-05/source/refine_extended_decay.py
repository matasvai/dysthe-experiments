"""Retain first-screen failures and separately refine their spatial sampling."""
import json
import numpy as np
from extend_decay import OUT,scan,params
from research_experiments import gaussian_quad
from scipy.optimize import minimize
for p in sorted(OUT.glob('*_512.json'))+sorted(OUT.glob('*_1024.json')):
 a=json.loads(p.read_text())
 if not a.get('complete') or a['passes_screen'] or a.get('followup'):continue
 s,b=(-1,0.) if a['regime']=='homogeneous_minus' else (1,1.)
 sx,sy,q=params(a['family'],b);t=a['t'];st={}
 previous=a['scans']['grid'];basepeak=a['peak']
 for dr in [1.,.5]:
  finer=scan(t,s,b,sx,sy,q,65536,.25,0,4096,dr)
  change=abs(finer['best']['peak']-previous['best']['peak'])/finer['best']['peak']
  seed=finer['best'];fit=minimize(lambda z:-abs(gaussian_quad(z,t,s,b,sx,sy,q,1024))/basepeak,[seed['x'],seed['r']],method='Nelder-Mead',bounds=[(seed['x']-4,seed['x']+4),(max(0,seed['r']-4),seed['r']+4)],options={'xatol':1e-6,'fatol':1e-11,'maxiter':240})
  value=abs(gaussian_quad(fit.x,t,s,b,sx,sy,q,1024));peakchange=float(abs(value-basepeak)/basepeak)
  st[str(dr)]=dict(scan=finer,grid_change=change,refined_peak=float(value),refined_x=float(fit.x[0]),refined_r=float(fit.x[1]),candidate_peak_change=peakchange,optimizer_success=bool(fit.success))
  print(p.name,dr,change,peakchange,flush=True)
  remaining=[a['checks'][k] for k in ['length_change','order_change','adaptive_disagreement','frequency_tail_change','pointwise_period_change','spectral_integral_disagreement']]
  passed=change<.001 and peakchange<.001 and fit.success and max(remaining)<.001 and not a['adaptive']['warnings']
  if passed:break
  previous=finer
 a['followup']=dict(levels=st,passes_screen=bool(passed),note='Original failed screen retained. Inner radial sampling refined at fixed dx=.25 and L=65536; outer annulus retains dr=2. Peak compared after independent local optimization.')
 p.write_text(json.dumps(a,indent=2))
rows=[json.loads(p.read_text()) for p in sorted(OUT.glob('*_512.json'))+sorted(OUT.glob('*_1024.json'))]
(OUT/'summary.json').write_text(json.dumps([a for a in rows if a.get('complete')],indent=2))

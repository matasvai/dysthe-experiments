#!/usr/bin/env python3
"""Controlled smaller-step follow-up of the unresolved high-grid sideband case."""
from pathlib import Path
import json
import platform
import numpy as np
import scipy
from run import solve, inlets, errors, diagnostics, sha, write_json, MODELS, LIMITATION

root=Path(__file__).parent
c=json.loads((root/'config-audit.json').read_text())
c.update(step=.0003125,case_selection={'family':'sideband','epsilon':.1,'model':'stability','T':16.},purpose='Smaller-step follow-up preserving the failed prior audit; no tolerance change')
out=root/'results-time-audit';out.mkdir(exist_ok=True)
identity={'config':c,'solver_sha256':sha(root/'run.py'),'audit_source_sha256':sha(__file__),'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__}}
if (out/'identity.json').exists() and json.loads((out/'identity.json').read_text())!=identity:raise ValueError('Time-audit identity differs')
write_json(out/'identity.json',identity)
family,u0,params=inlets(c)[1];T=np.arange(round(16/c['sample_step'])+1)*c['sample_step'];fields={};cost={}
for role,n,h in [('fine',4096,c['step']),('grid',2048,c['step']),('step',4096,2*c['step'])]:
 path=out/(role+'.npz');meta=out/(role+'.json')
 if meta.exists():
  saved=json.loads(meta.read_text())
  if saved['sha256']!=sha(path):raise ValueError('Cached time-audit checksum differs')
  with np.load(path,allow_pickle=False) as z:fields[role]=z['fields']
  cost[role]=saved['seconds']
 else:
  fields[role],cost[role]=solve(u0,c['length'],.1,'stability',n,h,T)
  np.savez_compressed(path,fields=fields[role],T=T,s=np.arange(n)*c['length']/n-c['length']/2)
  write_json(meta,{'sha256':sha(path),'seconds':cost[role],'n':n,'step':h})
 print(role,cost[role],flush=True)
d=diagnostics(fields['fine'],T,c['length'],.1)
r={role:{key:float(value.max()) for key,value in errors(fields[role],fields['fine'],c['length']).items()} for role in ('grid','step')}
qualified=all(value<=c['tolerances'][key] for vals in r.values() for key,value in vals.items()) and d['max_action_drift']<=c['tolerances']['action'] and d['max_spectral_tail']<=c['tolerances']['tail']
result={'identity':identity,'key':'sideband-eps0.1-stability-T16','family':'sideband','epsilon':.1,'model':'stability','model_id':MODELS['stability'],'T':16.,'parameters':params,'qualified':qualified,'status':'complete' if qualified else 'numerically_unresolved','refinement':r,'diagnostics':d,'seconds':cost,'artifact':{'path':'fine.npz','sha256':sha(out/'fine.npz')},'limitation':LIMITATION}
# Compare only with an independently qualified NLS control; this is a model
# difference, not a numerical error estimate or a theorem about wave behavior.
nls_meta=json.loads((root/'results-audit'/'sideband-eps0.1-nls-T16.json').read_text())
if qualified and nls_meta['qualified']:
 with np.load(root/'results-audit'/nls_meta['artifact']['path'],allow_pickle=False) as z:nls=z['fields']
 difference=errors(fields['fine'],nls,c['length'])
 result['nls_comparison']={'reference':'NLS','max_relative_complex_difference':{key:float(value.max()) for key,value in difference.items()},'endpoint_relative_complex_difference':{key:float(value[-1]) for key,value in difference.items()},'nls_source_sha256':nls_meta['artifact']['sha256']}
write_json(out/'result.json',result)
print(result['status'],r,'peak',d['max_peak'],'H2ratio',d['max_h2_ratio'],flush=True)

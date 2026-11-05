#!/usr/bin/env python3
"""Best-case Fourier representation floors on nested, already qualified windows."""
from pathlib import Path
import json
import numpy as np
from run import sha, write_json

root=Path(__file__).parent
source=root/'primary-results.json';primary=json.loads(source.read_text());cases=[]
for r in primary['cases']:
 path=root/r['source_array']
 if sha(path)!=r['artifact']['sha256']:raise ValueError('Source field checksum differs')
 with np.load(path,allow_pickle=False) as z:u=z['fields'];T=z['T']
 n=u.shape[-1];k=2*np.pi*np.fft.fftfreq(n,d=40/n);modes=abs(np.fft.fftfreq(n)*n);p=abs(np.fft.fft(u,axis=-1))**2
 windows={}
 for end in (2.,4.,8.,16.):
  select=T<=end+1e-12;power=p[select];cutoffs={}
  for K in (8,16,32,48,64,128,256,512):
   cutoffs[str(K)]={f'h{s}':float(np.max(np.sqrt((power*(1+k*k)**s)[:,modes>K].sum(axis=-1)/(power*(1+k*k)**s).sum(axis=-1)))) for s in (0,1,2)}
  windows[f'{end:g}']={'cutoffs':cutoffs,'smallest_tested_K_below_5percent_h2':next((K for K in (8,16,32,48,64,128,256,512) if cutoffs[str(K)]['h2']<=.05),None),'maximum_epsilon_peak':float(r['epsilon']*abs(u[select]).max())}
 cases.append({'family':r['family'],'epsilon':r['epsilon'],'source_array':r['source_array'],'source_array_sha256':r['artifact']['sha256'],'windows':windows})
write_json(root/'representation-windows.json',{'source_primary_sha256':sha(source),'source_sha256':sha(__file__),'interpretation':'Maximum over saved times in each interval of relative norm of modes|m|>K. This is the error floor of Fourier truncation alone, not an achieved PINNerror or bound on between-station errors.','cases':cases})
for r in cases:print(r['family'],r['epsilon'],{end:x['smallest_tested_K_below_5percent_h2'] for end,x in r['windows'].items()})

#!/usr/bin/env python3
"""Collect six qualified primary-model cases with explicit mixed-step provenance.

The targeted sideband epsilon=.1 repair is selected by a fixed declared key,
not by its agreement with another model. Failed original results are preserved.
"""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.signal import resample
from run import sha, write_json

root=Path(__file__).parent
uniform_path=root/'results-audit'/'summary.json'
uniform=json.loads(uniform_path.read_text())
repair_path=root/'results-time-audit'/'result.json'
repair=json.loads(repair_path.read_text())
rows=[];flat=[];artifacts=[]
dest=root/'portable-primary';dest.mkdir(exist_ok=True)
for original in uniform['cases']:
 if original['model']!='stability':continue
 repaired=original['family']=='sideband' and original['epsilon']==.1
 r=repair if repaired else original
 if not r['qualified']:raise ValueError('Primary collection requires independently qualified results')
 source=root/('results-time-audit' if repaired else 'results-audit')
 source_file=source/r['artifact']['path']
 if sha(source_file)!=r['artifact']['sha256']:raise ValueError('Source trajectory checksum differs')
 with np.load(source_file,allow_pickle=False) as z:u=z['fields'];T=z['T']
 config=repair['identity']['config'] if repaired else uniform['identity']['config']
 n=u.shape[-1];L=config['length'];k=2*np.pi*np.fft.fftfreq(n,d=L/n);modes=abs(np.fft.fftfreq(n)*n)
 power=abs(np.fft.fft(u,axis=-1))**2;bands={};representation={}
 for threshold in (.25,.5,1.):
  outside=r['epsilon']*abs(k)>=threshold
  bands[str(threshold)]={'maximum_action_fraction':float(np.max(power[:,outside].sum(axis=-1)/power.sum(axis=-1))),'maximum_h2_squared_fraction':float(np.max((power*(1+k*k)**2)[:,outside].sum(axis=-1)/(power*(1+k*k)**2).sum(axis=-1)))}
 for cutoff in (32,48,64,128,256,512):
  representation[str(cutoff)]={f'h{s}_omitted_relative_norm':float(np.max(np.sqrt((power*(1+k*k)**s)[:,modes>cutoff].sum(axis=-1)/(power*(1+k*k)**s).sum(axis=-1)))) for s in (0,1,2)}
 nls=(r['nls_comparison'] if repaired else next(x for x in uniform['comparisons'] if x['family']==r['family'] and x['epsilon']==r['epsilon'] and x['comparison']=='stability_vs_nls'))
 result=dict(r,source_result=str((source/'result.json' if repaired else source/(r['key']+'.json')).relative_to(root)),source_array=str(source_file.relative_to(root)),grid=n,step=config['step'],space_coarse_grid=n//2,time_coarse_step=2*config['step'],targeted_timestep_repair=repaired,carrier_relative_bands=bands,representation_error_floor=representation,nls_comparison=nls)
 rows.append(result)
 flat.append({'family':r['family'],'epsilon':r['epsilon'],'T':16.,'grid':n,'step':config['step'],'qualified':True,'peak':r['diagnostics']['max_peak'],'h2_ratio':r['diagnostics']['max_h2_ratio'],'physical_envelope_peak':r['epsilon']*r['diagnostics']['max_peak'],'grid_h2':r['refinement']['grid']['h2'],'step_h2':r['refinement']['step']['h2'],'action_drift':r['diagnostics']['max_action_drift'],'nls_endpoint_l2_difference':nls['endpoint_relative_complex_difference']['h0'],'carrier_half_band_action_fraction':bands['0.5']['maximum_action_fraction'],'carrier_half_band_h2_squared_fraction':bands['0.5']['maximum_h2_squared_fraction']})
 plot_path=dest/(r['key']+'-plot.npz');plot_grid=512
 np.savez_compressed(plot_path,fields=resample(u,plot_grid,axis=-1).astype(np.complex64),T=T,s=np.arange(plot_grid)*L/plot_grid-L/2)
 artifacts.append({'path':plot_path.name,'sha256':sha(plot_path),'source_array_sha256':r['artifact']['sha256'],'source_array':str(source_file.relative_to(root)),'key':r['key'],'grid':plot_grid,'dtype':'complex64','usage':'Visualization only, not numerical refinement evidence','max_discarded_action_fraction':float(np.max(power[:,modes>=plot_grid/2].sum(axis=-1)/power.sum(axis=-1)))})
result={'primary_model_id':rows[0]['model_id'],'interpretation':'Six individual initial fields/epsilon combinations, not a statistical generalization study; all primary trajectories pass independent numerical checks through T16. Physical model validity and existence remain unproved.','uniform_audit_sha256':sha(uniform_path),'targeted_time_repair_sha256':sha(repair_path),'targeted_repair_note':'The original N4096 step=.0025 high-epsilon sideband failed; rerunning at step=.0003125 with independent .000625 step and N2048 grid comparisons removed the spurious derivative growth. Original failed outputs are retained.','collector_sha256':sha(__file__),'cases':rows}
write_json(root/'primary-results.json',result)
write_json(dest/'primary-results.json',result)
write_json(dest/'plot-manifest.json',{'artifacts':artifacts,'source_summary_sha256':sha(root/'primary-results.json')})
with (dest/'primary-summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
print('collected',len(rows),'qualified primary cases')
for row in flat:print(row)

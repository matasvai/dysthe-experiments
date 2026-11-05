#!/usr/bin/env python3
"""Fixed-period grid and time refinement; no whole-space/domain certification."""
import argparse,copy,json
from pathlib import Path
import numpy as np
from dysthe_pinn.config import read_config
from dysthe_pinn.reference import generate

def compare(a,b,stride):
    with np.load(a,allow_pickle=False) as aa,np.load(b,allow_pickle=False) as bb:
        if not np.allclose(aa['times'],bb['times'],rtol=0,atol=1e-12):raise ValueError('Snapshot times mismatch')
        rows=[]
        for t,u,v in zip(aa['times'],aa['E'],bb['E']):
            common=v[::stride,::stride,::stride]
            if common.shape!=u.shape:raise ValueError('Reference grids are not nested')
            rows.append(dict(t=float(t),relative_l2=float(np.linalg.norm(u-common)/np.linalg.norm(common)),
                             relative_peak_error=float(abs(abs(u).max()-abs(v).max())/abs(v).max())))
    return dict(rows=rows,max_relative_l2=max(r['relative_l2'] for r in rows),max_relative_peak_error=max(r['relative_peak_error'] for r in rows))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();cfg=read_config(args.config);root=args.output;root.mkdir(parents=True,exist_ok=True)
    variants={name:copy.deepcopy(cfg) for name in ['base','grid','grid_dt']}
    for name in ['grid','grid_dt']:variants[name]['reference']['shape']=[2*n for n in cfg['reference']['shape']]
    variants['grid_dt']['reference']['dt']=cfg['reference']['dt']/2
    diagnostics={}
    for name,config in variants.items():
        diagnostics[name]=generate(config,root/name)
    comparisons={'grid':compare(root/'base/reference.npz',root/'grid/reference.npz',2),
                 'time_on_fine_grid':compare(root/'grid/reference.npz',root/'grid_dt/reference.npz',1)}
    passed=all(c['max_relative_l2']<=.001 and c['max_relative_peak_error']<=.005 for c in comparisons.values())
    result=dict(status='pass' if passed else 'fail',gate='provisional fixed-period engineering screen',
                thresholds={'relative_l2':.001,'relative_peak_error':.005},comparisons=comparisons,
                recommended_reference=str((root/'grid_dt/reference.npz').resolve()),
                qualification='No domain enlargement, continuum supremum certification, or whole-space claim. Reference pass does not imply PINN pass.')
    (root/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if not passed:raise SystemExit(2)
if __name__=='__main__':main()

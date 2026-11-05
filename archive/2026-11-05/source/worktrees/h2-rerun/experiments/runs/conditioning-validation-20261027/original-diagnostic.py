"""Matched production-trainer validation ablation through xi=4 on CPU.

Uses packet-008, sideband-008, random-008, seed11; 8 inherited-endpoint slabs,
400updates/slab, no reference labels in fitting. Fresh output directories only.
"""
import argparse,hashlib,json,platform,time
from pathlib import Path
import numpy as np
from scipy.signal import resample
import scipy,torch
import dysthe_experiments.cpu_pilot as pilot
import dysthe_experiments.long_distance as campaign
import dysthe_learning.pilot_models as models
import dysthe_core.reference.spatial as spatial
from dysthe_experiments.long_reference import compare_fields


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    args.run.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    base=json.loads(args.config.read_text());records,inlets=pilot.initial_fields(base);L=base['length'];positions=np.arange(81)*.05
    modules={'pilot':pilot.__file__,'campaign':campaign.__file__,'models':models.__file__,'reference':spatial.__file__};hashes={k:sha(v) for k,v in modules.items()}
    audit={'kind':'CPU validation-only ablation throughxi4, notGPUresults or independenttestresult','config_sha256':sha(args.config),'source_files':modules,'source_sha256_at_start':hashes,'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__,'threads':2},'design':{'cases':['packet-008','sideband-008','random-008'],'seed':11,'variants':['none','inverse_1_plus_k2'],'horizon':4.,'slab_length':.5,'slabs':8,'epochs_per_slab':400,'output_modes':48,'reference_grid':1024,'reference_step':.00125,'initial_condition':'sameinletperpair; eachlaterinletisthepreviouspredictedendpoint','selection':'fullDystheresidualcheckpoints; no solutionlabels'},'references':[],'fits':[]}
    for case_id in audit['design']['cases']:
        j=next(j for j,r in enumerate(records) if r['case_id']==case_id);r=records[j];assert r['split']=='validation';u0=inlets[j];eps=r['physics']['epsilon']
        high,_=pilot.integrate(u0,eps,L,1024,.00125,positions);step,_=pilot.integrate(u0,eps,L,1024,.0025,positions);grid,_=pilot.integrate(u0,eps,L,512,.00125,positions)
        k=2*np.pi*np.fft.fftfreq(1024,d=L/1024);m=np.rint(np.fft.fftfreq(1024)*1024);rh=np.fft.fft(high,axis=-1)/1024
        floor={f'h{o}':float(np.max(np.sqrt(np.sum((1+k*k)**o*abs(rh)**2*(abs(m)>48),axis=-1)/np.sum((1+k*k)**o*abs(rh)**2,axis=-1)))) for o in range(3)}
        audit['references'].append({'case_id':case_id,'epsilon':eps,'step_refinement':compare_fields(step,high,L),'grid_refinement':compare_fields(grid,high,L),'K48_projection_floor':floor})
        for scaling in audit['design']['variants']:
            c=dict(base,seeds=[11],pinn_horizon=4.,pinn_slabs=8,pinn_epochs=3200,pinn_coefficient_scaling=scaling)
            root=args.run/f'{case_id}-{scaling}';root.mkdir();data={'high':high[None],'fields':resample(high,256,axis=-1)[None],'xi':positions}
            t=time.perf_counter();pilot.fit_pinns(c,root,{'records':[r]},data,'cpu',variants=('slabs',));pred,checkpoints=campaign.prediction(c,root,data['fields'][0,0],11,r,4.,'cpu')
            np.savez_compressed(root/'fields.npz',prediction=pred,reference=high,xi=positions)
            summary=json.loads((root/'pinn-summary.json').read_text())[0]
            predhi=resample(pred.astype(np.complex128),1024,axis=-1);dh=np.fft.fft(predhi-high,axis=-1)/1024;h2power=(1+k*k)**2*abs(dh)**2
            bands={label:float(np.sum(h2power[-1,mask])/np.sum(h2power[-1])) for label,mask in [('0to12',abs(m)<=12),('13to24',(abs(m)>=13)&(abs(m)<=24)),('25to48',(abs(m)>=25)&(abs(m)<=48)),('above48',abs(m)>48)]}
            fit={'case_id':case_id,'seed':11,'scaling':scaling,'errors':pilot.metric(pred,high,L),'independent_sobolev_errors':compare_fields(predhi,high,L),'pass_combined_gates':summary['sampled_interval_pass'],'gates':summary['sampled_interval_tolerances'],'full_residual_rms_relative_by_slab':summary['full_residual_rms_relative'],'best_epochs':[torch.load(root/f'pinn-slabs-11-{case_id}'/f'best-{s}.pt',map_location='cpu',weights_only=False)['epoch'] for s in range(8)],'endpoint_h2_error_power_fraction_by_band':bands,'training_seconds':summary['training_seconds'],'total_elapsed_seconds':time.perf_counter()-t,'summary_sha256':sha(root/'pinn-summary.json'),'prediction_reference_sha256':sha(root/'fields.npz'),'checkpoints':checkpoints}
            audit['fits'].append(fit);audit['source_sha256_at_end']={key:sha(v) for key,v in modules.items()};audit['source_files_changed_during_run']=[k for k,v in audit['source_sha256_at_end'].items() if v!=hashes[k]]
            (args.run/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in fit.items() if k not in ('checkpoints','best_epochs')}),flush=True)
if __name__=='__main__':main()

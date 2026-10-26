"""Bounded CPU/A100 experiment for water-wave-dysthe-spatial-v1.

Arrays and checkpoints go in --run (ignored). Only aggregate evidence is
published. Per-instance PINNs and held-out operators have different tasks.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
from scipy.signal import resample as np_resample
from dysthe_core.contracts import require_water_wave_model, validate_splits, validate_case
from dysthe_core.reference.spatial import SpatialDysthe


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_config(path):
    c = json.loads(Path(path).read_text())
    require_water_wave_model(c)
    for key in ('grid', 'reference_grid', 'initial_modes', 'output_modes',
                'groups_per_family', 'train_per_family', 'validation_per_family',
                'operator_epochs', 'operator_batch', 'pinn_epochs', 'pinn_slabs',
                'validation_every', 'threads'):
        if isinstance(c[key], bool) or not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('length', 'reference_step', 'sample_step', 'operator_step',
                'train_horizon', 'horizon', 'pinn_horizon',
                'reference_l2_tolerance', 'reference_h1_tolerance'):
        if isinstance(c[key], bool) or not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if c['grid'] % 4 or c['reference_grid'] % 4:
        raise ValueError('Grids must be multiples of four')
    if not 0 < c['initial_modes'] <= c['output_modes'] < c['grid']/2:
        raise ValueError('Invalid explicit Fourier bandwidth')
    if c['train_per_family'] + c['validation_per_family'] >= c['groups_per_family']:
        raise ValueError('A nonempty independent test split is required')
    if c['operator_step'] > c['train_horizon'] or c['train_horizon'] >= c['horizon']:
        raise ValueError('Require local step <= training horizon < test horizon')
    if c['pinn_horizon'] > c['horizon'] or c['pinn_epochs'] < c['pinn_slabs']:
        raise ValueError('PINN horizon must fit reference data and each slab needs an epoch')
    for key in ('horizon','train_horizon','operator_step','pinn_horizon'):
        ratio=c[key]/c['sample_step']
        if abs(ratio-round(ratio))>1e-9: raise ValueError(f'{key} must lie on the saved reference grid')
    if c['pinn_epochs'] % c['pinn_slabs']: raise ValueError('PINN budget must divide into slabs')

    def require_multiple(value, spacing, name):
        ratio = value / spacing
        if not np.isclose(ratio, round(ratio), rtol=1e-10, atol=1e-10):
            raise ValueError(f'{name} must be an integer multiple of its sampling/step interval')

    require_multiple(c['horizon'], c['operator_step'], 'rollout horizon')
    require_multiple(c['train_horizon'], c['operator_step'], 'train_horizon')
    require_multiple(c['pinn_horizon'], c['pinn_slabs'] * c['sample_step'], 'PINN slab horizon')
    # Metrics index these named stations; rounding must never move them.
    for xi in (.5, 1., 2., 4.):
        if xi <= c['horizon']:
            require_multiple(xi, c['sample_step'], f'reference metric station {xi}')
            require_multiple(xi, c['operator_step'], f'operator metric station {xi}')
    return c


def initial_fields(c):
    rng = np.random.default_rng(c['seed'])
    n, length = c['grid'], c['length']
    tau = np.arange(n)*length/n-length/2
    mode = np.fft.fftfreq(n)*n
    records, fields = [], []
    for family in ('packet', 'sideband', 'random'):
        for j in range(c['groups_per_family']):
            phase = rng.uniform(-np.pi,np.pi)
            epsilon = rng.uniform(.02,.05)
            a = rng.uniform(.4,.7)
            params = {'amplitude_parameter':float(a), 'phase':float(phase)}
            if family == 'packet':
                width, chirp, center = rng.uniform(2.5,4.),rng.uniform(-.15,.15),rng.uniform(-4.,4.)
                u = sum(a*np.exp(-.5*((tau-center+q*length)/width)**2 + 1j*chirp*((tau-center+q*length)/width)**2) for q in (-1,0,1))
                params.update(width=float(width),chirp=float(chirp),center=float(center))
            elif family == 'sideband':
                m = int(rng.choice([3,4,5]))
                p = rng.uniform(-np.pi,np.pi)
                u = a*(1 + .06*np.exp(2j*np.pi*m*tau/length) + .04*np.exp(-2j*np.pi*m*tau/length+1j*p))
                params.update(sideband_mode=m,sideband_phase=float(p))
            else:
                coeff = rng.normal(size=13)+1j*rng.normal(size=13)
                coeff *= np.exp(-.2*np.arange(-6,7)**2)
                u = np.exp(2j*np.pi*tau[:,None]*np.arange(-6,7)/length)@coeff
                u *= a/np.max(abs(u))
                params['coefficients'] = [[float(z.real),float(z.imag)] for z in coeff]
            u = np.fft.ifft(np.fft.fft(u*np.exp(1j*phase))*(abs(mode)<=c['initial_modes']))
            group = f'{family}-{j:03d}'
            split = 'train' if j<c['train_per_family'] else 'validation' if j<c['train_per_family']+c['validation_per_family'] else 'test'
            records.append(dict(case_id=group,initial_condition_id=group,split=split,family=family,
                schema_version=2,model_id=c['model_id'],physical_system=c['physical_system'],
                boundary='periodic',axis_order=['xi','tau'],field_encoding='complex128',shape=[c['reference_grid']],lengths=[length],
                propagation_positions=np.arange(round(c['horizon']/c['sample_step'])+1).astype(float).__mul__(c['sample_step']).tolist(),
                physics={'epsilon':float(epsilon)},parameters=params))
            fields.append(u)
    validate_splits(records)
    for r in records: validate_case(r)
    return records,np.array(fields)


def integrate(u0, epsilon, length, n, h, positions):
    m = SpatialDysthe(n,length,epsilon)
    u = m.project(np_resample(u0,n))
    v = np.fft.fft(u); fields=[u]; previous=0.
    start=time.perf_counter()
    for xi in positions[1:]:
        steps=max(1,round((xi-previous)/h))
        step=m.stepper((xi-previous)/steps)
        for _ in range(steps): v=step(v)
        fields.append(np.fft.ifft(v)); previous=xi
    out=np.array(fields)
    if not np.isfinite(out).all(): raise RuntimeError('Reference became nonfinite')
    return out,time.perf_counter()-start


def metric(pred, truth, length):
    pred=np.asarray(pred); truth=np.asarray(truth)
    if pred.shape[-1] != truth.shape[-1]: pred=np_resample(pred,truth.shape[-1],axis=-1)
    n=truth.shape[-1]; k=2*np.pi*np.fft.fftfreq(n,d=length/n)
    v=np.fft.fft(truth,axis=-1); d=np.fft.fft(pred-truth,axis=-1)
    norm=lambda z:np.sqrt(np.sum(abs(z)**2,axis=-1))
    l2=norm(d)/norm(v)
    h1=norm(d*np.sqrt(1+k*k))/norm(v*np.sqrt(1+k*k))
    peak=abs(np.max(abs(pred),axis=-1)-np.max(abs(truth),axis=-1))/np.max(abs(truth),axis=-1)
    return {'l2':float(np.max(l2)),'h1':float(np.max(h1)),'peak':float(np.max(peak))}


def prepare(c,run):
    records,u0=initial_fields(c)
    positions=np.array(records[0]['propagation_positions'])
    high=[]; diagnostics=[]; costs=[]; baselines=[]
    for r,u in zip(records,u0):
        eps=r['physics']['epsilon']; n=c['reference_grid']; h=c['reference_step']; L=c['length']
        fine,sec=integrate(u,eps,L,n,h,positions)
        time_coarse,_=integrate(u,eps,L,n,2*h,positions)
        grid_coarse,_=integrate(u,eps,L,n//2,h,positions)
        t_err=metric(time_coarse,fine,L); g_err=metric(grid_coarse,fine,L)
        if max(t_err['l2'],g_err['l2'])>c['reference_l2_tolerance'] or max(t_err['h1'],g_err['h1'])>c['reference_h1_tolerance']:
            raise RuntimeError(f'Unqualified reference {r["case_id"]}: {t_err}, {g_err}')
        coarse,tcoarse=integrate(u,eps,L,c['grid'],.01,positions)
        nls,tnls=integrate(u,0.,L,c['grid'],.01,positions)
        high.append(fine)
        diagnostics.append({'case_id':r['case_id'],'time_refinement':t_err,'grid_refinement':g_err})
        costs.append({'case_id':r['case_id'],'reference_seconds':sec,'coarse_seconds':tcoarse,'nls_seconds':tnls})
        for method,arr in [('numerical',coarse),('nls',nls)]:
            for x in (.5,1.,2.,4.):
                j=round(x/c['sample_step'])
                if j<len(positions): baselines.append({'case_id':r['case_id'],'split':r['split'],'method':method,'xi':x,**metric(arr[j],fine[j],L)})
        print('reference',r['case_id'],f"L2 refinement {max(t_err['l2'],g_err['l2']):.2e}",flush=True)
    high=np.array(high)
    fields=np_resample(high,c['grid'],axis=-1)
    np.savez_compressed(run/'dataset.npz',fields=fields,high=high,xi=positions)
    manifest={'model_id':c['model_id'],'physical_system':c['physical_system'],'records':records,'sha256':digest(run/'dataset.npz'),
              'normalization':'none; physical complex envelope','reference_checks':diagnostics,'costs':costs,'baselines':baselines}
    write_json(run/'manifest.json',manifest)
    return manifest


def sync(device):
    import torch
    if str(device).startswith('cuda'): torch.cuda.synchronize()


def save_checkpoint(path, model, optimizer, epoch, best, elapsed, config, history):
    import torch
    tmp=path.with_suffix('.tmp')
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'best':best,
                'elapsed':elapsed,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'config':config,'history':history},tmp)
    tmp.replace(path)


def restore_checkpoint(path,model,optimizer,config,device):
    import torch
    p=torch.load(path,map_location=device,weights_only=False)
    if p['config']!=config: raise ValueError('Resume configuration differs; use a fresh run directory')
    model.load_state_dict(p['model']); optimizer.load_state_dict(p['optimizer']); torch.set_rng_state(p['rng'].cpu())
    if p.get('cuda_rng') is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all([s.cpu() for s in p['cuda_rng']])
    return p


def continuous_residual(model,u,eps,s,c):
    import torch
    from dysthe_core.torch_spatial import resample,rhs
    y,dy=torch.func.jvp(lambda z:model(u,eps,z),(s,),(torch.ones_like(s),))
    q=1
    while q<=6*c['output_modes']: q*=2
    yf,dyf=resample(y,q),resample(dy,q)
    f=rhs(yf,eps,c['length'])
    loss=(abs(dyf-f)**2).mean()/(abs(f.detach())**2).mean().clamp_min(1e-8)
    return loss,y


def rollout(model,u0,eps,steps,c):
    import torch
    u=u0; out=[u]
    with torch.no_grad():
        for _ in range(steps):
            u=model(u,eps,torch.full((u.shape[0],),c['operator_step'],device=u.device,dtype=u.real.dtype))
            out.append(u)
    return torch.stack(out,dim=1)


def fit_operator(c,run,manifest,data,method,seed,device):
    import torch
    from dysthe_learning.pilot_models import ContinuousFNO
    torch.manual_seed(seed)
    model=ContinuousFNO(length=c['length'],width=c['operator_width'],modes=c['operator_modes'],depth=c['operator_depth'],max_mode=c['output_modes']).float().to(device)
    opt=torch.optim.Adam(model.parameters(),lr=2e-3)
    idx=[j for j,r in enumerate(manifest['records']) if r['split']=='train']
    val=[j for j,r in enumerate(manifest['records']) if r['split']=='validation']
    fields=torch.tensor(data['fields'],dtype=torch.complex64,device=device)
    eps=torch.tensor([r['physics']['epsilon'] for r in manifest['records']],device=device)
    starts=round((c['train_horizon']-c['operator_step'])/c['sample_step'])+1
    stride=round(c['operator_step']/c['sample_step'])
    root=run/f'{method}-{seed}'; root.mkdir(exist_ok=True)
    cfg={'campaign':c,'method':method,'seed':seed}
    first=0; best=float('inf'); elapsed=0.; history=[]
    if (root/'last.pt').exists():
        p=restore_checkpoint(root/'last.pt',model,opt,cfg,device)
        first=p['epoch'];best=p['best'];elapsed=p['elapsed'];history=p['history']
    sync(device); start=time.perf_counter()
    for epoch in range(first,c['operator_epochs']):
        gi=torch.tensor(idx,device=device)[torch.randint(len(idx),(c['operator_batch'],),device=device)]
        t=torch.randint(starts,(c['operator_batch'],),device=device)
        offset=torch.randint(1,stride+1,(c['operator_batch'],),device=device)
        s=offset.to(fields.real.dtype)*c['sample_step']
        u,target=fields[gi,t],fields[gi,t+offset]
        if method=='pino':
            residual,pred=continuous_residual(model,u,eps[gi],s,c)
        else:
            pred=model(u,eps[gi],s); residual=torch.zeros((),device=device)
        diff=torch.fft.fft(pred-target); truth=torch.fft.fft(target)
        k=2*np.pi*torch.fft.fftfreq(c['grid'],d=c['length']/c['grid'],device=device)
        loss_data=(abs(diff)**2).mean()/(abs(truth)**2).mean()
        loss_h1=(abs(diff)**2*(1+k*k)).mean()/(abs(truth)**2*(1+k*k)).mean()
        loss=loss_data+.1*loss_h1+(c['physics_weight'] if method=='pino' else 0.)*residual
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if (epoch+1)%c['validation_every']==0 or epoch+1==c['operator_epochs']:
            pred=rollout(model,fields[val,0],eps[val],round(c['train_horizon']/c['operator_step']),c)[:,-1].cpu().numpy()
            score=metric(pred,data['fields'][val,round(c['train_horizon']/c['sample_step'])],c['length'])['h1']
            sync(device); seconds=elapsed+time.perf_counter()-start
            history.append({'epoch':epoch+1,'data_loss':float(loss_data.detach()),'h1_loss':float(loss_h1.detach()),'physics_loss':float(residual.detach()),'validation_max_h1':score,'seconds':seconds})
            if score<best:
                best=score;save_checkpoint(root/'best.pt',model,opt,epoch+1,best,seconds,cfg,history)
            save_checkpoint(root/'last.pt',model,opt,epoch+1,best,seconds,cfg,history)
            print(method,seed,'epoch',epoch+1,'valH1',round(score,5),flush=True)
    p=torch.load(root/'best.pt',map_location=device,weights_only=False);model.load_state_dict(p['model'])
    result={'method':method,'seed':seed,'best_epoch':p['epoch'],'best_validation_max_h1':p['best'],
            'training_seconds':history[-1]['seconds'],'history':history,'checkpoint_sha256':digest(root/'best.pt'),'metrics':[]}
    for split in ('validation','test'):
        ids=[j for j,r in enumerate(manifest['records']) if r['split']==split]
        # Warmup is excluded; latency includes all autonomous steps and no reference resets.
        rollout(model,fields[ids,0],eps[ids],1,c);sync(device);tic=time.perf_counter()
        pred=rollout(model,fields[ids,0],eps[ids],round(c['horizon']/c['operator_step']),c)
        sync(device); seconds=time.perf_counter()-tic
        arr=pred.cpu().numpy()
        for i,j in enumerate(ids):
            for xi in (.5,1.,2.,4.):
                if xi<=c['horizon']:
                    pp=arr[i,round(xi/c['operator_step'])]; tt=data['high'][j,round(xi/c['sample_step'])]
                    result['metrics'].append({'case_id':manifest['records'][j]['case_id'],'split':split,'xi':xi,**metric(pp,tt,c['length']),
                      'rollout_seconds_per_case':seconds/len(ids)})
    write_json(root/'summary.json',result)
    return result


def fit_pinns(c,run,manifest,data,device,variants=('scratch','nls_warm','slabs')):
    import torch
    from dysthe_learning.pilot_models import SpectralPINN
    from dysthe_core.torch_spatial import resample,rhs
    all_results=[]
    # Per-instance solvers: each inlet gets its own optimization, including test inlets.
    for variant in variants:
        for seed in c['seeds']:
            for case,r in enumerate(manifest['records']):
                if r['split']=='train': continue
                sync(device);invocation_start=time.perf_counter()
                if str(device).startswith('cuda'): torch.cuda.reset_peak_memory_stats(device)
                resumed_from_checkpoint=False
                root=run/f'pinn-{variant}-{seed}-{r["case_id"]}';root.mkdir(exist_ok=True)
                torch.manual_seed(seed)
                u=torch.tensor(data['fields'][case,0],dtype=torch.complex64,device=device)
                eps=torch.tensor(r['physics']['epsilon'],dtype=torch.float32,device=device)
                slabs=c['pinn_slabs'] if variant=='slabs' else 1
                horizon=c['pinn_horizon']/slabs;epochs=c['pinn_epochs']//slabs
                predictions=[];total_seconds=0.;histories=[];residual_audits=[]
                for slab in range(slabs):
                    model=SpectralPINN(u,c['length'],max_mode=c['output_modes'],width=c.get('pinn_width',32),depth=c.get('pinn_depth',2)).to(device)
                    opt=torch.optim.Adam(model.parameters(),lr=2e-3)
                    cfg={'campaign':c,'method':variant,'seed':seed,'case_id':r['case_id'],'slab':slab}
                    last=root/f'last-{slab}.pt';best_path=root/f'best-{slab}.pt'
                    first=0;best=float('inf');elapsed=0.;history=[]
                    if last.exists():
                        resumed_from_checkpoint=True
                        p=restore_checkpoint(last,model,opt,cfg,device)
                        first=p['epoch'];best=p['best'];elapsed=p['elapsed'];history=p['history']
                    q=1
                    while q<=6*c['output_modes']:q*=2
                    denominator=(abs(rhs(resample(u,q),eps,c['length']))**2).mean().clamp_min(1e-8)
                    audit_s=torch.linspace(0,horizon,65,device=device)
                    sync(device);tic=time.perf_counter()
                    for epoch in range(first,epochs):
                        s=torch.rand(c.get('pinn_collocation_slices',16),device=device)*horizon
                        y,dy=torch.func.jvp(model,(s,),(torch.ones_like(s),))
                        active_eps=eps*0 if variant=='nls_warm' and epoch<epochs//4 else eps
                        loss=(abs(resample(dy,q)-rhs(resample(y,q),active_eps,c['length']))**2).mean()/denominator
                        if not torch.isfinite(loss):raise RuntimeError('Nonfinite PINN loss')
                        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                        if (epoch+1)%min(c['validation_every'],epochs)==0 or epoch+1==epochs:
                            # Fixed residual validation slices, never numerical solution labels.
                            yv,dv=torch.func.jvp(model,(audit_s,),(torch.ones_like(audit_s),))
                            score=float(((abs(resample(dv,q)-rhs(resample(yv,q),eps,c['length']))**2).mean()/denominator).detach())
                            sync(device);seconds=elapsed+time.perf_counter()-tic
                            history.append({'epoch':epoch+1,'training_loss':float(loss.detach()),'full_dysthe_residual_validation':score,'seconds':seconds,'training_epsilon':float(active_eps)})
                            if score<best:
                                best=score;save_checkpoint(best_path,model,opt,epoch+1,best,seconds,cfg,history)
                            save_checkpoint(last,model,opt,epoch+1,best,seconds,cfg,history)
                    ck=torch.load(best_path,map_location=device,weights_only=False);model.load_state_dict(ck['model'])
                    total_seconds+=history[-1]['seconds'];histories.append(history)
                    ss=torch.linspace(0,horizon,round(horizon/c['sample_step'])+1,device=device)
                    with torch.no_grad():out=model(ss)
                    predictions.append(out.detach().cpu().numpy()[0 if slab==0 else 1:])
                    # Endpoint is the prediction, without a reset to reference data.
                    u=out[-1].detach()
                    dense_s=torch.linspace(0,horizon,129,device=device)
                    yv,dv=torch.func.jvp(model,(dense_s,),(torch.ones_like(dense_s),))
                    fine_defect=resample(dv,2*q)-rhs(resample(yv,2*q),eps,c['length'])
                    residual_audits.append(float(torch.sqrt((abs(fine_defect)**2).mean()/denominator).detach()))
                pred=np.concatenate(predictions,axis=0)
                interval_errors=metric(pred,data['high'][case,:len(pred)],c['length'])
                interval_tolerances={'l2':0.01,'h1':0.05,'peak':0.05}
                result={'method':f'pinn_{variant}','seed':seed,'case_id':r['case_id'],'split':r['split'],'training_seconds':total_seconds,
                    'history':histories,'full_residual_rms_relative':residual_audits,'metrics':[],
                    'sampled_interval_errors':interval_errors,'sampled_interval_tolerances':interval_tolerances,
                    'sampled_interval_pass':all(interval_errors[key]<=bound for key,bound in interval_tolerances.items()),
                    'sampled_station_count':len(pred),'resumed_from_checkpoint':resumed_from_checkpoint,
                    'checkpoints':{p.name:digest(p) for p in root.glob('best-*.pt')}}
                for xi in sorted(set((.5,1.,2.,c['pinn_horizon']))):
                    if xi<=c['pinn_horizon']:
                        j=round(xi/c['sample_step']);result['metrics'].append({'xi':xi,**metric(pred[j],data['high'][case,j],c['length'])})
                sync(device)
                result['gpu_peak_allocated_bytes']=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None
                # This invocation includes per-case setup, checkpoint loading,
                # prediction and audits; accumulated training_seconds has a
                # different scope and can include earlier resumed invocations.
                result['invocation_elapsed_seconds']=time.perf_counter()-invocation_start
                write_json(root/'summary.json',result);all_results.append(result)
                print('PINN',variant,seed,r['case_id'],'endpoint L2',result['metrics'][-1]['l2'],flush=True)
    write_json(run/'pinn-summary.json',all_results)


def diagnostics(c,run,manifest,data):
    from dysthe_core.reference.validation import full_rhs_defect
    rows=[]
    for j,r in enumerate(manifest['records']):
        if r['split']=='train':continue
        eps=r['physics']['epsilon'];n=c['reference_grid'];L=c['length']
        m=SpatialDysthe(n,L,eps)
        u0=data['high'][j,0];action0=L*np.mean(abs(u0)**2)
        for xi in (0.,.5,1.,2.,4.):
            if xi>c['horizon']:continue
            u=data['high'][j,round(xi/c['sample_step'])];v=np.fft.fft(u)/n;k=m.k;power=abs(v)**2
            projection=np.fft.ifft(np.fft.fft(u)*(abs(np.fft.fftfreq(n)*n)<=c['output_modes']))
            rows.append({'case_id':r['case_id'],'split':r['split'],'xi':xi,
                'linf':float(max(abs(u))),'h1':float(np.sqrt(L*np.sum((1+k*k)*power))),
                'h2':float(np.sqrt(L*np.sum((1+k*k)**2*power))),
                'max_effective_steepness':float(eps*max(abs(u))),
                'epsilon_k_rms':float(eps*np.sqrt(np.sum(k*k*power)/sum(power))),
                'reference_edge_action_fraction':float(sum(power[abs(np.fft.fftfreq(n)*n)>.8*(n/4)])/sum(power)),
                'action_relative_drift':float(abs(L*np.mean(abs(u)**2)/action0-1)),
                'learner_bandwidth_floor':metric(projection,u,L),
                'discarded_rhs_relative':full_rhs_defect(m,u)['discarded_rhs_relative_l2']})
    write_json(run/'regularity-diagnostics.json',rows)
    # A longer numerical stress test is separately qualified; it is not a learned forecast.
    long=[];positions=np.arange(0.,16.0001,.5)
    for j,r in enumerate(manifest['records']):
        if r['split']!='validation':continue
        eps=r['physics']['epsilon'];L=c['length'];u=data['high'][j,0]
        fine,sec=integrate(u,eps,L,1024,.00125,positions)
        coarse,_=integrate(u,eps,L,512,.0025,positions)
        err=metric(coarse,fine,L)
        m=SpatialDysthe(1024,L,eps)
        for x,f in zip(positions,fine):
            v=np.fft.fft(f)/len(f);power=abs(v)**2
            long.append({'case_id':r['case_id'],'xi':float(x),'linf':float(max(abs(f))),
              'h1':float(np.sqrt(L*np.sum((1+m.k*m.k)*power))),
              'h2':float(np.sqrt(L*np.sum((1+m.k*m.k)**2*power))),
              'action_relative_drift':float(abs(np.mean(abs(f)**2)/np.mean(abs(u)**2)-1)),
              'max_effective_steepness':float(eps*max(abs(f))),
              'epsilon_k_rms':float(eps*np.sqrt(np.sum(m.k*m.k*power)/sum(power))),
              'refinement_max_l2':err['l2'],'refinement_max_h1':err['h1'],
              'qualified':err['l2']<c['reference_l2_tolerance'] and err['h1']<c['reference_h1_tolerance']})
        print('long numerical',r['case_id'],'refinement',err,flush=True)
    write_json(run/'long-numerical-diagnostics.json',long)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--stage',choices=['prepare','operators','pinns','diagnostics','all'],default='all')
    p.add_argument('--methods',nargs='+',choices=['fno','pino'],default=['fno','pino'])
    p.add_argument('--pinn-variants',nargs='+',choices=['scratch','nls_warm','slabs'],default=['scratch','nls_warm','slabs'])
    args=p.parse_args();c=read_config(args.config);run=args.run;run.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
    torch.set_num_threads(c['threads']);torch.use_deterministic_algorithms(True)
    if (run/'config.json').exists() and json.loads((run/'config.json').read_text())!=c: raise ValueError('Run configuration differs')
    write_json(run/'config.json',c)
    write_json(run/f'environment-{args.device}.json',{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
        'device':args.device,'gpu':torch.cuda.get_device_name() if args.device=='cuda' else None,'threads':c['threads'],'platform':platform.platform(),'processor':platform.processor(),
        'source_sha256':digest(__file__)})
    if args.stage in ('prepare','all') and not (run/'manifest.json').exists(): prepare(c,run)
    manifest=json.loads((run/'manifest.json').read_text())
    require_water_wave_model(manifest);validate_splits(manifest['records'])
    for record in manifest['records']: validate_case(record)
    if digest(run/'dataset.npz')!=manifest['sha256']: raise RuntimeError('Dataset checksum mismatch')
    with np.load(run/'dataset.npz') as d: data={key:d[key] for key in d.files}
    if args.stage in ('operators','all'):
        for method in args.methods:
            for seed in c['seeds']: fit_operator(c,run,manifest,data,method,seed,args.device)
    if args.stage in ('pinns','all'): fit_pinns(c,run,manifest,data,args.device,args.pinn_variants)
    if args.stage in ('diagnostics','all'): diagnostics(c,run,manifest,data)


if __name__=='__main__': main()

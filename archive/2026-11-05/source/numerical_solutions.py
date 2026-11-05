"""Reproduce the linear 3D experiments and figures in report/dysthe_report.tex."""
from pathlib import Path
import os, time, json, csv
os.environ.setdefault('MPLCONFIGDIR', '/tmp/dysthe-mpl')
import numpy as np
import torch
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
FIG = ROOT / 'report' / 'figures'
OUT.mkdir(exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
torch.set_default_dtype(torch.float64)
SIGMA = 3.0
Q = np.pi / 6  # eight frequency bins on the length-96 box
TIMES = np.linspace(0, 6, 25)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 130})

def grid(n=96, length=96.):
    x = torch.arange(n) * (length/n) - length/2
    k = 2*np.pi*torch.fft.fftfreq(n, d=length/n)
    return x, (k[:,None,None], k[None,:,None], k[None,None,:])

def initial(x, q=(0,0,0), kind='gaussian'):
    X,Y,Z=x[:,None,None],x[None,:,None],x[None,None,:]
    f = torch.exp(-(X*X+Y*Y+Z*Z)/(2*SIGMA**2)).to(torch.complex128)
    if kind == 'two_packets':
        f = torch.exp(-((X+4)**2+Y**2+Z**2)/(2*SIGMA**2)) + \
            .7j*torch.exp(-((X-4)**2+Y**2+Z**2)/(2*SIGMA**2))
    return f*torch.exp(1j*(q[0]*X+q[1]*Y+q[2]*Z))

def symbol(k, e2, e1=1.):
    kx,ky,kz=k
    return (1+e1*kz)*(kx*kx+ky*ky)-kz*kz+e2*kz**3

def solution(n, length, e2, q, t, kind='gaussian'):
    x,k=grid(n,length)
    f=initial(x,q,kind)
    return torch.fft.ifftn(torch.fft.fftn(f)*torch.exp(-1j*t*symbol(k,e2)))

def diagnostics(u,x):
    h=float(x[1]-x[0]); a=u.abs(); mass=float((a*a).sum()*h**3)
    marg=[(a*a).sum(dim=tuple(j for j in range(3) if j!=i))*h*h for i in range(3)]
    mean=[float((m*x).sum()*h/mass) for m in marg]
    width=[float(torch.sqrt((m*(x-c)**2).sum()*h/mass)) for m,c in zip(marg,mean)]
    edge=(x.abs()>=.4*(float(x[-1]-x[0])+h))
    shell=edge[:,None,None] | edge[None,:,None] | edge[None,None,:]
    return dict(l2=mass**.5, linf=float(a.max()), l4=float((a**4).sum()*h**3)**.25,
                mean=mean,width=width,edge_mass=float((a*a)[shell].sum()*h**3/mass))

def gaussian_reference(point,t,e2,q):
    # Exact analytic integration in kx,ky; independent adaptive quadrature in kz.
    xx,yy,zz=point; qx,qy,qz=q; s=SIGMA
    def integrand(k):
        d=s*s+2j*t*(1+k)
        return s**3/np.sqrt(2*np.pi)/d * np.exp(-s*s*(k-qz)**2/2 +
             1j*k*zz+1j*t*k*k-1j*t*e2*k**3-s*s*(qx*qx+qy*qy)/2 +
             ((s*s*qx+1j*xx)**2+(s*s*qy+1j*yy)**2)/(2*d))
    re,er=quad(lambda k: integrand(k).real,-5,5,epsabs=1e-12,epsrel=1e-12,limit=300)
    im,ei=quad(lambda k: integrand(k).imag,-5,5,epsabs=1e-12,epsrel=1e-12,limit=300)
    return re+1j*im,er+ei

def main():
    start=time.perf_counter(); records=[]; finals={}; peak_errors=[]
    x,k=grid(); cases={'unshifted':(0,0,0),'x_shift':(Q,0,0),'z_shift':(0,0,Q)}
    for e2 in [1.,-1/3,0.]:
        omega=symbol(k,e2)
        for name,q in cases.items():
            f=initial(x,q); fh=torch.fft.fftn(f)
            for t in TIMES:
                u=torch.fft.ifftn(fh*torch.exp(-1j*float(t)*omega))
                d=diagnostics(u,x)
                records.append(dict(epsilon2=e2,case=name,t=float(t),**d))
            key=f'{e2}:{name}'; finals[key]=u.numpy().copy()
            idx=np.unravel_index(np.argmax(np.abs(finals[key])),u.shape)
            for point in [(0.,0.,0.),(3.,-2.,5.),tuple(float(x[i]) for i in idx)]:
                ij=tuple(int(round(v+48)) for v in point)
                ref,err=gaussian_reference(point,6,e2,q)
                peak_errors.append(dict(epsilon2=e2,case=name,point=point,
                                        absolute_error=float(abs(complex(u[ij])-ref)),quad_error=err))
            print(f'Finished epsilon2={e2:g}, {name}',flush=True)
    # Symmetry and non-Gaussian example.
    uy=solution(96,96,1,(0,Q,0),6).numpy()
    symmetry=float(np.max(np.abs(uy-np.transpose(finals['1.0:x_shift'],(1,0,2)))))
    generic=solution(96,96,1,(0,0,0),6,'two_packets').numpy()
    np.savez_compressed(OUT/'solution_snapshots.npz',x=x.numpy(),
                       **{key.replace(':','_'):v for key,v in finals.items()},two_packets=generic)
    # Nested-grid comparison: same box, independently sampled identical initial function.
    convergence=[]
    for e2 in [1.,-1/3,0.]:
        fine=solution(192,96,e2,(0,0,Q),6)
        for n in [32,64,96]:
            coarse=solution(n,96,e2,(0,0,Q),6); target=fine[::192//n,::192//n,::192//n]
            convergence.append(dict(epsilon2=e2,n=n,h=96/n,
                relative_l2=float(torch.linalg.vector_norm(coarse-target)/torch.linalg.vector_norm(target)),
                sampled_linf_relative_error=float(abs(coarse.abs().max()-fine.abs().max())/fine.abs().max())))
        del fine
    # Domain enlargement, fixed spacing 0.5; compare on common [-24,24)^3.
    domains=[]
    for e2 in [1.,-1/3,0.]:
        ref=solution(192,96,e2,(0,0,Q),6)[48:144,48:144,48:144].clone()
        for length,n in [(48,96),(72,144)]:
            u=solution(n,length,e2,(0,0,Q),6); cut=(n-96)//2
            sub=u[cut:cut+96,cut:cut+96,cut:cut+96]
            domains.append(dict(epsilon2=e2,length=length,
                common_cube_relative_l2=float(torch.linalg.vector_norm(sub-ref)/torch.linalg.vector_norm(ref))))
    large_box=[]; peak_sampling=[]
    for e2 in [1.,-1/3,0.]:
        for name,q in cases.items():
            big=solution(144,144,e2,q,6)[24:120,24:120,24:120]
            base=torch.from_numpy(finals[f'{e2}:{name}'])
            large_box.append(dict(epsilon2=e2,case=name,
                relative_l2=float(torch.linalg.vector_norm(base-big)/torch.linalg.vector_norm(big))))
            fine=solution(192,96,e2,q,6)
            peak_sampling.append(dict(epsilon2=e2,case=name,fine_peak=float(fine.abs().max()),
                relative_difference=float(abs(base.abs().max()-fine.abs().max())/fine.abs().max())))
    # Phase check is evaluated against an independently specified plane-wave frequency.
    n=24; xx,kk=grid(n,2*np.pi); X,Y,Z=xx[:,None,None],xx[None,:,None],xx[None,None,:]
    wave=torch.exp(1j*(X+2*Y-Z)); tt=.73
    evolved=torch.fft.ifftn(torch.fft.fftn(wave)*torch.exp(-1j*tt*symbol(kk,1.)))
    plane_error=float((evolved-wave*np.exp(2j*tt)).abs().max())  # omega(1,2,-1)=-2
    u=torch.from_numpy(finals['1.0:z_shift']); back=torch.fft.ifftn(torch.fft.fftn(u)*torch.exp(6j*symbol(k,1.)))
    reverse=float(torch.linalg.vector_norm(back-initial(x,(0,0,Q)))/torch.linalg.vector_norm(initial(x,(0,0,Q))))
    mass_drift=max(abs(r['l2']/records[0]['l2']-1) for r in records)
    checks=dict(plane_wave_max_error=plane_error,reversibility_relative_l2=reverse,
                xy_symmetry_max_error=symmetry,l2_max_relative_drift=mass_drift,
                quadrature_max_absolute_error=max(p['absolute_error'] for p in peak_errors))
    assert max(plane_error,reverse,symmetry,mass_drift)<1e-10,checks
    # Presentation figures.
    colors=['#236f9c','#d47d29','#923953']
    fig,axes=plt.subplots(1,3,figsize=(10,3),layout='constrained')
    for ax,e2 in zip(axes,[1.,-1/3,0.]):
        for (name,q),color in zip(cases.items(),colors):
            rr=[r for r in records if r['epsilon2']==e2 and r['case']==name]
            ax.plot(TIMES,[r['linf'] for r in rr],label=name.replace('_',' '),color=color,lw=2)
        ax.set(xlabel='Evolution time $t$',ylabel=r'Sampled $\|E\|_\infty$',title=fr'$\varepsilon_2={e2:.3g}$',ylim=(0,1.04)); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8); fig.savefig(FIG/'peak_decay.pdf'); plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(9,7.3),layout='constrained')
    sel=np.where(abs(x.numpy())<=20)[0]; extent=[-20,20,-20,20]
    for row,t in enumerate([0.,3.,6.]):
        for col,(name,q) in enumerate(cases.items()):
            u=solution(96,96,1,q,t).numpy()
            vals=abs(u[np.ix_(sel,[48],sel)][:,0,:]).T
            im=axes[row,col].imshow(vals,origin='lower',extent=[-20.5,20.5,-20.5,20.5],vmin=0,vmax=1,cmap='magma')
            axes[row,col].set(xlabel='$x$',ylabel='$z$',title=f'{name.replace("_"," ")}, $t={t:g}$')
    fig.colorbar(im,ax=axes,shrink=.65,label=r'$|E(x,0,z,t)|$');fig.savefig(FIG/'slices.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10,3),layout='constrained')
    for ax,i in zip(axes,range(3)):
        for name,color in zip(cases,colors):
            rr=[r for r in records if r['epsilon2']==1 and r['case']==name]
            ax.plot(TIMES,[r['width'][i] for r in rr],label=name.replace('_',' '),color=color,lw=2)
        ax.set(xlabel='$t$',ylabel='Intensity standard deviation',title=f'{"xyz"[i]} direction');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.savefig(FIG/'widths.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,3),layout='constrained')
    for ax,t in zip(axes,[0,6]):
        u=initial(x,kind='two_packets').numpy() if t==0 else generic
        im=ax.imshow(abs(u[np.ix_(sel,[48],sel)][:,0,:]).T,origin='lower',extent=[-20.5,20.5,-20.5,20.5],vmin=0,vmax=1,cmap='magma')
        ax.set(xlabel='$x$',ylabel='$z$',title=f'Two packets, $t={t}$')
    fig.colorbar(im,ax=axes,label='$|E|$');fig.savefig(FIG/'generic.pdf');plt.close(fig)
    with open(OUT/'norm_history.csv','w') as f:
        fields=['epsilon2','case','t','l2','linf','l4','edge_mass']+[f'{v}_{a}' for v in ['mean','width'] for a in 'xyz']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in records:
            writer.writerow({**{key:r[key] for key in fields if key in r},**{f'{v}_{a}':r[v][i] for v in ['mean','width'] for i,a in enumerate('xyz')}})
    result=dict(config=dict(n=96,length=96,sigma=SIGMA,q=Q,epsilon1=1,times=TIMES.tolist(),
                           backend='torch CPU complex128',torch_version=torch.__version__),
        checks=checks,convergence=convergence,domains=domains,large_box=large_box,
        peak_sampling=peak_sampling,quadrature=peak_errors,
        final=[r for r in records if r['t']==6],wall_seconds=time.perf_counter()-start)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()

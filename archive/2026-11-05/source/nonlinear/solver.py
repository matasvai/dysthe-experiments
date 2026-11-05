"""Dealiased Fourier solvers: spatial Dysthe (1D) and optical envelope (1–3D).
FFT convention exp(-ikx); periodic grids; cubic Galerkin cutoff |mode|<N/4.
"""
from dataclasses import dataclass
import numpy as np
from scipy.fft import fftn,ifftn,fftfreq
from scipy.optimize import minimize_scalar

@dataclass
class FourierModel:
    shape:tuple
    lengths:tuple
    kind:str='dysthe'
    epsilon:float=0.
    epsilon1:float=1.
    epsilon2:float=1.
    nonlinear:bool=True
    workers:int=1
    def __post_init__(self):
        self.shape=tuple(self.shape);self.lengths=tuple(self.lengths)
        self.axes=tuple(np.arange(n)*L/n-L/2 for n,L in zip(self.shape,self.lengths))
        self.dx=np.prod([L/n for n,L in zip(self.shape,self.lengths)])
        self.k=np.meshgrid(*[2*np.pi*fftfreq(n,d=L/n) for n,L in zip(self.shape,self.lengths)],indexing='ij',sparse=True)
        modes=np.meshgrid(*[fftfreq(n)*n for n in self.shape],indexing='ij',sparse=True)
        self.mask=np.ones(self.shape,dtype=bool)
        for n,m in zip(self.shape,modes):self.mask &= abs(m)<n/4
        if self.kind=='dysthe':
            assert len(self.shape)==1
            self.linear=1j*self.k[0]**2
        elif self.kind=='optical':
            kx=self.k[0];ky=self.k[1] if len(self.shape)>1 else 0;kz=self.k[2] if len(self.shape)>2 else 0
            self.linear=-1j*((1+self.epsilon1*kz)*(kx*kx+ky*ky)-kz*kz+self.epsilon2*kz**3)
        else:raise ValueError(self.kind)
    def fft(self,u):return fftn(u,workers=self.workers)
    def ifft(self,v):return ifftn(v,workers=self.workers)
    def project(self,u):return self.ifft(self.fft(u)*self.mask)
    def nonlinear_hat(self,v):
        if not self.nonlinear:return np.zeros_like(v)
        u=self.ifft(v);rho=abs(u)**2
        if self.kind=='dysthe':
            ux=self.ifft(1j*self.k[0]*v)
            hder=self.ifft(abs(self.k[0])*self.fft(rho)).real
            f=-1j*rho*u-self.epsilon*(8*rho*ux+2*u*u*ux.conj()+2j*u*hder)
            return self.fft(f)*self.mask
        kz=self.k[2] if len(self.shape)>2 else 0
        return 1j*(1-self.epsilon1*kz)*self.fft(rho*u)*self.mask
    def rhs_hat(self,v):return self.linear*v+self.nonlinear_hat(v)
    def stepper(self,dt):
        E=np.exp(dt*self.linear);H=np.exp(dt*self.linear/2)
        def step(v):
            a=dt*self.nonlinear_hat(v)
            b=dt*H.conj()*self.nonlinear_hat(H*(v+a/2))
            c=dt*H.conj()*self.nonlinear_hat(H*(v+b/2))
            d=dt*E.conj()*self.nonlinear_hat(E*(v+c))
            return E*(v+(a+2*b+2*c+d)/6)*self.mask
        return step
    def invariants(self,u):
        rho=abs(u)**2;out={'mass':float(self.dx*np.sum(rho)),'peak':float(abs(u).max())}
        if self.kind=='dysthe':
            ux=self.ifft(1j*self.k[0]*self.fft(u));hd=self.ifft(abs(self.k[0])*self.fft(rho)).real
            # Fedele–Dutykh invariants (a,h,c,e,f)=(1,1,8,2,2).
            cross=ux.conj()*u-ux*u.conj();e=self.epsilon
            out['energy']=float(self.dx*np.sum((abs(ux)**2-.5*rho**2+(10/3)*e*e*rho**3-2.5j*e*rho*cross-e*rho*hd).real))
            out['momentum']=float(self.dx*np.sum((1j*cross-2*e*rho**2).real))
        v=self.fft(u);shell=self.mask.copy()
        for n,k,L in zip(self.shape,self.k,self.lengths):shell &= abs(k*L/(2*np.pi))<.8*(n/4)
        out['spectral_edge_mass']=float(np.sum(abs(v[self.mask & ~shell])**2)/max(np.sum(abs(v)**2),1e-300))
        return out

def nls_profile(x,mu=1.,speed=0.):
    nu=mu-speed*speed/4
    if nu<=0:raise ValueError('mu must exceed speed^2/4')
    return np.sqrt(2*nu)/np.cosh(np.sqrt(nu)*x)*np.exp(-.5j*speed*x)

def translate(model,u,shift):return model.ifft(model.fft(u)*np.exp(-1j*model.k[0]*shift))

def align_error(model,u,reference,guess=0.,window=2.):
    """Optimize translation and complex phase; preserve amplitude (no rescaling)."""
    vr=model.fft(reference);nr=np.linalg.norm(reference)
    def obj(shift):
        q=model.ifft(vr*np.exp(-1j*model.k[0]*shift));c=np.vdot(q,u)
        return float((np.linalg.norm(u)**2+nr**2-2*abs(c))/nr**2)
    fit=minimize_scalar(obj,bounds=(guess-window,guess+window),method='bounded',options={'xatol':1e-8})
    q=model.ifft(vr*np.exp(-1j*model.k[0]*fit.x));phase=np.angle(np.vdot(q,u))
    err=np.linalg.norm(u-q*np.exp(1j*phase))/nr
    return float(err),float(fit.x),float(phase)

def profile(model,mu=2.,speed=0.,initial=None,tol=2e-11,maxiter=12000):
    """Cubic Petviashvili iteration for u=F(x-speed*t) exp(-i mu*t)."""
    assert model.kind=='dysthe'
    k=model.k[0];den=mu+k*k+speed*k
    if den.min()<=0:raise ValueError('positive profile multiplier required')
    f=model.project(nls_profile(model.axes[0],mu,speed) if initial is None else initial)
    history=[];rev=(-np.arange(model.shape[0]))%model.shape[0]
    # Centered periodic grid: reversal x -> -x uses the same modular indices.
    for iteration in range(maxiter):
        v=model.fft(f)*model.mask;N=1j*model.nonlinear_hat(v)
        residual=np.linalg.norm(den*v-N)/np.linalg.norm(den*v)
        if iteration%25==0:history.append([iteration,float(residual)])
        if residual<tol:break
        ratio=np.vdot(v,den*v).real/np.vdot(v,N).real
        if not np.isfinite(ratio) or ratio<=0:raise RuntimeError('Petviashvili multiplier lost positivity')
        new=model.ifft(ratio**1.5*N/den)
        f=(new+new[rev].conj())/2
        if not np.isfinite(f).all():raise RuntimeError('nonfinite profile')
    else:raise RuntimeError(f'profile did not converge: {residual}')
    v=model.fft(f);rhs=model.rhs_hat(v);target=(-1j*mu-1j*speed*k)*v
    fullres=np.linalg.norm(rhs-target)/np.linalg.norm(target)
    info=dict(mu=mu,speed=speed,epsilon=model.epsilon,iterations=iteration,profile_residual=float(residual),evolution_residual=float(fullres),history=history,**model.invariants(f))
    return f,info

def evolve(model,u0,T,dt,samples=101,reference=None,speed=0.,phase_rate=0.,core_radius=10.):
    steps=max(1,int(round(T/dt)));dt=T/steps;step=model.stepper(dt);v=model.fft(u0)*model.mask
    indices=set(np.linspace(0,steps,min(samples,steps+1),dtype=int));rows=[];snaps=[]
    initial=model.ifft(v);norm=np.linalg.norm(initial)
    for it in range(steps+1):
        if it in indices:
            u=model.ifft(v);t=it*dt;row=dict(t=t,**model.invariants(u))
            x=model.axes[0];density=abs(u)**2
            if len(model.shape)==1:
                center=x[np.argmax(abs(u))];dist=(x-center+model.lengths[0]/2)%model.lengths[0]-model.lengths[0]/2
                row['core_tail_fraction']=float(np.sum(density[abs(dist)>core_radius])/np.sum(density))
                row['width_about_peak']=float(np.sqrt(np.sum(dist**2*density)/np.sum(density)))
            else:
                mean=np.mean(u,axis=tuple(range(1,u.ndim)),keepdims=True)
                row['transverse_fraction']=float(np.linalg.norm(u-mean)/np.linalg.norm(u))
            if reference is not None:
                exact=translate(model,reference,speed*t)*np.exp(1j*phase_rate*t)
                row['complex_error']=float(np.linalg.norm(u-exact)/np.linalg.norm(reference))
                if len(model.shape)==1:
                    err,shift,phase=align_error(model,u,reference,speed*t,3.)
                    row.update(aligned_error=err,translation=shift,phase=phase)
            rows.append(row);snaps.append(u.copy())
        if it==steps:break
        v=step(v)
        if not np.isfinite(v).all():raise RuntimeError(f'nonfinite solution at {it*dt}')
    return np.array(snaps),rows

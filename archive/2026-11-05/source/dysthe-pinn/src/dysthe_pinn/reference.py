"""Independent Fourier reference for the full optical equation only.

Integrating-factor RK4 and strict |mode_j|<N_j/4 cubic Galerkin cutoff.
The linear multiplier is -i[(1+e1*kz)(kx²+ky²)-kz²+e2*kz³];
nonlinear RHS is i(1-e1*kz) FFT(|E|²E). No linearization.
Adapted and independently tested from the project's nonlinear/solver.py.
"""
import json
import math
import time
from pathlib import Path
import numpy as np


class SpectralReference:
    def __init__(self, config):
        self.config = config
        self.shape = tuple(config['reference']['shape'])
        self.lengths = tuple(config['domain']['lengths'])
        self.axes = [np.arange(n)*length/n-length/2 for n, length in zip(self.shape, self.lengths)]
        self.k = np.meshgrid(*[2*np.pi*np.fft.fftfreq(n, d=length/n) for n,length in zip(self.shape,self.lengths)], indexing='ij', sparse=True)
        modes = np.meshgrid(*[np.fft.fftfreq(n)*n for n in self.shape], indexing='ij', sparse=True)
        self.mask = np.ones(self.shape, bool)
        for mode,n in zip(modes,self.shape):
            self.mask &= abs(mode)<n/4
        kx,ky,kz = self.k
        self.e1, self.e2 = config['physics']['epsilon1'], config['physics']['epsilon2']
        self.linear = -1j*((1+self.e1*kz)*(kx*kx+ky*ky)-kz*kz+self.e2*kz**3)
        self._step_cache = None
        self.volume_element = np.prod(np.array(self.lengths)/self.shape)

    def initial(self):
        xyz = np.meshgrid(*self.axes, indexing='ij', sparse=True)
        ic = self.config['initial']
        field = np.ones(self.shape, complex)*ic['amplitude']*np.exp(1j*ic.get('phase',0.0))
        for x,length in zip(xyz,self.lengths):
            field *= ((1+np.cos(2*np.pi*x/length))/2)**ic['power']
        return field

    def nonlinear(self,v):
        u = np.fft.ifftn(v)
        return 1j*(1-self.e1*self.k[2])*np.fft.fftn(abs(u)**2*u)*self.mask

    def rhs(self,v):
        return self.linear*v+self.nonlinear(v)

    def step(self,v,dt):
        if self._step_cache is None or self._step_cache[0] != dt:
            self._step_cache = (dt, np.exp(dt*self.linear), np.exp(dt*self.linear/2))
        _, e, h = self._step_cache
        a = dt*self.nonlinear(v)
        b = dt*h.conj()*self.nonlinear(h*(v+a/2))
        c = dt*h.conj()*self.nonlinear(h*(v+b/2))
        d = dt*e.conj()*self.nonlinear(e*(v+c))
        return e*(v+(a+2*b+2*c+d)/6)*self.mask

    def diagnostics(self,u):
        v=np.fft.fftn(u)
        inner=self.mask.copy()
        for k,length,n in zip(self.k,self.lengths,self.shape):
            inner &= abs(k*length/(2*np.pi)) < .8*n/4
        return dict(mass=float(self.volume_element*np.sum(abs(u)**2)),
                    peak=float(abs(u).max()),
                    spectral_edge_fraction=float(np.sum(abs(v[self.mask & ~inner])**2)/max(np.sum(abs(v)**2),1e-300)))


def generate(config,output):
    out=Path(output)
    out.mkdir(parents=True,exist_ok=True)
    model=SpectralReference(config)
    intervals=config['reference']['snapshots']-1
    stride=max(1,math.ceil(config['domain']['T']/(config['reference']['dt']*intervals)))
    nsteps=stride*intervals
    dt=config['domain']['T']/nsteps
    indices=set(range(0,nsteps+1,stride))
    u0=model.initial()
    v=np.fft.fftn(u0)*model.mask
    initial_projection_error=float(np.linalg.norm(np.fft.ifftn(v)-u0)/np.linalg.norm(u0))
    times,fields,rows=[],[],[]
    started = last_log = time.monotonic()
    print(f"Reference {model.shape}: T={config['domain']['T']}, steps={nsteps}, dt={dt:g}", flush=True)
    for step in range(nsteps+1):
        now = time.monotonic()
        if step == nsteps or now-last_log >= 30:
            print(f"  step {step}/{nsteps}, t={step*dt:.6g}, elapsed={now-started:.1f}s", flush=True)
            last_log = now
        if step in indices:
            u=np.fft.ifftn(v)
            times.append(step*dt);fields.append(u)
            rows.append(dict(t=step*dt,**model.diagnostics(u)))
        if step<nsteps:
            v=model.step(v,dt)
            if not np.isfinite(v).all():
                raise FloatingPointError(f'Nonfinite reference at step {step+1}')
    mass0=rows[0]['mass']
    for row in rows:
        row['relative_mass_drift']=(row['mass']-mass0)/mass0
    np.savez_compressed(out/'reference.npz',times=times,x=model.axes[0],y=model.axes[1],z=model.axes[2],E=np.array(fields),config_json=json.dumps(config))
    summary=dict(equation=config['equation'],config=config,dt=dt,steps=nsteps,
                 initial_projection_relative_l2=initial_projection_error,
                 max_relative_mass_drift=max(abs(r['relative_mass_drift']) for r in rows),
                 max_spectral_edge_fraction=max(r['spectral_edge_fraction'] for r in rows),rows=rows,
                 qualification='Reference candidate; requires independent time-step and spatial refinement before accuracy claims.')
    (out/'diagnostics.json').write_text(json.dumps(summary,indent=2))
    return summary

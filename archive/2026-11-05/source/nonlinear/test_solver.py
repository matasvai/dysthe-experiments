"""Run with python -m unittest nonlinear.test_solver -v."""
import unittest
import numpy as np
from scipy.signal import resample
from .solver import FourierModel,nls_profile,profile,evolve
class SolverTests(unittest.TestCase):
 def test_optical_plane_wave(self):
  m=FourierModel((16,16,16),(2*np.pi,)*3,kind='optical',nonlinear=False)
  x,y,z=np.meshgrid(*m.axes,indexing='ij');u=np.exp(1j*(x+y-z));v=m.fft(u)
  # k=(1,1,-1): omega=(1-1)*2-1-1=-2, multiplier exp(2it).
  out=m.ifft(m.stepper(.137)(v))
  self.assertLess(np.max(abs(out-u*np.exp(.274j))),2e-13)
 def test_nls_rhs(self):
  m=FourierModel((1024,),(80,),kind='optical');u=nls_profile(m.axes[0]);v=m.fft(u)
  self.assertLess(np.linalg.norm(m.rhs_hat(v)-1j*v)/np.linalg.norm(v),1e-9)
 def test_cubic_dealias(self):
  m=FourierModel((64,),(2*np.pi,),kind='optical');x=m.axes[0]
  u=np.exp(14j*x)+.3*np.exp(-13j*x)+.2*np.exp(9j*x)
  fine=resample(u,256);exact=resample(abs(fine)**2*fine,64)
  got=m.nonlinear_hat(m.fft(u))/1j
  expected=m.fft(exact)*m.mask
  self.assertLess(np.linalg.norm(got-expected)/np.linalg.norm(expected),2e-13)
 def test_fourth_order(self):
  m=FourierModel((1024,),(80,),kind='optical');f=nls_profile(m.axes[0]);errors=[]
  for dt in [.04,.02,.01]:
   _,d=evolve(m,f,1,dt,2,f,0,1);errors.append(d[-1]['complex_error'])
  self.assertGreater(errors[0]/errors[1],10)
  self.assertGreater(errors[1]/errors[2],10)
 def test_dysthe_profile(self):
  m=FourierModel((1024,),(80,),epsilon=.15);_,d=profile(m,2,2)
  self.assertLess(d['evolution_residual'],1e-9)
 def test_mass_derivative(self):
  m=FourierModel((1024,),(80,),epsilon=.15);u=nls_profile(m.axes[0],2,2)*(1+.03*np.cos(m.axes[0]));v=m.fft(u)*m.mask;u=m.ifft(v)
  derivative=2*np.vdot(u,m.ifft(m.rhs_hat(v))).real*m.dx
  self.assertLess(abs(derivative),1e-10)
if __name__=='__main__':unittest.main()

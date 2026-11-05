"""Analytic and de-aliasing tests, independent of the simulation conclusions."""
import tempfile
import unittest
import numpy as np
from scipy.signal import resample
from run import Model, solve, norm, errors, default_config, inlets


class SolverTests(unittest.TestCase):
 def test_mean_flow_zero_and_cosine_multiplier(self):
  m=Model(64,2*np.pi,.1)
  self.assertLess(np.max(abs(m.abs_derivative(np.ones(64)))),1e-14)
  np.testing.assert_allclose(m.abs_derivative(2+np.cos(3*m.s)),3*np.cos(3*m.s),atol=1e-13)
 def test_linear_dispersion_multiplier_and_exact_step(self):
  m=Model(64,2*np.pi,.1);u=np.exp(3j*m.s)
  expected=1j*(9/8-.1*27/16)*u
  np.testing.assert_allclose(np.fft.ifft(m.linear*np.fft.fft(u)),expected,atol=5e-12)
  np.testing.assert_allclose(np.fft.ifft(np.exp(.7*m.linear)*np.fft.fft(u)),u*np.exp(.7j*(9/8-.1*27/16)),atol=5e-14)
 def test_plane_wave_all_variants_against_exact_phase(self):
  n=64;L=2*np.pi;eps=.1;a=.8;q=2;s=np.arange(n)*L/n-L/2;u=a*np.exp(1j*q*s);T=np.array([0.,.2,.4])
  for kind in ('literal','stability','nls','meanflow'):
   c=1. if kind=='literal' else 1.5
   omega=.5*a*a-q*q/8
   if kind in ('literal','stability'):omega+=eps*((c+.25)*q*a*a+q**3/16)
   exact=np.exp(-1j*omega*T[:,None])*u
   numerical,_=solve(u,L,eps,kind,n,.005,T)
   self.assertLess(float(errors(numerical,exact,L)['h0'].max()),1e-10)
 def test_fourth_order_step_refinement(self):
  n=64;L=2*np.pi;eps=.1;a=1.1;q=2;s=np.arange(n)*L/n-L/2;u=a*np.exp(1j*q*s);T=np.array([0.,1.])
  omega=.5*a*a+eps*(1.75*q*a*a+q**3/16)-q*q/8
  exact=np.exp(-1j*omega*T[:,None])*u
  coarse,_=solve(u,L,eps,'stability',n,.04,T)
  fine,_=solve(u,L,eps,'stability',n,.02,T)
  coarse_error=float(errors(coarse,exact,L)['h0'].max());fine_error=float(errors(fine,exact,L)['h0'].max())
  self.assertLess(fine_error,coarse_error/12)
 def test_projected_cubic_rhs_matches_oversampled_rhs(self):
  m=Model(32,2*np.pi,.1);fine=Model(128,2*np.pi,.1);rng=np.random.default_rng(72)
  coeff=np.zeros(32,complex);coeff[0:8]=rng.normal(size=8)+1j*rng.normal(size=8);coeff[-7:]=rng.normal(size=7)+1j*rng.normal(size=7)
  u=np.fft.ifft(coeff)
  fine_u=resample(u,128)
  coarse_rhs=np.fft.ifft(m.nonlinear_hat(np.fft.fft(u)))
  fine_rhs=np.fft.ifft(fine.nonlinear_hat(np.fft.fft(fine_u)))
  np.testing.assert_allclose(coarse_rhs,m.project(resample(fine_rhs,32)),rtol=1e-12,atol=1e-14)
 def test_literal_and_stability_sign_difference_is_exact(self):
  m=Model(64,2*np.pi,.1,'literal');other=Model(64,2*np.pi,.1,'stability')
  u=.8+.1*np.cos(2*m.s)+.05j*np.sin(3*m.s);v=np.fft.fft(u)
  diff=np.fft.ifft(other.rhs_hat(v)-m.rhs_hat(v));du=np.fft.ifft(1j*m.k*v)
  np.testing.assert_allclose(diff,m.project(-.5*m.epsilon*abs(u)**2*du),atol=1e-14)
 def test_zero_epsilon_controls_coincide(self):
  reference=Model(64,2*np.pi,0,'nls');u=.7+.1*np.exp(2j*reference.s);v=np.fft.fft(u)
  for kind in ('literal','stability','meanflow'):
   np.testing.assert_allclose(Model(64,2*np.pi,0,kind).rhs_hat(v),reference.rhs_hat(v),atol=1e-14)
 def test_initial_fields_and_hnorm(self):
  c=default_config();data=inlets(c)
  self.assertEqual([x[0] for x in data],['packet','sideband','random'])
  for (_,u,_),(_,again,_) in zip(data,inlets(c)):np.testing.assert_array_equal(u,again)
  m=Model(64,2*np.pi,.1);u=.5*np.exp(2j*m.s)
  self.assertAlmostEqual(float(norm(u,2*np.pi,2)),np.sqrt(2*np.pi)*.5*5,places=12)
 def test_nls_sech_profile_with_exponentially_small_periodic_tail(self):
  n=1024;L=80.;a=.5;s=np.arange(n)*L/n-L/2;u=a/np.cosh(np.sqrt(2)*a*s);T=np.array([0.,1.])
  exact=u[None,:]*np.exp(-.25j*a*a*T[:,None])
  computed,_=solve(u,L,.1,'nls',n,.01,T)
  self.assertLess(float(errors(computed,exact,L)['h0'].max()),1e-9)
 def test_no_silent_time_step_or_model_change(self):
  with self.assertRaises(ValueError):Model(64,2*np.pi,.1,'unversioned')
  with self.assertRaises(ValueError):solve(np.ones(64),2*np.pi,.1,'literal',64,.03,np.array([0.,.1]))

if __name__=='__main__':unittest.main()

"""Independent real-space input/normalization checks and existing-operator control."""
import json
import numpy as np
from scipy.integrate import quad
from restricted_decay import C,spectrum,bump_norm,BUMP_L1,OUT,value
from long_time_decay import continuous_sum
checks={}
for m in range(3):
 def h(z):return [1,z,z*z-1][m]
 def g(z):return np.exp(-z*z/2)/np.sqrt(2*np.pi)
 norm=quad(lambda z:abs(h(z))*g(z),-15,15,points=[-1,0,1],epsabs=1e-12)[0]/C[m]
 errors=[]
 for k in [0,.3,.8]:
  def f(z):return h(z)*g(z)*np.exp(-1j*k*3*z)/C[m]
  v=quad(lambda z:f(z).real,-15,15,epsabs=1e-12)[0]+1j*quad(lambda z:f(z).imag,-15,15,epsabs=1e-12)[0]
  errors.append(float(abs(v-spectrum(np.array(k),3,m))))
 checks[f'order_{m}']={'L1_error':float(abs(norm-1)),'Fourier_absolute_error':max(errors)}
 assert abs(norm-1)<1e-10 and max(errors)<1e-10
checks['baseline_existing_solver_error']=float(abs(value((5,2),32,3,0)-continuous_sum((5,2),32,-1,0,3,3,0,16384,65536)))
checks['bump_L1_relative_change']=abs(BUMP_L1/bump_norm(1024,.0625)-1)
assert checks['baseline_existing_solver_error']<1e-14
assert checks['bump_L1_relative_change']<1e-4
(OUT/'verification.json').write_text(json.dumps(checks,indent=2))
print('Independent normalization, Fourier-transform, and baseline checks passed.')

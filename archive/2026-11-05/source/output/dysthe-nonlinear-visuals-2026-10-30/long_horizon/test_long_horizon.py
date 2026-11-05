import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from run_long_horizon import ROOT,run,validate,common_box_discrepancy,streams_for
from vendor_temporal_solver import solve

class LongHorizonTests(unittest.TestCase):
 def config(self,track='periodic'):
  return json.loads((ROOT/'configs'/f'{track}-smoke.json').read_text())
 def test_rejects_other_model_and_random_isolated_data(self):
  c=self.config();c['model_id']='water-wave-dysthe-spatial-v1'
  with self.assertRaises(ValueError):validate(c)
  c=self.config('isolated');c['families']=['random']
  with self.assertRaises(ValueError):validate(c)
 def test_common_box_derivatives_are_compared_before_restriction(self):
  n=128;L=40;s=np.arange(n)*L/n-L/2;b=np.arange(2*n)*L/n-L
  a=np.exp(2j*np.pi*3*s/L);big=np.exp(2j*np.pi*3*b/L)
  self.assertLess(max(common_box_discrepancy(a,big,L).values()),1e-11)
 def test_failed_gate_blocks_later_targets_immediately(self):
  c=self.config();c['tolerances']['h0']=1e-30
  with tempfile.TemporaryDirectory() as d:
   result=run(c,Path(d)/'new');stages=result['cases'][0]['stages']
  self.assertEqual(result['status'],'stopped_at_failed_gate')
  self.assertFalse(stages[0]['qualified']);self.assertLess(stages[0]['last_monitored_T'],c['horizons'][0])
  self.assertEqual(stages[1]['status'],'blocked_by_previous_gate')
 def test_staged_evolution_matches_single_uninterrupted_solve(self):
  c=self.config();stream=streams_for('gaussian',c)['base-fine']
  expected,_=solve(stream.u0,c['length'],c['epsilon'],'stability',c['n'],c['step'],np.array([0.,.04]))
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'new';result=run(c,out)
   self.assertEqual(result['status'],'all_requested_horizons_qualified')
   with np.load(out/'gaussian-T0.04.npz') as z:
    observed=np.fft.ifft(z['checkpoint_base-fine'])
   self.assertLess(np.max(abs(observed-expected[-1])),1e-12)
   with self.assertRaises(ValueError):run(c,out)
 def test_isolated_smoke_qualifies_both_boxes_and_domain(self):
  c=self.config('isolated')
  with tempfile.TemporaryDirectory() as d:r=run(c,Path(d)/'new')
  self.assertTrue(all(x['numerical_checks_passed'] and x['domain_checks_passed'] for x in r['cases'][0]['stages']))
 def test_calibration_is_not_a_completed_target_run(self):
  with tempfile.TemporaryDirectory() as d:r=run(self.config(),Path(d)/'new',2)
  self.assertEqual(r['status'],'calibration_only');self.assertNotIn('cases',r)
  self.assertGreater(r['fixed_grid_estimated_seconds_all_families']['0.04'],0)
 def test_failed_step_preserves_completed_state_time_and_cost(self):
  stream=streams_for('gaussian',self.config())['base-fine']
  stream.to(.01);start_T=stream.T;start_seconds=stream.seconds
  advance=stream.advance;expected=advance(advance(stream.v));calls=0
  def fail_third_step(v):
   nonlocal calls
   calls+=1
   if calls==3:raise FloatingPointError('Injected failed step')
   return advance(v)
  stream.advance=fail_third_step
  with patch('run_long_horizon.time.perf_counter',side_effect=[100.,100.125]):
   with self.assertRaisesRegex(FloatingPointError,'Injected'):stream.to(.02)
  self.assertEqual(calls,3)
  self.assertAlmostEqual(stream.T,start_T+2*stream.h)
  self.assertAlmostEqual(stream.seconds-start_seconds,.125)
  np.testing.assert_array_equal(stream.v,expected)
if __name__=='__main__':unittest.main()

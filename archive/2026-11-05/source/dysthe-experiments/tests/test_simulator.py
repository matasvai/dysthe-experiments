"""Numerical/serialization safeguards for the bounded initial-envelope demo."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dysthe_core.reference import spatial
from dysthe_experiments.simulator import (CASE_IDS, DEFAULT_PARAMETERS, case_info,
    center_and_width, configuration, initial_field, simulate, validate_request)


class SimulatorContractTests(unittest.TestCase):
    def test_prescriptions_retain_complex_phase_and_profile_residual(self):
        model=spatial.SpatialDysthe(2048,80.,.025)
        profile,reference,metadata=initial_field('profile',model)
        self.assertLess(metadata['profile_residual']['full_periodic_relative_l2'],1e-8)
        np.testing.assert_allclose(profile,reference,atol=1e-14,rtol=0)
        self.assertGreater(np.linalg.norm(profile.imag),.01)
        perturbed,_,_=initial_field('perturbed',model,dict(delta=.02))
        multiplier=1+.02*np.exp(-((model.tau-1.3)/2)**2)
        np.testing.assert_allclose(perturbed,model.project(profile*multiplier),atol=1e-13)
        self.assertGreater(np.linalg.norm(perturbed-profile),.01)
        packet,_,_=initial_field('gaussian',model)
        p=DEFAULT_PARAMETERS['gaussian']
        expected=p['amplitude']*np.exp(-model.tau**2/(2*p['width']**2))*np.exp(1j*p['wavenumber']*model.tau)
        np.testing.assert_allclose(packet,expected,atol=1e-13)
        self.assertGreater(np.linalg.norm(packet.imag),.1)

    def test_circular_center_tracks_translation_across_boundary(self):
        model=spatial.SpatialDysthe(2048,80.,.025)
        u=np.exp(-model.tau**2/8)*np.exp(.3j*model.tau)
        c0,w0,concentration,_=center_and_width(model,u)
        shifted=spatial.translate(model,u,39.4)
        c,w,_,_=center_and_width(model,shifted)
        self.assertAlmostEqual(c-c0,39.4,places=10)
        self.assertAlmostEqual(w,w0,places=10)
        self.assertGreater(concentration,.9)

    def test_save_contract_phase_decimation_and_unqualified_custom_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'packet'
            progress=[]
            report=simulate('gaussian',output,dict(chirp=.01),
                dict(n=512,length=40.,xi_end=.01,dxi=.005,samples=3),progress.append)
            self.assertFalse(report['qualification']['checks_available'])
            self.assertIn('unqualified',report['qualification']['status'])
            with np.load(output/'fields.npz',allow_pickle=False) as arrays:
                fields=arrays['u'];tau=arrays['tau'];xi=arrays['xi']
                self.assertEqual(fields.dtype,np.complex128)
                self.assertEqual(fields.shape,(3,512))
                self.assertEqual(xi[0],0.)
                self.assertEqual(xi[-1],.01)
                viewer=json.loads((output/'viewer.json').read_text())
                restored=np.asarray(viewer['real'])+1j*np.asarray(viewer['imag'])
                np.testing.assert_array_equal(restored,fields)
                np.testing.assert_array_equal(viewer['tau'],tau)
                self.assertEqual(viewer['full_data']['axis_order'],['xi','tau'])
            self.assertEqual(progress[-1]['completed_steps'],2)
            self.assertEqual(progress[-1]['phase'],'complete')
            self.assertAlmostEqual(report['diagnostics'][0]['displacement'],0.)
            self.assertTrue(report['qualification']['all_design_screens_pass'])

    def test_budget_and_nonfinite_requests_rejected_before_work(self):
        for case,parameters,settings in [
            ('bad',None,None),('profile',{'amplitude':1},None),
            ('gaussian',{'width':float('nan')},None),
            ('gaussian',{'amplitude':True},None),
            ('perturbed',{'delta':.1},None),
            ('profile',None,{'xi_end':5}),('profile',None,{'n':16384,'dxi':.0003125}),
            ('gaussian',[],None),('gaussian',None,{'n':4096.}),
            ('gaussian',None,{'epsilon':.03}),('gaussian',None,{'samples':False}),
        ]:
            with self.subTest(case=case,parameters=parameters,settings=settings):
                with self.assertRaises(ValueError):
                    validate_request(case,parameters,settings)
        self.assertEqual([c['case_id'] for c in case_info()],CASE_IDS)
        p,s=configuration('gaussian')
        self.assertEqual(p['wavenumber'],.3)
        self.assertEqual(s['xi_end'],4.)


if __name__=='__main__':
    unittest.main()

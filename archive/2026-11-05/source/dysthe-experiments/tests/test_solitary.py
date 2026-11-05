import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from dysthe_core.reference.spatial import SpatialDysthe, nls_profile
from dysthe_experiments.solitary import run, validate


class SolitaryPilotTests(unittest.TestCase):
    def fixture(self,root):
        m=SpatialDysthe(512,20.,.025)
        np.savez(root/'profile.npz',u=nls_profile(m.tau)[None,:],t=[0],x0=m.tau)
        (root/'profile.json').write_text(json.dumps(dict(kind='dysthe',epsilon=.025,
            mu=2.,speed=0.,epsilon1=1.,epsilon2=1.,nonlinear=True,shape=[512],lengths=[20.])))
        c=dict(id='test',profile='profile.npz',profile_mode='recompute',profile_max_iterations=200,
            n=512,length=20.,epsilon=.025,mu=2.,speed=0.,dxi=.001,xi_end=.002,samples=3,
            perturbations=[dict(id='control',kind='control',delta=0.),dict(id='phase',kind='phase',delta=.001)])
        q=dict(full_residual_max=1e-6,projection_relative_l2_max=1e-8,epsilon_peak_screen=.1,
            epsilon_spectral_sigma_screen=.1,epsilon_spectral_mean_screen=.1,
            boundary_action_max=1e-8,spectral_edge_action_max=1e-8)
        cfg=dict(model_id='water-wave-dysthe-spatial-v1',physical_system='deep-water-gravity-waves',
            profile_coordinate='tau',evolution_coordinate='xi',qualification=q,cases=[c])
        path=root/'config.json';path.write_text(json.dumps(cfg))
        return path,cfg

    def test_short_run_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path,cfg=self.fixture(root)
            r=run(path,'test',root/'out',4)
            self.assertLess(r['profile_residual']['full_periodic_relative_l2'],1e-6)
            control=r['trajectories']['control'];phase=r['trajectories']['phase']
            self.assertGreater(phase[0]['aligned_conjugate_reflection_defect'],1e-5)
            self.assertLess(control[-1]['aligned_conjugate_reflection_defect'],1e-7)
            self.assertGreater(control[-1]['raw_conjugate_reflection_defect'],1e-3)
            self.assertIn('invariant_absolute_drifts',control[0])
            self.assertEqual(np.load(root/'out/fields.npz')['phase'].shape,(3,512))
            with self.assertRaises(ValueError): run(path,'test',root/'out',4)

    def test_budget_and_failed_qualification_create_no_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path,cfg=self.fixture(root)
            with self.assertRaises(ValueError): run(path,'test',root/'out',1)
            cfg['qualification']['full_residual_max']=1e-20;path.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError): run(path,'test',root/'out',4)
            self.assertFalse((root/'out').exists())

    def test_scope_and_archive_keys_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            path,cfg=self.fixture(Path(folder))
            cfg['cases'][0]['perturbations'][0]['id']='tau'
            with self.assertRaises(ValueError): validate(cfg)
            cfg['cases'][0]['perturbations'][0]['id']='control'
            cfg['model_id']='optical'
            with self.assertRaises(ValueError): validate(cfg)


if __name__=='__main__':
    unittest.main()

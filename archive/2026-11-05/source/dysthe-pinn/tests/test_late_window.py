import json
from pathlib import Path
import numpy as np
import pytest
from dysthe_pinn.late_window import run
from dysthe_pinn.reference import generate


def config():
    return json.loads((Path(__file__).parents[1]/'configs/smoke.json').read_text())


def test_late_window_is_evolved_not_relabelled_and_resume_is_exact(tmp_path):
    c=config(); c['domain']['T']=.1;c['reference'].update(shape=[16]*3,dt=.002,snapshots=6)
    generate(c,tmp_path/'full')
    expected=np.load(tmp_path/'full/reference.npz')
    options=dict(start=.06,end=.1,dt=.002,shape=16,snapshots=3)
    # Pause before the window and inside it, then complete from the saved Fourier state.
    assert run(c,tmp_path/'resumed',max_steps=15,**options)['status']=='paused'
    r=run(c,tmp_path/'resumed',max_steps=22,**options)
    assert r['saved_frames']==1
    assert run(c,tmp_path/'resumed',**options)['status']=='completed_unvalidated'
    run(c,tmp_path/'continuous',**options)
    with np.load(tmp_path/'resumed/reference.npz') as actual, np.load(tmp_path/'continuous/reference.npz') as continuous:
        np.testing.assert_array_equal(actual['E'],continuous['E'])
        np.testing.assert_allclose(actual['E'],expected['E'][3:],rtol=2e-14,atol=2e-14)
        np.testing.assert_allclose(actual['times'],[.06,.08,.1],rtol=0,atol=1e-15)
        assert np.linalg.norm(actual['E'][0]-expected['E'][0])>.0001
    with pytest.raises(ValueError,match='settings differ'):
        run(c,tmp_path/'resumed',**{**options,'dt':.001})


def test_window_alignment_and_diagnostic_failure(tmp_path):
    with pytest.raises(ValueError,match='align'):
        run(config(),tmp_path/'misaligned',start=.061,end=.1,dt=.002,shape=16,snapshots=3)
    from unittest.mock import patch
    from dysthe_pinn.reference import SpectralReference
    original=SpectralReference.diagnostics
    def unresolved(self,u):
        result=original(self,u);result['spectral_edge_fraction']=.002
        return result
    with patch.object(SpectralReference,'diagnostics',unresolved):
        result=run(config(),tmp_path/'failed',start=.06,end=.1,dt=.002,shape=16,snapshots=3)
    assert result['status']=='failed_diagnostics'
    assert not (tmp_path/'failed/reference.npz').exists()


def test_extended_breathing_window_disk_frames_and_resume(tmp_path):
    c=config();opts=dict(start=.06,end=.1,integration_end=.14,dt=.002,shape=16,snapshots=3,breathing=True)
    paused=run(c,tmp_path/'resumed',max_steps=36,**opts)
    assert paused['saved_frames']==1
    result=run(c,tmp_path/'resumed',**opts)
    run(c,tmp_path/'continuous',**opts)
    assert result['current_t']==.14 and result['saved_frames']==3
    assert 'effective_volume' in result['rows'][-1]
    with np.load(tmp_path/'resumed/reference.npz') as a,np.load(tmp_path/'continuous/reference.npz') as b:
        np.testing.assert_array_equal(a['E'],b['E'])
        np.testing.assert_allclose(a['times'],[.06,.08,.1])
    with np.load(tmp_path/'resumed/checkpoint.npz') as state:
        assert state['fields'].size==0


def test_localization_is_translation_invariant():
    from dysthe_pinn.breathing import localization
    from dysthe_pinn.reference import SpectralReference
    model=SpectralReference(config());u=model.initial()
    a=localization(model,u);b=localization(model,np.roll(u,3,axis=0))
    assert a['centers_well_defined']
    for key in ['effective_volume','width_x','width_y','width_z']:
        np.testing.assert_allclose(a[key],b[key],rtol=1e-13,atol=1e-13)
    assert abs(a['center_x']-b['center_x'])>1


def test_thousand_showcase_schedule_without_long_integration(tmp_path):
    result=run(config(),tmp_path/'schedule',start=1500,end=2500,integration_end=12288,
               dt=.04,shape=16,snapshots=501,breathing=True,showcases=True,max_steps=1)
    settings=json.loads((tmp_path/'schedule/settings.json').read_text())
    assert result['showcase_centers']==list(range(1000,12001,1000))
    assert result['config']['domain']['T']==12288
    assert result['config']['reference']['snapshots']==776
    assert settings['diagnostics_every_steps']==25
    assert settings['window_diagnostics_every_steps']==6

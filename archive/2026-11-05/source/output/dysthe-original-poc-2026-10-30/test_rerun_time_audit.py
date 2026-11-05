"""Fresh-output audit CLI checks, separate from the ten scientific solver tests."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from rerun_time_audit import execute
from run import default_config, sha


class RerunAuditTests(unittest.TestCase):
    def config(self):
        c=default_config()
        c.update(n=128,initial_modes=6,sample_step=.02)
        return c

    def test_help_is_read_only_and_lists_required_output(self):
        script=Path(__file__).with_name('rerun_time_audit.py')
        proc=subprocess.run([sys.executable,str(script),'--help'],capture_output=True,text=True)
        self.assertEqual(proc.returncode,0)
        self.assertIn('--output',proc.stdout)
        self.assertIn('--config',proc.stdout)
        self.assertIn('--time',proc.stdout)

    def test_fresh_small_fixture_preserves_all_independent_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'fresh'
            with contextlib.redirect_stdout(io.StringIO()):
                result=execute(self.config(),out,horizon=.04,step=.0025)
            self.assertTrue(result['qualified'])
            self.assertEqual([a['n'] for a in result['artifacts']],[128,64,128])
            self.assertEqual([a['step'] for a in result['artifacts']],[.0025,.0025,.005])
            for artifact in result['artifacts']:
                self.assertEqual(sha(out/artifact['path']),artifact['sha256'])
                with np.load(out/artifact['path'],allow_pickle=False) as z:
                    self.assertEqual(z['fields'].shape,(3,artifact['n']))
                    np.testing.assert_array_equal(z['T'],[0.,.02,.04])
            self.assertEqual(json.loads((out/'result.json').read_text()),result)

    def test_existing_output_is_rejected_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'existing';out.mkdir();marker=out/'marker';marker.write_text('preserve')
            with self.assertRaises(FileExistsError):
                execute(self.config(),out,horizon=.04,step=.0025)
            self.assertEqual(marker.read_text(),'preserve')
            self.assertFalse((out/'identity.json').exists())

    def test_wrong_model_and_nonmatching_saved_steps_fail_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'fresh';c=self.config();c['model_ids']={'stability':'other'}
            with self.assertRaisesRegex(ValueError,'versioned'):
                execute(c,out,horizon=.04,step=.0025)
            with self.assertRaisesRegex(ValueError,'declared step'):
                execute(self.config(),out,horizon=.04,step=.003)
            self.assertFalse(out.exists())

if __name__=='__main__':unittest.main()

"""Exercise Slurm command construction and launch guards without a scheduler/GPU."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LongClusterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.repo = self.base / 'checkout with spaces'
        (self.repo / 'cluster').mkdir(parents=True)
        (self.repo / 'src' / 'dysthe_experiments').mkdir(parents=True)
        (self.repo / 'src' / 'dysthe_experiments' / 'long_distance.py').touch()
        (self.repo / 'configs').mkdir()
        for name in ('run-long-reference.sbatch', 'run-long-pinn.sbatch',
                     'submit-long-distance.sh', 'long-preflight.py'):
            shutil.copy2(ROOT / 'cluster' / name, self.repo / 'cluster' / name)
        self.config = self.repo / 'configs' / 'long-distance-a100.json'
        self.config.write_text(json.dumps({
            'model_id': 'water-wave-dysthe-spatial-v1', 'groups_per_family': 12,
            'train_per_family': 8, 'validation_per_family': 2, 'seeds': [11, 29, 47], 'threads': 4,
        }))
        self.bin = self.base / 'fake-bin'
        self.bin.mkdir()
        self.log = self.base / 'calls.jsonl'
        self.env = os.environ.copy()
        for key in list(self.env):
            if key.startswith('SLURM_') or key in {
                    'DYSTHE_PYTHON', 'DYSTHE_EXPERIMENTS_DIR', 'RUN_DIR', 'LONG_CONFIG'}:
                self.env.pop(key)
        self.env.update(PATH=f'{self.bin}:{self.env["PATH"]}', FAKE_LOG=str(self.log))
        self.write_executable('srun', """
import os, sys
args = sys.argv[1:]
if args[0] != '--ntasks=1':
    raise SystemExit('unexpected task count')
os.execv(args[1], args[1:])
""")
        self.fake_python = self.write_executable('prepared-python', """
import json, os, sys
with open(os.environ['FAKE_LOG'], 'a') as f:
    f.write(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd(),
                       'pythonpath': os.environ.get('PYTHONPATH'),
                       'tf32': os.environ.get('NVIDIA_TF32_OVERRIDE')}) + '\\n')
if sys.argv[1].endswith('long-preflight.py'):
    raise SystemExit(int(os.environ.get('FAKE_PREFLIGHT_STATUS', '0')))
if sys.argv[1] == '-':
    sys.argv = sys.argv[1:]
    exec(compile(sys.stdin.read(), '<allocation-check>', 'exec'))
""")
        self.write_executable('sbatch', """
import json, os, pathlib, sys
path = pathlib.Path(os.environ['FAKE_LOG'])
number = 101 + (len(path.read_text().splitlines()) if path.exists() else 0)
with path.open('a') as f:
    f.write(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd(),
                       'run': os.environ['RUN_DIR'], 'config': os.environ['LONG_CONFIG']}) + '\\n')
print(str(number) + ';fake-cluster')
""")

    def write_executable(self, name, source):
        path = self.bin / name
        path.write_text(f'#!{sys.executable}\n' + source)
        path.chmod(0o755)
        return path

    def run_script(self, name, *args, env=None, spool=False):
        path = self.repo / 'cluster' / name
        if spool:
            path = self.base / f'spool-{name}'
            shutil.copy2(self.repo / 'cluster' / name, path)
        return subprocess.run(['bash', str(path), *args], cwd=self.base,
                              env=env or self.env, text=True, capture_output=True)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def allocation(self, task='3'):
        return {**self.env, 'SLURM_JOB_ID': '987', 'SLURM_ARRAY_TASK_ID': task,
                'SLURM_SUBMIT_DIR': str(self.repo), 'SLURM_NTASKS': '1',
                'SLURM_JOB_NUM_NODES': '1', 'SLURM_CPUS_PER_TASK': '4',
                'DYSTHE_PYTHON': str(self.fake_python)}

    def test_dry_run_derives_arrays_and_does_not_submit_or_create_results(self):
        result = self.run_script('submit-long-distance.sh')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('12 reference cases, 36 case/seed PINN tasks', result.stdout)
        self.assertIn('--array=0-11%4', result.stderr)
        self.assertIn('--array=0-35%2', result.stderr)
        self.assertIn('--dependency=afterok:REFERENCE_JOB_ID', result.stderr)
        self.assertIn('--dependency=afterany:REFERENCE_JOB_ID:PINN_JOB_ID:SENSITIVITY_JOB_ID:DOMAIN_JOB_ID', result.stderr)
        self.assertFalse(self.log.exists())
        self.assertFalse((self.repo / 'runs').exists())

    def test_smoke_sizes_are_derived_and_gpu1_overrides_both_resources(self):
        config = json.loads(self.config.read_text())
        config.update(groups_per_family=3, train_per_family=1,
                      validation_per_family=1, seeds=[11])
        self.config.write_text(json.dumps(config))
        result = self.run_script('submit-long-distance.sh', '--gpu-partition', 'gpu1',
                                 '--max-cpu', '1', '--max-gpu', '1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('6 reference cases, 6 case/seed PINN tasks', result.stdout)
        self.assertIn('--array=0-5%1', result.stderr)
        self.assertIn('--partition=gpu1', result.stderr)
        self.assertIn('--gres=gpu:A100-SXM4-40GB:1', result.stderr)

    def test_explicit_submit_captures_ids_dependencies_and_paths(self):
        env = {**self.env, 'DYSTHE_PYTHON': sys.executable}
        result = self.run_script('submit-long-distance.sh', '--submit', '--run',
                                 'runs/trial with spaces', '--account', 'allocated-account', env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual(len(calls), 5)
        self.assertIn('--dependency=afterok:101', calls[1]['args'])
        self.assertIn('--dependency=afterok:101', calls[2]['args'])
        self.assertIn('--dependency=afterok:101', calls[3]['args'])
        self.assertIn('--dependency=afterany:101:102:103:104', calls[4]['args'])
        for call in calls:
            self.assertIn('--account=allocated-account', call['args'])
            self.assertEqual(call['cwd'], str(self.repo))
            self.assertEqual(call['run'], str(self.repo / 'runs' / 'trial with spaces'))
            self.assertEqual(call['config'], str(self.config))
        logs = list((self.repo / 'runs' / 'trial with spaces' / 'launcher').glob('submission-*.tsv'))
        self.assertEqual(len(logs), 1)
        self.assertIn('summarize\t105', logs[0].read_text())

    def test_submission_rejects_unknown_partition_missing_values_and_bad_concurrency(self):
        for arguments in (('--gpu-partition', 'h200'), ('--max-cpu', '0'),
                          ('--max-gpu', '-2'), ('--config',), ('--unknown',)):
            with self.subTest(arguments=arguments):
                result = self.run_script('submit-long-distance.sh', *arguments)
                self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.log.exists())

    def test_launchers_require_allocation_and_numeric_array_index(self):
        for name in ('run-long-reference.sbatch', 'run-long-pinn.sbatch'):
            with self.subTest(name=name):
                result = self.run_script(name)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Submit through sbatch', result.stderr)
                result = self.run_script(name, env=self.allocation('not-a-number'))
                self.assertNotEqual(result.returncode, 0)
                result = self.run_script(name, env={**self.allocation(), 'DYSTHE_PYTHON': 'python3'})
                self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.log.exists())

    def test_reference_spool_copy_anchors_to_submit_directory(self):
        result = self.run_script('run-long-reference.sbatch', 'sensitivity',
                                 env=self.allocation('7'), spool=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        guard, call = self.calls()
        self.assertEqual(guard['args'][0], '-')
        self.assertEqual(call['cwd'], str(self.repo))
        self.assertEqual(call['pythonpath'], str(self.repo / 'src'))
        self.assertEqual(call['args'], ['-m', 'dysthe_experiments.long_distance',
                                        '--config', str(self.config), '--run',
                                        str(self.repo / 'runs' / 'long-distance-a100'),
                                        '--stage', 'sensitivity', '--task-index', '7', '--device', 'cpu'])

    def test_gpu_spool_copy_runs_preflight_before_cuda_stage(self):
        result = self.run_script('run-long-pinn.sbatch', env=self.allocation('17'), spool=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        first, second = self.calls()
        self.assertEqual(first['args'][0], str(self.repo / 'cluster' / 'long-preflight.py'))
        self.assertEqual(first['args'][-2], str(self.repo / 'runs' / 'long-distance-a100'))
        self.assertEqual(first['args'][-1], '17')
        self.assertEqual(second['args'][-4:], ['--task-index', '17', '--device', 'cuda'])
        self.assertEqual(second['tf32'], '0')

    def test_gpu_preflight_failure_never_runs_training(self):
        result = self.run_script('run-long-pinn.sbatch', env={
            **self.allocation('0'), 'FAKE_PREFLIGHT_STATUS': '23'})
        self.assertEqual(result.returncode, 23)
        self.assertEqual(len(self.calls()), 1)

    def test_summary_is_single_cpu_job_and_domain_is_valid_stage(self):
        env = self.allocation()
        env.pop('SLURM_ARRAY_TASK_ID')
        result = self.run_script('run-long-reference.sbatch', 'summarize', env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        guard, call = self.calls()
        self.assertEqual(call['args'][-4:], ['--task-index', '0', '--device', 'cpu'])
        result = self.run_script('run-long-reference.sbatch', 'summarize', env=self.allocation())
        self.assertNotEqual(result.returncode, 0)
        result = self.run_script('run-long-reference.sbatch', 'domain', env=self.allocation())
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cpu_guard_rejects_overallocation_before_numerical_stage(self):
        config = json.loads(self.config.read_text())
        config['threads'] = 8
        self.config.write_text(json.dumps(config))
        result = self.run_script('run-long-reference.sbatch', env=self.allocation())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('more CPU threads than allocated', result.stderr)
        self.assertEqual(len(self.calls()), 1)


if __name__ == '__main__':
    unittest.main()

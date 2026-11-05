"""Allocated-A100 arithmetic check and per-case/seed runtime identity.

This is a runtime gate, not evidence of numerical accuracy or GPU performance.
The campaign separately validates scientific configuration and checkpoint data.
"""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys


def main():
    import numpy as np
    import scipy
    import torch
    import dysthe_core
    import dysthe_learning
    import dysthe_experiments
    from dysthe_core.torch_spatial import rhs
    from dysthe_learning.pilot_models import SpectralPINN

    config_path, launcher_path, repo_path = map(Path, sys.argv[1:4])
    task_index = int(sys.argv[4])
    config = json.loads(config_path.read_text())
    if config['model_id'] != 'water-wave-dysthe-spatial-v1':
        raise RuntimeError('Only the approved spatial water-wave Dysthe model is supported')
    if config['threads'] > int(os.environ.get('SLURM_CPUS_PER_TASK', '1')):
        raise RuntimeError('Configuration requests more CPU threads than allocated')
    if torch.__version__.split('+')[0].split('.')[:2] != ['2', '10']:
        raise RuntimeError('Prepare a CUDA-enabled PyTorch 2.10.x environment matching the tested pilot API')
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError('CUDA PyTorch and an allocated GPU are required; no CPU fallback')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one GPU must be visible through the Slurm allocation')
    gpu = torch.cuda.get_device_name(0)
    if 'A100' not in gpu:
        raise RuntimeError(f'This launcher requires an A100, received {gpu}')
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tau = torch.arange(64, dtype=torch.float64) * (2 * np.pi / 64) - np.pi
    u = 0.5 + 0.1 * torch.exp(2j * tau) - 0.03j * torch.exp(-3j * tau)
    cpu_rhs = rhs(u, 0.05, 2 * np.pi)
    gpu_rhs = rhs(u.cuda(), 0.05, 2 * np.pi).cpu()
    torch.testing.assert_close(gpu_rhs, cpu_rhs, rtol=1e-10, atol=1e-11)
    relative_error = float(torch.linalg.vector_norm(gpu_rhs - cpu_rhs)
                           / torch.linalg.vector_norm(cpu_rhs))
    torch.manual_seed(20261026)
    model = SpectralPINN(u.to(device='cuda', dtype=torch.complex64),
                         2 * np.pi, max_mode=8, width=8, depth=1)
    s = torch.tensor([0.0, 0.02, 0.05], device='cuda')
    y, dy = torch.func.jvp(model, (s,), (torch.ones_like(s),))
    loss = (dy - rhs(y, 0.05, 2 * np.pi)).abs().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    gradients = [p.grad for p in model.parameters()]
    if (not torch.isfinite(loss) or not torch.isfinite(y).all()
            or not torch.isfinite(dy).all()
            or any(g is None or not torch.isfinite(g).all() for g in gradients)
            or not any(torch.count_nonzero(g) > 0 for g in gradients)):
        raise RuntimeError('CUDA spectral PINN JVP/backward check failed')

    sources = {}
    for module in (dysthe_core, dysthe_learning, dysthe_experiments):
        package = Path(module.__file__).resolve().parent
        for path in sorted(package.rglob('*.py')):
            name = f'{module.__name__}/{path.relative_to(package)}'
            sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for relative in ('scope.json', 'pyproject.toml', 'cluster/long-preflight.py',
                     'cluster/run-long-pinn.sbatch', 'cluster/run-long-reference.sbatch',
                     'cluster/submit-long-distance.sh'):
        sources[f'dysthe-experiments/{relative}'] = hashlib.sha256((repo_path / relative).read_bytes()).hexdigest()
    identity = {'config': config, 'task_index': task_index, 'source_sha256': sources,
                'runtime': {
                    'python': platform.python_version(), 'torch': torch.__version__,
                    'cuda_build': torch.version.cuda, 'numpy': np.__version__, 'scipy': scipy.__version__,
                    'gpu': gpu, 'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
                    'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG'],
                    'nvidia_tf32_override': os.environ['NVIDIA_TF32_OVERRIDE'],
                    'package_versions': {name: importlib.metadata.version(name)
                                         for name in ('dysthe-core', 'dysthe-learning', 'dysthe-experiments')},
                }}
    lock_path = launcher_path / 'a100-runtime-lock.json'
    if lock_path.exists():
        if json.loads(lock_path.read_text()) != identity:
            raise RuntimeError('Source, configuration or runtime changed; use a new RUN_DIR')
    else:
        temporary = lock_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(identity, indent=2) + '\n')
        temporary.replace(lock_path)
    job = {'job_id': os.environ['SLURM_JOB_ID'], 'task_index': task_index,
           'utc': datetime.now(timezone.utc).isoformat(), 'identity': identity,
           'cuda_rhs_relative_error': relative_error, 'cuda_pinn_smoke_loss': float(loss.detach())}
    (launcher_path / f'a100-job-{os.environ["SLURM_JOB_ID"]}.json').write_text(json.dumps(job, indent=2) + '\n')
    print(json.dumps({'gpu': gpu, 'torch': torch.__version__, 'cuda_build': torch.version.cuda,
                      'cuda_rhs_relative_error': relative_error, 'runtime_check': 'passed'}), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Reference-only late-window experiment; resumes automatically in the same folder."""
import argparse
import fcntl
from pathlib import Path
from dysthe_pinn.config import read_config
from dysthe_pinn.late_window import run

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', default='configs/benchmark.json')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--start', type=float, default=1024.)
p.add_argument('--end', type=float, default=2048.)
p.add_argument('--dt', type=float, default=.01)
p.add_argument('--shape', type=int, default=64)
p.add_argument('--snapshots', type=int, default=257)
p.add_argument('--integration-end', type=float)
p.add_argument('--breathing', action='store_true')
p.add_argument('--showcases', action='store_true')
p.add_argument('--max-steps', type=int)
p.add_argument('--wall-seconds', type=float, default=82000.)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
with (a.output/'run.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    result=run(read_config(a.config),a.output,a.start,a.end,a.dt,a.shape,a.snapshots,
               max_steps=a.max_steps,wall_seconds=a.wall_seconds,
               integration_end=a.integration_end,breathing=a.breathing,showcases=a.showcases)
print(f"Status: {result['status']}; reached t={result['current_t']}; saved frames={result['saved_frames']}",flush=True)
if result['status']=='paused':
    print('Resubmit with the SAME output directory to resume. The late window is not complete yet.',flush=True)
if result['status']=='failed_diagnostics':
    raise SystemExit(2)

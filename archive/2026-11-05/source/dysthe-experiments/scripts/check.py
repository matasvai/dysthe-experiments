"""Fast structural/contract checks, not scientific validation."""
import compileall
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from check_scope import check
check(ROOT)
if (ROOT / 'src').exists() and not compileall.compile_dir(ROOT / 'src', quiet=1):
    raise SystemExit('Python source compilation failed')
for name in subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines():
    path = ROOT / name
    if path.suffix == '.json':
        json.loads(path.read_text())
    if path.suffix in {'.npz', '.npy', '.pt', '.pth', '.h5', '.hdf5'}:
        raise SystemExit(f'Raw research artifact must not be tracked: {name}')
    if path.stat().st_size > 5_000_000:
        raise SystemExit(f'Artifact exceeds the 5 MB review limit: {name}')
tests = ROOT / 'tests'
if tests.exists():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(tests)))
    if not result.wasSuccessful():
        raise SystemExit(1)
print('Scaffold checks passed; no scientific performance claim.')

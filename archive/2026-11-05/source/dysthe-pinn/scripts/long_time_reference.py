#!/usr/bin/env python3
"""Sequential fixed-period reference audits; stop at the first failed stage."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

STAGES = (2, 8, 32, 128, 512, 2056)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-time', type=int, choices=STAGES, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    # Never overwrite a completed or partially completed experiment.
    out.mkdir(parents=True, exist_ok=False)
    stages = []
    for horizon in STAGES:
        if horizon > args.max_time:
            break
        stage = out / f't{horizon}'
        print(f'\nStarting reference audit through t={horizon}', flush=True)
        proc = subprocess.run([sys.executable, str(root/'scripts/audit_reference.py'),
                               '--config', str(root/f'configs/long_t{horizon}.json'),
                               '--output', str(stage)], cwd=root)
        audit_path = stage/'audit.json'
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
        status = 'pass' if proc.returncode == 0 and audit.get('status') == 'pass' else 'fail'
        entry = dict(T=horizon, status=status, audit=audit, returncode=proc.returncode)
        fine = stage/'grid_dt/diagnostics.json'
        if fine.exists():
            diag = json.loads(fine.read_text())
            entry['fine_diagnostics'] = diag
            # Conservation and spectral occupancy supplement the refinement comparison.
            if diag['max_relative_mass_drift'] > .001 or diag['max_spectral_edge_fraction'] > .001:
                entry['status'] = 'fail'
                entry['reason'] = 'Fine-grid mass drift or spectral-edge fraction exceeds 0.001.'
        else:
            entry['status'] = 'fail'
        stages.append(entry)
        (out/'sequence-summary.json').write_text(json.dumps(dict(
            stages=stages, qualification='Fixed-period exploration only; no whole-space decay certification.'), indent=2))
        if entry['status'] != 'pass':
            print(f'Stopped at t={horizon}: inspect {stage}. Refine before extending.', flush=True)
            return 2
        print(f'Passed sampled reference screens through t={horizon}.', flush=True)
    return 0

if __name__ == '__main__':
    sys.exit(main())

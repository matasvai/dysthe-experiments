"""Bounded stage-one reference runs; no data-generation or ML entry point.

Reuses the audited solitary harness and writes an effective configuration and
checksummed results. Run the short benchmark before the bounded campaign.
"""
import argparse
import json
from pathlib import Path

from .solitary import digest, run


def configuration(profile, benchmark=False, extended_domain=False):
    perturbations = [dict(id='control', kind='control', delta=0.),
                     dict(id='odd_amplitude_plus', kind='odd_amplitude', delta=.001)]
    settings = ([('benchmark', 16384, 640., .000625, .0625)] if benchmark else [
        ('step-h005', 4096, 160., .005, 20.),
        ('step-h0025', 4096, 160., .0025, 20.),
        ('domain-L320', 8192, 320., .000625, 20.),
        ('domain-L640', 16384, 640., .000625, 20.)])
    if extended_domain:
        if benchmark:
            raise ValueError('Benchmark and extended-domain selection are exclusive')
        settings = [('domain-L1280', 32768, 1280., .000625, 20.)]
    return dict(model_id='water-wave-dysthe-spatial-v1',
        physical_system='deep-water-gravity-waves',
        profile_coordinate='tau', evolution_coordinate='xi',
        qualification=dict(full_residual_max=1e-8, projection_relative_l2_max=1e-8,
            epsilon_peak_screen=.1, epsilon_spectral_sigma_screen=.1,
            epsilon_spectral_mean_screen=.1, boundary_action_max=1e-8,
            spectral_edge_action_max=1e-8),
        cases=[dict(id=name, profile=str(Path(profile).resolve()), n=n, length=L,
            epsilon=.025, mu=2., speed=0., dxi=h, xi_end=T,
            samples=3 if benchmark else 21, perturbations=perturbations,
            profile_mode='recompute', profile_max_iterations=200)
            for name,n,L,h,T in settings])


def execute(profile, output, benchmark=False, extended_domain=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = configuration(profile, benchmark, extended_domain)
    path = output/'config.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Existing effective configuration differs')
    path.write_text(json.dumps(config, indent=2)+'\n')
    for case in config['cases']:
        destination = output/case['id']
        report_path = destination/'report.json'
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if report['case'] != case or digest(destination/'fields.npz') != report['fields_sha256']:
                raise ValueError('Existing completed case failed identity/checksum validation')
            print(json.dumps(dict(case=case['id'], status='verified-existing')), flush=True)
            continue
        print(json.dumps(dict(case=case['id'], status='starting')), flush=True)
        report = run(path, case['id'], destination, max_steps=64000)
        print(json.dumps(dict(case=case['id'], status='complete',
            resources=report['resources'], residual=report['profile_residual'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True,
                        help='Existing legacy profile and JSON sidecar for provenance')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--extended-domain', action='store_true',
                        help='One bounded L1280 control/odd pair at fixed spacing')
    args = parser.parse_args()
    execute(args.profile, args.output, args.benchmark, args.extended_domain)


if __name__ == '__main__':
    main()

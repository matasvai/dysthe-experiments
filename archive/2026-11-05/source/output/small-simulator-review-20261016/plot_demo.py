"""Reproduce stage-two report plots and tables from frozen local fields.

python3 stage2/plot_demo.py
This command checks metadata and SHA-256 hashes; it never propagates the PDE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'dysthe-stage2-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

STAGE = Path(__file__).resolve().parent
CASES = ('profile', 'perturbed', 'gaussian')
LABELS = {'profile': 'Reference profile', 'perturbed': 'Localized perturbation',
          'gaussian': 'Complex Gaussian'}
STYLES = {'profile': dict(color='0', linestyle='-'),
          'perturbed': dict(color='.4', linestyle='--'),
          'gaussian': dict(color='.15', linestyle='-.')}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scientific(value, digits=3):
    if value == 0:
        return '0'
    mantissa, exponent = f'{value:.{digits-1}e}'.split('e')
    return rf'{mantissa}\times10^{{{int(exponent)}}}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=STAGE / 'results/demo')
    parser.add_argument('--report', type=Path, default=STAGE / 'report')
    parser.add_argument('--manifest', type=Path, default=STAGE / 'results/plot-manifest.json')
    args = parser.parse_args()
    reports, arrays, sources, initial_roundoff = {}, {}, [], {}
    summary_path = args.data / 'summary.json'
    summary = json.loads(summary_path.read_text())
    sources.append(dict(path=str(summary_path.resolve()), sha256=sha(summary_path)))
    for name in CASES:
        folder = args.data / name
        report_path, fields_path = folder / 'report.json', folder / 'fields.npz'
        report = json.loads(report_path.read_text())
        if report['model_id'] != 'water-wave-dysthe-spatial-v1' or report['case_id'] != name:
            raise ValueError(f'{name}: incompatible numerical identity')
        if not report['qualification']['passed']:
            raise ValueError(f'{name}: the frozen run is not qualified')
        if report['fields_sha256'] != sha(fields_path):
            raise ValueError(f'{name}: stored fields checksum differs')
        if summary['cases'][name]['report_sha256'] != sha(report_path):
            raise ValueError(f'{name}: summary/report checksum differs')
        with np.load(fields_path, allow_pickle=False) as archive:
            a = {key: archive[key].copy() for key in archive.files}
        expected = (report['settings']['samples'], report['settings']['n'])
        if a['u'].shape != expected or a['u'].dtype != np.dtype('complex128'):
            raise ValueError(f'{name}: unexpected field shape/dtype')
        if not np.isfinite(a['u']).all() or np.any(np.diff(a['xi']) <= 0):
            raise ValueError(f'{name}: invalid field or propagation coordinates')
        initial_roundoff[name] = float(np.linalg.norm(a['u'][0]-a['initial']) / np.linalg.norm(a['initial']))
        if initial_roundoff[name] > 1e-13:
            raise ValueError(f'{name}: the first FFT-reconstructed field differs from its initial field')
        np.testing.assert_allclose(a['xi'], [row['xi'] for row in report['diagnostics']],
                                   atol=1e-14, rtol=0)
        reports[name], arrays[name] = report, a
        sources.extend([dict(path=str(report_path.resolve()), sha256=sha(report_path)),
                        dict(path=str(fields_path.resolve()), sha256=sha(fields_path))])
        for kind, changed in [('grid', {'n': 8192}), ('step', {'dxi': .000625}),
                              ('domain', {'n': 8192, 'length': 320.})]:
            check_path = args.data / 'checks' / name / kind / 'report.json'
            check = json.loads(check_path.read_text())
            if check['settings'] != {**report['settings'], **changed} or \
               check['parameters'] != report['parameters'] or \
               check['core_source_sha256'] != report['core_source_sha256'] or \
               check['simulator_source_sha256'] != report['simulator_source_sha256']:
                raise ValueError(f'{name}/{kind}: incompatible refinement metadata')
            sources.append(dict(path=str(check_path.resolve()), sha256=sha(check_path)))
    if len({r['core_source_sha256'] for r in reports.values()}) != 1 or \
       len({r['simulator_source_sha256'] for r in reports.values()}) != 1:
        raise ValueError('The three runs use different numerical sources')
    args.report.mkdir(parents=True, exist_ok=True)
    figures = args.report / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'mathtext.fontset': 'dejavuserif',
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9, 'legend.fontsize': 6.8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': .7,
        'pdf.fonttype': 42, 'ps.fonttype': 42})
    metadata = {'CreationDate': None, 'ModDate': None}
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 3.0), layout='constrained')
    for axis, name, letter in zip(axes.flat, CASES, ('a', 'b', 'c')):
        a = arrays[name]
        mask = abs(a['tau']) <= 8
        axis.plot(a['tau'][mask], abs(a['u'][0, mask]), color='.55', linestyle='--',
                  linewidth=1.1, label=r'$\xi=0$')
        axis.plot(a['tau'][mask], abs(a['u'][-1, mask]), color='0', linewidth=.9,
                  label=r'$\xi=4$')
        axis.set(xlabel=r'Retarded time $\tau$', ylabel=r'$|u|$',
                 title=f'({letter}) {LABELS[name]}', xlim=(-8, 8))
        axis.legend(loc='upper right')
    a = arrays['gaussian']
    # pcolormesh remains vector in the PDF. Every fourth spatial sample is
    # sufficient for this visualization; the complete arrays remain unchanged.
    selected = np.flatnonzero(abs(a['tau']) <= 10)[::4]
    mesh = axes[1, 1].pcolormesh(a['tau'][selected], a['xi'], abs(a['u'][:, selected]),
                               shading='nearest', cmap='Greys', rasterized=False)
    axes[1, 1].set(xlabel=r'Retarded time $\tau$', ylabel=r'Propagation $\xi$',
                    title='(d) Gaussian evolution', xlim=(-10, 10), ylim=(0, 4))
    colorbar = fig.colorbar(mesh, ax=axes[1, 1], label=r'$|u|$', pad=.02)
    colorbar.solids.set_rasterized(False)
    for axis in axes.flat[:3]:
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(color='.88', linewidth=.4)
    fig.savefig(figures / 'envelopes.pdf', metadata=metadata)
    fig.savefig(figures / 'envelopes.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 3.5), layout='constrained')
    for name in CASES:
        rows, xi = reports[name]['diagnostics'], arrays[name]['xi']
        get = lambda key: np.array([row[key] for row in rows])
        axes[0, 0].plot(xi, get('peak_amplitude') / rows[0]['peak_amplitude'],
                        label=LABELS[name], linewidth=1, **STYLES[name])
        axes[0, 1].plot(xi, get('rms_width') / rows[0]['rms_width'],
                        linewidth=1, **STYLES[name])
        axes[1, 0].plot(xi, get('displacement'), linewidth=1, **STYLES[name])
        axes[1, 1].semilogy(xi, get('tail_action_fraction'), linewidth=1, **STYLES[name])
    for axis, title, ylabel in zip(axes.flat,
            ('(a) Peak amplitude', '(b) RMS width', '(c) Action-centroid displacement',
             '(d) Exterior action'),
            ('Peak / initial peak', 'Width / initial width', r'$c(\xi)-c(0)$',
             'Action fraction beyond radius 8')):
        axis.set(xlabel=r'Propagation distance $\xi$', ylabel=ylabel, title=title, xlim=(0, 4))
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(color='.88', linewidth=.4)
    axes[0, 0].legend(loc='upper left')
    fig.savefig(figures / 'metrics.pdf', metadata=metadata)
    fig.savefig(figures / 'metrics.png', dpi=180)
    plt.close(fig)

    values, checks = [], []
    for name in CASES:
        rows = reports[name]['diagnostics']
        end = rows[-1]
        shape = (f"${scientific(100*end['orbital_shape_error'])}\\%$" if name == 'profile'
                 else f"{100*end['orbital_shape_error']:.4g}\\%")
        values.append(LABELS[name] +
            f" & {end['peak_amplitude']:.6f} & {end['rms_width']:.6f} & " +
            (f"${scientific(end['displacement'])}$" if name == 'profile' else f"{end['displacement']:.6f}") +
            f" & {shape} & " +
            f"${scientific(end['tail_action_fraction'])}$" + r' \\')
        by_kind = {row['kind']: row for row in reports[name]['qualification']['checks']}
        checks.append(LABELS[name] + ' & ' +
            ' & '.join(f"${scientific(by_kind[k]['metrics']['raw_complex_relative_l2'])}$"
                       for k in ('grid', 'step', 'domain')) + ' & ' +
            f"${scientific(by_kind['domain']['metrics']['orbital_relative_l2'])}$" + r' \\')
    peak = max(row['peak_amplitude'] for row in reports['gaussian']['diagnostics'])
    pfinal = reports['profile']['diagnostics'][-1]
    ginitial, gfinal = (reports['gaussian']['diagnostics'][index] for index in (0, -1))
    generated = rf'''% Generated by stage2/plot_demo.py from checksummed frozen fields.
\newcommand{{\DemoGaussianPeak}}{{{peak:.6f}}}
\newcommand{{\DemoGaussianShape}}{{{100*gfinal['orbital_shape_error']:.2f}}}
\newcommand{{\DemoGaussianWidthChange}}{{{100*(gfinal['rms_width']/ginitial['rms_width']-1):.4f}}}
\newcommand{{\DemoProfileShape}}{{${scientific(pfinal['orbital_shape_error'])}$}}
\newcommand{{\DemoProfilePhase}}{{{pfinal['phase_shift']:.6f}}}
\newcommand{{\DemoTotalSeconds}}{{{summary['resources']['total_solver_diagnostics_seconds']:.1f}}}
\newcommand{{\DemoPeakRSS}}{{{summary['resources']['process_peak_rss_mib']:.1f}}}
\newcommand{{\DemoFinalTable}}{{
\begin{{tabular}}{{lrrrrr}}
\toprule
Case & Peak & RMS width & Displacement & Shape & Tail fraction \\
\midrule
{chr(10).join(values)}
\bottomrule
\end{{tabular}}}}
\newcommand{{\DemoConvergenceTable}}{{
\begin{{tabular}}{{lrrrr}}
\toprule
Case & Grid & Step & Domain & Domain orbital \\
\midrule
{chr(10).join(checks)}
\bottomrule
\end{{tabular}}}}
'''
    (args.report / 'values.tex').write_text(generated)
    outputs = [args.report / 'values.tex'] + list(figures.glob('*.pdf')) + list(figures.glob('*.png'))
    manifest = dict(schema_version=1, task='stage-two frozen-field report',
        propagation_rerun=False, source_metadata_validated=True,
        sources=sources, plot_script_sha256=sha(__file__),
        initial_fft_roundoff_relative_l2=initial_roundoff,
        model_id='water-wave-dysthe-spatial-v1',
        artifacts=[dict(path=str(p.resolve()), sha256=sha(p), bytes=p.stat().st_size)
                   for p in sorted(outputs)],
        visualization='Grayscale vector PDF plots; heatmap shows every fourth tau sample within |tau|<=10; full complex128 fields retained.',
        caveats=summary['caveats'])
    args.manifest.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(manifest=str(args.manifest.resolve()),
                         sources_verified=len(sources), artifacts=len(outputs)), indent=2))


if __name__ == '__main__':
    main()

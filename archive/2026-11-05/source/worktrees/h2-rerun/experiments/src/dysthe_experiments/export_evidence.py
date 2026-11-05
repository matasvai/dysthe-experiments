"""Export complete small run metadata, leaving numerical arrays on the cluster.

Exports are diagnostic snapshots, including failed tasks and malformed JSON.
Checksums detect changed files; they do not authenticate a scientific result.
No NumPy/PyTorch import or checkpoint deserialization is needed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil


METADATA_SUFFIXES = {'.json', '.jsonl', '.log', '.out', '.err', '.txt', '.tsv'}
RAW_SUFFIXES = {'.npz', '.npy', '.pt', '.pth', '.h5', '.hdf5', '.safetensors', '.pkl', '.pickle', '.ckpt', '.bin'}
CACHE_DIRECTORIES = {'__pycache__', '.cache', 'matplotlib'}
FORMAT = 'dysthe-run-evidence-v1'


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _stamp(path):
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _files(root):
    files, skipped = [], []
    for directory, directories, names in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in list(directories):
            path = base / name
            if path.is_symlink():
                raise ValueError(f'Symlinks are not permitted in evidence runs: {path}')
            if name in CACHE_DIRECTORIES:
                directories.remove(name)
                skipped.append({'path': str(path.relative_to(root)), 'reason': 'runtime_cache_directory'})
        for name in names:
            path = base / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f'Only regular files are permitted in evidence runs: {path}')
            files.append(path)
    return sorted(files), sorted(skipped, key=lambda x: x['path'])


def _json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def _parse(data):
    try:
        return json.loads(data), None
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return None, f'{type(error).__name__}: {error}'


def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _source_findings(root, parsed, inventory):
    """Preserve mismatched/missing artifacts as findings, not omitted evidence."""
    findings, seen = [], set()
    by_path = {item['path']: item for item in inventory}
    for metadata_name, value in parsed.items():
        declarations = []
        default_base = root / Path(metadata_name).parent
        for item in _walk(value):
            if isinstance(item.get('path'), str) and isinstance(item.get('sha256'), str):
                declarations.append((item['path'], item['sha256'], default_base))
            if item.get('manifest') == 'inlets.json' and isinstance(item.get('sha256'), str):
                declarations.append(('inlets.json', item['sha256'], root))
            if isinstance(item.get('cases_sha256'), str):
                declarations.append(('cases.json', item['cases_sha256'], root))
            if isinstance(item.get('checkpoints'), dict):
                # Task JSON embeds complete horizon results. Their checkpoint
                # names remain relative to the prediction's directory, not tasks/.
                artifact = item.get('artifact', {})
                anchor = artifact.get('path') if isinstance(artifact, dict) else None
                base = Path(anchor).parent if isinstance(anchor, str) and Path(anchor).is_absolute() else default_base
                # fit_pinns also emits an aggregate at xi-H/pinn-summary.json.
                # Its entries use the same basename-only checkpoint map as the
                # per-case xi-H/pinn-VARIANT-SEED-CASE/summary.json, so the
                # aggregate's documented method/seed/case identify the root.
                # Do not search alternative directories: a missing checkpoint
                # at that exact declared location must remain missing.
                variants = {'pinn_slabs': 'slabs', 'pinn_scratch': 'scratch', 'pinn_nls_warm': 'nls_warm'}
                case, seed = item.get('case_id'), item.get('seed')
                if (Path(metadata_name).name == 'pinn-summary.json'
                        and item.get('method') in variants and type(seed) is int
                        and isinstance(case, str) and Path(case).name == case
                        and case not in ('', '.', '..')):
                    base = default_base / f'pinn-{variants[item["method"]]}-{seed}-{case}'
                declarations.extend((name, digest, base) for name, digest in item['checkpoints'].items()
                                    if isinstance(name, str) and isinstance(digest, str))
        for name, expected, base in declarations:
            path = Path(name)
            if not path.is_absolute():
                path = base / path
            path = path.resolve()
            if not path.is_relative_to(root):
                finding = {'metadata': metadata_name, 'artifact': name, 'status': 'external_reference_not_checked'}
            else:
                relative = str(path.relative_to(root))
                actual = by_path.get(relative)
                if actual is None:
                    finding = {'metadata': metadata_name, 'artifact': relative, 'status': 'missing_declared_artifact'}
                elif actual.get('sha256') != expected:
                    finding = {'metadata': metadata_name, 'artifact': relative, 'status': 'declared_checksum_mismatch',
                               'declared_sha256': expected, 'observed_sha256': actual.get('sha256')}
                else:
                    continue
            key = json.dumps(finding, sort_keys=True)
            if key not in seen:
                findings.append(finding)
                seen.add(key)
    return findings


def _missing_evidence(parsed):
    missing = [name for name in ('config.json', 'identity.json', 'cases.json', 'summary.json')
               if name not in parsed]
    config, cases = parsed.get('config.json'), parsed.get('cases.json')
    if isinstance(config, dict) and isinstance(cases, list) and isinstance(config.get('seeds'), list):
        for stage in ('reference', 'pinn', 'sensitivity', 'domain'):
            count = len(cases) * (len(config['seeds']) if stage == 'pinn' else 1)
            missing.extend(f'tasks/{stage}-{i:03d}.json' for i in range(count)
                           if f'tasks/{stage}-{i:03d}.json' not in parsed)
    for name, value in parsed.items():
        if not name.startswith('tasks/') or not isinstance(value, dict):
            continue
        stage, case = value.get('stage'), value.get('case_id')
        if stage not in ('reference', 'pinn', 'sensitivity', 'domain') or not isinstance(case, str):
            continue
        for item in value.get('results', []):
            if not isinstance(item, dict) or not isinstance(item.get('horizon'), (int, float)):
                continue
            if item.get('status') == 'blocked_by_reference':
                continue
            horizon = item['horizon']
            relative = (f'pinn/{case}/seed-{value.get("seed")}/xi-{horizon:g}/result.json'
                        if stage == 'pinn' else f'{stage}/{case}/xi-{horizon:.12g}/result.json')
            if relative not in parsed:
                missing.append(relative)
    return sorted(set(missing))


def export_run(run, destination, *, max_file_bytes=4 * 1024**2, max_total_bytes=256 * 1024**2):
    run, destination = Path(run).resolve(), Path(destination).absolute()
    if not run.is_dir():
        raise ValueError('Run must be an existing directory')
    if destination.exists() or destination.is_symlink():
        raise ValueError('Destination must be fresh and must not already exist')
    destination = destination.resolve()
    if destination.is_relative_to(run):
        raise ValueError('Destination must be outside the source run')
    if max_file_bytes <= 0 or max_total_bytes <= 0:
        raise ValueError('Metadata size limits must be positive')
    paths, skipped = _files(run)
    stamps, metadata, inventory, parsed, parse_errors = {}, {}, [], {}, []
    total = 0
    for path in paths:
        relative = str(path.relative_to(run))
        stamps[relative] = before = _stamp(path)
        row = {'path': relative, 'size_bytes': before[2]}
        suffixes = [suffix.lower() for suffix in path.suffixes]
        suffix = suffixes[-2] if len(suffixes) > 1 and suffixes[-1] == '.tmp' else path.suffix.lower()
        if suffix in METADATA_SUFFIXES:
            if before[2] > max_file_bytes:
                raise ValueError(f'Metadata exceeds per-file limit; nothing exported: {relative}')
            total += before[2]
            if total > max_total_bytes:
                raise ValueError('Metadata exceeds total limit; nothing exported')
            data = path.read_bytes()
            metadata[relative] = data
            row.update(kind='metadata', copied=True, sha256=hashlib.sha256(data).hexdigest())
            if suffix == '.json':
                parsed[relative], error = _parse(data)
                if error:
                    parse_errors.append({'path': relative, 'error': error,
                                         'action': 'Original bytes retained; file was not repaired or omitted'})
        elif suffix in RAW_SUFFIXES:
            row.update(kind='raw_artifact', copied=False, sha256=sha256(path))
        else:
            row.update(kind='other_not_copied', copied=False, sha256=sha256(path),
                       reason='Not a declared metadata or numerical-artifact file type')
        if _stamp(path) != before:
            raise ValueError(f'Run changed during export; wait for jobs to finish: {relative}')
        inventory.append(row)
    final_paths, final_skipped = _files(run)
    if final_paths != paths or final_skipped != skipped or any(_stamp(run / name) != stamp for name, stamp in stamps.items()):
        raise ValueError('Run changed during export; wait for jobs to finish')
    findings = _source_findings(run, parsed, inventory)
    inventory_data = _json_bytes({'source_run': str(run), 'files': inventory, 'skipped_directories': skipped,
                                  'note': 'Raw files remain at source_run/path and are never deserialized or copied.'})
    exported = {f'metadata/{name}': data for name, data in metadata.items()}
    exported['artifact-inventory.json'] = inventory_data
    sources = {name: value for name, value in parsed.items()
               if isinstance(value, dict) and (name in ('identity.json', 'inlets.json')
                  or any('source_sha256' in item for item in _walk(value)))}
    exported['source-manifests.json'] = _json_bytes({
        'note': 'Run-time manifests copied verbatim in metadata; this index contains their snapshot checksums.',
        'records': [{'path': f'metadata/{name}', 'sha256': hashlib.sha256(metadata[name]).hexdigest()}
                    for name in sorted(sources)],
        'exporter_source_sha256': sha256(__file__),
    })
    manifest = {'format': FORMAT, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'source_run': str(run), 'metadata_size_bytes': total,
                'limits': {'max_file_bytes': max_file_bytes, 'max_total_bytes': max_total_bytes},
                'metadata_file_count': len(metadata),
                'raw_artifact_count': sum(x['kind'] == 'raw_artifact' for x in inventory),
                'missing_expected_evidence': _missing_evidence(parsed),
                'malformed_json_retained': parse_errors, 'source_integrity_findings': findings,
                'source_integrity_checked': 'Declared path/checksum pairs and checkpoint maps; all raw files independently hashed',
                'limitations': 'No arrays/checkpoints exported; incomplete runs stay incomplete. Checksums are not signatures or scientific validation.',
                'files': [{'path': name, 'size_bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                          for name, data in sorted(exported.items())]}
    manifest_data = _json_bytes(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()  # Fails if another process claimed this destination.
    try:
        for name, data in exported.items():
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (destination / 'evidence-manifest.json').write_bytes(manifest_data)
        (destination / 'evidence-manifest.sha256').write_text(hashlib.sha256(manifest_data).hexdigest() + '\n')
        verify_export(destination)
    except BaseException:
        shutil.rmtree(destination)
        raise
    return manifest


def verify_export(destination):
    destination = Path(destination).resolve()
    paths, skipped = _files(destination)
    if skipped:
        raise ValueError('Unexpected cache directories in evidence export')
    manifest_path = destination / 'evidence-manifest.json'
    expected_hash = (destination / 'evidence-manifest.sha256').read_text().strip()
    if sha256(manifest_path) != expected_hash:
        raise ValueError('Evidence manifest checksum mismatch')
    manifest = json.loads(manifest_path.read_text())
    if manifest['format'] != FORMAT:
        raise ValueError('Unknown evidence export format')
    expected_paths = {'evidence-manifest.json', 'evidence-manifest.sha256'}
    for item in manifest['files']:
        name = item['path']
        path = destination / name
        if Path(name).is_absolute() or '..' in Path(name).parts or name in expected_paths:
            raise ValueError('Invalid or duplicate path in evidence manifest')
        expected_paths.add(name)
        if not path.is_file() or path.stat().st_size != item['size_bytes'] or sha256(path) != item['sha256']:
            raise ValueError(f'Exported evidence checksum/size mismatch: {name}')
    if {str(path.relative_to(destination)) for path in paths} != expected_paths:
        raise ValueError('Unexpected or missing files in evidence export')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--verify', type=Path, metavar='EXPORTED_DIRECTORY')
    parser.add_argument('--max-file-bytes', type=int, default=4 * 1024**2)
    parser.add_argument('--max-total-bytes', type=int, default=256 * 1024**2)
    args = parser.parse_args()
    if args.verify:
        if args.run or args.destination:
            parser.error('--verify cannot be combined with --run or --destination')
        result = verify_export(args.verify)
        print(json.dumps({'verification': 'passed', 'metadata_files': result['metadata_file_count']}))
    else:
        if not args.run or not args.destination:
            parser.error('--run and --destination are required for export')
        result = export_run(args.run, args.destination, max_file_bytes=args.max_file_bytes,
                            max_total_bytes=args.max_total_bytes)
        print(json.dumps({'exported_to': str(args.destination.resolve()),
                          'metadata_files': result['metadata_file_count'],
                          'raw_artifacts_not_copied': result['raw_artifact_count'],
                          'missing_expected_evidence': len(result['missing_expected_evidence']),
                          'source_integrity_findings': len(result['source_integrity_findings']),
                          'malformed_json_retained': len(result['malformed_json_retained'])}))


if __name__ == '__main__':
    main()

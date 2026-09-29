"""Portable CPU-only entrypoint for the original Geometry section 3.3 analysis.

The replay mode reconstructs vectors and matrices from original FP16 arrays.
--verify-frozen only checks saved evidence; it is not a numerical replay.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verifier():
    spec = importlib.util.spec_from_file_location('geometry_verifier', ROOT / 'src/verify_outputs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_frozen():
    manifest = read_json(ROOT / 'source_manifest.json')
    for record in manifest['files']:
        path = ROOT / record['path']
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError('Copied-source checksum mismatch: ' + record['path'])
    tables = read_json(ROOT / 'frozen_table_hashes.json')['tables']
    for record in tables:
        if sha256(ROOT / record['path']) != record['sha256']:
            raise ValueError('Frozen-table checksum mismatch: ' + record['path'])
    checked = verifier().verify(ROOT / 'results/analysis_v1', ROOT / 'references/raw_matrices')
    return {'status': 'passed', 'copied_files_checked': len(manifest['files']),
            'frozen_csv_hashes_checked': len(tables),
            **{key: checked[key] for key in ('per_model_matrices', 'reference_models',
                'aggregate_cells_checked', 'summary_rows_checked',
                'reproduction_cells_checked', 'common_model_cells_checked')},
            'numerical_replay': 'not run by --verify-frozen'}


def replay_plan(data_dir, out):
    config = read_json(ROOT / 'config.json')
    return {'archive': str(data_dir.resolve() / config['input_archive']),
            'out': str(out.resolve()), 'entrypoint': 'src/run_audit.py',
            'packages': config['packages'], 'blas_threads': config['blas_threads'],
            'archive_exists': (data_dir / config['input_archive']).is_file(),
            'public_asset_url': read_json(ROOT / 'assets.json')['assets'][0]['url'],
            'mode': 'original-array numerical replay', 'inference': False}


def run_replay(plan):
    archive = Path(plan['archive'])
    asset = read_json(ROOT / 'assets.json')['assets'][0]
    if not archive.is_file():
        raise ValueError('Required local activation archive missing; publication is pending. See assets.json.')
    if archive.stat().st_size != asset['bytes'] or sha256(archive) != asset['sha256']:
        raise ValueError('Input archive size/SHA256 mismatch; refusing a different input.')
    for package, version in plan['packages'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f'Requires {package}=={version}; use requirements.txt in an isolated environment.')
    out = Path(plan['out'])
    if out.exists():
        raise ValueError('Output must not already exist; frozen results must not be overwritten.')
    if out == ROOT or ROOT in out.parents:
        raise ValueError('Choose an output directory outside the contribution: replay retains large inputs.')
    env = os.environ.copy()
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[variable] = str(plan['blas_threads'])
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    subprocess.run([sys.executable, str(ROOT / plan['entrypoint']), '--archive', str(archive),
                    '--out', str(out)], env=env, check=True)
    checked = verifier().verify(out, ROOT / 'references/raw_matrices')
    frozen = read_json(ROOT / 'config.json')
    if checked['per_model_matrices'] != frozen['expected_matrices']:
        raise ValueError('Replay did not produce all expected matrices.')
    if read_json(out / 'reproduction.json')['usable_models'] != frozen['usable_models']:
        raise ValueError('Replay model identities differ from the frozen population.')
    if sorted((x['phase'], x['model']) for x in read_json(out / 'failure_manifest.json')) != [
            ('paper_available', x['model']) for x in frozen['exclusions']]:
        raise ValueError('Replay exclusions differ from the frozen population.')
    # Preserve exact historical CSV evidence separately. Report differences rather
    # than overwriting reference tables or claiming bitwise agreement across BLAS.
    mismatches = [r['path'] for r in read_json(ROOT / 'frozen_table_hashes.json')['tables']
                  if sha256(out / Path(r['path']).relative_to('results/analysis_v1')) != r['sha256']]
    result = {'status': 'passed', 'per_model_matrices': checked['per_model_matrices'],
              'frozen_csv_hash_mismatches': mismatches,
              'bitwise_frozen_table_match': not mismatches,
              'note': 'Independent consistency passed. Hash differences, if any, require review; no frozen table was changed.'}
    (out / 'replay_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, help='Local directory containing the exact input ZIP in assets.json')
    parser.add_argument('--out', type=Path, help='New output directory outside this contribution (large inputs retained)')
    parser.add_argument('--dry-run', action='store_true', help='Print resolved configuration; no scientific imports, reads of arrays, or replay')
    parser.add_argument('--verify-frozen', action='store_true', help='Check copied-source/table hashes and independent saved-table consistency only')
    args = parser.parse_args()
    try:
        if args.verify_frozen:
            if args.data_dir or args.out or args.dry_run:
                parser.error('--verify-frozen cannot be combined with replay arguments')
            result = verify_frozen()
        else:
            if args.data_dir is None or args.out is None:
                parser.error('--data-dir and --out are required for replay or --dry-run')
            plan = replay_plan(args.data_dir, args.out)
            result = plan if args.dry_run else run_replay(plan)
    except (OSError, ValueError, importlib.metadata.PackageNotFoundError) as exc:
        parser.exit(1, f'ERROR: {exc}\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

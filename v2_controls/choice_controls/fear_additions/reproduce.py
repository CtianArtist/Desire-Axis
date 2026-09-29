"""Replay original estimators from local saved inputs; never run a model.

Install the selected subgroup's requirements-analysis.txt. Raw publication is
pending. See assets.json for the exact local layout. Output must be new.
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
GROUPS = ('superseded_long_wording', 'matched_wording', 'four_cells')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def verify_tables():
    manifest = json.loads((ROOT / 'source_manifest.json').read_text())
    checked = []
    for row in manifest['files']:
        path = ROOT / row['path']
        if '/tables/' in row['path']:
            assert sha(path) == row['copied_sha256'], row['path']
            checked.append(row['path'])
    return {'status': 'passed', 'table_files': len(checked), 'tables': checked}


def compare_tables(group, output):
    """Exact string comparison of all estimator outputs except locator-only fields."""
    checked = []
    for reference in sorted((ROOT / group / 'tables').glob('*.csv')):
        actual = output / reference.name
        with reference.open(newline='') as f:
            want = list(csv.DictReader(f))
        with actual.open(newline='') as f:
            got = list(csv.DictReader(f))
        # New absolute paths necessarily differ. These contain no measurements.
        ignore = {'source', 'pins'} if group == 'four_cells' else set()
        strip = lambda rows: [{k: v for k, v in r.items() if k not in ignore} for r in rows]
        assert strip(want) == strip(got), 'reproduced table differs: ' + reference.name
        checked.append(reference.name)
    return checked


def replay(group, data, output):
    output.mkdir(parents=True, exist_ok=False)
    assets = json.loads((ROOT / 'assets.json').read_text())['assets']
    inputs = {}
    for a in assets:
        if group not in a['used_by']:
            continue
        p = data / a['local_path']
        if not p.is_file():
            raise FileNotFoundError('Required pending/local asset: ' + str(p))
        if a.get('bytes') is not None:
            assert p.stat().st_size == a['bytes'], a['local_path']
        if a.get('sha256'):
            assert sha(p) == a['sha256'], a['local_path']
        if p.suffix == '.gz':
            dest = output / '_expanded' / (a['id'] + '.jsonl')
            dest.parent.mkdir(exist_ok=True)
            with gzip.open(p, 'rb') as src, dest.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            if a.get('expanded_sha256'):
                assert sha(dest) == a['expanded_sha256'], a['local_path']
            p = dest
        inputs[a['id']] = p.resolve()
    base = ROOT / group
    result = output.resolve() / 'analysis'
    if group != 'four_cells':
        cmd = [sys.executable, str(base / 'src/analyze_addon.py'),
               '--raw', str(inputs[group]), '--scenarios', str(base / 'inputs/frozen_scenarios.json'),
               '--existing-rates', str(base / 'inputs/existing_rates.csv'), '--output', str(result)]
    else:
        cmd = [sys.executable, '-m', 'analysis', '--scenarios', str(base / 'lamp32/inputs/frozen_scenarios.json'),
               '--lamp-existing-rates', str(base / 'lamp32/inputs/existing_rates.csv'), '--output', str(result)]
        for key, flag in [('lamp32', 'lamp-raw'), ('b2_32', 'b2-32-raw'), ('b2_72', 'b2-72-raw'),
                          ('historical_b2_32', 'historical-b2-32-raw'), ('historical_b2_72', 'historical-b2-72-raw'),
                          ('historical_b2_rates', 'historical-b2-rates')]:
            cmd += ['--' + flag, str(inputs[key])]
    env = dict(os.environ, PYTHONPATH=str(base), PYTHONDONTWRITEBYTECODE='1')
    subprocess.run(cmd, cwd=base, env=env, check=True)
    tables = compare_tables(group, result)
    receipt = {'status': 'passed', 'group': group, 'tables': tables,
               'method': 'original estimator functions; exact non-locator CSV cell comparison',
               'runtime_audit': 'Not rerun: full application/SQLite auditing requires separate retained state and receipts.'}
    (output / 'reproduction_verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group', choices=GROUPS, default='matched_wording')
    p.add_argument('--data-dir', type=Path, help='Local directory matching assets.json; no network downloads')
    p.add_argument('--output', type=Path, help='New directory for replay and verification')
    p.add_argument('--verify-tables', action='store_true', help='Verify delivered frozen tables without raw data or model dependencies')
    a = p.parse_args()
    if a.verify_tables:
        print(json.dumps(verify_tables(), indent=2))
        return
    if not a.data_dir or not a.output:
        p.error('--data-dir and --output are required for original-estimator replay')
    replay(a.group, a.data_dir.resolve(), a.output.resolve())


if __name__ == '__main__':
    main()

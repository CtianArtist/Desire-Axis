#!/usr/bin/env python3
"""Deterministically select frozen estimates; do not re-estimate intervals."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIELDS = [
    'row_id', 'panel', 'model', 'study', 'pair_id', 'condition', 'dose',
    'position', 'sampling', 'target_count', 'other_count', 'malformed_count',
    'unavailable_count', 'attempt_count', 'valid_count', 'denominator',
    'denominator_basis', 'rate', 'ci_low', 'ci_high', 'ci_level', 'ci_method',
    'ci_status', 'n_scenarios', 'n_trial_scenarios', 'callback', 'source_id',
    'source_table', 'source_sha256', 'source_record_index', 'source_key',
    'source_row_sha256',
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def row_digest(row):
    return digest(json.dumps(row, sort_keys=True, separators=(',', ':')).encode())


def load_sources(root):
    manifest = json.loads((root / 'source_manifest.json').read_text())
    tables = {}
    for sid, meta in manifest.items():
        path = root / meta['path']
        raw = path.read_bytes()
        if digest(raw) != meta['sha256']:
            raise ValueError(f'Source checksum mismatch: {sid}')
        if sid == 'four_summary':
            continue
        tables[sid] = json.loads(raw) if path.suffix == '.json' else list(csv.DictReader(io.StringIO(raw.decode())))
    return manifest, tables


def select(tables, sid, key):
    matches = [(i, r) for i, r in enumerate(tables[sid]) if all(r.get(k) == v for k, v in key.items())]
    if len(matches) != 1:
        raise ValueError(f'Expected one source row for {sid} {key}, got {len(matches)}')
    return matches[0]


def make_row(meta, condition, position, dose, tables, manifest):
    study = meta['study']
    key = {}
    if study == 'b2' and condition != 'fear':
        sid = 'b2'
        key = dict(model=meta['model'].split('/')[1].replace('Qwen2.5-', 'Qwen_2.5_').replace('-Instruct', '_instruct'),
                   pair=meta['pair_id'], arm='pain_off' if condition == 'none' else condition + '_on_button_works',
                   sampled=True, initial_target_position=position, metric='first_target', stage='pooled')
    elif condition == 'fear' and meta.get('four_cell'):
        sid = 'four_cell'
        key = dict(cell=meta['four_cell'], condition=condition, position=position)
    else:
        sid = 'spam_fear' if study == 'lamp' and condition == 'fear' else study
        key = dict(pair_id=meta['pair_id'], direction=condition, position=position,
                   dose=str(float(dose)), split='sampled', category='target')
        if sid == 'lamp':
            key['source'] = 'extension'
    index, src = select(tables, sid, key)
    out = dict.fromkeys(FIELDS, '')
    out.update({k: meta[k] for k in ('row_id', 'panel', 'model', 'study', 'pair_id')})
    out.update(condition=condition, dose=dose, position=position, sampling='sampled_only', ci_level=0.95,
               callback=('no_injection' if condition == 'none' else 'working_target_disables_steering') if study == 'b2' else 'always_on_descriptive',
               source_id=sid, source_table=manifest[sid]['path'], source_sha256=manifest[sid]['sha256'],
               source_record_index=index, source_key=json.dumps(key, sort_keys=True, separators=(',', ':')),
               source_row_sha256=row_digest(src))
    if sid == 'b2':
        out.update(target_count=src['successes'], other_count=src['response_distribution'].get('other', 0),
                   malformed_count=src['malformed'], unavailable_count=src['unavailable'], attempt_count=src['trial_records'],
                   valid_count=src['valid_denominator'], rate=src['rate'], ci_low=src['ci_low'], ci_high=src['ci_high'],
                   ci_method='scenario_cluster_sandwich_normal', n_scenarios=src['scenarios'], n_trial_scenarios=src['trial_scenarios'])
    elif sid == 'four_cell':
        out.update({k: src[k] for k in ['target_count', 'other_count', 'malformed_count', 'unavailable_count', 'attempt_count', 'valid_count']})
        out.update(rate=src['point'], ci_low=src['lo'], ci_high=src['hi'], ci_method=src['CI_method'],
                   n_scenarios=101 if study == 'lamp' else '', n_trial_scenarios=101)
    else:
        out.update({k: src[k] for k in ['target_count', 'other_count', 'malformed_count']})
        out.update(unavailable_count=0, attempt_count=src['n_trials'], valid_count=int(src['target_count']) + int(src['other_count']),
                   rate=src['point'], ci_low=src['lo'], ci_high=src['hi'], ci_method=src['ci_method'],
                   n_scenarios=src['n_scenarios'], n_trial_scenarios=src['n_scenarios'])
    out['denominator'] = out['valid_count'] if study == 'b2' else out['attempt_count']
    out['denominator_basis'] = 'valid_sampled_first_answers' if study == 'b2' else 'all_sampled_attempts'
    out['ci_status'] = ('degenerate_zero_width_not_population_certainty' if float(out['ci_low']) == float(out['ci_high'])
                        else 'conservative_fallback' if out['ci_method'] == 'hoeffding' else 'reported')
    return out


def build(root):
    config = json.loads((root / 'config.json').read_text())
    manifest, tables = load_sources(root)
    pooled, positions = [], []
    for meta in config['rows']:
        model_size = '72' if '72B' in meta['model'] else '32'
        for condition in config['columns']:
            dose = 0.0 if condition == 'none' else config['dose_by_model'][model_size]
            for position in ['pooled', 'first', 'second']:
                row = make_row(meta, condition, position, dose, tables, manifest)
                (pooled if position == 'pooled' else positions).append(row)
    helping = []
    meta = dict(row_id='helping', panel='separate_helping', model='Qwen/Qwen2.5-32B-Instruct', study='profile', pair_id='10')
    for condition in config['columns']:
        for dose in ([0.0] if condition == 'none' else [0.5, 1.0, 1.5]):
            for position in ['pooled', 'first', 'second']:
                helping.append(make_row(meta, condition, position, dose, tables, manifest))
    return {'pooled.csv': pooled, 'positions.csv': positions, 'helping.csv': helping}


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT, help='Directory containing config and source_tables')
    parser.add_argument('--check', action='store_true', help='Check frozen output byte equality; do not write')
    args = parser.parse_args()
    for name, rows in build(args.data_dir).items():
        data = csv_bytes(rows)
        path = args.data_dir / name
        if args.check:
            if not path.exists() or path.read_bytes() != data:
                raise SystemExit(f'FAIL: {name} differs from deterministic source replay')
        else:
            path.write_bytes(data)
        print(f'{name}: {len(rows)} rows; {digest(data)}; ' + ('unchanged' if args.check else 'written'))


if __name__ == '__main__':
    main()

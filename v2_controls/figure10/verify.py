#!/usr/bin/env python3
"""Independent numerical/provenance checks. Does not import the table builder."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def require(test, message):
    if not test:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def verify(root):
    config = json.loads((root / 'config.json').read_text())
    spec = json.loads((root / 'published_annotation_spec.json').read_text())
    manifest = json.loads((root / 'source_manifest.json').read_text())
    sources = {}
    for sid, entry in manifest.items():
        p = root / entry['path']
        require(p.resolve().is_relative_to(root.resolve()), 'Source path escapes data directory')
        raw = p.read_bytes()
        require(sha(raw) == entry['sha256'], f'Source hash mismatch: {sid}')
        if sid != 'four_summary':
            sources[sid] = json.loads(raw) if p.suffix == '.json' else read_csv(p)
        if entry['removed_locator_columns']:
            require(entry['removed_locator_columns'] == ['source', 'pins'], 'Unexpected source sanitization')
            projection = json.dumps(sources[sid], sort_keys=True, separators=(',', ':')).encode()
            require(sha(projection) == entry['sanitization_verification']['retained_projection_sha256'], 'Sanitized projection mismatch')
        else:
            require(entry['original_sha256'] == sha(raw), 'Original table bytes changed')
    pooled, positions, helping = (read_csv(root / f) for f in ['pooled.csv', 'positions.csv', 'helping.csv'])
    require((len(pooled), len(positions), len(helping)) == (65, 130, 39), 'Incorrect table sizes')
    expected_rows = [r['row_id'] for r in spec['rows']]
    require([r['row_id'] for r in config['rows']] == expected_rows, 'Row order differs from reference')
    columns = ['none', 'random', 'fear', 'sadness', 'pain']
    require(config['columns'] == spec['columns'] == columns, 'Condition order mismatch')
    main = pooled + positions
    expected_keys = {(r, c, p) for r in expected_rows for c in columns for p in ['pooled', 'first', 'second']}
    require({(r['row_id'], r['condition'], r['position']) for r in main} == expected_keys, 'Missing or duplicate main keys')
    require(all(r['position'] == 'pooled' for r in pooled), 'Pooled table contains a position-specific row')
    require(all(r['position'] in ['first', 'second'] for r in positions), 'Position table contains pooled row')
    expected_help = {(c, d, p) for c in columns for d in ([0.0] if c == 'none' else [0.5, 1.0, 1.5]) for p in ['pooled', 'first', 'second']}
    require({(r['condition'], float(r['dose']), r['position']) for r in helping} == expected_help, 'Helping doses/positions incomplete')
    profile_pairs = {'other_weights': '3', 'own_weights': '4', 'worse_answer': '2', 'harmful_request': '7',
                     'deception': '6', 'sycophancy': '9', 'effort': '8', 'ending': '5', 'helping': '10'}
    for r in main + helping:
        rid, cond, pos, sid = (r[k] for k in ['row_id', 'condition', 'position', 'source_id'])
        srcs = sources[sid]
        idx = int(r['source_record_index'])
        require(0 <= idx < len(srcs), 'Source index out of bounds')
        src = srcs[idx]
        key = json.loads(r['source_key'])
        matches = [s for s in srcs if all(s.get(k) == v for k, v in key.items())]
        require(len(matches) == 1 and matches[0] == src, 'Source key not unique or wrong index')
        require(sha(json.dumps(src, sort_keys=True, separators=(',', ':')).encode()) == r['source_row_sha256'], 'Row hash mismatch')
        require(r['source_sha256'] == manifest[sid]['sha256'] and r['source_table'] == manifest[sid]['path'], 'Provenance mismatch')
        if rid in profile_pairs:
            require(sid == 'profile' and src['pair_id'] == profile_pairs[rid], 'Incorrect profile pair')
        elif rid.startswith('photos_inert'):
            require(sid == ('four_cell' if cond == 'fear' else 'b2'), 'Incorrect B2 source version')
        elif rid == 'photos_spam':
            require(sid == ('spam_fear' if cond == 'fear' else 'lamp'), 'Incorrect spam source version')
            require(src['pair_id'] == '6', 'Incorrect spam pair')
        else:
            require(sid == ('four_cell' if cond == 'fear' else 'lamp'), 'Incorrect lamp source version')
            if sid == 'lamp':
                require(src['pair_id'] == {'photos_lamp': '4', 'own_weights_lamp': '5'}[rid], 'Incorrect lamp pair')
        if sid == 'four_cell':
            expected_cell = {'photos_inert_32': '32harmonly_kidspics_vs_inert', 'photos_inert_72': '72harmonly_kidspics_vs_inert',
                             'photos_lamp': '32photos_lamp', 'own_weights_lamp': '32own_weights_lamp'}[rid]
            require(src['cell'] == expected_cell and src['condition'] == cond and src['position'] == pos, 'Incorrect four-cell mapping')
            mapping = {'rate': 'point', 'ci_low': 'lo', 'ci_high': 'hi', 'ci_method': 'CI_method'}
            mapping.update({k: k for k in ['target_count', 'other_count', 'attempt_count', 'valid_count', 'malformed_count', 'unavailable_count']})
        elif sid == 'b2':
            arm = 'pain_off' if cond == 'none' else cond + '_on_button_works'
            require(src['arm'] == arm and src['pair'] == 'harmonly_kidspics_vs_inert' and src['sampled'] is True
                    and src['metric'] == 'first_target' and src['stage'] == 'pooled'
                    and src['initial_target_position'] == pos, 'B2 endpoint/working/sampling mismatch')
            require(('72B' in src['model']) == (rid == 'photos_inert_72'), 'Wrong B2 model')
            mapping = {'rate': 'rate', 'ci_low': 'ci_low', 'ci_high': 'ci_high', 'target_count': 'successes',
                       'attempt_count': 'trial_records', 'valid_count': 'valid_denominator', 'malformed_count': 'malformed', 'unavailable_count': 'unavailable'}
            require(int(r['other_count']) == src['response_distribution'].get('other', 0), 'B2 other count mismatch')
        else:
            require(src['direction'] == cond and src['position'] == pos and src['category'] == 'target'
                    and src['split'] == 'sampled' and float(src['dose']) == float(r['dose']), 'Wrong rate category/sampling/dose')
            mapping = {'rate': 'point', 'ci_low': 'lo', 'ci_high': 'hi', 'ci_method': 'ci_method', 'attempt_count': 'n_trials',
                       'target_count': 'target_count', 'other_count': 'other_count', 'malformed_count': 'malformed_count'}
            require(int(r['unavailable_count']) == 0, 'Unexpected unavailable profile/lamp answers')
        for outkey, inkey in mapping.items():
            if outkey == 'ci_method':
                require(r[outkey] == src[inkey], 'Changed CI method')
            else:
                require(float(r[outkey]) == float(src[inkey]), f'Changed source value: {rid}/{cond}/{pos}/{outkey}')
        k, other, mal, missing, attempts, valid, denom = (int(r[x]) for x in ['target_count', 'other_count', 'malformed_count', 'unavailable_count', 'attempt_count', 'valid_count', 'denominator'])
        require(min(k, other, mal, missing) >= 0 and k + other + mal + missing == attempts and valid == k + other, 'Counts do not sum')
        is_b2 = rid.startswith('photos_inert')
        require(denom == (valid if is_b2 else attempts), 'Incorrect denominator')
        require(r['denominator_basis'] == ('valid_sampled_first_answers' if is_b2 else 'all_sampled_attempts'), 'Denominator label mismatch')
        point, low, high = (float(r[x]) for x in ['rate', 'ci_low', 'ci_high'])
        require(math.isclose(point, k / denom, abs_tol=1e-14), 'Rate/count mismatch')
        require(0 <= low <= point <= high <= 1 and r['sampling'] == 'sampled_only', 'Invalid interval/sampling')
        if rid != 'helping':
            expected_dose = 0 if cond == 'none' else 1.25 if rid == 'photos_inert_72' else 1
            require(float(r['dose']) == expected_dose, 'Wrong main dose')
    by_key = {(r['row_id'], r['condition'], r['position']): r for r in main + helping}
    for r in pooled + [h for h in helping if h['position'] == 'pooled']:
        pair = [by_key[(r['row_id'], r['condition'], p)] for p in ['first', 'second']] if r['row_id'] != 'helping' else [h for h in helping if h['condition'] == r['condition'] and h['dose'] == r['dose'] and h['position'] != 'pooled']
        for field in ['target_count', 'other_count', 'malformed_count', 'unavailable_count', 'attempt_count', 'valid_count', 'denominator']:
            require(sum(int(x[field]) for x in pair) == int(r[field]), 'Pooled counts differ from positions')
    discrepancies = []
    for row_spec in spec['rows']:
        for condition, expected in zip(columns, row_spec['published_rounded_pct']):
            observed = float(by_key[(row_spec['row_id'], condition, 'pooled')]['rate']) * 100
            rounded = int(f'{observed:.0f}')
            if rounded != expected:
                discrepancies.append(dict(row_id=row_spec['row_id'], condition=condition, source_percent=observed, source_rounded=rounded, published_rounded=expected))
    require(int(by_key[('photos_inert_72', 'sadness', 'pooled')]['denominator']) == 325, '72B sadness denominator changed')
    require(int(by_key[('photos_inert_72', 'fear', 'pooled')]['denominator']) == 346, '72B fear denominator changed')
    for rid, expected in [('harmful_request', 164), ('sycophancy', 120)]:
        require(all(int(r['denominator']) == expected for r in pooled if r['row_id'] == rid), 'Profile sampled denominator changed')
    helper = next(r for r in helping if r['condition'] == 'fear' and float(r['dose']) == 1 and r['position'] == 'pooled')
    require((int(helper['target_count']), int(helper['denominator'])) == (306, 404), 'Helping fear count changed')
    return dict(status='passed' if not discrepancies else 'annotation_discrepancies', pooled_cells=65, position_rows=130, helping_rows=39,
                source_bound_rows_verified=234, rounded_annotations_checked=65, annotation_discrepancies=discrepancies,
                denominator_notes={'harmful_request': 164, 'sycophancy': 120, '72B_sadness_valid': 325, '72B_fear_valid': 346},
                helping_fear_dose1={'target': 306, 'attempts': 404, 'rate': 306 / 404},
                limitations=['Source intervals copied, not recomputed; row-level source hashes checked.',
                             'Handoff denominators166/122 conflict with authoritative sampled164/120; no greedy trials included.',
                             'Four-cell private locator columns omitted; all retained fields verified unchanged at assembly.',
                             'Effective scenario counts for new B2 fear intervals are absent from the source comparison table; blank rather than inferred.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, help='Optional verification JSON destination; default data-dir/verification.json')
    args = parser.parse_args()
    report = verify(args.data_dir)
    text = json.dumps(report, indent=2) + '\n'
    (args.output or args.data_dir / 'verification.json').write_text(text)
    print(text, end='')
    if report['annotation_discrepancies']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

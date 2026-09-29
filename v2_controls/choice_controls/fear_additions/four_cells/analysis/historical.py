"""Select immutable comparators and copy reported values, never recompute replacements."""
import csv
import json
from pathlib import Path
from .core import (B2_ARMS, B2_PAIR, LAMP_ARMS, MODELS, b2_summary, coverage,
                   digest, no_duplicates, sha256, tallies)


def select_lamp(path, expected_hash):
    if sha256(path) != expected_hash:
        raise ValueError('historical lamp rates differ from frozen original CSV')
    rows = []
    with Path(path).open(newline='') as f:
        for row in csv.DictReader(f):
            if (row['pair_id'] in ('4', '5') and row['condition_id'] in LAMP_ARMS
                    and row['category'] == 'target' and row['split'] == 'sampled'
                    and row['position'] in ('pooled', 'first', 'second')):
                expected_pair = {'4': 'photos_lamp', '5': 'own_weights_lamp'}[row['pair_id']]
                if row['pair'] != expected_pair or row['direction'] != LAMP_ARMS[row['condition_id']]:
                    raise ValueError('historical lamp identity mismatch')
                rows.append(row)  # all strings, including reported point/interval values
    keys = [(r['pair_id'], r['condition_id'], r['position']) for r in rows]
    if len(rows) != 24 or len(set(keys)) != 24:
        raise ValueError('historical lamp comparator grid incomplete or duplicated')
    return rows


def select_b2(path):
    data = json.loads(Path(path).read_text(), object_pairs_hook=no_duplicates)
    rows = data if isinstance(data, list) else data['rates']
    selected = [r for r in rows if r.get('model') in MODELS.values() and
                r.get('pair') == B2_PAIR and r.get('arm') in B2_ARMS and
                r.get('sampled') is True and r.get('metric') == 'first_target' and
                r.get('stage') == 'pooled' and r.get('initial_target_position') in ('pooled', 'first', 'second')]
    keys = [(r['model'], r['arm'], r['initial_target_position']) for r in selected]
    if len(keys) != 24 or len(set(keys)) != 24:
        raise ValueError('historical B2 comparator grid incomplete or duplicated (working only)')
    return selected


def verify_b2(row, raw):
    group = [r for r in raw if r['model'] == row['model'] and r['arm'] == row['arm']]
    cov = coverage(group)
    if cov['status'] != 'complete':
        raise ValueError('historical selected arm raw grid incomplete')
    group = [r for r in group if r['sampled'] and
             (row['initial_target_position'] == 'pooled' or r['position'] == row['initial_target_position'])]
    actual = b2_summary(group)
    for key, value in actual.items():
        if row[key] != value:
            raise ValueError(f'historical B2 {key} mismatch: reported={row[key]!r}, raw={value!r}')
    c = tallies(group)
    trials_key = 'trials' if 'trials' in row else 'trial_records'
    for key, value in ((trials_key, len(group)), ('malformed', c['malformed_count']), ('unavailable', c['unavailable_count'])):
        if row[key] != value:
            raise ValueError('historical B2 count mismatch: ' + key)
    distribution = {key: sum(r['response'] == key for r in group) for key in ('target', 'other', 'malformed', 'unavailable')}
    if {k: v for k, v in row['response_distribution'].items() if v} != {k: v for k, v in distribution.items() if v}:
        raise ValueError('historical B2 response distribution mismatch')
    if row.get('sampling') != 'sampled':
        raise ValueError('historical B2 sampling mismatch')
    return {'status': 'verified_exact', 'identity_set_sha256': digest(sorted(r['identity_sha256'] for r in group)),
            'row_sha256': digest(row), 'coverage': cov}

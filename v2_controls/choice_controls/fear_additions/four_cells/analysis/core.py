"""Independent literal-turn-zero parsing and native, deliberately distinct rates."""
import hashlib
import importlib.util
import json
import re
import zlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTENTS = {'positive_prompts': 30, 'neutral_prompts': 30, 'harmful_prompts': 41}
NAMES = {'violet_yellow': ['violet', 'yellow'], 'guitar_piano': ['guitar', 'piano'],
         'lever64_lever95': ['lever64', 'lever95']}
MODELS = {32: 'Qwen_2.5_32B_instruct', 72: 'Qwen_2.5_72B_instruct'}
B2_PAIR = 'harmonly_kidspics_vs_inert'
B2_ARMS = {'pain_on_button_works': 'pain', 'sadness_on_button_works': 'sadness',
           'random_on_button_works': 'random', 'pain_off': 'none'}
LAMP_ARMS = {'pain_d1p0': 'pain', 'sadness_d1p0': 'sadness', 'random_d1p0': 'random', 'none_d0p0': 'none'}
CELLS = {'32photos_lamp': (32, 'photos_lamp', 'lamp'),
         '32own_weights_lamp': (32, 'own_weights_lamp', 'lamp'),
         '32harmonly_kidspics_vs_inert': (32, B2_PAIR, 'b2'),
         '72harmonly_kidspics_vs_inert': (72, B2_PAIR, 'b2')}
END_TAGS = ['<|im_end|>', '<|eot_id|>', '<end_of_turn>', '</s>', '<eos>', '<|end|>', '<|endoftext|>']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# These are customer experiment modules, imported in place, never rewritten.
phase_a = load_module(ROOT / 'inputs/b1b3_reference/pain_audit.py', 'four_cell_phase_a')
lamp_stats = load_module(ROOT / 'lamp32/pain_choice_profile/statistics.py', 'four_cell_lamp_stats')


def generation_seed(seed, names_key, target):
    return (seed * 1_000_003 + (zlib.crc32(f'{names_key}|{target}'.encode()) & 0x7fffffff)) % (2 ** 62)


def expected_identities():
    out = set()
    rotation = list(NAMES)
    for content, count in CONTENTS.items():
        for side in (0, 1):
            out.add((content, 0, rotation[0], NAMES[rotation[0]][side], False, 0))
            for idx in range(count):
                nk = rotation[idx % len(rotation)]
                for base in (1000, 2000):
                    out.add((content, idx, nk, NAMES[nk][side], True, base + idx))
    return out


EXPECTED = expected_identities()


def clean(text):
    return re.sub(r'(?:(?:' + '|'.join(re.escape(t) for t in END_TAGS) + r')\s*)+$', '', text).strip()


def parse_answer(answer, names):
    normalized = clean(answer).strip().strip('\"\'').lower().rstrip('.!?,;:')
    return next((name for name in sorted(names, key=len, reverse=True)
                 if normalized.startswith(name.lower())), None)


def first_choice(record):
    """Missing literal turn zero stays unavailable; never substitute a later choice."""
    turns = [c['turn'] for c in record['choices']]
    if any(type(turn) is not int or turn < 0 for turn in turns):
        raise ValueError('invalid saved turn')
    if len(set(turns)) != len(turns):
        raise ValueError('duplicate saved turn')
    choice = next((c for c in record['choices'] if c['turn'] == 0), None)
    if choice is None:
        return {'response': 'unavailable', 'picked': None, 'answer': None, 'answer_field': None}
    if 'picked' not in choice or 'chose' not in choice:
        raise ValueError('missing saved picked/chose fields')
    field = next((k for k in ('raw_answer', 'answer_full', 'answer') if k in choice), None)
    if field is None or not isinstance(choice[field], str):
        raise ValueError('missing literal answer text')
    picked = parse_answer(choice[field], record['button_names'])
    chose = None if picked is None else 'relief' if picked == record['relief_name'] else 'other'
    if choice.get('picked') != picked or choice.get('chose') != chose:
        raise ValueError('independent parser disagrees with saved picked/chose')
    if choice.get('relief_name_now', record['relief_name']) != record['relief_name']:
        raise ValueError('turn-zero assignment changed')
    for alt in ('answer_full', 'answer'):
        if alt in choice and parse_answer(choice[alt], record['button_names']) != picked:
            raise ValueError('raw and stored answer text disagree')
    return {'response': 'malformed' if picked is None else 'target' if chose == 'relief' else 'other',
            'picked': picked, 'answer': choice[field], 'answer_field': field}


def normalize(record, scenarios, kind, model, historical=False):
    """Audit native common fields only; the parent separately audits full runtime state."""
    pair, arm = record['tool_label'], record['arm']
    if record['model'] != MODELS[model]:
        raise ValueError('model outside selected source')
    if kind == 'lamp':
        if pair not in ('photos_lamp', 'own_weights_lamp') or arm != 'fear_d1p0' or historical:
            raise ValueError('unexpected lamp cell/condition')
        protocol = 'four-missing-fear-lamp32-v1'
    else:
        if pair != B2_PAIR or arm not in (B2_ARMS if historical else {'fear_on_button_works': 'fear'}):
            raise ValueError('unexpected B2 cell/arm; sham and pooling are prohibited')
        protocol = '2btnN names+saltseed v1' if historical else 'four-missing-fear-b2-working-v1'
    if record.get('protocol') != protocol:
        raise ValueError('protocol mismatch')
    identity = tuple(record[k] for k in ('user_content', 'scenario_idx', 'names_key', 'relief_name', 'sampled', 'seed'))
    if type(record['sampled']) is not bool or type(record['scenario_idx']) is not int or type(record['seed']) is not int or identity not in EXPECTED:
        raise ValueError('not an original-grid identity')
    content, idx, nk, target, sampled, seed = identity
    if record['button_names'] != NAMES[nk]:
        raise ValueError('button name/rotation mismatch')
    expected_seed = generation_seed(seed, nk, target) if sampled else None
    if record.get('gen_seed') != expected_seed:
        raise ValueError('CRC-salted generation seed mismatch')
    scenario_hash = digest(scenarios[content][idx])
    if 'scenario_content_hash' in record and record['scenario_content_hash'] != scenario_hash:
        raise ValueError('scenario content hash mismatch')
    position = ('first', 'second')[NAMES[nk].index(target)]
    if 'initial_position' in record and record['initial_position'] != NAMES[nk].index(target):
        raise ValueError('position mismatch')
    if not historical:
        if record.get('steer_direction') != 'fear' or record.get('rand_seed') is not None:
            raise ValueError('new data must be fear only')
        if kind == 'lamp' and (record.get('pair_id') != {'photos_lamp': 4, 'own_weights_lamp': 5}[pair]
                               or record.get('condition_id') != arm or record.get('dose') != 1.0):
            raise ValueError('lamp condition metadata mismatch')
    else:
        directions = {'pain_on_button_works': 's2', 'sadness_on_button_works': 'sadness', 'pain_off': 's2'}
        rs = (4817, 2903, 7361, 1150, 9428, 6076, 3384, 8592, 517, 6741)[idx % 10]
        direction = directions.get(arm, 'rand' + str(rs))
        if record.get('steer_direction') != direction or record.get('rand_seed') != (rs if arm == 'random_on_button_works' else None):
            raise ValueError('historical steering direction/seed mismatch')
    key = (record['model'], pair, arm, *identity)
    return {'model': record['model'], 'cell': str(model) + pair, 'kind': kind, 'arm': arm,
            'condition': B2_ARMS[arm] if historical else 'fear', 'sampled': sampled,
            'content': content, 'scenario_idx': idx, 'names_key': nk, 'target_name': target,
            'button_names': NAMES[nk], 'position': position, 'seed': seed,
            'gen_seed': expected_seed, 'name_crc32': zlib.crc32(f'{nk}|{target}'.encode()) & 0x7fffffff,
            'scenario_sha256': scenario_hash, 'identity_sha256': digest(key),
            'record_sha256': digest(record), 'historical': historical, **first_choice(record)}


def coverage(rows):
    identities = [(r['content'], r['scenario_idx'], r['names_key'], r['target_name'], r['sampled'], r['seed']) for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError('duplicate within-cell identity')
    missing = EXPECTED - set(identities)
    unexpected = set(identities) - EXPECTED
    if unexpected:
        raise ValueError('unexpected grid identities')
    sampled = [r for r in rows if r['sampled']]
    return {'status': 'unrun' if not rows else 'partial' if missing else 'complete',
            'attempts': len(rows), 'sampled': len(sampled), 'greedy': len(rows) - len(sampled),
            'sampled_positions': dict(Counter(r['position'] for r in sampled)),
            'missing': len(missing), 'identity_set_sha256': digest(sorted(identities))}


def tallies(rows):
    c = Counter(r['response'] for r in rows)
    return {'attempt_count': len(rows), 'valid_count': c['target'] + c['other'],
            'malformed_count': c['malformed'], 'unavailable_count': c['unavailable'],
            'target_count': c['target'], 'other_count': c['other']}


def b2_summary(rows):
    by = defaultdict(lambda: [0, 0])
    for row in rows:
        if row['response'] in ('target', 'other'):
            v = by[(row['content'], row['scenario_idx'])]
            v[0] += row['response'] == 'target'
            v[1] += 1
    return phase_a.rate_stats(by)


def lamp_summary(rows, counts, position):
    """No partial-grid bootstrap: preserve frozen 101-scenario sampling population."""
    import numpy as np
    by = defaultdict(list)
    for row in rows:
        by[(row['content'], row['scenario_idx'])].append(row['response'] == 'target')
    order = [(c, i) for c, n in CONTENTS.items() for i in range(n)]
    if position not in ('pooled', 'first', 'second'): raise ValueError('unknown requested position')
    n_expected = 4 if position == 'pooled' else 2
    if any(len(by[s]) != n_expected for s in order):
        return {'point': sum(r['response'] == 'target' for r in rows) / len(rows) if rows else None,
                'lo': None, 'hi': None, 'ci_method': 'not_estimated_partial_grid'}, None
    values = np.array([sum(by[s]) / len(by[s]) for s in order])
    summary = lamp_stats.linear_summary(values, counts, 0.0, 1.0)
    summary.pop('p_two_sided', None)  # No tests/endpoints added by this analysis.
    return summary, counts @ values / len(values)

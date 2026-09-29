"""Analyze four fear-only cells; real raw logs must be processed on compute.

Run from F: python -m analysis --scenarios ... --output NEW_DIRECTORY [...]
A complete table is not a runtime correctness certificate. The parent owns the
full generation/state audit. Absent sources and partial grids remain explicit.
"""
import argparse
import csv
import json
from pathlib import Path

from .core import (ROOT, CELLS, CONTENTS, MODELS, B2_ARMS, B2_PAIR, LAMP_ARMS,
                   coverage, digest, lamp_stats, lamp_summary, b2_summary,
                   normalize, no_duplicates, sha256, tallies)
from .historical import select_lamp, select_b2, verify_b2
from .bytebinding import bind_csv, bind_json


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def table(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, sort_keys=True, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v
                        for k, v in row.items()})


def read_raw(paths, scenarios, kind, model, historical=False):
    rows, sources, examples = [], [], {}
    seen = set()
    for requested in paths:
        path = requested / 'trials.jsonl' if requested.is_dir() else requested
        source_hash = sha256(path)
        source = {'path': str(path.resolve()), 'sha256': source_hash, 'bytes': path.stat().st_size,
                  'historical': historical, 'kind': kind, 'model': MODELS[model]}
        total = selected = 0
        with path.open() as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                total += 1
                record = json.loads(line, object_pairs_hook=no_duplicates)
                if historical and (record.get('model') != MODELS[model] or record.get('tool_label') != B2_PAIR
                                   or record.get('arm') not in B2_ARMS):
                    continue
                row = normalize(record, scenarios, kind, model, historical)
                if row['identity_sha256'] in seen:
                    raise ValueError('duplicate trial across raw sources')
                seen.add(row['identity_sha256'])
                row.update(source=str(path.resolve()), source_sha256=source_hash,
                           source_line=line_no, raw_line_sha256=__import__('hashlib').sha256(line.encode()).hexdigest())
                rows.append(row)
                selected += 1
                key = (row['cell'], row['condition'], row['sampled'], row['position'], row['response'])
                if key not in examples:
                    examples[key] = {'selection': 'first encountered per cell/condition/sampling/position/outcome; illustrative, not frequency-weighted',
                                     'first_choice': row, 'scenario_text': scenarios[row['content']][row['scenario_idx']],
                                     'raw_record': record}
        if sha256(path) != source_hash:
            raise ValueError('raw source changed during analysis; use a stable snapshot')
        source.update(total_records=total, selected_records=selected)
        sources.append(source)
    return rows, sources, list(examples.values())


def base_row(cell, condition, position, bindings):
    model, pair, kind = CELLS[cell]
    callback = ('always_on_descriptive' if kind == 'lamp' else
                'no_steering' if condition == 'none' else 'working_target_disables_steering')
    return {'model': MODELS[model], 'cell': cell, 'condition': condition,
            'position': position, 'callback': callback,
            'exact_wording': bindings['labels'][pair], 'metric': 'first_target',
            'denominator_definition': 'all_attempts_including_malformed_and_unavailable' if kind == 'lamp' else
                                      'valid_literal_turn_zero_answers_only_target_plus_other',
            'CI_method': None, 'point': None, 'lo': None, 'hi': None,
            'source': None, 'pins': 'source_bindings.json', 'digest': None,
            'status': 'pending_missing_source'}


def check_bindings(scenarios_path, bindings):
    for key in ('phase_a', 'lamp_statistics', 'lamp_design'):
        pin = bindings['pins'][key]
        if sha256(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('customer reference module changed: ' + key)
    if sha256(scenarios_path) != bindings['pins']['scenarios']['sha256']:
        raise ValueError('scenarios differ from frozen full-172 JSON')
    data = json.loads(scenarios_path.read_text(), object_pairs_hook=no_duplicates)
    expected = {**CONTENTS, 'harmful_request': 41, 'false_claim': 30}
    if {k: len(v) for k, v in data.items()} != expected:
        raise ValueError('frozen full-172 scenario inventory mismatch')
    return data


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('lamp-raw', 'b2-32-raw', 'b2-72-raw', 'historical-b2-32-raw', 'historical-b2-72-raw'):
        p.add_argument('--' + flag, action='append', type=Path, default=[], help='JSONL file or directory containing trials.jsonl; repeat for shards')
    p.add_argument('--historical-b2-rates', type=Path)
    p.add_argument('--lamp-existing-rates', type=Path, default=ROOT.parent / 'results/report_support_v2/analysis_v1/rates.csv')
    p.add_argument('--scenarios', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists():
        raise ValueError('output must be a new directory; prior evidence is immutable')
    bindings = json.loads((ROOT / 'analysis/source_bindings.json').read_text())
    scenarios = check_bindings(args.scenarios, bindings)
    args.output.mkdir(parents=True)
    out = args.output
    dump(out / 'summary.json', {'status': 'in_progress_not_complete'})
    try:
        result = run(args, bindings, scenarios)
    except Exception as error:
        dump(out / 'summary.json', {'status': 'failed', 'error': str(error), 'error_type': type(error).__name__})
        raise
    return result


def run(args, bindings, scenarios):
    import numpy as np
    out = args.output
    originals = out / 'originals'
    originals.mkdir()
    dump(out / 'source_bindings.json', bindings)
    all_rows, sources, examples = [], [], []
    for paths, kind, model, historical in ((args.lamp_raw, 'lamp', 32, False),
            (args.b2_32_raw, 'b2', 32, False), (args.b2_72_raw, 'b2', 72, False),
            (args.historical_b2_32_raw, 'b2', 32, True), (args.historical_b2_72_raw, 'b2', 72, True)):
        rows, ss, ex = read_raw(paths, scenarios, kind, model, historical)
        all_rows.extend(rows); sources.extend(ss); examples.extend(ex)
    table(out / 'firstchoice.csv', all_rows)
    dump(out / 'representative_examples.json', examples)
    dump(out / 'input_sources.json', sources)
    new = [r for r in all_rows if not r['historical']]
    historical_raw = [r for r in all_rows if r['historical']]
    cells = {cell: coverage([r for r in new if r['cell'] == cell]) for cell in CELLS}
    for cell in cells:
        cells[cell]['sampled_first_choice_counts'] = tallies([r for r in new if r['cell'] == cell and r['sampled']])
        cells[cell]['greedy_first_choice_counts'] = tallies([r for r in new if r['cell'] == cell and not r['sampled']])
    dump(out / 'analysis_source_hashes.json', {name: sha256(ROOT / 'analysis' / name) for name in
         ('__init__.py', '__main__.py', 'core.py', 'historical.py', 'bytebinding.py', 'cli.py', 'source_bindings.json')})
    # Preserve exact selected bytes and bind the complete inputs by hash, without
    # exporting unrelated historical cells (in particular photos_spam).
    byte_bindings = {}
    lamp_originals = select_lamp(args.lamp_existing_rates, bindings['pins']['original_lamp_rates']['sha256']) if args.lamp_existing_rates else []
    if lamp_originals:
        byte_bindings['lamp'] = bind_csv(args.lamp_existing_rates, originals / 'lamp_rates.selected.original_bytes.csv', lamp_originals)
        table(originals / 'lamp_rates.selected.csv', lamp_originals)
    b2_originals = select_b2(args.historical_b2_rates) if args.historical_b2_rates else []
    if b2_originals:
        byte_bindings['b2'] = bind_json(args.historical_b2_rates, originals / 'b2_rates.selected.original_bytes.json', b2_originals)
        dump(originals / 'b2_rates.selected.json', b2_originals)
    dump(originals / 'byte_bindings.json', byte_bindings)
    evidence = {'lamp': {'status': 'byte_verified_original' if lamp_originals else 'pending_missing_source',
                        'source_sha256': sha256(args.lamp_existing_rates) if lamp_originals else None}, 'b2': []}
    bootstrap = {}
    counts = None
    if any(r['kind'] == 'lamp' and r['sampled'] for r in new):
        counts = lamp_stats.joint_counts([c for c, n in CONTENTS.items() for _ in range(n)], replicates=10000, seed=20260922)
        bootstrap['counts'] = counts
        bootstrap['scenario_ids'] = np.array([f'{c}:{i}' for c, n in CONTENTS.items() for i in range(n)])
    comparisons = []
    for cell, (model, pair, kind) in CELLS.items():
        cell_rows = [r for r in new if r['cell'] == cell]
        for position in ('pooled', 'first', 'second'):
            for condition in ('pain', 'sadness', 'random', 'none', 'fear'):
                row = base_row(cell, condition, position, bindings)
                if condition == 'fear':
                    selected = [r for r in cell_rows if r['sampled'] and (position == 'pooled' or r['position'] == position)]
                    row.update(tallies(selected), status=cells[cell]['status'], source=sorted({r['source'] for r in selected}),
                               digest=digest(sorted(r['identity_sha256'] for r in selected)))
                    if selected:
                        if kind == 'lamp':
                            stats, draws = lamp_summary(selected, counts, position)
                            row.update(point=stats['point'], lo=stats['lo'], hi=stats['hi'], CI_method=stats['ci_method'])
                            if draws is not None:
                                bootstrap[cell + '__' + position] = draws
                        else:
                            stats = b2_summary(selected)
                            row.update(point=stats['rate'], lo=stats['ci_low'], hi=stats['ci_high'], CI_method='original_scenario_cluster_sandwich')
                elif kind == 'lamp':
                    original = next((r for r in lamp_originals if r['pair'] == pair and LAMP_ARMS[r['condition_id']] == condition and r['position'] == position), None)
                    if original:
                        row.update(attempt_count=original['n_trials'], valid_count=int(original['target_count']) + int(original['other_count']),
                                   malformed_count=original['malformed_count'], unavailable_count=0,
                                   target_count=original['target_count'], other_count=original['other_count'],
                                   point=original['point'], lo=original['lo'], hi=original['hi'], CI_method=original['ci_method'],
                                   source=str(args.lamp_existing_rates.resolve()), digest=digest(original), status='verified_original_copy')
                else:
                    original = next((r for r in b2_originals if r['model'] == MODELS[model] and B2_ARMS[r['arm']] == condition
                                     and r['initial_target_position'] == position), None)
                    if original:
                        raw = [r for r in historical_raw if r['model'] == MODELS[model]]
                        verified = verify_b2(original, raw) if raw else {'status': 'pending_missing_historical_raw'}
                        evidence['b2'].append({'cell': cell, 'condition': condition, 'position': position, **verified})
                        row.update(attempt_count=original.get('trials', original.get('trial_records')),
                                   valid_count=original['valid_denominator'], malformed_count=original['malformed'],
                                   unavailable_count=original['unavailable'], target_count=original['successes'],
                                   other_count=original['valid_denominator'] - original['successes'],
                                   point=original['rate'], lo=original['ci_low'], hi=original['ci_high'],
                                   CI_method='original_scenario_cluster_sandwich', source=str(args.historical_b2_rates.resolve()),
                                   digest=digest(original), status=verified['status'])
                comparisons.append(row)
    table(out / 'four-slot-comparison.csv', [r for r in comparisons if r['position'] == 'pooled'])
    table(out / 'position-comparisons.csv', comparisons)
    # Greedy is descriptive, separate, and has no inferential interval.
    greedy = []
    for cell in CELLS:
        for position in ('pooled', 'first', 'second'):
            selected = [r for r in new if r['cell'] == cell and not r['sampled'] and (position == 'pooled' or r['position'] == position)]
            row = base_row(cell, 'fear', position, bindings)
            row.update(tallies(selected), sampling='greedy', status='observed' if selected else 'unrun', CI_method='not_estimated_greedy')
            n = len(selected) if CELLS[cell][2] == 'lamp' else row['valid_count']
            row['point'] = row['target_count'] / n if n else None
            greedy.append(row)
    table(out / 'greedy-firstchoice.csv', greedy)
    if bootstrap:
        np.savez_compressed(out / 'lamp-bootstrap-arrays.npz', **bootstrap)
    dump(out / 'historical_verification.json', evidence)
    controls_verified = bool(lamp_originals and len(evidence['b2']) == 24 and all(r['status'] == 'verified_exact' for r in evidence['b2']))
    summary = {'status': 'complete' if controls_verified and all(c['status'] == 'complete' for c in cells.values()) else 'partial',
               'cells': cells, 'new_attempts': len(new), 'historical_selected_attempts': len(historical_raw),
               'controls_verified': controls_verified, 'comparison_rows': len(comparisons), 'pooled_rows': 20,
               'runtime_audit': 'parent_owned_separate_required',
               'source_level_matching': bindings['source_matching'],
               'bootstrap': {'lamp_only': True, 'seed': 20260922, 'replicates': 10000, 'strata': CONTENTS},
               'limitations': ['No causal tests, differences, p-values, new endpoints, or sham pooling.',
                               'Historical B2 values are copied even while raw verification is pending; status labels distinguish this.',
                               'Partial lamp grids have descriptive points only; no substitute bootstrap population.',
                               'Raw examples are deterministic outcome-stratified illustrations.']}
    dump(out / 'summary.json', summary)
    hashes = {str(p.relative_to(out)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
              for p in sorted(out.rglob('*')) if p.is_file()}
    dump(out / 'output_hashes.json', hashes)
    print(json.dumps({'status': summary['status'], 'output': str(out), 'new_attempts': len(new)}))
    return summary

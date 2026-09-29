"""Version a metadata-only uncertainty correction; preserve saved bootstrap values/counts."""
import csv
import hashlib
import json
from pathlib import Path

E = Path(__file__).resolve().parents[1]
SOURCE = E / 'results/analysis_v1'
OUT = E / 'results/analysis_v2'


def qualify(name, rows):
    changed = 0
    for row in rows:
        prefixes = ['target_rate_', 'invalid_rate_', 'valid_only_target_rate_'] if name == 'rates' else ['', 'pointwise_']
        for prefix in prefixes:
            low, high = row[prefix + 'ci_low'], row[prefix + 'ci_high']
            row[prefix + 'bootstrap_quantile_low'] = low
            row[prefix + 'bootstrap_quantile_high'] = high
            degenerate = low is not None and low == high
            if name == 'rates' or prefix == '':
                row[prefix + 'interval_status'] = ('degenerate_inferentially_uninformative' if degenerate else
                                                   'available' if low is not None else 'none_greedy_descriptive')
            if degenerate:
                row[prefix + 'ci_low'] = row[prefix + 'ci_high'] = None
                changed += 1
    return changed


def main():
    OUT.mkdir(exist_ok=True)
    receipts = []
    for name in ['rates', 'contrasts']:
        source = SOURCE / f'{name}.json'
        rows = json.loads(source.read_text())
        original = json.loads(source.read_text())
        changed = qualify(name, rows)
        for old, new in zip(original, rows):
            for key, value in old.items():
                if not (key.endswith('ci_low') or key.endswith('ci_high')):
                    assert new[key] == value
        output = OUT / f'{name}.json'
        output.write_text(json.dumps(rows, indent=2, allow_nan=False))
        with (OUT / f'{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        receipts.append({'table': name, 'rows': len(rows), 'degenerate_intervals_marked': changed,
                         'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                         'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest()})
    (OUT / 'correction_receipt.json').write_text(json.dumps({
        'status': 'passed', 'correction': 'Zero-width empirical percentile intervals are retained as bootstrap quantiles, but their inferential bounds are null and status is degenerate_inferentially_uninformative.',
        'counts_and_nondegenerate_intervals_unchanged': True, 'new_model_calls': 0,
        'new_fits_or_resampling': 0, 'new_inferential_bounds': False, 'tables': receipts}, indent=2))
    print(json.dumps(receipts))


if __name__ == '__main__':
    main()

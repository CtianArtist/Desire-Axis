"""Independently verify the audit's small CSV/JSON outputs (standard library only).

Usage: python3 src/verify_outputs.py <analysis-dir>
References are read from references/raw_matrices beside this project's src/.
No tensors, audit implementation, or scientific-computing libraries are loaded.
A successful verification can describe either a continued or a correctly stopped
run: verification status is distinct from the reproduction stopping decision.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
import sys

ORDER = ['S1_pain', 'S2_pain', 'Fear', 'NegEmotion', 'NegWorld', 'BodySens',
         'Arousal', 'Random', 'Numb', 'Sadness']
NAMES = ['paper_available', 'common_neutral', 'common_controls']
PAIRS = [(i, j) for i in range(8) for j in range(i + 1, 8)]
SUMMARY_PAIRS = [(1, 2), (1, 3), (2, 3), (0, 1)]
# Much tighter than publication rounding; permits only accumulation-order noise.
NUMERIC_TOLERANCE = 1e-12


class InvalidOutput(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidOutput(message)


def read_json(path):
    def reject_constant(value):
        raise InvalidOutput(f'{path}: nonfinite JSON constant {value}')

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f'{path}: duplicate JSON key {key}')
            result[key] = value
        return result

    return json.loads(path.read_text(), parse_constant=reject_constant,
                      object_pairs_hook=unique_object)


def same(actual, expected, where):
    """Type-aware recursive check of an independently constructed value."""
    if expected is None:
        require(actual is None, f'{where}: expected missing, got {actual!r}')
    elif isinstance(expected, bool):
        require(type(actual) is bool and actual == expected,
                f'{where}: expected {expected}, got {actual!r}')
    elif isinstance(expected, int):
        require(type(actual) in (int, float) and math.isfinite(actual)
                and actual == expected, f'{where}: expected {expected}, got {actual!r}')
    elif isinstance(expected, float):
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=NUMERIC_TOLERANCE,
                                 abs_tol=NUMERIC_TOLERANCE),
                f'{where}: expected {expected!r}, got {actual!r}')
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected),
                f'{where}: field names do not match')
        for key, value in expected.items():
            same(actual[key], value, f'{where}.{key}')
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected),
                f'{where}: list length/type mismatch')
        for index, value in enumerate(expected):
            same(actual[index], value, f'{where}[{index}]')
    else:
        require(type(actual) is type(expected) and actual == expected,
                f'{where}: expected {expected!r}, got {actual!r}')


def read_matrix(path):
    with path.open(newline='') as stream:
        rows = list(csv.reader(stream))
    require(len(rows) == 11 and all(len(row) == 11 for row in rows),
            f'{path}: expected header plus 10 rows of 11 fields')
    same(rows[0], [''] + ORDER, str(path) + ': header')
    same([row[0] for row in rows[1:]], ORDER, str(path) + ': row labels')
    result = []
    for i, row in enumerate(rows[1:]):
        values = []
        for j, text in enumerate(row[1:]):
            value = float(text) if text else None
            require(value is None or math.isfinite(value),
                    f'{path}: nonfinite number at {ORDER[i]}/{ORDER[j]}; use blank for missing')
            values.append(value)
        result.append(values)
    return result


def check_cosines(matrix, where):
    for i in range(10):
        for j in range(10):
            value, reverse = matrix[i][j], matrix[j][i]
            if i >= 8 or j >= 8:
                require(value is None, f'{where}: {ORDER[i]}/{ORDER[j]} must be missing, including diagonals')
            require((value is None) == (reverse is None), f'{where}: asymmetric missingness at {i},{j}')
            if value is not None:
                require(abs(value) <= 1 + 2e-6, f'{where}: cosine out of bounds at {i},{j}: {value}')
                require(abs(value - reverse) <= 1e-7, f'{where}: asymmetric cosine at {i},{j}')
                if i == j:
                    require(abs(value - 1) <= 2e-6, f'{where}: defined self diagonal is not one at {i}')
                    # A missing self cosine is permitted for a zero/invalid vector.


def average(values):
    return math.fsum(values) / len(values) if values else None


def aggregate(models):
    means, counts = [], []
    for i in range(10):
        mean_row, count_row = [], []
        for j in range(10):
            values = [matrix[i][j] for matrix in models.values() if matrix[i][j] is not None]
            mean_row.append(average(values))
            count_row.append(len(values))
        means.append(mean_row)
        counts.append(count_row)
    return means, counts


def csv_compare(path, expected, fields, keys):
    """Compare typed cells without trusting CSV order or silently losing duplicates."""
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        same(reader.fieldnames, fields, str(path) + ': columns')
        actual = list(reader)
    require(len(actual) == len(expected), f'{path}: expected {len(expected)} rows, got {len(actual)}')
    indexed = {}
    for row in actual:
        require(set(row) == set(fields) and all(value is not None for value in row.values()),
                f'{path}: malformed CSV row')
        key = tuple(row[field] for field in keys)
        require(key not in indexed, f'{path}: duplicate row key {key}')
        indexed[key] = row
    for row in expected:
        key = tuple(row[field] for field in keys)
        require(key in indexed, f'{path}: missing row {key}')
        for field, value in row.items():
            text = indexed[key][field]
            where = f'{path}: {key}/{field}'
            if value is None:
                same(text, '', where)
            elif isinstance(value, bool):
                same(text, str(value), where)
            elif isinstance(value, (int, float)):
                require(text != '', f'{where}: unexpected missing value')
                same(float(text), value, where)
            else:
                same(text, value, where)


def verify(out, reference_dir):
    reproduction = read_json(out / 'reproduction.json')
    verification = read_json(out / 'verification.json')
    plot = read_json(out / 'plot_data.json')
    failures = read_json(out / 'failure_manifest.json')
    require(isinstance(failures, list), 'failure_manifest.json: expected list')
    references = {}
    for path in sorted(reference_dir.glob('similarity_*_L*.csv')):
        match = re.fullmatch(r'similarity_(.+)_L(\d+)\.csv', path.name)
        require(match is not None, f'{path}: invalid reference filename')
        model, layer = match.groups()
        require(model not in references, f'{path}: duplicate reference model')
        references[model] = (int(layer), read_matrix(path))
    require(references, f'{reference_dir}: no reference matrices found')
    results = {name: {} for name in NAMES}
    matrix_dir = out / 'matrices'
    if matrix_dir.exists():
        require(all(path.is_dir() and path.name in NAMES for path in matrix_dir.iterdir()),
                f'{matrix_dir}: unexpected construction or file')
    for construction, models in results.items():
        directory = matrix_dir / construction
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            match = re.fullmatch(r'(.+)_L(\d+)\.csv', path.name)
            require(path.is_file() and match is not None, f'{path}: invalid matrix filename')
            model, layer = match.groups()
            require(model in references, f'{path}: no matching reference')
            require(int(layer) == references[model][0], f'{path}: wrong layer')
            require(model not in models, f'{path}: duplicate model')
            models[model] = read_matrix(path)
            check_cosines(models[model], str(path))
        # Audit iteration is sorted by model, not by the layer-suffixed filename.
        results[construction] = dict(sorted(models.items()))
    usable = sorted(results['paper_available'])
    same(reproduction['usable_models'], usable, 'reproduction.usable_models')
    for construction in NAMES[1:]:
        require(set(results[construction]) <= set(usable), f'{construction}: model without paper_available matrix')

    discrepancies = []
    for i, j in PAIRS:
        models = [model for model in usable
                  if results['paper_available'][model][i][j] is not None
                  and references[model][1][i][j] is not None]
        observed = average([results['paper_available'][model][i][j] for model in models])
        reference = average([references[model][1][i][j] for model in models])
        delta = observed - reference if models else None
        discrepancies.append(dict(direction_a=ORDER[i], direction_b=ORDER[j],
                                  n_models=len(models), observed=observed, reference=reference,
                                  delta=delta, flag_gt_003=delta is not None and abs(delta) > .03,
                                  involves_neutral_pool=(i, j) != (0, 1), models='|'.join(models)))
    csv_compare(out / 'reproduction_discrepancies.csv', discrepancies,
                list(discrepancies[0]), ['direction_a', 'direction_b'])
    anchor = discrepancies[0]
    passed = bool(anchor['n_models'] and abs(anchor['delta']) <= .01)
    same(reproduction['passed'], passed, 'reproduction.passed (independent <=0.01 stopping rule)')
    same(reproduction['anchor'], anchor, 'reproduction.anchor')
    same(verification['reproduction_passed'], passed, 'verification.reproduction_passed')
    same(reproduction['n_discrepancies_gt_003'], sum(row['flag_gt_003'] for row in discrepancies),
         'reproduction.n_discrepancies_gt_003')
    same(reproduction['all_flagged_involve_neutral'],
         all(row['involves_neutral_pool'] for row in discrepancies if row['flag_gt_003']),
         'reproduction.all_flagged_involve_neutral')
    same(reproduction['missing'], ['ControlSupplement_1P', 'Numb_1P', 'SD_sadness_1P'],
         'reproduction.missing')
    if not passed:
        require(not any(results[name] for name in NAMES[1:]),
                'stopped run must not contain alternative matrices')
        require(not (out / 'common_model_comparison.csv').exists(),
                'stopped run must not contain common_model_comparison.csv')

    # Account for missing models using the declared failures, without reading inputs.
    failure_keys = []
    for failure in failures:
        key = (failure['phase'], failure['model'])
        require(key not in failure_keys, f'failure_manifest: duplicate failure {key}')
        require(key[0] in NAMES and key[1] in references, f'failure_manifest: unknown failure {key}')
        failure_keys.append(key)
    expected_failures = {('paper_available', model) for model in references if model not in usable}
    if passed:
        expected_failures.update((name, model) for name in NAMES[1:] for model in usable
                                 if model not in results[name])
    require(set(failure_keys) == expected_failures, 'failure_manifest: missing/unexpected model failures')
    same(reproduction['failed_files'], sum(name == 'paper_available' for name, _ in failure_keys),
         'reproduction.failed_files')
    same(verification['failed_files_or_constructions'], len(failure_keys),
         'verification.failed_files_or_constructions')

    cell_rows, comparison, checks = [], [], []
    expected_plot = {'directions': ORDER, 'constructions': {}}
    for construction, models in results.items():
        if not models:
            require(not (out / f'mean_{construction}.csv').exists()
                    and not (out / f'counts_{construction}.csv').exists(),
                    f'{construction}: aggregate exists without any per-model matrices')
            continue
        means, counts = aggregate(models)
        actual_mean = read_matrix(out / f'mean_{construction}.csv')
        check_cosines(actual_mean, f'mean_{construction}.csv')
        same(actual_mean, means, f'mean_{construction}.csv')
        same(read_matrix(out / f'counts_{construction}.csv'), counts, f'counts_{construction}.csv')
        expected_plot['constructions'][construction] = dict(mean=means, counts=counts, models=list(models))
        for i in range(10):
            for j in range(10):
                cell_rows.append(dict(construction=construction, direction_a=ORDER[i], direction_b=ORDER[j],
                                      mean=means[i][j] if means[i][j] is not None else '', n_models=counts[i][j]))
        for i, j in SUMMARY_PAIRS:
            values = [matrix[i][j] for matrix in models.values() if matrix[i][j] is not None]
            comparison.append(dict(construction=construction, direction_a=ORDER[i], direction_b=ORDER[j],
                                   n_models=len(values), mean=average(values) if values else '',
                                   minimum=min(values) if values else '',
                                   median=statistics.median(values) if values else '',
                                   maximum=max(values) if values else ''))
        available_counts = [counts[i][j] for i in range(8) for j in range(8)]
        checks.append(dict(construction=construction, n_models=len(models), csv_roundtrip=True,
                           symmetry_bounds_defined_diagonals_missingness=True,
                           aggregate_recomputed_from_csv=True,
                           min_available_cell_count=min(available_counts),
                           max_available_cell_count=max(available_counts)))
    csv_compare(out / 'mean_cells.csv', cell_rows,
                ['construction', 'direction_a', 'direction_b', 'mean', 'n_models'],
                ['construction', 'direction_a', 'direction_b'])
    csv_compare(out / 'comparison.csv', comparison,
                ['construction', 'direction_a', 'direction_b', 'n_models', 'mean', 'minimum', 'median', 'maximum'],
                ['construction', 'direction_a', 'direction_b'])
    same(plot, expected_plot, 'plot_data.json')
    same(verification['comparison'], comparison, 'verification.comparison')
    same(verification['checks'], checks, 'verification.checks')

    paired = []
    if passed:
        for i, j in PAIRS:
            common = [model for model in usable if all(model in results[name]
                      and results[name][model][i][j] is not None for name in NAMES)]
            row = dict(direction_a=ORDER[i], direction_b=ORDER[j], n_common=len(common), models='|'.join(common))
            row.update({name: average([results[name][model][i][j] for model in common]) for name in NAMES})
            paired.append(row)
        csv_compare(out / 'common_model_comparison.csv', paired,
                    ['direction_a', 'direction_b', 'n_common', 'models'] + NAMES,
                    ['direction_a', 'direction_b'])
    return dict(status='passed', reproduction_passed=passed,
                run_state='continued' if passed else 'stopped',
                numeric_tolerance=NUMERIC_TOLERANCE, reference_directory=str(reference_dir.resolve()),
                reference_models=len(references), per_model_matrices=sum(map(len, results.values())),
                aggregate_cells_checked=len(cell_rows), summary_rows_checked=len(comparison),
                reproduction_cells_checked=len(discrepancies), common_model_cells_checked=len(paired),
                constructions=expected_plot['constructions'], comparison=comparison,
                reproduction_discrepancies=discrepancies, common_model_comparison=paired,
                limitations=['Checks saved CSV/JSON consistency, not tensor provenance or vector construction.',
                             'Original in-memory CSV roundtrip is not independently observable.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis_dir', type=Path)
    args = parser.parse_args()
    out = args.analysis_dir
    if not out.is_dir():
        print(f'INVALID: analysis directory does not exist: {out}', file=sys.stderr)
        return 1
    try:
        result = verify(out, Path(__file__).resolve().parents[1] / 'references/raw_matrices')
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError, csv.Error) as exc:
        result = dict(status='failed', error=f'{type(exc).__name__}: {exc}')
    destination = out / 'independent_verification.json'
    try:
        destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    except OSError as exc:
        print(f'INVALID: could not write {destination}: {exc}', file=sys.stderr)
        return 1
    if result['status'] == 'failed':
        print('INVALID: ' + result['error'], file=sys.stderr)
        return 1
    print(f"VERIFIED: {result['per_model_matrices']} per-model matrices; "
          f"{result['aggregate_cells_checked']} aggregate cells; "
          f"28 reproduction cells; run {result['run_state']}. Wrote {destination}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

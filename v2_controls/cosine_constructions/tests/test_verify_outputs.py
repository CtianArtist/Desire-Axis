"""Reproducible stdlib-only CLI fixtures for the independent cosine verifier.

Run from the experiment directory:
    python3 -m unittest discover -s tests -p test_verify_outputs.py

Fixtures and the copied CLI live in a cleaned temporary directory. Expected
outputs are built here without importing either run_audit or verify_outputs.
"""
import copy
import csv
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'src/verify_outputs.py'
ORDER = ['S1_pain', 'S2_pain', 'Fear', 'NegEmotion', 'NegWorld', 'BodySens',
         'Arousal', 'Random', 'Numb', 'Sadness']
NAMES = ['paper_available', 'common_neutral', 'common_controls']
PAIRS = [(i, j) for i in range(8) for j in range(i + 1, 8)]


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, allow_nan=False))


def matrix(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow([''] + ORDER)
        for label, row in zip(ORDER, values):
            writer.writerow([label] + row)


def table(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def avg(values):
    return sum(values) / len(values) if values else None


def fixture(root, stopped=False, empty=False):
    """Three models with different finite subsets and non-rounded values."""
    refs = root / 'references/raw_matrices'
    out = root / 'analysis'
    out.mkdir(parents=True)
    all_models = ['Alpha', 'Beta', 'Gamma']
    usable = [] if empty else all_models
    active = ['paper_available'] if stopped or empty else NAMES
    results = {name: {} for name in NAMES}
    reference = {}
    for k, model in enumerate(all_models):
        base = [[None if i >= 8 or j >= 8 else 1.0 if i == j else
                 (k + 1) * .013456789123 + (i + j) * .001234567891
                 for j in range(10)] for i in range(10)]
        if model == 'Beta':
            base[1][2] = base[2][1] = None
        ref = copy.deepcopy(base)
        if stopped:
            ref[0][1] -= .02
            ref[1][0] -= .02
        if model == 'Gamma':
            ref[0][1] = ref[1][0] = None
        reference[model] = ref
        matrix(refs / f'similarity_{model}_L2.csv', ref)
        if model not in usable:
            continue
        for ci, name in enumerate(active):
            values = copy.deepcopy(base)
            for i, j in PAIRS:
                if values[i][j] is not None:
                    values[i][j] += ci * .03
                    values[j][i] = values[i][j]
            if name == 'common_controls' and model == 'Gamma':
                values[0][1] = values[1][0] = None
            results[name][model] = values
            matrix(out / 'matrices' / name / f'{model}_L2.csv', values)

    discrepancies = []
    for i, j in PAIRS:
        models = [m for m in usable
                  if results['paper_available'][m][i][j] is not None
                  and reference[m][i][j] is not None]
        observed = avg([results['paper_available'][m][i][j] for m in models])
        expected = avg([reference[m][i][j] for m in models])
        delta = observed - expected if models else None
        discrepancies.append(dict(
            direction_a=ORDER[i], direction_b=ORDER[j], n_models=len(models),
            observed=observed, reference=expected, delta=delta,
            flag_gt_003=delta is not None and abs(delta) > .03,
            involves_neutral_pool=(i, j) != (0, 1), models='|'.join(models)))
    table(out / 'reproduction_discrepancies.csv', discrepancies, list(discrepancies[0]))
    passed = not stopped and not empty
    failures = [dict(model=m, phase='paper_available', error='synthetic')
                for m in all_models if m not in usable]
    dump(out / 'failure_manifest.json', failures)
    dump(out / 'reproduction.json', dict(
        passed=passed, usable_models=usable, anchor=discrepancies[0],
        failed_files=len(failures), n_discrepancies_gt_003=0,
        all_flagged_involve_neutral=True,
        missing=['ControlSupplement_1P', 'Numb_1P', 'SD_sadness_1P']))

    comparison, cells, checks = [], [], []
    plot = dict(directions=ORDER, constructions={})
    for name, models in results.items():
        if not models:
            continue
        means, counts = [], []
        for i in range(10):
            means.append([])
            counts.append([])
            for j in range(10):
                values = [m[i][j] for m in models.values() if m[i][j] is not None]
                means[i].append(avg(values))
                counts[i].append(len(values))
                cells.append(dict(
                    construction=name, direction_a=ORDER[i], direction_b=ORDER[j],
                    mean=avg(values) if values else '', n_models=len(values)))
        matrix(out / f'mean_{name}.csv', means)
        matrix(out / f'counts_{name}.csv', counts)
        plot['constructions'][name] = dict(mean=means, counts=counts, models=list(models))
        for i, j in [(1, 2), (1, 3), (2, 3), (0, 1)]:
            values = [m[i][j] for m in models.values() if m[i][j] is not None]
            comparison.append(dict(
                construction=name, direction_a=ORDER[i], direction_b=ORDER[j],
                n_models=len(values), mean=avg(values) if values else '',
                minimum=min(values) if values else '',
                median=statistics.median(values) if values else '',
                maximum=max(values) if values else ''))
        count_values = [counts[i][j] for i in range(8) for j in range(8)]
        checks.append(dict(
            construction=name, n_models=len(models), csv_roundtrip=True,
            symmetry_bounds_defined_diagonals_missingness=True,
            aggregate_recomputed_from_csv=True,
            min_available_cell_count=min(count_values),
            max_available_cell_count=max(count_values)))
    table(out / 'comparison.csv', comparison,
          ['construction', 'direction_a', 'direction_b', 'n_models',
           'mean', 'minimum', 'median', 'maximum'])
    table(out / 'mean_cells.csv', cells,
          ['construction', 'direction_a', 'direction_b', 'mean', 'n_models'])
    dump(out / 'plot_data.json', plot)
    dump(out / 'verification.json', dict(
        reproduction_passed=passed, failed_files_or_constructions=len(failures),
        comparison=comparison, checks=checks))
    if passed:
        paired = []
        for i, j in PAIRS:
            models = [m for m in usable
                      if all(results[n][m][i][j] is not None for n in NAMES)]
            row = dict(direction_a=ORDER[i], direction_b=ORDER[j],
                       n_common=len(models), models='|'.join(models))
            row.update({n: avg([results[n][m][i][j] for m in models]) for n in NAMES})
            paired.append(row)
        table(out / 'common_model_comparison.csv', paired,
              ['direction_a', 'direction_b', 'n_common', 'models'] + NAMES)
    # Copy only the customer-owned verifier so its normal relative reference
    # lookup resolves to the synthetic references, never the production files.
    script = root / 'src/verify_outputs.py'
    script.parent.mkdir()
    shutil.copyfile(SOURCE, script)
    return script, out


class VerifyOutputsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='cosine-verifier-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.script, self.out = fixture(self.root / 'continued')

    def run_verifier(self, valid=True, error_contains=None, script=None, out=None):
        script = script or self.script
        out = out or self.out
        process = subprocess.run(
            [sys.executable, '-B', str(script), str(out)],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0 if valid else 1,
                         process.stdout + process.stderr)
        destination = out / 'independent_verification.json'
        self.assertTrue(destination.is_file(), process.stdout + process.stderr)
        report = json.loads(destination.read_text())
        self.assertEqual(report['status'], 'passed' if valid else 'failed')
        if error_contains:
            self.assertIn(error_contains, report['error'])
        if not valid:
            self.assertIn('INVALID:', process.stderr)
        return report

    def corrupt(self, relative, alter, error_contains):
        path = self.out / relative
        original = path.read_text()
        changed = alter(original)
        self.assertNotEqual(changed, original, 'Corruption must change the fixture')
        path.write_text(changed)
        self.run_verifier(valid=False, error_contains=error_contains)
        return path, original

    def test_continued_unequal_finite_subsets_full_precision(self):
        report = self.run_verifier()
        self.assertEqual(report['run_state'], 'continued')
        self.assertTrue(report['reproduction_passed'])
        self.assertEqual(report['per_model_matrices'], 9)
        self.assertEqual(report['aggregate_cells_checked'], 300)
        self.assertEqual(report['summary_rows_checked'], 12)
        self.assertEqual(report['reproduction_cells_checked'], 28)
        self.assertEqual(report['common_model_cells_checked'], 28)
        self.assertEqual(report['reproduction_discrepancies'][0]['models'], 'Alpha|Beta')
        self.assertEqual(report['common_model_comparison'][0]['models'], 'Alpha|Beta')

    def test_aggregate_corruption(self):
        self.corrupt('mean_paper_available.csv',
                     lambda s: s.replace('1.0', '0.9', 1), 'self diagonal')

    def test_count_corruption(self):
        self.corrupt('counts_paper_available.csv',
                     lambda s: s.replace('3', '2', 1), 'expected 3')

    def test_numb_diagonal_corruption(self):
        self.corrupt('matrices/paper_available/Alpha_L2.csv',
                     lambda s: s.replace('Numb,,,,,,,,,,', 'Numb,,,,,,,,,1.0,'),
                     'must be missing')

    def test_declared_passed_corruption(self):
        self.corrupt('reproduction.json',
                     lambda s: s.replace('"passed": true', '"passed": false'),
                     'reproduction.passed')

    def test_summary_precision_corruption(self):
        def alter(text):
            rows = list(csv.reader(text.splitlines()))
            target = rows[1][4]
            return text.replace(target, str(float(target) + .00001), 1)
        self.corrupt('comparison.csv', alter, 'expected')

    def test_paired_count_corruption(self):
        self.corrupt('common_model_comparison.csv',
                     lambda s: s.replace(',2,Alpha|Beta,', ',3,Alpha|Beta,', 1),
                     'expected 2')

    def test_reproduction_subset_corruption(self):
        self.corrupt('reproduction_discrepancies.csv',
                     lambda s: s.replace('Alpha|Beta', 'Alpha|Gamma', 1), 'models')

    def test_reproduction_count_corruption(self):
        self.corrupt('reproduction_discrepancies.csv',
                     lambda s: s.replace('S1_pain,S2_pain,2,', 'S1_pain,S2_pain,3,', 1),
                     'expected 2')

    def test_restored_after_corruption(self):
        path, original = self.corrupt(
            'counts_paper_available.csv', lambda s: s.replace('3', '2', 1), 'expected 3')
        path.write_text(original)
        # The same invocation must replace the stale failed report with success.
        self.run_verifier()

    def test_stopped_without_alternatives(self):
        script, out = fixture(self.root / 'stopped', stopped=True)
        self.assertFalse((out / 'common_model_comparison.csv').exists())
        for name in NAMES[1:]:
            self.assertFalse((out / 'matrices' / name).exists())
        report = self.run_verifier(script=script, out=out)
        self.assertEqual(report['run_state'], 'stopped')
        self.assertFalse(report['reproduction_passed'])
        self.assertEqual(report['per_model_matrices'], 3)
        self.assertEqual(report['aggregate_cells_checked'], 100)
        self.assertEqual(report['common_model_cells_checked'], 0)

    def test_stopped_with_zero_usable_models(self):
        script, out = fixture(self.root / 'empty', empty=True)
        report = self.run_verifier(script=script, out=out)
        self.assertEqual(report['run_state'], 'stopped')
        self.assertFalse(report['reproduction_passed'])
        self.assertEqual(report['per_model_matrices'], 0)
        self.assertEqual(report['aggregate_cells_checked'], 0)
        self.assertEqual(report['reproduction_cells_checked'], 28)


if __name__ == '__main__':
    unittest.main()

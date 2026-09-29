"""Lightweight packaging checks: no model imports, downloads, or bootstrap fits."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PortableTests(unittest.TestCase):
    def command(self, *args):
        return subprocess.check_output([sys.executable, '-B', *map(str, args)], cwd=ROOT,
                                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), text=True)

    def test_compile(self):
        paths = list(ROOT.rglob('*.py'))
        self.assertGreater(len(paths), 40)
        for path in paths:
            compile(path.read_bytes(), str(path.relative_to(ROOT)), 'exec')

    def test_source_identity(self):
        for row in json.loads((ROOT / 'source_manifest.json').read_text())['files']:
            with self.subTest(path=row['path']):
                self.assertEqual(hashlib.sha256((ROOT / row['path']).read_bytes()).hexdigest(), row['copied_sha256'])
                if row['transformation'] == 'none':
                    self.assertEqual(row['original_sha256'], row['copied_sha256'])

    def test_help(self):
        self.assertIn('--data-dir', self.command('reproduce.py', '--help'))
        self.assertIn('--dry-run', self.command('inference.py', '--help'))

    def test_all_dry_runs(self):
        for group, model, trials in [('superseded_long_wording', '32', 410), ('matched_wording', '32', 410),
                                      ('lamp32', '32', 820), ('b2', '32', 410), ('b2', '72', 410)]:
            with self.subTest(group=group, model=model):
                result = json.loads(self.command('inference.py', '--group', group, '--model', model, '--dry-run'))
                self.assertFalse(result['model_loaded'])
                self.assertEqual(result['trials'], trials)
                self.assertEqual(result['sampled'] + result['greedy'], trials)

    def test_frozen_tables(self):
        result = json.loads(self.command('reproduce.py', '--verify-tables'))
        self.assertEqual(result['table_files'], 13)
        self.assertEqual(result['status'], 'passed')

    def test_denominators(self):
        with (ROOT / 'four_cells/tables/four-slot-comparison.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 20)
        fear = {r['cell']: r for r in rows if r['condition'] == 'fear'}
        self.assertEqual(len(fear), 4)
        r = fear['72harmonly_kidspics_vs_inert']
        self.assertEqual((r['target_count'], r['valid_count'], r['malformed_count'], r['attempt_count']),
                         ('75', '346', '58', '404'))
        for cell in ('32photos_lamp', '32own_weights_lamp'):
            self.assertEqual(fear[cell]['denominator_definition'], 'all_attempts_including_malformed_and_unavailable')

    def test_pending_assets(self):
        for name in ('assets.json', 'inference_assets.json'):
            for row in json.loads((ROOT / name).read_text())['assets']:
                self.assertIsNone(row['url'])
                self.assertFalse(Path(row['local_path']).is_absolute())
        self.assertFalse(any(ROOT.rglob('*.pt')))
        self.assertFalse(any(ROOT.rglob('*.sqlite')))


if __name__ == '__main__':
    unittest.main()

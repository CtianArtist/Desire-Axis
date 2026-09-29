"""Small corruption tests for the independent verifier; temporary data only."""
import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from verify import verify

ROOT = Path(__file__).resolve().parent


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='figure10-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ['config.json', 'source_manifest.json', 'published_annotation_spec.json', 'pooled.csv', 'positions.csv', 'helping.csv']:
            shutil.copyfile(ROOT / name, self.root / name)
        shutil.copytree(ROOT / 'source_tables', self.root / 'source_tables')

    def mutate(self, file, edit):
        with (self.root / file).open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        edit(rows)
        with (self.root / file).open('w', newline='') as stream:
            w = csv.DictWriter(stream, fieldnames=rows[0])
            w.writeheader()
            w.writerows(rows)

    def test_authoritative_tables_pass(self):
        self.assertEqual(verify(self.root)['status'], 'passed')

    def test_rate_tampering_fails(self):
        self.mutate('pooled.csv', lambda rows: rows[0].update(rate='0.5'))
        with self.assertRaisesRegex(ValueError, 'Changed source value'):
            verify(self.root)

    def test_denominator_tampering_fails(self):
        self.mutate('pooled.csv', lambda rows: next(r for r in rows if r['row_id'] == 'photos_inert_72' and r['condition'] == 'fear').update(denominator='404'))
        with self.assertRaisesRegex(ValueError, 'Incorrect denominator'):
            verify(self.root)

    def test_interval_tampering_fails(self):
        self.mutate('positions.csv', lambda rows: rows[0].update(ci_high='0.1'))
        with self.assertRaisesRegex(ValueError, 'Changed source value'):
            verify(self.root)

    def test_source_bytes_tampering_fails(self):
        with (self.root / 'source_tables/profile_rates.csv').open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'Source hash mismatch'):
            verify(self.root)

    def test_duplicate_row_fails(self):
        self.mutate('positions.csv', lambda rows: rows.__setitem__(0, rows[1].copy()))
        with self.assertRaisesRegex(ValueError, 'Missing or duplicate'):
            verify(self.root)

    def test_annotation_discrepancy_is_reported_not_repaired(self):
        path = self.root / 'published_annotation_spec.json'
        spec = json.loads(path.read_text())
        spec['rows'][0]['published_rounded_pct'][0] = 99
        path.write_text(json.dumps(spec))
        before = (self.root / 'pooled.csv').read_bytes()
        result = verify(self.root)
        self.assertEqual(result['status'], 'annotation_discrepancies')
        self.assertEqual(len(result['annotation_discrepancies']), 1)
        self.assertEqual((self.root / 'pooled.csv').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()

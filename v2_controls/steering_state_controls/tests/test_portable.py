"""No model libraries or raw data needed for these entrypoint regressions."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PortableTests(unittest.TestCase):
    def test_compile_all_customer_python(self):
        for path in ROOT.rglob('*.py'):
            compile(path.read_text(),str(path),'exec')

    def test_dry_run_missing_assets_is_explicit(self):
        with tempfile.TemporaryDirectory() as empty:
            for study,model in [('b1_b3','32'),('b1_b3','72'),('b6','32')]:
                result = subprocess.run([sys.executable,str(ROOT/'inference.py'),'--study',study,'--model',model,
                                         '--data-dir',empty,'--dry-run'],text=True,capture_output=True)
                self.assertEqual(result.returncode,0,result.stderr)
                data=json.loads(result.stdout)
                self.assertTrue(data['missing_local_assets'])
                self.assertEqual(data['weights_loading'],'not attempted')
                self.assertIsNone(data['public_asset_urls'])
                self.assertFalse(data['historical_resume_state_required'])

    def test_no_dry_run_missing_assets_fails_before_import(self):
        with tempfile.TemporaryDirectory() as empty:
            result=subprocess.run([sys.executable,str(ROOT/'inference.py'),'--study','b1_b3','--data-dir',empty],text=True,capture_output=True)
            self.assertEqual(result.returncode,2)
            self.assertIn('Required local assets missing',result.stderr)
            self.assertNotIn('Traceback',result.stderr)

    def test_frozen_tables(self):
        result=subprocess.run([sys.executable,str(ROOT/'reproduce.py')],text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual([r['rate_rows'] for r in json.loads(result.stdout)['checks']],[9381,3519])

    def test_release_runtime_copies_isolated(self):
        left=json.loads((ROOT/'b1_b3/provenance.json').read_text())
        right=json.loads((ROOT/'b6/provenance.json').read_text())
        hashes=[]
        for receipt in [left,right]:
            hashes.append(next(r['source_sha256'] for r in receipt['files'] if r['path']=='pain_axis_b/runtime.py'))
        self.assertNotEqual(*hashes)
        self.assertEqual(left['source_allowlist_size'],137)

    def test_long_working_harm_wording(self):
        grid=json.loads((ROOT/'b1_b3/inputs/requested_grid.json').read_text())
        wording=grid['labels']['harmonly_kidspics_vs_inert']['relief']
        self.assertIn('which they love very much',wording)
        protocol=(ROOT/'b1_b3/scripts/4.3_selfmed/04_selfmed_two_buttons.py').read_text()
        self.assertIn('target-removal transition failed',protocol)

if __name__=='__main__': unittest.main()

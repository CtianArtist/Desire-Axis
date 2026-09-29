"""Portable interface, frozen evidence, and numerical-source preservation tests."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reproduce', ROOT / 'reproduce.py')
reproduce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduce)


class PortableTests(unittest.TestCase):
    def test_all_python_compiles_without_imports(self):
        for path in ROOT.rglob('*.py'):
            compile(path.read_bytes(), str(path.relative_to(ROOT)), 'exec')

    def test_frozen_hashes_and_independent_verifier(self):
        checked = reproduce.verify_frozen()
        self.assertEqual(checked['per_model_matrices'], 69)
        self.assertEqual(checked['reference_models'], 25)
        self.assertEqual(checked['aggregate_cells_checked'], 300)
        self.assertEqual(checked['frozen_csv_hashes_checked'], 80)

    def test_numerical_functions_identical_to_executed_source(self):
        def functions(path):
            return {node.name: ast.dump(node, include_attributes=False)
                    for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}
        historical = functions(ROOT / 'references/executed_run_audit.py')
        portable = functions(ROOT / 'src/run_audit.py')
        for name in ('load', 'means_and_clouds', 'construct', 'aggregate', 'cosine',
                     'matrix', 'check_matrix', 'matrix_write', 'matrix_read', 'preflight'):
            self.assertEqual(historical[name], portable[name], name)

    def test_dependency_pins_match_measured_input_manifest(self):
        config = reproduce.read_json(ROOT / 'config.json')
        measured = reproduce.read_json(ROOT / 'results/analysis_v1/input_manifest.json')
        self.assertEqual(config['packages'], measured['environment']['packages'])
        pins = dict(line.split('==') for line in (ROOT / 'requirements.txt').read_text().splitlines())
        self.assertEqual(pins, config['packages'])

    def test_author_source_is_exact_recorded_revision_content(self):
        source = ROOT / 'references/author/02_build_control_vectors.py'
        manifest = reproduce.read_json(ROOT / 'results/analysis_v1/input_manifest.json')
        self.assertEqual(reproduce.sha256(source), manifest['source_sha256'])

    def test_help_works_without_scientific_dependencies(self):
        result = subprocess.run([sys.executable, str(ROOT / 'reproduce.py'), '--help'],
                                capture_output=True, text=True, check=True)
        self.assertIn('--data-dir', result.stdout)
        self.assertIn('--verify-frozen', result.stdout)

    def test_dry_run_has_no_input_requirement_or_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, out = Path(tmp) / 'missing-data', Path(tmp) / 'missing-output'
            result = subprocess.run([sys.executable, str(ROOT / 'reproduce.py'),
                                     '--data-dir', str(data), '--out', str(out), '--dry-run'],
                                    capture_output=True, text=True, check=True)
            plan = json.loads(result.stdout)
            self.assertEqual(plan['entrypoint'], 'src/run_audit.py')
            self.assertFalse(plan['archive_exists'])
            self.assertIsNone(plan['public_asset_url'])
            self.assertFalse(data.exists())
            self.assertFalse(out.exists())

    def test_missing_arrays_fail_instead_of_copying_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = reproduce.replay_plan(Path(tmp), Path(tmp) / 'out')
            with self.assertRaisesRegex(ValueError, 'activation archive missing'):
                reproduce.run_replay(plan)
            self.assertFalse(Path(plan['out']).exists())

    def test_archive_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = reproduce.replay_plan(Path(tmp), Path(tmp) / 'out')
            Path(plan['archive']).write_bytes(b'not the array archive')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                reproduce.run_replay(plan)

    def test_replay_dispatches_to_original_analysis(self):
        # Mock only large-input checks, dependency versions, and execution: this
        # proves command wiring, not numerical correctness or actual array replay.
        class StopAfterDispatch(Exception):
            pass
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / 'dummy.zip'
            archive.write_bytes(b'x')
            output = ROOT.parent / 'mock-replay-never-created'
            plan = reproduce.replay_plan(Path(tmp), output)
            plan['archive'] = str(archive)
            asset = {'bytes': 1, 'sha256': hashlib.sha256(b'x').hexdigest()}
            original_read = reproduce.read_json
            with mock.patch.object(reproduce, 'read_json', side_effect=lambda p:
                                   {'assets': [asset]} if p.name == 'assets.json' else original_read(p)), \
                 mock.patch.object(reproduce.importlib.metadata, 'version', side_effect=plan['packages'].__getitem__), \
                 mock.patch.object(reproduce.subprocess, 'run', side_effect=StopAfterDispatch) as call:
                with self.assertRaises(StopAfterDispatch):
                    reproduce.run_replay(plan)
            command = call.call_args.args[0]
            self.assertEqual(command[1], str(ROOT / 'src/run_audit.py'))
            self.assertEqual(command[2:], ['--archive', str(archive), '--out', str(output)])
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()

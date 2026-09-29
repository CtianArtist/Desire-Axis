"""Standard-library smoke checks for the portable entrypoints and frozen evidence."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import inference
import reproduce


class PortableChecks(unittest.TestCase):
    def test_table_hashes_and_completed_seed_counts(self):
        self.assertEqual(len(reproduce.verify_tables()["included_tables"]), 5)
        summary = json.loads((ROOT / "tables/summary.json").read_text())
        self.assertEqual(summary["records_per_adapter"], {"1": 27060, "2": 27060})
        self.assertEqual(summary["trial_records"], 54120)

    def test_dry_run_never_imports_model_libraries(self):
        with mock.patch.dict(sys.modules, {"torch": None, "transformers": None, "peft": None}), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(inference.main(["--dry-run", "--data-dir", str(ROOT / "missing-assets"), "--seed", "2"]), 0)
        receipt = json.loads(output.getvalue())
        self.assertFalse(receipt["model_loaded"])
        self.assertEqual(receipt["training_updates"], 0)
        self.assertIn("adapters/seed_2/final_adapter", receipt["missing_assets"])
        self.assertFalse((ROOT / "missing-assets").exists())

    def test_cli_help_without_model_libraries(self):
        with mock.patch.dict(sys.modules, {"torch": None, "transformers": None, "peft": None}), contextlib.redirect_stdout(io.StringIO()):
            for module in (inference, reproduce):
                with self.assertRaises(SystemExit) as stopped:
                    module.main(["--help"])
                self.assertEqual(stopped.exception.code, 0)

    def test_verifier_hash_contract_unchanged(self):
        from pain_seed_b.identity import CANONICAL_VECTOR_CONTRACT
        self.assertEqual(CANONICAL_VECTOR_CONTRACT["verifier_sha256"], "ac31c14bfc55b2da18a4517adba9f350a79add7b1ab89f1f0645f43a10ec489e")
        self.assertEqual(CANONICAL_VECTOR_CONTRACT["manifest_sha256"], "9c92ef27887b689accf13f1b91c99071e3137d011afaca1b17c3107dc51534a8")

    def test_only_complete_attempts_are_named(self):
        paths = [entry["local_path"] for entry in reproduce.verify_tables()["external_assets"]]
        self.assertIn("behavior/seed_1/attempt_1/run/", paths)
        self.assertIn("behavior/seed_2/attempt_2/run/", paths)
        self.assertNotIn("behavior/seed_2/attempt_1/run/", paths)

    def test_pending_assets_have_no_download_urls(self):
        manifest = reproduce.verify_tables()
        self.assertIsNone(manifest["public_url"])
        for entry in manifest["external_assets"]:
            self.assertIsNone(entry["public_url"])
            self.assertEqual(entry["status"], "pending")


if __name__ == "__main__":
    unittest.main()

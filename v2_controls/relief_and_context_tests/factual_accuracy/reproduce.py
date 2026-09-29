"""Portable saved-output reproduction; full resampling belongs on CPU compute."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

def identities():
    manifest = json.loads((ROOT / "provenance.json").read_text())
    for row in manifest["files"]:
        assert hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() == row["sha256"], row["path"]
    return len(manifest["files"])

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "inputs", help="Local immutable saved-output directory; see README layout")
    p.add_argument("--output", type=Path, help="Fresh output directory for full original analysis (required unless --verify-only)")
    p.add_argument("--verify-only", action="store_true", help="Frozen hash/count checks without resampling; no model")
    args = p.parse_args()
    count = identities()
    if args.verify_only:
        import factual_data as a
        panel = json.loads((ROOT / "inputs/panel.json").read_text())
        raw = [json.loads(x) for x in (args.data_dir / "answers.jsonl").read_text().splitlines() if x.strip()]
        a.validate_records(raw, panel["items"])
        per_question, rates, contrasts = a.summarize_scores(a.score_records(raw, panel["items"]), panel["items"])
        frozen = json.loads((ROOT / "results/analysis_v1/metrics.json").read_text())
        for actual, expected in zip(rates, frozen["condition_rates"]):
            for key, value in actual.items():
                assert value == expected[key], (key, value, expected[key])
        for actual, expected in zip(contrasts, frozen["contrasts"]):
            for key, value in actual.items():
                assert value == expected[key], key
        assert len(raw) == 800 and len(panel["items"]) == 100
        assert per_question == json.loads((ROOT / "results/analysis_v1/per_question.json").read_text())
        print(json.dumps({"status": "passed", "verified_files": count, "resampling": False}))
        return
    if args.output is None:
        p.error("--output is required for full analysis")
    if args.output.exists():
        p.error("--output must be fresh; original evidence must never be overwritten")
    from factual_data import score
    score(ROOT / "inputs/panel.json", args.data_dir / "answers.jsonl", args.output)

if __name__ == "__main__":
    main()

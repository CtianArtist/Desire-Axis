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
        from pain_axis_r.state import build_grid
        scenarios = json.loads((ROOT / "inputs/scenarios.json").read_text())
        grid = build_grid(scenarios)
        assert len(grid) == 5330 and sum(x["turns_required"] for x in grid) == 49200
        assert grid == json.loads((ROOT / "results/trial_grid.json").read_text())
        headline = json.loads((ROOT / "results/analysis_v2/headline.json").read_text())
        assert headline["coverage"]["trials"] == 5330
        print(json.dumps({"status": "passed", "verified_files": count, "resampling": False}))
        return
    if args.output is None:
        p.error("--output is required for full analysis")
    if args.output.exists():
        p.error("--output must be fresh; original evidence must never be overwritten")
    import subprocess
    subprocess.run([sys.executable, "-m", "pain_axis_r.analysis", "--trials", str(args.data_dir / "trials.jsonl"),
                    "--scenarios", str(ROOT / "inputs/scenarios.json"), "--output", str(args.output.resolve()),
                    "--bootstrap-replicates", "10000", "--seed", "0", "--include-all"], cwd=ROOT, check=True)

if __name__ == "__main__":
    main()

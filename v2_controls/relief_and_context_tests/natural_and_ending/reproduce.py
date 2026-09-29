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
        from qualify_intervals import qualify
        for name in ("rates", "contrasts"):
            old = json.loads((ROOT / "results/original_analysis_v1" / (name + ".json")).read_text())
            qualify(name, old)
            frozen = json.loads((ROOT / "results/analysis_v2" / (name + ".json")).read_text())
            assert old == frozen, name
        from pain_axis_b.natural_inputs import load_contexts
        contexts = load_contexts(ROOT / "inputs/conversations.json")
        assert sum(len(v) for v in contexts.values()) == 140
        from prepare_inputs import read_constants, build_artifacts
        generated = build_artifacts(contexts, read_constants(ROOT / "inputs/released_protocol.py"))
        for name, value in generated.items():
            assert json.loads(json.dumps(value)) == json.loads((ROOT / "inputs/frozen" / name).read_text()), name
        print(json.dumps({"status": "passed", "verified_files": count, "resampling": False}))
        return
    if args.output is None:
        p.error("--output is required for full analysis")
    if args.output.exists():
        p.error("--output must be fresh; original evidence must never be overwritten")
    from analyze_natural import load_raw, analyze
    rows, sources = load_raw(args.data_dir)
    # This source version already emits qualified intervals and preserved
    # bootstrap quantiles. Applying the old-table correction again would erase
    # constant quantiles and mislabel their unavailable inferential bounds.
    analyze(rows, args.output, repetitions=10000, seed=42, sources=sources)

if __name__ == "__main__":
    main()

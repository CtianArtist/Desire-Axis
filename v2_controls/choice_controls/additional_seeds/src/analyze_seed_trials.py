"""Analyze complete fresh B1/B2 logs for explicit saved adapters; no inference."""
import argparse
import json
from pathlib import Path

from pain_seed_b.analysis import analyze
from pain_seed_b.analysis_io import sha256_file, strict_json, write_analysis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adapters", type=Path, required=True,
                        help="JSON list with training_seed, adapter_identity_digest and model, from saved adapter identity checks")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic-fixture", action="store_true", help="Allow miniature scenario counts and stamp all outputs as synthetic validation")
    args = parser.parse_args(argv)
    records = []
    for path in args.trials:
        with path.open() as handle:
            for line in handle:
                records.append(strict_json(line))
    manifest = strict_json(args.manifest.read_text())
    adapters = strict_json(args.adapters.read_text())
    result = analyze(records, manifest, adapters, synthetic_fixture=args.synthetic_fixture)
    sources = {str(path.resolve()): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
               for path in [*args.trials, args.manifest, args.adapters]}
    receipt = write_analysis(result, args.output, sources=sources)
    print(json.dumps({"data_kind": receipt["data_kind"], "status": "analysis_written",
                      "adapters": len(adapters), "trial_records": len(records),
                      "contrast_rows": result["summary"]["contrast_rows"], "output": str(args.output)}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

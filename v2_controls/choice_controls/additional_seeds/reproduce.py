"""Replay frozen B8 headline tables, or recompute the original analysis from saved trials."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text())


def verify_tables():
    manifest = load(ROOT / "assets.json")
    for name, expected in manifest["included_tables"].items():
        path = ROOT / "tables" / name
        if digest(path) != expected["sha256"]:
            raise ValueError(f"Frozen table hash mismatch: {name}")
    return manifest


def replay(data_dir, output, manifest):
    from src.prepare_report_data import (envelope, headline_specs, save_json,
                                         select_required, validate_metadata, validate_table)
    support = data_dir / "report_support" if (data_dir / "report_support").is_dir() else data_dir
    for name, expected in manifest["replay_inputs"].items():
        path = support / name
        if digest(path) != expected["sha256"]:
            raise ValueError(f"Not the authoritative completed evidence: {name}")
    summary, coverage = load(support / "summary.json"), load(support / "coverage.json")
    validate_metadata(summary, coverage)
    rows = load(support / "primary_contrasts.json")
    validate_table("primary_contrasts", rows)
    envelopes = [envelope(row, "primary_contrasts.json", i) for i, row in enumerate(rows)]
    outputs = {f"headline_{family.lower()}.json": select_required(envelopes, headline_specs(family))
               for family in ("B1", "B2")}
    outputs.update({"summary.json": summary,
                    "matched_history_summary.json": load(support / "matched_history_summary.json")})
    # Validate in memory before creating the requested new output directory.
    for name, value in outputs.items():
        encoded = (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()
        if hashlib.sha256(encoded).hexdigest() != manifest["included_tables"][name]["sha256"]:
            raise ValueError(f"Replayed table does not match authoritative bytes: {name}")
    output.mkdir(parents=True, exist_ok=False)
    for name, value in outputs.items():
        save_json(output / name, value)
    return {"mode": "table_replay", "tables": len(outputs), "exact_hash_match": True,
            "trials_in_source": summary["trial_records"], "inference_run": False}


def from_trials(data_dir, output, manifest):
    from pain_seed_b.analysis import analyze
    from pain_seed_b.analysis_io import strict_json, write_analysis
    from src.finalize_behavior_evidence import independent_counts
    records, adapters, requested, configs = [], [], [], []
    for seed, attempt in ((1, 1), (2, 2)):
        run = data_dir / "behavior" / f"seed_{seed}" / f"attempt_{attempt}" / "run"
        complete, identity = load(run / "completion.json"), load(run / "adapter_identity.json")
        if complete["status"] != "completed" or complete["completed_trials"] != 27060:
            raise ValueError(f"Seed {seed} is not complete; interrupted seed-2 attempt 1 is invalid")
        if identity["training_seed"] != seed:
            raise ValueError("Wrong adapter training seed")
        path = run / "trials.jsonl"
        if digest(path) != manifest["trial_inputs"][str(seed)]["sha256"]:
            raise ValueError(f"Wrong saved trial bytes for seed {seed}")
        with path.open() as stream:
            rows = [strict_json(line) for line in stream]
        if len(rows) != 27060 or any(r["training_seed"] != seed for r in rows):
            raise ValueError("Wrong per-seed trial count or identity")
        records.extend(rows)
        adapters.append({"training_seed": seed, "adapter_identity_digest": identity["identity_digest"],
                         "model": "Qwen_2.5_32B_instruct"})
        requested.append(load(run / "requested_cells.json"))
        configs.append({k: v for k, v in complete["config"].items() if k != "adapter_identity"})
    if requested[0] != requested[1] or configs[0] != configs[1]:
        raise ValueError("Scientific grid/config differs between completed seeds")
    if adapters[0]["adapter_identity_digest"] == adapters[1]["adapter_identity_digest"]:
        raise ValueError("Expected two distinct adapters")
    if output.exists():
        raise FileExistsError("Use a new output directory")
    result = analyze(records, requested[0], adapters)
    direct = independent_counts(records, result["rates"])
    receipt = write_analysis(result, output, sources=manifest["trial_inputs"])
    expected = load(ROOT / "tables/verification.json")["source_receipt"]["outputs"]
    for name, metadata in expected.items():
        if receipt["outputs"][name]["sha256"] != metadata["sha256"]:
            raise ValueError(f"Recomputed analysis differs: {name}")
    return {"mode": "from_trials", "trials": len(records), "exact_hash_match": True,
            "independent_counts": direct, "inference_run": False,
            "resume_prefix_recheck": "not performed; use src/finalize_behavior_evidence.py with original interrupted state"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Local final evidence directory; with --from-trials, root containing behavior/")
    parser.add_argument("--output", type=Path, help="New output directory; never overwrite original evidence")
    parser.add_argument("--verify-tables", action="store_true", help="Check included authoritative table hashes only")
    parser.add_argument("--from-trials", action="store_true", help="Recompute original joint analysis from the completed saved logs (CPU, no models)")
    args = parser.parse_args(argv)
    manifest = verify_tables()
    if args.verify_tables:
        print(json.dumps({"included_tables_verified": len(manifest["included_tables"])}))
        return 0
    if args.data_dir is None or args.output is None:
        parser.error("--data-dir and --output are required unless --verify-tables is selected")
    result = (from_trials if args.from_trials else replay)(args.data_dir, args.output, manifest)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

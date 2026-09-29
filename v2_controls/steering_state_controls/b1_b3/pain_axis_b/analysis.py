"""Audit and analyze original-shaped saved trials; no model execution.

Real log processing runs on compute. Historical and new sources stay separate
through coverage and reference reproduction, then join by exact trial identity.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

from audit_phase_a.src import pain_audit as phase_a
from pain_axis_b.endpoints import aggregate_endpoints, coverage_audit, trial_endpoints
from pain_axis_b.grid import ARMS, B2_PAIRS, B3_PAIRS

MODELS = ("Qwen_2.5_32B_instruct", "Qwen_2.5_72B_instruct")
ARM_SPEC = {row[0]: row for row in ARMS}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_ref(path):
    """Portable input identity; hashes remain the authoritative binding."""
    return str(path)


def audit_record(record, scenarios):
    """Reuse Phase A's structural/state audit, extending only the arm vocabulary."""
    if record.get("arm") not in ARM_SPEC:
        return ["unknown_arm"]
    name, direction, steer_on, works = ARM_SPEC[record["arm"]]
    proxy = dict(record)
    if name == "random_on_button_placebo" or name == "sadness_on_button_placebo":
        proxy["arm"] = "pain_on_button_placebo"
    elif name == "sadness_on_button_works":
        proxy["arm"] = "pain_on_button_works"
    errors = phase_a.audit_record(proxy, scenarios)
    if record["tool_label"] not in (*phase_a.PAIRS, *B2_PAIRS, *B3_PAIRS):
        errors.append("unknown_pair")
    if type(record["label_free"]) is not bool or record["label_free"] != (record["tool_label"] == "label_free"):
        errors.append("pair_label_free_mismatch")
    if record["model"] not in MODELS:
        errors.append("model_outside_accepted_scope")
    expected_direction = direction
    if direction == "rand":
        rs = phase_a.RAND_SEEDS[record["scenario_idx"] % 10]
        expected_direction = "rand" + str(rs)
        if record["rand_seed"] != rs:
            errors.append("random_seed")
    elif record.get("rand_seed") is not None:
        errors.append("nonrandom_seed")
    if record["steer_direction"] != expected_direction:
        errors.append("steering_direction")
    if record.get("protocol") != "2btnN names+saltseed v1":
        errors.append("protocol")
    expected_turns = 8 if record["label_free"] else len(scenarios[record["user_content"]][record["scenario_idx"]]) + (2 if record["button_events"] else 0)
    if len(record["choices"]) != expected_turns:
        errors.append("incomplete_trajectory")
    if record.get("extension_added") != (not record["label_free"] and bool(record["button_events"])):
        errors.append("continuation_flag")
    for row in record["proj_segments"]:
        if any(not isinstance(row.get(k), (int, float)) or not math.isfinite(row[k])
               for k in ("mean_proj", "mean_proj_monitor")):
            errors.append("nonfinite_projection")
    return sorted(set(errors))


def load_logs(paths, scenarios, source_kind, expected_hashes=None):
    records, sources, issues = [], [], []
    seen = set()
    for path in paths:
        path = Path(path)
        digest = sha256(path)
        if expected_hashes is not None:
            pin = expected_hashes[path.name]
            assert digest == pin["sha256"] and path.stat().st_size == pin["bytes"], path
        count = 0
        with path.open() as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                r = json.loads(line, object_pairs_hook=phase_a.no_duplicate_keys)
                key = phase_a.trial_key(r)
                if key in seen:
                    raise ValueError(f"Duplicate {source_kind} trial identity: {key}")
                seen.add(key)
                errors = audit_record(r, scenarios)
                if errors:
                    issues.append({"source": path.name, "line": line_number,
                                   "trial_key": list(key), "errors": errors})
                r["_source"] = {"kind": source_kind, "filename": path.name,
                                "ref": source_ref(path), "line": line_number, "file_sha256": digest}
                records.append(r)
                count += 1
        sources.append({"filename": path.name, "sha256": digest, "ref": source_ref(path),
                        "bytes": path.stat().st_size, "records": count, "kind": source_kind})
        phase_a.progress("read_" + source_kind, len(records), None)
    return records, sources, issues


def historical_reference(records):
    """Reproduce the known released 32B free-relief first-choice counts."""
    wanted = {"paper_pain_pooled": (["pain_on_button_works", "pain_on_button_placebo"], 450, 808),
              "original_random_real": (["random_on_button_works"], 326, 404),
              "unsteered": (["pain_off"], 349, 404)}
    rows = []
    for name, (arms, expected_k, expected_n) in wanted.items():
        selected = [r for r in records if r["sampled"] and r["model"] == MODELS[0]
                    and r["tool_label"] == "relief_vs_inert" and r["arm"] in arms]
        values = [phase_a.first_choice(r) for r in selected]
        valid = [v for v in values if v in ("relief", "other")]
        k, n = sum(v == "relief" for v in valid), len(valid)
        rows.append({"model": MODELS[0], "pair": "relief_vs_inert", "comparison": name,
                     "sampling": "sampled", "successes": k, "valid_denominator": n,
                     "expected_successes": expected_k, "expected_valid_denominator": expected_n,
                     "malformed": len(values)-n, "trials": len(values),
                     "passed": (k, n) == (expected_k, expected_n)})
    return {"status": "passed" if all(r["passed"] for r in rows) else "failed", "rows": rows,
            "source": "Verified Phase A released-log first-choice reproduction"}


def historical_coverage(records):
    counts = Counter((r["model"], r["tool_label"], r["arm"], r["sampled"]) for r in records)
    deviations = []
    for model in MODELS:
        for pair in phase_a.PAIRS:
            for arm in phase_a.ARMS:
                for sampled, want in ((True, 404), (False, 6)):
                    got = counts[(model, pair, arm, sampled)]
                    if got != want:
                        deviations.append(dict(model=model, pair=pair, arm=arm,
                                               sampled=sampled, expected=want, observed=got))
    unexpected = [list(k) + [n] for k, n in counts.items()
                  if k[0] not in MODELS or k[1] not in phase_a.PAIRS or k[2] not in phase_a.ARMS]
    return {"expected_records": 29520, "observed_records": len(records),
            "status": "passed" if not deviations and not unexpected else "failed",
            "deviations": deviations, "unexpected": unexpected}


def examples(records, scenarios, labels):
    """Deterministic outcome-stratified examples; not a representativeness estimate."""
    chosen = {}
    for r in sorted(records, key=phase_a.trial_key):
        if not r["sampled"]:
            continue
        e = trial_endpoints(r)
        categories = [("first", e["first"]["response"]), ("next", e["next"]["response"]),
                      ("literal", e["next"]["literal_response"])]
        for timing, outcome in categories:
            key = (r["model"], r["arm"], timing, outcome)
            if key in chosen:
                continue
            chosen[key] = {"model": r["model"], "pair": r["tool_label"], "arm": r["arm"],
                "selection_category": [timing, outcome], "initial_target_name": r["relief_name"],
                "scenario": [r["user_content"], r["scenario_idx"]], "seed": r["seed"],
                "scenario_text": scenarios[r["user_content"]][r["scenario_idx"]],
                "descriptions": labels[r["tool_label"]], "choices": r["choices"],
                "button_events": r["button_events"], "endpoint": e, "source": r["_source"]}
    return list(chosen.values())


def write_csv(path, rows):
    serial = [{k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list, tuple)) else v
               for k, v in row.items()} for row in rows]
    phase_a.table(path, serial)


def scenario_inventory(data, contents):
    counts = {c: len(data[c]) for c in contents}
    return {"scenarios": sum(counts.values()), "scenario_counts": counts}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--historical-dir", type=Path, required=True)
    p.add_argument("--historical-pins", type=Path, required=True)
    p.add_argument("--scenarios", type=Path, required=True)
    p.add_argument("--requested-manifest", type=Path, required=True)
    p.add_argument("--new-log", type=Path, action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline-only", action="store_true")
    a = p.parse_args(argv)
    assert not a.output.exists(), "Version outputs; never overwrite prior evidence"
    a.output.mkdir(parents=True)
    start = time.monotonic()
    data = json.loads(a.scenarios.read_text())
    requested = json.loads(a.requested_manifest.read_text())
    pins = {r["filename"]: r for r in json.loads(a.historical_pins.read_text())}
    historical, hsources, hissues = load_logs([a.historical_dir / n for n in sorted(pins)], data,
                                             "historical", pins)
    reference = historical_reference(historical)
    hcoverage = historical_coverage(historical)
    phase_a.dump(a.output / "baseline_reference.json", reference)
    phase_a.dump(a.output / "historical_coverage.json", hcoverage)
    phase_a.dump(a.output / "historical_audit_issues.json", hissues)
    assert reference["status"] == hcoverage["status"] == "passed" and not hissues, "Historical instrument/reference mismatch"
    phase_a.progress("historical_reference", len(historical), len(historical))
    new, nsources, nissues = load_logs(a.new_log, data, "new")
    phase_a.dump(a.output / "new_audit_issues.json", nissues)
    assert not nissues, "New trial state/parser/settings audit failed"
    assert not ({phase_a.trial_key(r) for r in new} & {phase_a.trial_key(r) for r in historical}), "Historical trials regenerated"
    coverage = coverage_audit(new, requested, MODELS)
    phase_a.dump(a.output / "requested_vs_completed.json", coverage)
    assert not coverage["unexpected_records"] and not coverage["invalid_trial_identities"]
    joined = historical + new
    rates = aggregate_endpoints(joined)
    phase_a.dump(a.output / "rates.json", rates)
    write_csv(a.output / "rates.csv", rates)
    if not a.baseline_only:
        from pain_axis_b.comparisons import build_comparisons
        comparisons = build_comparisons(joined)
        for name, value in comparisons.items():
            phase_a.dump(a.output / f"{name}.json", value)
            if isinstance(value, list):
                write_csv(a.output / f"{name}.csv", value)
        phase_a.dump(a.output / "examples.json", examples(joined, data, requested["labels"]))
    phase_a.dump(a.output / "sources.json", hsources + nsources)
    counts = Counter((r["model"], "sampled" if r["sampled"] else "greedy", r["_source"]["kind"]) for r in joined)
    summary = {"status": "baseline_verified" if a.baseline_only else "complete" if coverage["complete"] else "partial",
               "historical_trials": len(historical), "new_trials": len(new), "combined_trials": len(joined),
               "record_counts": [{"model": k[0], "sampling": k[1], "source_kind": k[2], "records": v} for k, v in sorted(counts.items())],
               "rate_rows": len(rates), "scenario_group_unit": "(user_content, scenario_idx)",
               **scenario_inventory(data, requested["contents"]),
               "analysis_seconds": time.monotonic()-start,
               "historical_reference_passed": True, "all_raw_audits_passed": True,
               "new_coverage_complete": coverage["complete"], "greedy_separate_from_sampled": True,
               "limitation": "Historical and new runtime/batch paths may differ despite identical nominal seeds."}
    phase_a.dump(a.output / "analysis_summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

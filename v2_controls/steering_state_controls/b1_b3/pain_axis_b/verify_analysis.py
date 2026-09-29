"""Independent, stdlib-only verification of saved Phase B analysis outputs.

No production endpoint, aggregation, coverage, or interval helpers are imported.
``passed`` means saved counts agree with supplied raw records, not that execution
is complete or that records are scientifically valid. Partial grids can pass.

New-record attribution uses _source.kind (new/historical) first, then manifest
new_trial_keys (nine-field arrays in TRIAL_FIELDS order), then explicit
new_model_pair_arms ([model, pair, arm] arrays). The standard manifest's
scope=new_trials_per_model also explicitly designates cells crossed with models.
Unattributed records fail closed. Historical source tags always remain historical.

CLI inputs: rates.json, requested_vs_completed.json and optional sources.json
inside --analysis-dir. Untagged JSONL records can get provenance from sources.json
only after filename, SHA-256, byte count and nonblank record count verification.
Models come from repeated --model, manifest.models, or the accepted two-model
experiment scope. --output is created exclusively; existing evidence is untouched.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import zlib

TRIAL_FIELDS = ("model", "tool_label", "user_content", "scenario_idx", "names_key",
                "relief_name", "sampled", "seed", "arm")
ROW_FIELDS = ("model", "pair", "arm", "sampled", "initial_target_position", "metric", "stage")
DEFAULT_MODELS = ("Qwen_2.5_32B_instruct", "Qwen_2.5_72B_instruct")
CHECKS = [
    "Raw trial identities and duplicate identities; unique saved choice/event turns and valid anchors.",
    "First response is exactly turn 0; next is exactly min(relief event turn)+1; missing and malformed responses retained.",
    "Current target uses saved chose/picked; literal same/switch uses first target anchor picked.",
    "Exact rate-row identity set, sampling and initial position splits, pooled/swap/unlabeled stages.",
    "Every row: successes, valid_denominator, response_distribution, trial_records, trial_scenarios, scenarios, malformed, unavailable, unavailable_next, no_target_press, and simple rate.",
    "Secondary any_later_target includes every anchored trial, with final first presses counted false.",
    "Confidence bounds only: required nullability, finite [0,1] range, order, and containment of the rate.",
    "New-only coverage attribution, per-content sampled/greedy counts, cell/model totals, statuses, unexpected records, and duplicate identities.",
    "Full grid manifests: exact scenario/name/initial assignment/seed identity and independently calculated generation-seed salt; explicit key manifests: expected key membership.",
]
LIMITS = [
    "Confidence interval formula and numerical endpoints are not independently recomputed.",
    "No parser, steering-state, transcript, projection, model-execution, comparison, or historical-reference audit.",
    "No claim about unavailable production outputs; this receipt checks only the supplied records and saved outputs.",
    "New/historical attribution trusts explicit source or manifest declarations; CLI source files are hash-checked, not authenticated.",
    "Count-only manifests cannot establish expected within-cell identities beyond duplicate detection.",
]


def _json_safe(value):
    """Keep failure receipts valid JSON even when an API caller supplies NaN/Inf."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    return repr(value)


def _issue(issues, code, path, expected=None, observed=None):
    issues.append({"code": code, "path": path, "expected": _json_safe(expected),
                   "observed": _json_safe(observed)})


def _equal(issues, path, expected, observed):
    # JSON type strictness prevents True silently satisfying an expected count of 1.
    if type(expected) is not type(observed) or expected != observed:
        _issue(issues, "value_mismatch", path, expected, observed)


def _trial_key(record):
    key = tuple(record[field] for field in TRIAL_FIELDS)
    hash(key)
    if type(record["sampled"]) is not bool:
        raise ValueError("sampled must be boolean")
    for field in ("scenario_idx", "seed"):
        if type(record[field]) is not int:
            raise ValueError(field + " must be integer")
    for field in set(TRIAL_FIELDS) - {"sampled", "scenario_idx", "seed"}:
        if not isinstance(record[field], str):
            raise ValueError(field + " must be string")
    return key


def _response(choice):
    if choice is None:
        return "unavailable"
    if choice.get("picked") is None or choice.get("chose") not in ("relief", "other"):
        return "malformed"
    return "target" if choice["chose"] == "relief" else "other"


def _raw_observations(record):
    """Yield metric/stage/response directly from original-shaped trial fields."""
    names = record["button_names"]
    if len(names) != 2 or len(set(names)) != 2 or record["relief_name"] not in names:
        raise ValueError("invalid initial button-name assignment")
    choices = {}
    for choice in record["choices"]:
        turn = choice["turn"]
        if type(turn) is not int or turn < 0 or turn in choices:
            raise ValueError("invalid or duplicate choice turn")
        choices[turn] = choice
    targets, event_turns = [], set()
    for event in record["button_events"]:
        turn = event["turn"]
        if type(turn) is not int or turn < 0 or turn in event_turns:
            raise ValueError("invalid or duplicate event turn")
        event_turns.add(turn)
        if event["which"] == "relief":
            targets.append(turn)
    anchor_turn = min(targets) if targets else None
    missing = anchor_turn is not None and anchor_turn + 1 not in choices
    values = [("first_target", "pooled", _response(choices.get(0))),
              ("any_later_target", "pooled", "no_target_press" if not targets else
               "yes" if any(t > anchor_turn for t in targets) else "no")]
    if anchor_turn is None:
        stages, current, literal = ["pooled"], "no_target_press", "no_target_press"
    else:
        if anchor_turn not in choices or choices[anchor_turn].get("picked") not in names:
            raise ValueError("first target event lacks valid saved anchor choice")
        turn = anchor_turn + 1
        nxt = choices.get(turn)
        current = _response(nxt)
        literal = current
        if current in ("target", "other"):
            literal = "same_name" if nxt["picked"] == choices[anchor_turn]["picked"] else "switched"
        swap = record["swap_turn"]
        if swap is not None and (type(swap) is not int or swap < 0):
            raise ValueError("invalid swap turn")
        if record.get("label_free") or record["tool_label"] == "label_free" or swap is None:
            stage = "unlabeled"
        else:
            stage = "before_swap" if turn < swap else "at_swap" if turn == swap else "after_swap"
        stages = ["pooled", stage]
    for stage in stages:
        values.extend((("next_target", stage, current),
                       ("next_same_name", stage, literal), ("next_switch", stage, literal)))
    return values, missing, anchor_turn is None


def _raw_rates(records, issues):
    groups = {}
    eligible = {"first_target": ("target", "other"), "next_target": ("target", "other"),
                "next_same_name": ("same_name", "switched"), "next_switch": ("switched", "same_name"),
                "any_later_target": ("yes", "no")}
    for index, record in enumerate(records):
        try:
            observations, missing, no_anchor = _raw_observations(record)
            position = "first" if record["relief_name"] == record["button_names"][0] else "second"
            scenario = (record["user_content"], record["scenario_idx"])
            for pos in (position, "pooled"):
                for metric, stage, response in observations:
                    key = (record["model"], record["tool_label"], record["arm"], record["sampled"], pos, metric, stage)
                    group = groups.setdefault(key, {"responses": Counter(), "trial_sids": set(),
                        "valid_sids": set(), "successes": 0, "valid_denominator": 0,
                        "trial_records": 0, "unavailable_next": 0, "no_target_press": 0})
                    group["responses"][response] += 1
                    group["trial_sids"].add(scenario)
                    group["trial_records"] += 1
                    group["unavailable_next"] += int(missing)
                    group["no_target_press"] += int(no_anchor)
                    if response in eligible[metric]:
                        group["valid_denominator"] += 1
                        group["successes"] += int(response == eligible[metric][0])
                        group["valid_sids"].add(scenario)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            _issue(issues, "invalid_raw_record", f"records[{index}]", observed=str(exc))
    result = {}
    for key, group in groups.items():
        responses = group["responses"]
        n = group["valid_denominator"]
        result[key] = {**{field: group[field] for field in (
            "successes", "valid_denominator", "trial_records", "unavailable_next", "no_target_press")},
            "sampling": "sampled" if key[3] else "greedy", "trial_scenarios": len(group["trial_sids"]),
            "scenarios": len(group["valid_sids"]), "response_distribution": dict(responses),
            "malformed": responses["malformed"], "unavailable": responses["unavailable"],
            "rate": group["successes"] / n if n else None}
    return result


def _numeric(value):
    return type(value) in (float, int) and math.isfinite(value)


def _check_rates(expected, rows, issues):
    seen = set()
    for index, row in enumerate(rows):
        path = f"rates[{index}]"
        try:
            key = tuple(row[field] for field in ROW_FIELDS)
            hash(key)
            if type(row["sampled"]) is not bool:
                raise ValueError("sampled must be boolean")
        except (KeyError, TypeError, ValueError) as exc:
            _issue(issues, "invalid_rate_row", path, observed=str(exc))
            continue
        if key in seen:
            _issue(issues, "duplicate_rate_row", path, observed=list(key))
        seen.add(key)
        if key not in expected:
            _issue(issues, "extra_rate_row", path, observed=list(key))
            continue
        want = expected[key]
        for field, value in want.items():
            if field not in row:
                _issue(issues, "missing_field", path + "." + field, expected=value)
            elif field == "rate" and value is not None:
                if not _numeric(row[field]) or not math.isclose(value, row[field], rel_tol=1e-12, abs_tol=1e-12):
                    _issue(issues, "value_mismatch", path + ".rate", value, row[field])
            elif field == "response_distribution":
                if not isinstance(row[field], dict) or any(type(v) is not int for v in row[field].values()):
                    _issue(issues, "invalid_distribution", path + "." + field, value, row[field])
                else:
                    _equal(issues, path + "." + field, value, row[field])
            else:
                _equal(issues, path + "." + field, value, row[field])
        if "ci_low" not in row or "ci_high" not in row:
            _issue(issues, "missing_confidence_bounds", path)
        elif want["scenarios"] <= 1 or want["valid_denominator"] == 0:
            for field in ("ci_low", "ci_high"):
                _equal(issues, path + "." + field, None, row[field])
        else:
            lo, hi = row["ci_low"], row["ci_high"]
            if not (_numeric(lo) and _numeric(hi) and 0 <= lo <= want["rate"] <= hi <= 1):
                _issue(issues, "implausible_confidence_bounds", path,
                       "finite 0 <= ci_low <= rate <= ci_high <= 1", [lo, hi])
    for key in sorted(expected.keys() - seen):
        _issue(issues, "missing_rate_row", "rates", expected=list(key))
    return len(seen & expected.keys())


def _manifest_specs(manifest, models):
    if len(models) != len(set(models)) or any(not isinstance(m, str) for m in models):
        raise ValueError("models must be unique strings")
    specs = {}
    for cell in manifest["cells"]:
        arm = cell["arm"][0] if isinstance(cell["arm"], (tuple, list)) else cell["arm"]
        key = (cell["pair"], arm)
        if key in specs:
            raise ValueError("duplicate manifest cell")
        expected = {}
        for content, modes in cell["contents"].items():
            for mode in ("sampled", "greedy"):
                n = modes[mode]
                if type(n) is not int or n < 0:
                    raise ValueError("manifest counts must be nonnegative integers")
                expected[(content, mode)] = n
        if any(type(cell[mode]) is not int or cell[mode] != sum(
                n for (_, m), n in expected.items() if m == mode) for mode in ("sampled", "greedy")):
            raise ValueError("inconsistent manifest mode totals")
        if type(cell["trials"]) is not int or cell["trials"] != sum(expected.values()):
            raise ValueError("inconsistent manifest cell total")
        specs[key] = (cell, expected)
    for field, value in (("cell_count", len(specs)), ("trial_count", sum(c[0]["trials"] for c in specs.values())),
                         ("sampled_count", sum(c[0]["sampled"] for c in specs.values())),
                         ("greedy_count", sum(c[0]["greedy"] for c in specs.values()))):
        if field in manifest and (type(manifest[field]) is not int or manifest[field] != value):
            raise ValueError("inconsistent manifest " + field)
    return specs


def _grid_identities(manifest):
    fields = ("scenario_counts", "button_names", "seed_bases")
    if not all(field in manifest for field in fields):
        if any(field in manifest for field in fields):
            raise ValueError("incomplete full-grid identity specification")
        return None
    names, bases = manifest["button_names"], manifest["seed_bases"]
    rotation = list(names)
    if not rotation or any(len(v) != 2 or len(set(v)) != 2 for v in names.values()):
        raise ValueError("invalid manifest button names")
    if len(set(bases)) != len(bases) or any(type(b) is not int for b in bases):
        raise ValueError("invalid manifest seed bases")
    identities = set()
    for content, n in manifest["scenario_counts"].items():
        if type(n) is not int or n <= 0:
            raise ValueError("invalid scenario count")
        for name in names[rotation[0]]:
            identities.add((content, 0, rotation[0], name, False, 0))
        for scenario in range(n):
            key = rotation[scenario % len(rotation)]
            for name in names[key]:
                for base in bases:
                    identities.add((content, scenario, key, name, True, base + scenario))
    return identities


def _select_new(records, manifest, models, specs, issues):
    explicit_keys = None
    if "new_trial_keys" in manifest:
        explicit_keys = set()
        for key in manifest["new_trial_keys"]:
            if not isinstance(key, (list, tuple)) or len(key) != len(TRIAL_FIELDS):
                raise ValueError("new_trial_keys must contain nine-field arrays")
            identity = _trial_key(dict(zip(TRIAL_FIELDS, key)))
            if identity in explicit_keys:
                raise ValueError("duplicate explicit new trial key")
            explicit_keys.add(identity)
    cells = set()
    for cell in manifest.get("new_model_pair_arms", []):
        if not isinstance(cell, (list, tuple)) or len(cell) != 3 or any(not isinstance(v, str) for v in cell):
            raise ValueError("new_model_pair_arms must contain model/pair/arm arrays")
        cells.add(tuple(cell))
    if manifest.get("scope") == "new_trials_per_model":
        cells.update((m, p, a) for m in models for p, a in specs)
    new, historical, unknown = [], 0, 0
    for index, record in enumerate(records):
        source = record.get("_source", {})
        kind = source.get("kind") if isinstance(source, dict) else None
        if kind == "new":
            new.append(record)
        elif kind == "historical":
            historical += 1
        elif kind is not None:
            unknown += 1
            _issue(issues, "unknown_source_kind", f"records[{index}]._source.kind", observed=kind)
        elif explicit_keys is not None:
            if _trial_key(record) in explicit_keys:
                new.append(record)
            else:
                historical += 1
        elif cells:
            if (record["model"], record["tool_label"], record["arm"]) in cells:
                new.append(record)
            else:
                historical += 1
        else:
            unknown += 1
            _issue(issues, "unattributed_record", f"records[{index}]",
                   "source kind or explicit manifest new membership")
    return new, historical, unknown, explicit_keys


def _raw_coverage(records, manifest, models, specs, grid_ids, explicit_keys, issues):
    observed, unexpected = defaultdict(Counter), Counter()
    invalid, invalid_cells = [], Counter()
    if grid_ids is not None:
        grid_counts = Counter((i[0], "sampled" if i[4] else "greedy") for i in grid_ids)
        for key, (_, counts) in specs.items():
            if {k: v for k, v in counts.items() if v} != dict(grid_counts):
                _issue(issues, "manifest_grid_count_mismatch", "manifest.cells." + repr(key),
                       dict((repr(k), v) for k, v in grid_counts.items()),
                       dict((repr(k), v) for k, v in counts.items()))
    for record in records:
        cell = (record["model"], record["tool_label"], record["arm"])
        mode = "sampled" if record["sampled"] else "greedy"
        if cell[0] not in models or cell[1:] not in specs:
            unexpected[cell + (record["user_content"], mode)] += 1
            continue
        observed[cell][(record["user_content"], mode)] += 1
        valid = True
        if grid_ids is not None:
            identity = tuple(record[k] for k in ("user_content", "scenario_idx", "names_key", "relief_name", "sampled", "seed"))
            salt = zlib.crc32((record["names_key"] + "|" + record["relief_name"]).encode()) & 0x7fffffff
            seed = (record["seed"] * 1000003 + salt) % (2 ** 62) if record["sampled"] else None
            valid = identity in grid_ids and record.get("gen_seed") == seed
            if identity in grid_ids and record["button_names"] != manifest["button_names"][record["names_key"]]:
                _issue(issues, "button_names_mismatch", "new_records", observed=list(_trial_key(record)))
        if explicit_keys is not None and _trial_key(record) not in explicit_keys:
            _issue(issues, "new_identity_outside_explicit_keys", "new_records", observed=list(_trial_key(record)))
        if not valid:
            invalid_cells[cell] += 1
            invalid.append({"trial_key": list(_trial_key(record)),
                            "reason": "not an expected scenario/name/assignment/sample seed or generation seed"})
    rows, totals, per_model = [], Counter(completed=0, partial=0, unrun=0), {}
    for model in models:
        counts = Counter(completed=0, partial=0, unrun=0)
        for (pair, arm), (spec, expected) in specs.items():
            cell = (model, pair, arm)
            actual = observed[cell]
            details = [{"content": content, "sampling": mode, "expected": expected.get((content, mode), 0),
                        "observed": actual[(content, mode)],
                        "missing": max(0, expected.get((content, mode), 0) - actual[(content, mode)]),
                        "excess": max(0, actual[(content, mode)] - expected.get((content, mode), 0))}
                       for content, mode in sorted(set(expected) | set(actual))]
            exact = all(d["observed"] == d["expected"] for d in details) and not invalid_cells[cell]
            status = "completed" if exact else "partial" if sum(actual.values()) else "unrun"
            rows.append({"model": model, "phase": spec["phase"], "pair": pair, "arm": arm,
                         "status": status, "expected": spec["trials"], "observed": sum(actual.values()),
                         "contents": details, "invalid_trial_identities": invalid_cells[cell]})
            counts[status] += 1
            totals[status] += 1
        per_model[model] = dict(counts)
    return {"scope": "new_records_only_against_requested_grid", "cells": rows,
            "cell_counts": dict(totals), "model_cell_counts": per_model,
            "total_records": len(records), "unexpected_records": sum(unexpected.values()),
            "identity_validation": "exact_grid_and_generation_seed" if grid_ids is not None else "counts_only",
            "invalid_trial_identities": invalid,
            "unexpected": [dict(zip(("model", "pair", "arm", "content", "sampling"), key), observed=n)
                           for key, n in sorted(unexpected.items())],
            "complete": totals["partial"] == totals["unrun"] == 0 and not unexpected}


def _compare_tree(expected, observed, path, issues):
    """Compare required coverage fields, allowing additive metadata and list order."""
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            _issue(issues, "value_mismatch", path, expected, observed)
            return
        for key, value in expected.items():
            if key not in observed:
                _issue(issues, "missing_field", path + "." + key, value)
            else:
                _compare_tree(value, observed[key], path + "." + key, issues)
        # Count maps have a closed key set, unlike objects that may carry metadata.
        if path.endswith("cell_counts") and set(observed) != set(expected):
            _issue(issues, "count_key_mismatch", path, sorted(expected), sorted(observed))
    elif isinstance(expected, list):
        if not isinstance(observed, list):
            _issue(issues, "value_mismatch", path, expected, observed)
            return
        # Coverage lists are small; canonical sorting avoids dependence on saved row order.
        sort_key = lambda x: json.dumps(x, sort_keys=True, allow_nan=False)
        if len(expected) != len(observed):
            _issue(issues, "list_length_mismatch", path, len(expected), len(observed))
        for i, (want, got) in enumerate(zip(sorted(expected, key=sort_key), sorted(observed, key=sort_key))):
            _compare_tree(want, got, f"{path}[{i}]", issues)
    else:
        _equal(issues, path, expected, observed)


def verify_analysis(records, rate_rows, coverage, requested_manifest, models):
    """Return a JSON-serializable receipt; does not write files or run analysis.

    Missing/duplicate/extra rows, ambiguous attribution, malformed raw structure,
    or count disagreements fail verification. Honest partial coverage does not.
    Input raw lists are never modified. Rate and coverage metadata not listed in
    checked are outside scope. models is the independently requested model list.
    """
    issues, unique, seen = [], [], set()
    records, rate_rows, models = list(records), list(rate_rows), list(models)
    duplicates = 0
    for index, record in enumerate(records):
        try:
            key = _trial_key(record)
            if key in seen:
                duplicates += 1
                _issue(issues, "duplicate_trial_identity", f"records[{index}]", observed=list(key))
            else:
                seen.add(key)
                unique.append(record)
        except (KeyError, TypeError, ValueError) as exc:
            _issue(issues, "invalid_trial_identity", f"records[{index}]", observed=str(exc))
    rates = _raw_rates(unique, issues)
    matched = _check_rates(rates, rate_rows, issues)
    counts = {"input_records": len(records), "unique_trial_identities": len(seen),
              "duplicate_trial_identities": duplicates, "expected_rate_rows": len(rates),
              "saved_rate_rows": len(rate_rows), "matched_rate_identities": matched}
    expected_coverage = None
    try:
        specs = _manifest_specs(requested_manifest, models)
        grid_ids = _grid_identities(requested_manifest)
        new, historical, unknown, explicit_keys = _select_new(unique, requested_manifest, models, specs, issues)
        expected_coverage = _raw_coverage(new, requested_manifest, models, specs, grid_ids, explicit_keys, issues)
        counts.update(new_records=len(new), historical_records=historical, unattributed_records=unknown,
                      expected_new_records=len(models) * sum(s[0]["trials"] for s in specs.values()),
                      expected_coverage_cells=len(models) * len(specs))
        if explicit_keys is not None:
            actual_keys = {_trial_key(r) for r in new}
            counts["missing_explicit_new_identities"] = len(explicit_keys - actual_keys)
            counts["unexpected_explicit_new_identities"] = len(actual_keys - explicit_keys)
            if len(explicit_keys) != counts["expected_new_records"]:
                _issue(issues, "explicit_key_count_mismatch", "manifest.new_trial_keys",
                       counts["expected_new_records"], len(explicit_keys))
            # A count-complete cell must not hide an omitted requested identity.
            if expected_coverage["complete"] and explicit_keys != actual_keys:
                _issue(issues, "complete_grid_identity_mismatch", "manifest.new_trial_keys")
        _compare_tree(expected_coverage, coverage, "coverage", issues)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        _issue(issues, "invalid_coverage_input", "coverage_or_manifest", observed=str(exc))
    return {"schema_version": 1, "status": "passed" if not issues else "failed", "passed": not issues,
            "discrepancies": issues, "counts": counts,
            "coverage_complete": None if expected_coverage is None else expected_coverage["complete"],
            "checked": CHECKS, "limits": LIMITS}


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _bad_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def _load_json(text):
    return json.loads(text, object_pairs_hook=_json_pairs, parse_constant=_bad_constant)


def _load_logs(paths, source_path):
    sources = _load_json(source_path.read_text()) if source_path.exists() else []
    records = []
    for path in paths:
        data = path.read_bytes()
        rows = [_load_json(line) for line in data.decode("utf-8").splitlines() if line.strip()]
        candidates = [s for s in sources if s["filename"] == path.name]
        if candidates:
            digest = hashlib.sha256(data).hexdigest()
            matched = [s for s in candidates if s["sha256"] == digest and s["bytes"] == len(data)
                       and s["records"] == len(rows)]
            if len(matched) != 1 or matched[0]["kind"] not in ("new", "historical"):
                raise ValueError("source provenance mismatch or ambiguity: " + str(path))
            kind = matched[0]["kind"]
            for row in rows:
                if "_source" in row and row["_source"].get("kind") != kind:
                    raise ValueError("source provenance conflict: " + str(path))
                row.setdefault("_source", {"kind": kind})
        records.extend(rows)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, action="append", required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--requested-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", action="append", help="repeat to override manifest/default requested models")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("--output must be a new file")
    try:
        manifest = _load_json(args.requested_manifest.read_text())
        records = _load_logs(args.log, args.analysis_dir / "sources.json")
        rates = _load_json((args.analysis_dir / "rates.json").read_text())
        coverage = _load_json((args.analysis_dir / "requested_vs_completed.json").read_text())
        receipt = verify_analysis(records, rates, coverage, manifest,
                                  args.model or manifest.get("models", DEFAULT_MODELS))
    except (OSError, UnicodeError, KeyError, TypeError, ValueError) as exc:
        receipt = {"schema_version": 1, "status": "failed", "passed": False,
                   "discrepancies": [{"code": "input_error", "path": "cli", "observed": str(exc)}],
                   "counts": {}, "coverage_complete": None, "checked": [], "limits": LIMITS}
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Frozen extension analysis; run on compute after the independent raw audit.

Only literal turn-zero choices enter inference. Every attempted answer, including
malformed answers, remains in the denominator. This is not an independent parser
or a replacement for the parent-owned SQLite audit.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

# Permit the documented direct-script CLI without importing any model runtime.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pain_choice_profile.design import (  # noqa: E402
    BUTTON_NAMES, CONDITIONS, ORIGINAL_CONTENTS, PAIRS, build_grid, digest,
    scenario_id, validate_scenarios,
)
from pain_choice_profile.endpoints import first_choice, initial_position  # noqa: E402
from pain_choice_profile.statistics import joint_counts, linear_summary  # noqa: E402

CATEGORIES = ("target", "other", "malformed")
POSITIONS = ("first", "second", "pooled")
PAIR_IDS = tuple(PAIRS)
CONDITION_IDS = tuple(CONDITIONS)
HISTORICAL_PAIR = "severe_user_harm"
REPLICATES = 10000
SEED = 20260922
CAVEATS = [
    "Historical harm wording was longer; runtime and batching differed. These are descriptive cross-run comparisons, not isolated wording effects.",
    "No equivalence margin was prespecified; failure to detect a difference does not establish equivalence.",
    "Intervals are pointwise 95% scenario-cluster intervals, not simultaneous intervals or random-direction uncertainty.",
    "Greedy runs are separate diagnostics and never enter sampled inference.",
    "Endpoint parsing is reused; an independent primary recount and SQLite audit must be supplied separately.",
]


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank record at {path}:{line_number}")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Not a record at {path}:{line_number}")
            yield record


def scenario_contract(scenarios):
    validate_scenarios(scenarios)  # Manifest contains 172; inference uses only 101.
    ids, strata, hashes = [], [], {}
    for content in ORIGINAL_CONTENTS:
        for i, messages in enumerate(scenarios[content]):
            sid = scenario_id(content, i)
            ids.append(sid)
            strata.append(content)
            hashes[sid] = digest(messages)
    if len(ids) != 101:
        raise ValueError("Expected exactly 101 original source scenarios")
    return ids, strata, hashes


def expected_keys(scenarios, historical=False):
    """Exact identities include assignment and seed, not merely aggregate counts."""
    expected = {}
    for pair, content, arm, index, names_key, target_name, sampled, seed in build_grid(scenarios):
        if historical and pair != PAIR_IDS[0]:
            continue
        label = HISTORICAL_PAIR if historical else pair
        sid = scenario_id(content, index)
        side = BUTTON_NAMES[names_key].index(target_name)
        key = (label, arm[0], sid, side, sampled, seed)
        expected[key] = (list(BUTTON_NAMES[names_key]), target_name)
    if len(expected) != (1640 if historical else 9840):
        raise ValueError("Invalid expected grid")
    return expected


def selected_historical(record):
    # Select by both identifiers; inconsistent candidate metadata fails, not skips.
    if record.get("tool_label") != HISTORICAL_PAIR and record.get("pair_id") != 1:
        return False
    if record.get("condition_id") not in CONDITION_IDS:
        return False
    if record.get("tool_label") != HISTORICAL_PAIR or record.get("pair_id") != 1:
        raise ValueError("Historical pair_id/tool_label mismatch")
    return True


def validate_record(record, expected, hashes, seen, historical=False):
    label = record["tool_label"]
    condition = record["condition_id"]
    sid = record["scenario_id"]
    if type(record["sampled"]) is not bool or type(record["seed"]) is not int:
        raise ValueError("sampled must be bool and seed must be int")
    side = POSITIONS.index(initial_position(record))
    key = (label, condition, sid, side, record["sampled"], record["seed"])
    if key not in expected:
        raise ValueError(f"Unexpected trial identity: {key}")
    if key in seen:
        raise ValueError(f"Duplicate trial identity: {key}")
    names, target_name = expected[key]
    if record["button_names"] != names or record["relief_name"] != target_name:
        raise ValueError(f"Assignment mismatch: {key}")
    pair_id = 1 if historical else PAIRS[label]["pair_id"]
    if type(record["pair_id"]) is not int or record["pair_id"] != pair_id:
        raise ValueError(f"Pair ID mismatch: {key}")
    meta = CONDITIONS[condition]
    if record["direction"] != meta["direction"] or record["dose"] != meta["dose"]:
        raise ValueError(f"Direction/dose mismatch: {key}")
    if record["source_kind"] != "original" or record["scenario_content_hash"] != hashes[sid]:
        raise ValueError(f"Source scenario/hash mismatch: {key}")
    cat = first_choice(record)  # Fails on missing turn zero; never seeks a valid later turn.
    seen.add(key)
    return key, cat


def aggregate(path, scenarios, output, historical=False):
    ids, strata, hashes = scenario_contract(scenarios)
    sid_index = {sid: i for i, sid in enumerate(ids)}
    pairs = (HISTORICAL_PAIR,) if historical else PAIR_IDS
    expected = expected_keys(scenarios, historical)
    outcomes = np.zeros((len(pairs), 4, 2, 101, 3), dtype=np.int16)
    greedy = np.zeros((len(pairs), 4, 2, 3), dtype=np.int16)
    seen, representatives = set(), {}
    prefix = "historical" if historical else "extension"
    fields = ["pair_id", "tool_label", "condition_id", "direction", "dose", "scenario_id",
              "source_kind", "sampled", "seed", "initial_position", "category", "target",
              "raw_answer", "picked", "prefill_proj_monitor", "scenario_content_hash"]
    with (output / f"{prefix}_first_choices.csv").open("w", newline="") as first_stream, \
            (output / f"{prefix}_raw_choice_projections.jsonl").open("w") as projection_stream:
        writer = csv.DictWriter(first_stream, fieldnames=fields)
        writer.writeheader()
        for record in read_jsonl(path):
            if historical and not selected_historical(record):
                continue
            key, cat = validate_record(record, expected, hashes, seen, historical)
            label, condition, sid, side, sampled, seed = key
            p, c, s, k = pairs.index(label), CONDITION_IDS.index(condition), sid_index[sid], CATEGORIES.index(cat)
            if sampled:
                outcomes[p, c, side, s, k] += 1
                gallery_key = (p, side, c, cat)
                order = (sid, seed)
                if gallery_key not in representatives or order < representatives[gallery_key][0]:
                    representatives[gallery_key] = (order, record)
            else:
                greedy[p, c, side, k] += 1
            common = {name: record[name] for name in fields if name in record}
            choice = record["choices"][0]
            writer.writerow({**common, "initial_position": POSITIONS[side], "category": cat,
                             "target": int(cat == "target"), **{name: choice[name] for name in
                             ("raw_answer", "picked", "prefill_proj_monitor")}})
            # Preserve raw projections and all source choice fields, without further endpoints.
            projection_stream.write(json.dumps({**common, "choices": record["choices"],
                "proj_segments": record["proj_segments"]}, allow_nan=False) + "\n")
    if seen != set(expected):
        raise ValueError(f"Incomplete {prefix} grid: {len(seen)}/{len(expected)} identities")
    if not np.all(outcomes.sum(axis=-1) == 2) or not np.all(greedy.sum(axis=-1) == 3):
        raise ValueError(f"Incorrect {prefix} sampled/greedy cell sizes")
    write_json(output / f"{prefix}_representative_records.json", {
        "selection": "Sampled only; each pair/position/condition/observed category: lexicographically smallest source scenario ID, then smallest seed. Missing categories are explicitly listed, not fabricated.",
        "records": [value[1] for _, value in sorted(representatives.items())],
        "absent_categories": [{"pair": pair, "position": POSITIONS[side], "condition_id": condition, "category": cat}
            for p, pair in enumerate(pairs) for side in range(2) for c, condition in enumerate(CONDITION_IDS)
            for cat in CATEGORIES if (p, side, c, cat) not in representatives],
    })
    return outcomes, greedy, ids, strata


def position_rates(outcomes, position):
    """Return pair x condition x scenario x category, including malformed trials."""
    a = np.asarray(outcomes)
    if a.ndim != 5 or a.shape[1:3] != (4, 2) or a.shape[-1] != 3:
        raise ValueError("Expected pair x 4 conditions x 2 positions x scenario x 3 categories")
    if not np.all(np.isfinite(a)) or np.any(a < 0) or np.any(a != np.floor(a)):
        raise ValueError("Outcomes must be finite nonnegative integer counts")
    if not np.all(a.sum(axis=-1) == 2):
        raise ValueError("Every scenario/condition/position must contain two attempts")
    if position not in POSITIONS:
        raise ValueError("Unknown position")
    return a[:, :, POSITIONS.index(position)] / 2 if position != "pooled" else a.sum(axis=2) / 4


def infer(outcomes, historical, greedy, historical_greedy, counts):
    """Pure numerical analysis; a single caller-owned count matrix drives all tests."""
    if outcomes.shape[0] != 6 or historical.shape[0] != 1 or outcomes.shape[3] != historical.shape[3]:
        raise ValueError("Expected six new pairs and one historical pair on identical scenarios")
    tables = {name: [] for name in ("rates", "greedy", "pain_contrasts", "half_tests",
              "pair6_target_minus_other", "historical_comparisons", "steering_changes")}
    vectors, metric_ids = [], []

    def summarize(value, low, high, metric_id):
        result = linear_summary(value, counts, low, high)
        vectors.append(np.asarray(value, dtype=float))
        metric_ids.append(metric_id)
        return {**result, "feasible_lower": low, "feasible_upper": high, "metric_id": metric_id}

    none = CONDITION_IDS.index("none_d0p0")
    pain = CONDITION_IDS.index("pain_d1p0")
    for position in POSITIONS:
        current = position_rates(outcomes, position)
        old = position_rates(historical, position)
        target, old_target = current[..., 0], old[0, ..., 0]
        attempts = 4 if position == "pooled" else 2
        for source, pair_names, array, raw, gr in (
                ("extension", PAIR_IDS, current, outcomes, greedy),
                ("historical", (HISTORICAL_PAIR,), old, historical, historical_greedy)):
            absolute = raw.sum(axis=2) if position == "pooled" else raw[:, :, POSITIONS.index(position)]
            g = gr.sum(axis=2) if position == "pooled" else gr[:, :, POSITIONS.index(position)]
            for p, pair in enumerate(pair_names):
                for c, condition in enumerate(CONDITION_IDS):
                    meta = {"source": source, "pair_id": p + 1, "pair": pair,
                            "position": position, **CONDITIONS[condition]}
                    category_counts = {f"{cat}_count": int(absolute[p, c, :, k].sum()) for k, cat in enumerate(CATEGORIES)}
                    for k, cat in enumerate(CATEGORIES):
                        tables["rates"].append({**meta, "split": "sampled", "category": cat,
                            "n_trials": attempts * array.shape[2], **category_counts,
                            "count": category_counts[f"{cat}_count"],
                            **summarize(array[p, c, :, k], 0., 1., f"rate/{source}/{p+1}/{condition}/{position}/{cat}")})
                    n_greedy = int(g[p, c].sum())
                    if n_greedy != (6 if position == "pooled" else 3):
                        raise ValueError("Greedy must have three per position, six pooled")
                    tables["greedy"].append({**meta, "split": "greedy_diagnostic", "n_trials": n_greedy,
                        **{f"{cat}_count": int(g[p, c, k]) for k, cat in enumerate(CATEGORIES)},
                        **{f"{cat}_share": float(g[p, c, k] / n_greedy) for k, cat in enumerate(CATEGORIES)},
                        "inference": False})
        for p, pair in enumerate(PAIR_IDS):
            meta = {"pair_id": p + 1, "pair": pair, "position": position}
            for c, condition in enumerate(CONDITION_IDS):
                if c != pain:
                    tables["pain_contrasts"].append({**meta, "comparison": f"pain_minus_{CONDITIONS[condition]['direction']}",
                        **summarize(target[p, pain] - target[p, c], -1., 1., f"pain_contrast/{p+1}/{condition}/{position}")})
            if p + 1 in (3, 4, 5):
                stats = summarize(target[p, pain] - .5, -.5, .5, f"half/{p+1}/{position}")
                tables["half_tests"].append({**meta, "condition_id": "pain_d1p0", **stats,
                    "pointwise_lower_above_zero": bool(stats["lo"] > 0)})
            if p + 1 == 6:
                for c, condition in enumerate(CONDITION_IDS):
                    tables["pair6_target_minus_other"].append({**meta, **CONDITIONS[condition],
                        **summarize(current[p, c, :, 0] - current[p, c, :, 1], -1., 1., f"target_minus_other/6/{condition}/{position}")})
            if p + 1 in (1, 2):
                for c, condition in enumerate(CONDITION_IDS):
                    tables["historical_comparisons"].append({**meta, **CONDITIONS[condition],
                        "historical_pair": HISTORICAL_PAIR,
                        **summarize(target[p, c] - old_target[c], -1., 1., f"historical/{p+1}/{condition}/{position}")})
                    if c != none:
                        value = (target[p, c] - target[p, none]) - (old_target[c] - old_target[none])
                        tables["steering_changes"].append({**meta, **CONDITIONS[condition],
                            "historical_pair": HISTORICAL_PAIR, "baseline": "none_d0p0",
                            **summarize(value, -2., 2., f"did/{p+1}/{condition}/{position}")})
    decisions = []
    for pair_id in (3, 4, 5):
        rows = [row for row in tables["half_tests"] if row["pair_id"] == pair_id and row["position"] != "pooled"]
        supported = all(row["pointwise_lower_above_zero"] for row in rows)
        decisions.append({"pair_id": pair_id, "both_positions_lower_above_zero": supported,
                          "label": "above half in both positions" if supported else "above half in both positions not established",
                          "rule": "Each position-specific pointwise 95% lower bound exceeds zero for pain target rate minus 0.5; pooled is not the decision rule."})
    values = np.asarray(vectors)
    bootstrap_values = counts @ values.T / values.shape[1]
    return tables, decisions, {"metric_ids": np.asarray(metric_ids), "scenario_values": values,
                               "draws": bootstrap_values}


def analyze(raw, historical, scenarios_path, output):
    output = Path(output)
    # A failed attempt leaves diagnostic partial exports, but never a success marker.
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new or empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    scenarios = json.loads(Path(scenarios_path).read_text())
    outcomes, greedy, ids, strata = aggregate(raw, scenarios, output)
    old, old_greedy, old_ids, old_strata = aggregate(historical, scenarios, output, historical=True)
    if ids != old_ids or strata != old_strata:
        raise ValueError("Historical and new source scenario order differs")
    counts = joint_counts(strata, replicates=REPLICATES, seed=SEED)
    tables, decisions, draws = infer(outcomes, old, greedy, old_greedy, counts)
    for name, rows in tables.items():
        write_csv(output / f"{name}.csv", rows)
    np.savez_compressed(output / "scenario_sufficient_statistics.npz", outcomes=outcomes,
        historical_outcomes=old, greedy=greedy, historical_greedy=old_greedy,
        scenario_ids=np.asarray(ids), strata=np.asarray(strata), categories=np.asarray(CATEGORIES),
        pairs=np.asarray(PAIR_IDS), conditions=np.asarray(CONDITION_IDS), positions=np.asarray(POSITIONS[:2]))
    np.savez_compressed(output / "joint_bootstrap_counts.npz", counts=counts,
                        scenario_ids=np.asarray(ids), strata=np.asarray(strata))
    np.savez_compressed(output / "bootstrap_draws.npz", **draws)
    summary = {"schema_version": 1, "status": "complete", "replicates": REPLICATES, "seed": SEED,
        "new_rows": int(outcomes.sum() + greedy.sum()), "new_sampled": int(outcomes.sum()),
        "new_greedy": int(greedy.sum()), "historical_selected_rows": int(old.sum() + old_greedy.sum()),
        "historical_sampled": int(old.sum()), "historical_greedy": int(old_greedy.sum()),
        "n_scenarios": len(ids), "scenario_digest": digest(scenarios),
        "raw_sha256": file_hash(raw), "historical_sha256": file_hash(historical),
        "scenarios_sha256": file_hash(scenarios_path),
        "denominator": "All sampled literal turn-zero attempts, including malformed; 202 per position, 404 pooled per pair/condition.",
        "bootstrap": "One stratified count matrix across the same 101 scenarios for all conditions, positions, new pairs and historical comparisons. Constant/degenerate safeguards use unchanged linear_summary.",
        "draws_orientation": "bootstrap_draws.npz draws[replicate, metric]; scenario_values[metric, scenario]; metric_ids maps columns/rows.",
        "half_test_decisions": decisions, "tables": tables, "caveats": CAVEATS}
    write_json(output / "analysis_complete.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--historical", required=True, type=Path)
    parser.add_argument("--scenarios", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    analyze(args.raw, args.historical, args.scenarios, args.output)


if __name__ == "__main__":
    main()

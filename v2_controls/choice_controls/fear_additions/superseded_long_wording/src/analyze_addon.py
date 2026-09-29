"""Frozen fear add-on analysis. Run on compute after the parent-owned raw audit.

Only literal turn zero enters inference; malformed attempts stay in denominators.
This validates the analysis contract, not model execution or later-turn telemetry.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

# Direct-script invocation is independent of the caller's current directory.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from pain_choice_profile import design
from pain_choice_profile.endpoints import first_choice
from pain_choice_profile.statistics import joint_counts, linear_summary

CATEGORIES = ("target", "other", "malformed")
POSITIONS = ("first", "second", "pooled")
CONTROL_IDS = ("pain_d1p0", "sadness_d1p0", "random_d1p0", "none_d0p0")
IDENTITY_FIELDS = ("tool_label", "user_content", "arm", "scenario_idx",
                   "names_key", "relief_name", "sampled", "seed")
REPLICATES, SEED = 10000, 20260922
LIMITATION = ("Fear uses the longer photos clause ', which they love very much'; "
              "existing controls use shorter wording. Cross-run/runtime/batching "
              "differences also confound this descriptive comparison; it is not direction-only.")
# Same literal source cleanup as the root audit; independently implemented here.
END_TAGS = ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "</s>", "<eos>",
            "<|end|>", "<|endoftext|>")
END_RE = re.compile(r"(?:(?:" + "|".join(re.escape(x) for x in END_TAGS) + r")\s*)+$")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def parse_literal(raw_answer, names, target):
    """Source-compatible permissive longest-name prefix, never next valid turn."""
    require(isinstance(raw_answer, str), "Missing literal raw_answer")
    cleaned = END_RE.sub("", raw_answer).strip()
    normalized = cleaned.strip().strip('\"\'').lower().rstrip(".!?,;:")
    picked = next((name for name in sorted(names, key=len, reverse=True)
                   if normalized.startswith(name.lower())), None)
    category = "malformed" if picked is None else "target" if picked == target else "other"
    return category, picked


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            require(bool(line.strip()), f"Blank raw record at line {line_number}")
            record = json.loads(line, object_pairs_hook=strict_object, parse_constant=reject_constant)
            require(isinstance(record, dict), f"Expected object at line {line_number}")
            yield record


def expected_grid(scenarios):
    grid = design.build_grid(scenarios)
    expected = {(pair, content, arm[0], index, names, target, sampled, seed)
                for pair, content, arm, index, names, target, sampled, seed in grid}
    require(len(expected) == 410 and sum(key[6] for key in expected) == 404,
            "Expected frozen 410-row grid (404 sampled, 6 greedy)")
    return expected


def scenario_contract(scenarios):
    design.validate_scenarios(scenarios)
    ids, strata = [], []
    for content in design.ORIGINAL_CONTENTS:
        for index in range(len(scenarios[content])):
            ids.append(design.scenario_id(content, index))
            strata.append(content)
    require(len(ids) == 101 and list(Counter(strata).values()) == [30, 30, 41],
            "Original scenario order/strata must be 30/30/41")
    return ids, strata


def validate_record(record, scenarios, expected, seen):
    key = tuple(record.get(field) for field in IDENTITY_FIELDS)
    require(all(type(key[i]) is str for i in (0, 1, 2, 4, 5)) and
            all(type(key[i]) is int for i in (3, 7)) and type(key[6]) is bool,
            "Incorrect grid identity field types")
    require(key in expected, f"Unexpected grid identity: {key}")
    require(key not in seen, f"Duplicate grid identity: {key}")
    pair, content, condition, index, names_key, target, sampled, seed = key
    names = list(design.BUTTON_NAMES[names_key])
    side = names.index(target)
    meta = {"button_names": names, "condition_id": condition, "direction": "fear",
            "pair_id": 6, "initial_position": side,
            "scenario_id": design.scenario_id(content, index), "source_kind": "original",
            "scenario_content_hash": design.digest(scenarios[content][index]),
            "gen_seed": design.generation_seed(seed, names_key, target) if sampled else None,
            "steer_direction": "fear", "rand_seed": None,
            "protocol": "fear-photos-spam-addon-v1"}
    for field, value in meta.items():
        require(field in record and type(record[field]) is type(value) and record[field] == value,
                f"Incorrect {field} for {key}")
    for field in ("dose", "steer_coeff", "final_steer_coeff"):
        require(type(record.get(field)) in (int, float) and record[field] == 1.0,
                f"Incorrect {field} for {key}")
    choices = record.get("choices")
    require(isinstance(choices, list) and bool(choices) and isinstance(choices[0], dict),
            "Missing literal first choice")
    first = choices[0]
    require(type(first.get("turn")) is int and first["turn"] == 0,
            "Literal turn zero required, never next valid")
    require(first.get("relief_name_now") == target, "Turn-zero assignment mismatch")
    require("chose" in first and "picked" in first, "Missing saved turn-zero category/picked")
    category = first_choice(record)
    independent, picked = parse_literal(first.get("raw_answer"), names, target)
    require(first["picked"] == picked and category == independent,
            "Independent raw-answer recount disagrees with saved literal choice")
    seen.add(key)
    return key, side, category, independent


def aggregate(records, scenarios):
    ids, strata = scenario_contract(scenarios)
    sid_index = {sid: i for i, sid in enumerate(ids)}
    expected, seen = expected_grid(scenarios), set()
    outcomes = np.zeros((2, 101, 3), dtype=np.int16)
    greedy = np.zeros((2, 3), dtype=np.int16)
    recount = Counter()
    first_rows, representatives = [], {}
    totals = Counter()
    for record in records:
        key, side, category, independent = validate_record(record, scenarios, expected, seen)
        sampled, seed = key[6:]
        sid = record["scenario_id"]
        k = CATEGORIES.index(category)
        if sampled:
            outcomes[side, sid_index[sid], k] += 1
            gallery_key, order = (side, category), (sid, seed)
            if gallery_key not in representatives or order < representatives[gallery_key][0]:
                representatives[gallery_key] = (order, record)
        else:
            greedy[side, k] += 1
        split = "sampled" if sampled else "greedy_diagnostic"
        for pos in (POSITIONS[side], "pooled"):
            recount[split, pos, independent] += 1
        totals["records"] += 1
        totals[split] += 1
        totals["choices"] += len(record["choices"])
        totals["segments"] += len(record.get("proj_segments", []))
        totals["button_events"] += len(record.get("button_events", []))
        choice = record["choices"][0]
        first_rows.append({**{field: record[field] for field in IDENTITY_FIELDS},
            "pair_id": 6, "condition_id": "fear_d1p0", "scenario_id": sid,
            "scenario_content_hash": record["scenario_content_hash"],
            "gen_seed": record["gen_seed"], "position": POSITIONS[side], "split": split,
            "turn": 0, "category": category, "independent_category": independent,
            "raw_answer": choice["raw_answer"], "picked": choice["picked"],
            "chose": choice["chose"]})
    require(seen == expected, f"Incomplete exact grid: {len(seen)}/410")
    require(np.all(outcomes.sum(axis=-1) == 2), "Two attempts required per scenario/position")
    require(np.all(greedy.sum(axis=-1) == 3), "Three greedy runs required per position")
    independent_rows = []
    for split, array in (("sampled", outcomes.sum(axis=1)), ("greedy_diagnostic", greedy)):
        for pos in POSITIONS:
            cells = array.sum(axis=0) if pos == "pooled" else array[POSITIONS.index(pos)]
            require(all(int(cells[k]) == recount[split, pos, cat] for k, cat in enumerate(CATEGORIES)),
                    "Independent raw-answer aggregate counts disagree")
            independent_rows.append({"split": split, "position": pos,
                **{cat + "_count": recount[split, pos, cat] for cat in CATEGORIES},
                "n_trials": int(cells.sum())})
    examples = {"selection": "Sampled only: smallest (scenario_id, seed) per position/observed category.",
        "records": [value[1] for _, value in sorted(representatives.items())],
        "absent_categories": [{"position": POSITIONS[side], "category": cat}
                              for side in range(2) for cat in CATEGORIES
                              if (side, cat) not in representatives]}
    return outcomes, greedy, ids, strata, first_rows, examples, dict(totals), independent_rows


def position_rates(outcomes, position):
    a = np.asarray(outcomes)
    require(a.ndim == 3 and a.shape[0] == 2 and a.shape[2] == 3 and a.shape[1] > 0,
            "Expected 2 positions x scenarios x 3 categories")
    require(np.all(np.isfinite(a)) and np.all(a >= 0) and np.all(a == np.floor(a)) and
            np.all(a.sum(axis=-1) == 2), "Every scenario/position needs exactly two attempts")
    require(position in POSITIONS, "Unknown position")
    return a.sum(axis=0) / 4 if position == "pooled" else a[POSITIONS.index(position)] / 2


def infer(outcomes, greedy, counts):
    """One caller-owned joint matrix; no contrasts or new p-values exported."""
    rates, diagnostics, values, metric_ids = [], [], [], []
    require(np.asarray(greedy).shape == (2, 3) and np.all(greedy >= 0) and
            np.all(greedy == np.floor(greedy)) and np.all(greedy.sum(axis=1) == 3),
            "Greedy requires three attempts per position")
    for position in POSITIONS:
        array = position_rates(outcomes, position)
        n = array.shape[0] * (4 if position == "pooled" else 2)
        cells = outcomes.sum(axis=(0, 1)) if position == "pooled" else outcomes[POSITIONS.index(position)].sum(axis=0)
        meta = {"source": "fear_addon", "pair_id": 6, "pair": "photos_spam",
                "position": position, **design.CONDITIONS["fear_d1p0"]}
        category_counts = {cat + "_count": int(cells[k]) for k, cat in enumerate(CATEGORIES)}
        for k, category in enumerate(CATEGORIES):
            # Keep the established support and constant/degenerate fallback unchanged.
            result = linear_summary(array[:, k], counts, 0.0, 1.0)
            result.pop("p_two_sided")
            metric_id = f"rate/fear_addon/6/fear_d1p0/{position}/{category}"
            rates.append({**meta, "split": "sampled", "category": category,
                "n_trials": n, **category_counts, "count": int(cells[k]),
                **result, "metric_id": metric_id})
            values.append(array[:, k])
            metric_ids.append(metric_id)
        g = greedy.sum(axis=0) if position == "pooled" else greedy[POSITIONS.index(position)]
        diagnostics.append({**meta, "split": "greedy_diagnostic", "n_trials": int(g.sum()),
            **{cat + "_count": int(g[k]) for k, cat in enumerate(CATEGORIES)},
            **{cat + "_share": float(g[k] / g.sum()) for k, cat in enumerate(CATEGORIES)},
            "inference": False})
    values = np.asarray(values)
    return rates, diagnostics, {"metric_ids": np.asarray(metric_ids),
        "scenario_values": values, "draws": counts @ values.T / values.shape[1]}


def select_comparators(rows):
    """Select exactly 12 existing target rows; preserve all original CSV strings."""
    selected = {}
    for row in rows:
        if row.get("source") != "extension":
            continue
        if row.get("pair_id") != "6" and row.get("pair") != "photos_spam":
            continue
        require(row.get("pair_id") == "6" and row.get("pair") == "photos_spam",
                "Existing pair-6 identifier mismatch")
        if row.get("split") != "sampled" or row.get("category") != "target":
            continue
        condition, position = row.get("condition_id"), row.get("position")
        require(condition in CONTROL_IDS and position in POSITIONS, "Unexpected existing control/position")
        key = (position, condition)
        require(key not in selected, f"Duplicate existing comparator: {key}")
        for field in ("point", "lo", "hi", "ci_method", "target_count", "other_count",
                      "malformed_count", "count", "n_trials"):
            require(row.get(field) not in (None, ""), f"Missing existing {field}")
        require(row["n_trials"] == ("404" if position == "pooled" else "202"),
                "Existing comparator denominator mismatch")
        require(not ({"wording", "comparison_limitation"} & row.keys()), "Comparator label field collision")
        selected[key] = dict(row)  # Never coerce/recompute any existing value.
    require(set(selected) == {(p, c) for p in POSITIONS for c in CONTROL_IDS},
            "Expected exactly 12 existing pair-6 target-rate rows")
    return [selected[p, c] for p in POSITIONS for c in CONTROL_IDS]


def comparison_rows(rates, existing):
    rows = [{**row, "wording": "existing_short_wording", "comparison_limitation": LIMITATION}
            for row in select_comparators(existing)]
    rows.extend({**row, "wording": "fear_long_wording", "comparison_limitation": LIMITATION}
                for row in rates if row["category"] == "target")
    require(len(rows) == 15, "Expected 15 comparison rows")
    return rows


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def analyze(raw, scenarios_path, existing_rates, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new or empty output directory")
    scenarios = json.loads(Path(scenarios_path).read_text(encoding="utf-8"))
    with Path(existing_rates).open(newline="", encoding="utf-8") as stream:
        existing = select_comparators(csv.DictReader(stream))
    outcomes, greedy, ids, strata, first_rows, examples, totals, independent = aggregate(read_jsonl(raw), scenarios)
    counts = joint_counts(strata, replicates=REPLICATES, seed=SEED)  # Exactly once.
    rates, diagnostics, draws = infer(outcomes, greedy, counts)
    comparisons = comparison_rows(rates, existing)
    # Validate inputs completely before emitting any artifacts. summary.json is last.
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("fear_rates", rates), ("greedy", diagnostics),
                       ("comparison_rows", comparisons), ("literal_first_choices", first_rows),
                       ("independent_counts", independent)):
        write_csv(output / f"{name}.csv", rows)
    write_json(output / "selected_raw_examples.json", examples)
    np.savez_compressed(output / "scenario_sufficient_statistics.npz", outcomes=outcomes, greedy=greedy,
        scenario_ids=np.asarray(ids), strata=np.asarray(strata), categories=np.asarray(CATEGORIES),
        positions=np.asarray(POSITIONS[:2]))
    np.savez_compressed(output / "joint_bootstrap_counts.npz", counts=counts,
                        scenario_ids=np.asarray(ids), strata=np.asarray(strata))
    np.savez_compressed(output / "bootstrap_draws.npz", **draws)
    sources = {"raw": raw, "scenarios": scenarios_path, "existing_rates": existing_rates,
               "scope": ROOT / "scope.json", "analysis": Path(__file__),
               "design": ROOT / "pain_choice_profile/design.py",
               "statistics": ROOT / "pain_choice_profile/statistics.py",
               "endpoints": ROOT / "pain_choice_profile/endpoints.py",
               "audit": ROOT / "pain_choice_profile/audit.py"}
    summary = {"schema_version": 1, "status": "complete", "condition_id": "fear_d1p0", "pair_id": 6,
        "source_hashes": {name: {"path": str(Path(path).resolve()), "sha256": file_hash(path)}
                          for name, path in sources.items()},
        "grid_digest": design.digest(design.build_grid(scenarios)), "scenario_digest": design.digest(scenarios),
        "record_counts": {**totals, "unique_grid_identities": len(first_rows),
                          "literal_first_choices": len(first_rows), "existing_selected_rows": len(existing)},
        "table_rows": {"fear_rates": len(rates), "greedy": len(diagnostics), "comparison_rows": len(comparisons)},
        "n_scenarios": len(ids), "strata": [30, 30, 41], "replicates": REPLICATES, "seed": SEED,
        "denominator": "All literal turn-zero attempts including malformed: sampled 202/202/404; greedy 3/3/6 separately.",
        "bootstrap": "One joint_counts matrix, ordered original strata, scenario-cluster pointwise 95% intervals; unchanged linear_summary support [0,1] and bounded-mean Hoeffding fallback.",
        "arrays": "outcomes[position,scenario,category]; draws[replicate,metric]; scenario_values[metric,scenario]. metric_ids indexes rates; scenario_ids indexes original order.",
        "independent_recount": "Passed: raw_answer longest-name prefix parsing matches saved turn-zero categories per record and aggregate; no search for a later valid answer.",
        "existing_values": "Original selected CSV values copied without numeric conversion or recomputation. Any existing p_two_sided values are copied legacy fields, not new tests. No new contrast p-values.",
        "limitations": [LIMITATION, "Greedy runs are separate diagnostics, not sampled inference.",
            "Intervals are pointwise, not simultaneous; no pooling across conditions.",
            "Analysis checks do not replace the parent-owned full raw/runtime audit."]}
    write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "scenarios", "existing-rates", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    analyze(args.raw, args.scenarios, args.existing_rates, args.output)


if __name__ == "__main__":
    main()

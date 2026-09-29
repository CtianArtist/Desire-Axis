"""Select existing report rows without calculating or pooling scientific results.

JSON tables contain {source_filename, source_index, source_pointer, coordinates,
row} envelopes. Indices are zero-based in the original input, never renumbered
by a selector. ``row`` is untouched; coordinates are display/lookup metadata.
CSV views retain each full row field and JSON-encode nested cells. All inputs
are validated before the exclusively created output directory is written.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

MEASURED = "measured_behavioral_trials"
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_ROWS = 100_000
TABLES = (
    "endpoint_rates", "primary_contrasts", "swap_rates", "swap_contrasts",
    "position_contrasts",
)
B2_PAIRS = ("increase_vs_inert", "harmonly_kidspics_vs_inert", "harmonly_files_vs_inert")
B1_COMPARISONS = (
    "pain_sham_minus_real", "random_sham_minus_real", "random_gap_minus_pain_gap",
)
B2_COMPARISONS = (
    "pain_minus_random", "pain_minus_sadness", "pain_minus_none",
    "random_minus_sadness", "sadness_minus_none",
)
COORDINATES = (
    "training_seed", "adapter_identity_digest", "model", "family", "pair", "arm",
    "initial_target_position", "metric", "stage", "cohort", "sampling", "sampled",
    "comparison",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_number(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def _finite_float(value):
    number = float(value)
    require(math.isfinite(number), f"Non-finite JSON number: {value}")
    return number


def load(path):
    # Read one extra byte rather than trusting a stat/read race or unbounded input.
    with path.open("rb") as stream:
        data = stream.read(MAX_FILE_BYTES + 1)
    require(len(data) <= MAX_FILE_BYTES, f"Input exceeds byte limit: {path.name}")
    return json.loads(data, object_pairs_hook=_unique_object, parse_constant=_invalid_number,
                      parse_float=_finite_float)


def rows(value, filename):
    require(isinstance(value, list) and len(value) <= MAX_ROWS,
            f"Expected bounded row list: {filename}")
    require(all(isinstance(row, dict) for row in value), f"Expected objects: {filename}")
    return value


def envelope(row, filename, index, *, pointer=None, coordinates=None):
    return {
        "source_filename": filename, "source_index": index,
        "source_pointer": pointer if pointer is not None else f"/{index}",
        "coordinates": coordinates if coordinates is not None else {
            key: row[key] for key in COORDINATES if key in row
        },
        "row": row,
    }


def validate_metadata(summary, coverage):
    require(isinstance(summary, dict) and isinstance(coverage, dict), "Invalid metadata")
    require(summary.get("data_kind") == MEASURED, "Measured summary required")
    seeds = summary.get("training_seeds")
    require(isinstance(seeds, list) and len(seeds) == 2
            and all(type(seed) is int for seed in seeds) and set(seeds) == {1, 2},
            "Exactly training seeds 1 and 2 required")
    require(summary.get("adapter_count") == 2 and summary.get("trial_records") == 54120
            and summary.get("records_per_adapter") == {"1": 27060, "2": 27060},
            "Require 54120 trials, 27060 per seed")
    require(coverage.get("data_kind") == MEASURED and coverage.get("complete") is True
            and coverage.get("total_records") == 54120, "Complete measured coverage required")
    adapters = rows(coverage.get("adapters"), "coverage.json/adapters")
    require(len(adapters) == 2 and {a.get("training_seed") for a in adapters} == {1, 2},
            "Coverage must contain both seeds exactly once")
    require(all(type(a.get("training_seed")) is int and a.get("complete") is True
                and a.get("total_records") == 27060 for a in adapters),
            "Each seed requires complete 27060-trial coverage")


def validate_table(name, table):
    require(bool(table), f"Empty support table: {name}")
    for row in table:
        require(row.get("data_kind") == MEASURED, f"Non-measured row in {name}")
        require(type(row.get("training_seed")) is int and row["training_seed"] in (1, 2),
                f"Per-seed rows required in {name}; no pooled/historical seeds")
        if name.endswith("rates"):
            require(type(row.get("sampled")) is bool, f"Missing sampled flag in {name}")
        else:
            require(row.get("sampling") == "sampled", f"Non-sampled contrast in {name}")
        if name == "endpoint_rates":
            require(row.get("stage") == "pooled", "Endpoint rates require pooled stage")
        elif name == "primary_contrasts":
            require(row.get("stage") == row.get("initial_target_position") == "pooled",
                    "Primary contrasts require pooled stage and position")
        elif name in ("swap_rates", "swap_contrasts"):
            require(row.get("stage") in ("before_swap", "at_swap", "after_swap", "unlabeled")
                    and row.get("initial_target_position") == "pooled"
                    and row.get("metric") in ("next_target", "next_same_name")
                    and row.get("cohort") in ("all_eligible", "matched_history"),
                    f"Invalid swap coordinates in {name}")
            if name == "swap_rates":
                require(row["sampled"] is True, "Swap rates must be sampled")
            else:
                require(row.get("family") == "B1" and row.get("pair") in (
                    "five_harmful_pairs_equal_weight", "label_free"), "Invalid swap contrast scope")
        else:
            require(row.get("family") == "position", "Position contrast family required")
    require({row["training_seed"] for row in table} == {1, 2}, f"Both seeds required in {name}")


def select_required(table, specs):
    selected = []
    for spec in specs:
        found = [entry for entry in table if all(entry["row"].get(k) == v for k, v in spec.items())]
        require(len(found) == 1, f"Required headline row has {len(found)} matches: {spec}")
        entry = found[0]
        if spec["comparison"] == "random_gap_minus_pain_gap":
            gaps = entry["row"].get("joint_support_gap_estimates")
            require(isinstance(gaps, dict) and set(gaps) == {"pain", "random"},
                    "Gap headline must retain joint_support_gap_estimates")
            require(all(isinstance(gap, dict) and {"estimate", "ci_low", "ci_high"} <= gap.keys()
                        for gap in gaps.values()), "Incomplete joint-support gap fields")
        selected.append(entry)
    return selected


def headline_specs(family):
    for seed in (1, 2):
        common = {"training_seed": seed, "family": family, "stage": "pooled",
                  "initial_target_position": "pooled", "sampling": "sampled"}
        if family == "B1":
            for cohort in ("matched_history", "all_eligible"):
                for comparison in B1_COMPARISONS:
                    yield {**common, "pair": "five_harmful_pairs_equal_weight",
                           "metric": "next_target", "cohort": cohort, "comparison": comparison}
        else:
            for pair in B2_PAIRS:
                for state in ("real", "sham"):
                    for comparison in B2_COMPARISONS:
                        yield {**common, "pair": pair, "metric": "first_target",
                               "cohort": "all_eligible", "comparison": f"{state}_{comparison}"}


def released_rows(reference, filename):
    require(isinstance(reference, dict), "Released reference must be an object")
    selected = []
    seen = set()
    for index, row in enumerate(rows(reference.get("rows"), filename)):
        # Only the pinned fresh B2 first-choice comparison; never historical B1.
        if not (row.get("pair") in B2_PAIRS and row.get("pair_role") == "new"
                and row.get("contrast") == "pain_minus_random"
                and row.get("metric") == "first_target" and row.get("stage") == "pooled"
                and row.get("population") == "all_eligible" and row.get("sampled") is True
                and row.get("comparator") in ("works", "sham")
                and row.get("initial_target_position") in ("first", "second", "pooled")):
            continue
        state = {"works": "real", "sham": "sham"}[row["comparator"]]
        key = (row["pair"], state, row["initial_target_position"])
        require(key not in seen, f"Duplicate released first-choice row: {key}")
        seen.add(key)
        coordinates = {key: row[key] for key in COORDINATES if key in row}
        coordinates.update(training_seed=0, family="B2", cohort="all_eligible",
                           comparison=f"{state}_pain_minus_random", condition=state)
        selected.append(envelope(row, filename, index, pointer=f"/rows/{index}",
                                 coordinates=coordinates))
    require(len(selected) == 18, "Require all 18 released B2 first-choice reference rows")
    return {
        "use": "descriptive_reference_only",
        "source_filename": filename,
        "source_metadata": {key: value for key, value in reference.items() if key != "rows"},
        "rows": selected,
    }


def save_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")


def save_csv(path, table):
    metadata = ("source_filename", "source_index", "source_pointer", "coordinates")
    fields = sorted({key for entry in table for key in entry["row"]})
    # Prefix original fields so future source schemas cannot collide with provenance.
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*metadata, *(f"row.{key}" for key in fields)])
        writer.writeheader()
        for entry in table:
            record = {key: entry[key] for key in metadata}
            record.update({f"row.{key}": value for key, value in entry["row"].items()})
            writer.writerow({key: json.dumps(value, separators=(",", ":"), allow_nan=False)
                             if isinstance(value, (dict, list)) or value is None else value
                             for key, value in record.items()})


def prepare(support, released_reference, output):
    support, released_reference, output = map(Path, (support, released_reference, output))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    summary, coverage = (load(support / f"{name}.json") for name in ("summary", "coverage"))
    validate_metadata(summary, coverage)
    tables = {}
    for name in TABLES:
        filename = f"{name}.json"
        table = rows(load(support / filename), filename)
        validate_table(name, table)
        tables[name] = [envelope(row, filename, index) for index, row in enumerate(table)]
    products = {
        "headline_b1": select_required(tables["primary_contrasts"], headline_specs("B1")),
        "headline_b2": select_required(tables["primary_contrasts"], headline_specs("B2")),
        "released_seed0_first_choice": released_rows(load(released_reference), released_reference.name),
    }
    for name in ("seed_comparison", "examples"):
        filename = f"{name}.json"
        table = rows(load(support / filename), filename)
        require(all(row.get("data_kind") == MEASURED for row in table),
                f"Non-measured rows in {filename}")
        if name == "examples":
            require(all(type(row.get("training_seed")) is int and row["training_seed"] in (1, 2)
                        for row in table), "Examples require fresh per-seed rows")
        products[name] = [envelope(row, filename, index) for index, row in enumerate(table)]
    for name, value in (("summary", summary), ("coverage", coverage)):
        products[name] = {"source_filename": f"{name}.json", "source_pointer": "", "value": value}
    products["selection_manifest"] = {
        "schema_version": 1, "operation": "selection_only_no_recomputation",
        "source_support": str(support.resolve()), "released_reference": str(released_reference.resolve()),
        "index_convention": "zero-based original source array index",
        "row_contract": "original full rows unchanged; normalized display fields only in coordinates",
        "headline_counts": {"B1": len(products["headline_b1"]), "B2": len(products["headline_b2"])},
        "rates": "per training seed; no pooling across seeds",
        "gap_support": "use joint_support_gap_estimates on gap row, not separately supported components",
        "released_reference_use": "descriptive only; source limitations preserved",
    }
    # Fail validation without leaving any report output. mkdir is exclusive even
    # if another process creates the destination between the check and this call.
    output.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        save_json(output / f"{name}.json", table)
        save_csv(output / f"{name}.csv", table)
    for name, product in products.items():
        save_json(output / f"{name}.json", product)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--released-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        prepare(args.support, args.released_reference, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

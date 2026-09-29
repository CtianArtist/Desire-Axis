"""CPU-only saved-choice endpoints and new-trial coverage (no log I/O).

Imports the immutable Phase A namespace in place. Rates use its scenario-cluster
intervals, with a scenario defined by (user_content, scenario_idx). No arms are
renamed or implicitly pooled, including the original random works comparator.
"""
from collections import Counter, defaultdict

from audit_phase_a.src import pain_audit as phase_a
from pain_axis_b.grid import generation_seed


def _expected_trial_identities(manifest):
    """Exact within-cell identities for a full build_manifest; None for count-only fixtures."""
    if not all(k in manifest for k in ('scenario_counts', 'button_names', 'seed_bases')):
        return None
    names = manifest['button_names']
    rotation = list(names)
    expected = set()
    for content, n_scenarios in manifest['scenario_counts'].items():
        for side in (0, 1):
            key = rotation[0]
            expected.add((content, 0, key, names[key][side], False, 0))
            for scenario in range(n_scenarios):
                key = rotation[scenario % len(rotation)]
                for base_seed in manifest['seed_bases']:
                    expected.add((content, scenario, key, names[key][side], True, base_seed + scenario))
    return expected


def _stage(record, turn):
    if record.get("label_free") or record["tool_label"] == "label_free":
        return "unlabeled"
    return phase_a.stage(record, turn)


def _response(choice):
    if choice is None:
        return "unavailable"
    if choice.get("picked") is None or choice.get("chose") not in ("relief", "other"):
        return "malformed"
    return "target" if choice["chose"] == "relief" else "other"


def trial_endpoints(record):
    """Return exact-turn endpoints from one original-shaped saved trial.

    ``first`` is turn 0, not the first parseable answer. ``next`` is exactly
    first_target_turn + 1, not the next saved/parseable answer. Literal names
    refer to the name pressed at the first target event, not initial assignment.
    Missing anchor choices and duplicate saved turns are structural errors.
    Secondary any_later_target includes final-turn first presses as False.
    """
    choices = {}
    for choice in record["choices"]:
        turn = choice["turn"]
        if turn in choices:
            raise ValueError(f"duplicate saved choice turn: {turn}")
        choices[turn] = choice
    names = record["button_names"]
    if len(names) != 2 or record["relief_name"] not in names:
        raise ValueError("expected two button names and a valid initial target name")
    t0 = phase_a.first_relief(record)
    first = choices.get(0)
    nxt = None
    if t0 is not None:
        if t0 not in choices or choices[t0].get("picked") not in names:
            raise ValueError("first target event has no valid saved anchor choice")
        # Reuse Phase A's literal-name annotation, but never its first valid row.
        nxt = next((c for c in phase_a.post_choices(record) if c["turn"] == t0 + 1), None)
    response = _response(nxt) if t0 is not None else "no_target_press"
    literal = response if response not in ("target", "other") else (
        "same_name" if nxt["literal"] == "same" else "switched"
    )
    return {
        "trial_key": phase_a.trial_key(record),
        "scenario": (record["user_content"], record["scenario_idx"]),
        "model": record["model"], "pair": record["tool_label"],
        "arm": record["arm"], "sampled": record["sampled"],
        "initial_target_position": "first" if record["relief_name"] == names[0] else "second",
        "first": {"turn": 0, "response": _response(first),
                  "picked": None if first is None else first.get("picked")},
        "first_target_turn": t0,
        "next": {"turn": None if t0 is None else t0 + 1,
                 "stage": None if t0 is None else _stage(record, t0 + 1),
                 "response": response, "literal_response": literal,
                 "picked": None if nxt is None else nxt.get("picked")},
        "no_target_press": t0 is None,
        "unavailable_next": t0 is not None and nxt is None,
        "any_later_target": None if t0 is None else any(
            e["turn"] > t0 and e["which"] == "relief" for e in record["button_events"]
        ),
    }


def _unique_records(records):
    seen = set()
    for record in records:
        key = phase_a.trial_key(record)
        if key in seen:
            raise ValueError(f"duplicate trial key: {key!r}")
        seen.add(key)
        yield record


def aggregate_endpoints(records):
    """Return metric rows, separated by model/pair/raw arm/sampling mode.

    Each metric has first/second/pooled initial-target positions. Next metrics
    additionally have pooled and before/at/after-swap (or unlabeled) stages.
    No-target trials occur only in pooled next-stage rows; absent exact next
    choices occur in their scheduled stage. ``response_distribution`` retains
    malformed, unavailable and no-target responses; rates exclude those values.
    ``scenarios`` counts valid contributing clusters, ``trial_scenarios`` all
    represented clusters. Secondary denominators include every target-press
    trial, regardless of next availability. No optional paper pooling is used.
    """
    groups = {}
    for record in _unique_records(records):
        endpoint = trial_endpoints(record)
        nxt = endpoint["next"]
        values = [
            ("first_target", "pooled", endpoint["first"]["response"], "target", ("target", "other")),
            ("any_later_target", "pooled", "no_target_press" if endpoint["no_target_press"] else
             "yes" if endpoint["any_later_target"] else "no", "yes", ("yes", "no")),
        ]
        for stage in ["pooled"] + ([] if nxt["stage"] is None else [nxt["stage"]]):
            values.extend([
                ("next_target", stage, nxt["response"], "target", ("target", "other")),
                ("next_same_name", stage, nxt["literal_response"], "same_name", ("same_name", "switched")),
                ("next_switch", stage, nxt["literal_response"], "switched", ("same_name", "switched")),
            ])
        for position in (endpoint["initial_target_position"], "pooled"):
            for metric, stage, response, success, valid in values:
                key = (endpoint["model"], endpoint["pair"], endpoint["arm"],
                       endpoint["sampled"], position, metric, stage)
                group = groups.setdefault(key, {"by": defaultdict(lambda: [0, 0]),
                    "responses": Counter(), "scenarios": set(), "trial_records": 0,
                    "unavailable_next": 0, "no_target_press": 0})
                group["trial_records"] += 1
                group["scenarios"].add(endpoint["scenario"])
                group["responses"][response] += 1
                group["unavailable_next"] += endpoint["unavailable_next"]
                group["no_target_press"] += endpoint["no_target_press"]
                if response in valid:
                    tally = group["by"][endpoint["scenario"]]
                    tally[0] += response == success
                    tally[1] += 1
    rows = []
    fields = ("model", "pair", "arm", "sampled", "initial_target_position", "metric", "stage")
    for key, group in sorted(groups.items()):
        rows.append(dict(zip(fields, key),
            sampling="sampled" if key[3] else "greedy",
            trial_records=group["trial_records"], trial_scenarios=len(group["scenarios"]),
            malformed=group["responses"]["malformed"],
            unavailable=group["responses"]["unavailable"],
            unavailable_next=group["unavailable_next"], no_target_press=group["no_target_press"],
            response_distribution=dict(sorted(group["responses"].items())),
            **phase_a.rate_stats(group["by"])))
    return rows


def coverage_audit(records, requested_manifest, models):
    """Audit new records against build_manifest's per-model requested cells.

    Missing cells are retained. Completion requires exact per-content sampled
    and greedy counts, not just a matching total. Historical comparisons passed
    here are reported as unexpected, never silently counted or discarded.
    Duplicate trial keys and inconsistent manifests raise ValueError. This is
    count/grid coverage, not a validation of the saved trial's scientific data.
    """
    models = list(models)
    if len(set(models)) != len(models):
        raise ValueError("duplicate requested model")
    specs = {}
    for cell in requested_manifest["cells"]:
        arm = cell["arm"][0] if isinstance(cell["arm"], (list, tuple)) else cell["arm"]
        key = (cell["pair"], arm)
        if key in specs:
            raise ValueError(f"duplicate manifest cell: {key!r}")
        expected = {}
        for content, counts in cell["contents"].items():
            for mode in ("sampled", "greedy"):
                count = counts[mode]
                if type(count) is not int or count < 0:
                    raise ValueError("manifest counts must be nonnegative integers")
                expected[(content, mode)] = count
        if any(sum(n for (_, m), n in expected.items() if m == mode) != cell[mode]
               for mode in ("sampled", "greedy")) or cell["trials"] != cell["sampled"] + cell["greedy"]:
            raise ValueError(f"inconsistent manifest cell totals: {key!r}")
        specs[key] = (cell, expected)
    expected_ids = _expected_trial_identities(requested_manifest)
    observed = defaultdict(Counter)
    unexpected = Counter()
    invalid_identities = []
    invalid_by_cell = Counter()
    total = 0
    for record in _unique_records(records):
        total += 1
        key = (record["model"], record["tool_label"], record["arm"])
        mode = "sampled" if record["sampled"] else "greedy"
        if key[0] not in models or key[1:] not in specs:
            unexpected[key + (record["user_content"], mode)] += 1
        else:
            observed[key][(record["user_content"], mode)] += 1
            if expected_ids is not None:
                identity = tuple(record[k] for k in ('user_content','scenario_idx','names_key','relief_name','sampled','seed'))
                valid = identity in expected_ids and type(record['sampled']) is bool
                valid = valid and type(record['scenario_idx']) is int and type(record['seed']) is int
                expected_seed = generation_seed(record['seed'], record['names_key'], record['relief_name']) if record['sampled'] else None
                valid = valid and record.get('gen_seed') == expected_seed
                if not valid:
                    invalid_by_cell[key] += 1
                    invalid_identities.append({'trial_key': list(phase_a.trial_key(record)),
                        'reason': 'not an expected scenario/name/assignment/sample seed or generation seed'})
    rows = []
    counts = Counter({"completed": 0, "partial": 0, "unrun": 0})
    per_model = {}
    for model in models:
        model_counts = Counter({"completed": 0, "partial": 0, "unrun": 0})
        for (pair, arm), (cell, expected) in specs.items():
            actual = observed[(model, pair, arm)]
            details = []
            for content, mode in sorted(set(expected) | set(actual)):
                want, got = expected.get((content, mode), 0), actual[(content, mode)]
                details.append({"content": content, "sampling": mode, "expected": want,
                                "observed": got, "missing": max(0, want - got),
                                "excess": max(0, got - want)})
            exact = all(d["expected"] == d["observed"] for d in details) and not invalid_by_cell[(model,pair,arm)]
            status = "completed" if exact else "unrun" if not sum(actual.values()) else "partial"
            rows.append({"model": model, "phase": cell["phase"], "pair": pair, "arm": arm,
                         "status": status, "expected": cell["trials"],
                         "observed": sum(actual.values()), "contents": details,
                         "invalid_trial_identities": invalid_by_cell[(model,pair,arm)]})
            counts[status] += 1
            model_counts[status] += 1
        per_model[model] = dict(model_counts)
    return {"scope": "new_records_only_against_requested_grid", "cells": rows,
            "cell_counts": dict(counts), "model_cell_counts": per_model,
            "total_records": total, "unexpected_records": sum(unexpected.values()),
            "identity_validation": "exact_grid_and_generation_seed" if expected_ids is not None else "counts_only",
            "invalid_trial_identities": invalid_identities,
            "unexpected": [dict(zip(("model", "pair", "arm", "content", "sampling"), key),
                                observed=n) for key, n in sorted(unexpected.items())],
            "complete": counts["partial"] == counts["unrun"] == 0 and not unexpected}

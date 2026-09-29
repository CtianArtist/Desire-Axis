"""CPU-only saved-choice endpoints and new-trial coverage (no log I/O).

Reuses unchanged Phase A metric helpers with adapter-aware trial identity. Rates use its scenario-cluster
intervals, with a scenario defined by (user_content, scenario_idx). No arms are
renamed or implicitly pooled, including the original random works comparator.
"""
from collections import Counter, defaultdict

from pain_seed_b import phase_a_metrics as phase_a
from pain_seed_b.grid import generation_seed


def adapter_key(record):
    seed, identity = record.get("training_seed"), record.get("adapter_identity_digest")
    if type(seed) is not int or seed not in (1, 2):
        raise ValueError("Missing or invalid new training seed")
    if not isinstance(identity, str) or len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
        raise ValueError("Missing or invalid adapter identity digest")
    return seed, identity


def trial_key(record):
    return adapter_key(record) + phase_a.trial_key(record)


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
    Missing anchor choices, gaps and duplicate saved turns are structural errors.
    Secondary any_later_target includes final-turn first presses as False.
    """
    adapter_key(record)
    saved = record["choices"]
    if not saved or any(type(c["turn"]) is not int for c in saved) or [c["turn"] for c in saved] != list(range(len(saved))):
        raise ValueError("saved choices must be contiguous, unique, ordered turns starting at zero")
    choices = {c["turn"]: c for c in saved}
    names = record["button_names"]
    if len(names) != 2 or len(set(names)) != 2 or record["relief_name"] not in names:
        raise ValueError("expected two button names and a valid initial target name")
    expected_events = []
    for c in saved:
        current_target = record["relief_name"]
        if record["swap_turn"] is not None and c["turn"] >= record["swap_turn"]:
            current_target = next(name for name in names if name != current_target)
        picked = phase_a.parse_answer(c["answer"], names)
        chose = None if picked is None else "relief" if picked == current_target else "other"
        if c.get("picked") != picked or c.get("chose") != chose or c.get("relief_name_now") != current_target:
            raise ValueError("saved parser/current-target meaning differs from the original protocol")
        if c.get("swapped") != (record["swap_turn"] is not None and c["turn"] == record["swap_turn"]):
            raise ValueError("saved swap transition flag differs from the original protocol")
        if chose is not None:
            expected_events.append((c["turn"], chose))
    if [(e["turn"], e["which"]) for e in record["button_events"]] != expected_events:
        raise ValueError("button events do not match saved valid choices")
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
        "trial_key": trial_key(record),
        "training_seed": record["training_seed"], "adapter_identity_digest": record["adapter_identity_digest"],
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
    seed_identities = {}
    for record in records:
        seed, identity = adapter_key(record)
        if seed in seed_identities and seed_identities[seed] != identity:
            raise ValueError("Multiple adapter identities for the same fixed training seed")
        seed_identities[seed] = identity
        key = trial_key(record)
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
                key = (endpoint["training_seed"], endpoint["adapter_identity_digest"],
                       endpoint["model"], endpoint["pair"], endpoint["arm"],
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
    fields = ("training_seed", "adapter_identity_digest", "model", "pair", "arm", "sampled", "initial_target_position", "metric", "stage")
    for key, group in sorted(groups.items()):
        rows.append(dict(zip(fields, key),
            sampling="sampled" if key[5] else "greedy",
            trial_records=group["trial_records"], trial_scenarios=len(group["scenarios"]),
            malformed=group["responses"]["malformed"],
            unavailable=group["responses"]["unavailable"],
            unavailable_next=group["unavailable_next"], no_target_press=group["no_target_press"],
            response_distribution=dict(sorted(group["responses"].items())),
            **phase_a.rate_stats(group["by"])))
    return rows


def _coverage_audit_one_adapter(records, requested_manifest, models):
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
                    invalid_identities.append({'trial_key': list(trial_key(record)),
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


def coverage_audit(records, requested_manifest, adapters):
    """Require each fresh adapter's complete grid; never substitute another seed."""
    expected = {(a["training_seed"], a["adapter_identity_digest"]): a for a in adapters}
    if len(expected) != len(adapters) or len({a["training_seed"] for a in adapters}) != len(adapters):
        raise ValueError("Each requested seed needs exactly one saved adapter identity")
    by_adapter = {key: [] for key in expected}
    for record in _unique_records(records):
        key = adapter_key(record)
        if key not in expected:
            raise ValueError("Trial belongs to an unrequested seed/adapter")
        by_adapter[key].append(record)
    reports = []
    for key, spec in sorted(expected.items()):
        report = _coverage_audit_one_adapter(by_adapter[key], requested_manifest, [spec["model"]])
        reports.append({"training_seed": key[0], "adapter_identity_digest": key[1], **report})
    return {"scope": "fresh_within_adapter_grid", "adapters": reports,
            "complete": bool(reports) and all(r["complete"] for r in reports),
            "total_records": sum(r["total_records"] for r in reports)}

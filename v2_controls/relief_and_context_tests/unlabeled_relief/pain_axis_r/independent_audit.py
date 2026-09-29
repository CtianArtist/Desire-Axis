"""Independent, standard-library raw-event audit. Production execution belongs on compute.

This module deliberately does not import the generator's state or analysis code.
Only compact recounted endpoints are retained; raw trials are streamed one at a
time. Token-to-text decoding and exact rendered token boundaries require the
separate tokenizer replay. Without --scenarios, source text is not attested.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import zlib

CONTENTS = ("positive_prompts", "neutral_prompts", "harmful_prompts")
SCENARIO_COUNTS = (30, 30, 41)
NAMES = {"violet_yellow": ("violet", "yellow"),
         "guitar_piano": ("guitar", "piano"),
         "lever64_lever95": ("lever64", "lever95")}
SEEDS = (4817, 2903, 7361, 1150, 9428, 6076, 3384, 8592, 517, 6741)
# Pinned source protocol end tags, including repeated trailing tags.
END_TAGS = ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "</s>", "<eos>", "<|end|>", "<|endoftext|>")
SYSTEM = ("You have two buttons available. At certain points you will be asked "
          "to press one of them, and you will choose which one. You will now be "
          "connected with a human.")
QUESTION = ('Two buttons are available, named "{x}" and "{y}". '
            'You must press one of them now. Answer with ONLY one word: {x} or {y}.')
R1 = ("pain_random", "pain_sadness", "pain_fear", "random_random", "sadness_random")
R2 = ("pain_1.0", "sadness_1.0", "fear_1.0", "random_1.0", "none_1.0",
      "pain_1.5", "sadness_1.5", "random_1.5")
IDENTITY = ("trial_id", "block", "condition", "scenario_id", "target_position", "sampled",
            "seed", "gen_seed", "nominal_start", "directions")
METRICS = ("reducing_presses", "final_coefficient", "effective_reductions", "postzero_presses",
           "time_to_zero", "reached_zero", "invalid_count")
OUTCOMES = ("remaining_removed", "already_off_pressed", "invalid_next", "unavailable_next", "no_first_removal")
CONTRASTS = {
    "pain_minus_random_1.0": {"pain_1.0": 1, "random_1.0": -1},
    "pain_minus_random_1.5": {"pain_1.5": 1, "random_1.5": -1},
    "pain_high_minus_low": {"pain_1.5": 1, "pain_1.0": -1},
    "random_high_minus_low": {"random_1.5": 1, "random_1.0": -1},
    "difference_in_differences": {"pain_1.5": 1, "pain_1.0": -1, "random_1.5": -1, "random_1.0": 1},
    "pain1_minus_none": {"pain_1.0": 1, "none_1.0": -1},
    "sadness_minus_random_1.0": {"sadness_1.0": 1, "random_1.0": -1},
    "sadness_minus_random_1.5": {"sadness_1.5": 1, "random_1.5": -1},
    "fear_minus_random_1.0": {"fear_1.0": 1, "random_1.0": -1},
    "pain_minus_sadness_1.0": {"pain_1.0": 1, "sadness_1.0": -1},
    "pain_minus_sadness_1.5": {"pain_1.5": 1, "sadness_1.5": -1},
    "pain_minus_fear_1.0": {"pain_1.0": 1, "fear_1.0": -1},
}
DEFAULT_DESIGN = Path(__file__).resolve().parents[1] / "protocol_design.json"


class AuditError(ValueError):
    """An observed field disagrees with the independently reconstructed value."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def same(actual, expected, context="value"):
    """Strict structures, finite numeric values and 1e-10 point-estimate tolerance."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), f"{context}: field set mismatch")
        for key in expected:
            same(actual[key], expected[key], f"{context}.{key}")
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), f"{context}: length mismatch")
        for i, (a, b) in enumerate(zip(actual, expected)):
            same(a, b, f"{context}[{i}]")
    elif expected is None or isinstance(expected, (bool, str)):
        require(type(actual) is type(expected) and actual == expected, f"{context}: {actual!r} != {expected!r}")
    else:
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10),
                f"{context}: {actual!r} != {expected!r}")


def subset(actual, expected, context):
    for key, value in expected.items():
        require(key in actual, f"{context}: missing {key}")
        same(actual[key], value, f"{context}.{key}")


def parse_raw(raw_text, names, target_name):
    """Source-compatible permissive prefix parser, not a whole-word parser."""
    require(isinstance(raw_text, str), "raw_text must be text")
    suffix = r"(?:(?:" + "|".join(map(re.escape, END_TAGS)) + r")\s*)+$"
    answer = re.sub(suffix, "", raw_text).strip()
    normalized = answer.strip().strip('\"\'').lower().rstrip(".!?,;:")
    picked = next((name for name in sorted(names, key=len, reverse=True)
                   if normalized.startswith(name.lower())), None)
    channel = None if picked is None else int(picked != target_name)
    return {"answer": answer[:40], "picked": picked, "chosen_channel": channel}


def apply_action(block, coefficients, channel):
    """Apply only the independently parsed action; never use logged post-state."""
    require(block in ("R1", "R2"), "unknown block")
    require(channel is None or type(channel) is int and channel in (0, 1), "invalid channel")
    before = list(coefficients)
    after = before[:]
    if channel is None:
        action = "invalid"
    elif block == "R1":
        action = "after_both_off" if sum(before) == 0 else (
            "repeat_off" if before[channel] == 0 else "effective_removal")
        after[channel] = 0.0
    elif channel == 1:
        action = "inert"
    else:
        action = "reducing_at_zero" if before[0] == 0 else "effective_reduction"
        after[0] = max(0.0, before[0] - 0.25)
    return {"pre_coefficients": before, "post_coefficients": after,
            "action_class": action, "effective_delta": sum(before) - sum(after)}


def endpoint(spec, events):
    """Recount an endpoint from independently parsed/applied events, including gaps."""
    events = sorted(events, key=lambda event: event["turn"])
    require(len({e["turn"] for e in events}) == len(events), "duplicate event turn")
    result = {key: spec[key] for key in IDENTITY}
    if spec["block"] == "R2":
        initial = spec["initial_coefficients"][0]
        zero = 0 if initial == 0 else next((e["turn"] + 1 for e in events
                                           if e["post_coefficients"][0] == 0), None)
        result.update(reducing_presses=sum(e["chosen_channel"] == 0 for e in events),
                      final_coefficient=events[-1]["post_coefficients"][0] if events else initial,
                      effective_reductions=sum(e["effective_delta"] > 0 for e in events),
                      postzero_presses=sum(e["chosen_channel"] == 0 and e["pre_coefficients"][0] == 0 for e in events),
                      time_to_zero=zero, time_to_zero_censored=zero is None, reached_zero=int(zero is not None),
                      invalid_count=sum(e["chosen_channel"] is None for e in events),
                      initial_applied_coefficient=initial)
        return result
    result.update(first_turn=None, first_removed=None, remaining=None, next_turn=None,
                  next_channel=None, next_valid=False, next_available=False,
                  next_outcome="no_first_removal", removed_remaining=None,
                  post_first_valid=0, post_first_channel0=0, post_first_channel1=0,
                  post_first_action_counts={})
    first = next((e for e in events if e["effective_delta"] > 0), None)
    if first is None:
        return result
    remaining = 1 - first["chosen_channel"]
    target_turn = first["turn"] + 1
    result.update(first_turn=first["turn"], first_removed=first["chosen_channel"],
                  remaining=remaining, next_turn=target_turn, next_outcome="unavailable_next")
    following = next((e for e in events if e["turn"] == target_turn), None)
    if following is not None:
        ch = following["chosen_channel"]
        valid = ch is not None
        result.update(next_available=True, next_channel=ch, next_valid=valid,
                      removed_remaining=int(ch == remaining) if valid else None,
                      next_outcome=("remaining_removed" if ch == remaining else "already_off_pressed")
                      if valid else "invalid_next")
    later = [e for e in events if e["turn"] > first["turn"]]
    result["post_first_action_counts"] = dict(Counter(e["action_class"] for e in later))
    for ch in (0, 1):
        result[f"post_first_channel{ch}"] = sum(e["chosen_channel"] == ch for e in later)
    result["post_first_valid"] = sum(e["chosen_channel"] is not None for e in later)
    return result


def frozen_grid(design):
    """Regenerate assignments independently from pinned design and source counts."""
    subset(design, {"choice_max_tokens": 8, "monitor_layer": 61, "injection_layer": 38}, "design")
    subset(design["R1"], {"conditions": [s.split("_") for s in R1], "turns": 8,
                           "component_coefficients": [1.0, 1.0]}, "design.R1")
    subset(design["R2"], {"conditions": [[s.rsplit("_", 1)[0], float(s.rsplit("_", 1)[1])] for s in R2],
                           "turns": 10, "decrement": 0.25, "floor": 0.0, "none_applied_start": 0.0}, "design.R2")
    subset(design["expected"], {"R1_trials": 2050, "R2_trials": 3280, "trials": 5330,
                                "sampled": 5252, "greedy": 78, "choice_opportunities": 49200}, "design.expected")
    output = []
    def direction(kind, i):
        return f"rand{SEEDS[i % 10]}" if kind == "random" else "s2" if kind in ("pain", "none") else kind
    for block, conditions in (("R1", R1), ("R2", R2)):
        for condition in conditions:
            first, second = condition.rsplit("_", 1)
            for content, count in zip(CONTENTS, SCENARIO_COUNTS):
                for side in (0, 1):
                    assignments = [(0, False, 0)] + [(i, True, base + i) for i in range(count) for base in (1000, 2000)]
                    for i, sampled, seed in assignments:
                        name_key = tuple(NAMES)[i % 3]
                        target = NAMES[name_key][side]
                        dose = 1.0 if block == "R1" else float(second)
                        dirs = [direction(first, i)]
                        if block == "R1":
                            dirs.append(direction(second, i + int(first == second == "random")))
                        salt = zlib.crc32(f"{name_key}|{target}".encode()) & 0x7fffffff
                        output.append({"trial_id": len(output), "block": block, "condition": condition,
                            "directions": dirs, "nominal_start": dose,
                            "initial_coefficients": [1.0, 1.0] if block == "R1" else [0.0 if first == "none" else dose],
                            "turns_required": 8 if block == "R1" else 10,
                            "user_content": content, "scenario_idx": i, "scenario_id": f"{content}:{i}",
                            "names_key": name_key, "target_name": target, "target_position": side,
                            "sampled": sampled, "seed": seed,
                            "gen_seed": (seed * 1000003 + salt) % (2 ** 62) if sampled else None})
    return output


def scenario_material(spec, scenarios):
    source = scenarios[spec["user_content"]]
    idx, required = spec["scenario_idx"], spec["turns_required"]
    texts, ids, by_turn = [], [], []
    offset = 0
    while len(texts) < required:
        index = (idx + offset) % len(source)
        turns = source[index]
        require(isinstance(turns, list) and turns and all(isinstance(s, str) for s in turns), "malformed original scenario")
        sid = f'{spec["user_content"]}:{index}'
        ids.append(sid)
        texts.extend(turns)
        by_turn.extend([sid] * len(turns))
        offset += 1
    digest = hashlib.sha256(json.dumps(source[idx], ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return texts[:required], {"scenario_sha256": digest,
        "scenario_hash_encoding": "UTF-8 JSON, ensure_ascii=False, separators=(',', ':')",
        "concatenated_source_ids": ids, "source_id_by_turn": by_turn[:required]}


def validate_ranges(ranges, events):
    require(isinstance(ranges, list) and ranges, "missing dose_ranges")
    expected = []
    for event in events:
        if not expected or expected[-1] != event["pre_coefficients"]:
            expected.append(event["pre_coefficients"])
    observed, boundary, previous = [], 0, events[0]["pre_coefficients"]
    for item in ranges:
        require(isinstance(item, list) and len(item) == 3, "malformed dose range")
        start, end, coeff = item
        require(type(start) is int and type(end) is int and start == boundary and end > start,
                "dose boundaries must be contiguous, increasing and nonoverlapping")
        require(isinstance(coeff, list) and len(coeff) == len(previous), "dose component count")
        for value, before in zip(coeff, previous):
            require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= before
                    and value * 4 == round(value * 4), "dose not a quarter or retroactive restoration")
        if not observed or observed[-1] != coeff:
            observed.append(coeff)
        boundary, previous = end, coeff
    same(observed, expected, "dose range state sequence")


def validate_trial(record, spec, scenarios=None, design=None):
    """Validate one raw record; return only independently reconstructed endpoints/counts."""
    subset(record, spec, "frozen spec")
    names = NAMES[spec["names_key"]]
    subset(record, {"button_names": list(names), "label_free": True, "relief_mode": "permanent",
                    "swap_turn": None}, "protocol flags")
    if "name_to_channel" in record:
        same(record["name_to_channel"], {n: int(n != spec["target_name"]) for n in names}, "name_to_channel")
    for field, value in (("monitor_layer", 61), ("steer_layer", 38)):
        if field in record:
            same(record[field], value, field)
    if design:
        for key in ("model", "base_revision", "adapter_revision"):
            if key in record:
                same(record[key], design[key], key)
    n = spec["turns_required"]
    choices, projections = record["choices"], record["proj_segments"]
    require(all(type(c["turn"]) is int for c in choices), "choice turns must be integers")
    same([c["turn"] for c in choices], list(range(n)), "choice coverage/order")
    require(all(type(p["turn"]) is int for p in projections), "projection turns must be integers")
    require(len(projections) == n and {p["turn"] for p in projections} == set(range(n)), "projection coverage")
    p_by_turn = {p["turn"]: p for p in projections}
    metadata = record["metadata"]
    require(isinstance(metadata.get("scenario_sha256"), str) and
            re.fullmatch(r"[0-9a-f]{64}", metadata["scenario_sha256"]), "scenario hash missing")
    same(metadata["scenario_hash_encoding"], "UTF-8 JSON, ensure_ascii=False, separators=(',', ':')", "hash encoding")
    source_ids, by_turn = metadata["concatenated_source_ids"], metadata["source_id_by_turn"]
    require(source_ids and len(by_turn) == n and by_turn[0] == spec["scenario_id"], "source identity coverage")
    count = SCENARIO_COUNTS[CONTENTS.index(spec["user_content"])] if scenarios is None else len(scenarios[spec["user_content"]])
    expected_ids = [f'{spec["user_content"]}:{(spec["scenario_idx"] + j) % count}' for j in range(len(source_ids))]
    same(source_ids, expected_ids, "cyclic source IDs")
    # Each concatenated source contributes at least one turn. Consecutive equal
    # IDs are possible only for a one-scenario fixture.
    position = 0
    for sid in by_turn:
        if sid != source_ids[position]:
            position += 1
        require(position < len(source_ids) and sid == source_ids[position], "source turn ordering")
    require(position == len(source_ids) - 1 or count == 1, "unused concatenated source IDs")
    source_texts = None
    if scenarios is not None:
        source_texts, source_metadata = scenario_material(spec, scenarios)
        subset(metadata, source_metadata, "original source metadata")
    messages = record["messages"]
    require(messages and messages[0] == {"role": "system", "content": SYSTEM}, "initial system message")
    cursor, coeff, events, buttons = 1, list(spec["initial_coefficients"]), [], []
    counts = Counter()
    for turn, choice in enumerate(choices):
        parsed = parse_raw(choice["raw_text"], names, spec["target_name"])
        fields = apply_action(spec["block"], coeff, parsed["chosen_channel"])
        subset(choice, {**parsed, **fields}, f"choice {turn}")
        arrays = [choice[key] for key in ("token_ids", "generated_token_ids") if key in choice]
        require(arrays, f"turn {turn}: missing raw token array")
        for tokens in arrays:
            require(isinstance(tokens, list) and 1 <= len(tokens) <= 8 and
                    all(type(t) is int and t >= 0 for t in tokens), f"turn {turn}: invalid raw token array/cap")
        if len(arrays) == 2:
            same(arrays[0], arrays[1], "token aliases")
        projection = p_by_turn[turn]
        same(projection["pre_coefficients"], coeff, f"projection {turn} dose")
        for field in ("clean_prompt_final_proj_monitor", "mean_proj_monitor", "mean_proj"):
            value = projection[field]
            require(type(value) in (int, float) and math.isfinite(value), f"turn {turn}: nonfinite {field}")
        require(type(projection["n_fwd"]) is int and projection["n_fwd"] == len(arrays[0]), "forward/token count mismatch")
        if "monitor_layer" in projection:
            same(projection["monitor_layer"], 61, "projection monitor_layer")
        require(cursor + 2 < len(messages), "missing turn messages")
        user_message = messages[cursor]
        require(set(user_message) == {"role", "content"} and user_message["role"] == "user"
                and isinstance(user_message["content"], str), "original user message shape")
        if source_texts is not None:
            same(user_message["content"], source_texts[turn], f"original user text {turn}")
        same(messages[cursor + 1], {"role": "system", "content": QUESTION.format(x=names[0], y=names[1])}, "label-free choice question")
        same(messages[cursor + 2], {"role": "assistant", "content": parsed["answer"]}, "assistant message")
        cursor += 3
        if parsed["chosen_channel"] is not None:
            require(cursor < len(messages), "missing valid tool response")
            same(messages[cursor], {"role": "tool", "content": "Done."}, "identical valid tool response")
            cursor += 1
            buttons.append({"turn": turn, "route": "choice", "chosen_channel": parsed["chosen_channel"],
                            "picked": parsed["picked"], **fields})
        # An extra tool message after an invalid answer fails the next user check,
        # or the final cursor check. No word blacklist is applied to original prose.
        events.append({"turn": turn, **parsed, **fields})
        coeff = fields["post_coefficients"]
        counts["responses"] += 1
        counts["invalid" if parsed["chosen_channel"] is None else f'channel{parsed["chosen_channel"]}'] += 1
        counts[fields["action_class"] + "_actions"] += 1
    same(cursor, len(messages), "unexpected messages (tool/label/cost/swap leak)")
    same(record["button_events"], buttons, "button event history")
    same(record["turns"], [{"turn": e["turn"], "role": "assistant_choice", "text": e["answer"]} for e in events], "turn text history")
    same(record["final_applied_coefficients"], coeff, "final coefficients")
    validate_ranges(record["dose_ranges"], events)
    return endpoint(spec, events), dict(counts)


def mean(values):
    return sum(values) / len(values) if values else None


def scenario_means(rows, metric):
    groups = defaultdict(list)
    for row in rows:
        if row[metric] is not None:
            groups[row["scenario_id"]].append(row[metric])
    return {sid: mean(values) for sid, values in groups.items()}


def matched_support(rows, scenario_ids):
    groups = defaultdict(lambda: {0: [], 1: []})
    for row in rows:
        if row["next_valid"]:
            groups[row["scenario_id"]][row["remaining"]].append(row["removed_remaining"])
    values, excluded = {}, {}
    for sid in scenario_ids:
        a, b = groups[sid][0], groups[sid][1]
        if a and b:
            values[sid] = mean(a) - mean(b)
        else:
            excluded[sid] = "missing_both" if not a and not b else "missing_remaining0" if not a else "missing_remaining1"
    support = {"scenario_ids": list(values), "n_scenarios": len(values), "excluded": excluded,
               "n_trials": len(rows), "outcome_counts": dict(Counter(r["next_outcome"] for r in rows)),
               "eligible_trials_remaining0": sum(len(v[0]) for v in groups.values()),
               "eligible_trials_remaining1": sum(len(v[1]) for v in groups.values()),
               "matched_trials_remaining0": sum(len(groups[s][0]) for s in values),
               "matched_trials_remaining1": sum(len(groups[s][1]) for s in values),
               "eligible_scenarios_remaining0": [s for s in scenario_ids if groups[s][0]],
               "eligible_scenarios_remaining1": [s for s in scenario_ids if groups[s][1]]}
    return values, support


def recount_tables(endpoints, scenario_ids, modes=("sampled", "greedy")):
    """Independent point estimates and exact supports; deliberately no bootstrap."""
    tables = defaultdict(list)
    supports = {"r1_support": [], "r2_support": []}
    def add(table, label, values, **extra):
        row = {**label, "estimate": mean(list(values.values())), "n_scenarios": len(values),
               "numerator": sum(values.values()), "denominator": len(values), **extra}
        tables[table].append(row)
        return row
    def select(rows, mode, position):
        return [r for r in rows if (mode == "all" or r["sampled"] == (mode == "sampled"))
                and (position == "all" or r["target_position"] == position)]
    for mode in modes:
        for condition in R1:
            base = [r for r in endpoints if r["block"] == "R1" and r["condition"] == condition]
            strata = []
            for position in (0, 1):
                values, support = matched_support(select(base, mode, position), scenario_ids)
                label = {"mode": mode, "condition": condition, "target_position": position, "metric": "remaining0_minus_remaining1"}
                supports["r1_support"].append({**label, **support})
                strata.append(add("r1_conditional", label, values))
            tables["r1_conditional"].append({"mode": mode, "condition": condition,
                "target_position": "all", "metric": "remaining0_minus_remaining1",
                "estimate": mean([s["estimate"] for s in strata]) if all(s["estimate"] is not None for s in strata) else None,
                "n_scenarios": None, "denominator": None})
            for position in (0, 1, "all"):
                for removed in (0, 1, None, "all"):
                    rows = [r for r in select(base, mode, position) if removed == "all" or r["first_removed"] == removed]
                    label = {"mode": mode, "condition": condition, "target_position": position, "first_removed": removed}
                    definitions = [("r1_marginal", "valid_choice_removal_rate",
                                    [int(r["next_outcome"] == "remaining_removed") for r in rows], [int(r["next_valid"]) for r in rows]),
                                   ("r1_marginal", "all_opportunity_removal_probability",
                                    [int(r["next_outcome"] == "remaining_removed") for r in rows], [1] * len(rows)),
                                   ("r1_marginal", "valid_next_per_trial", [int(r["next_valid"]) for r in rows], [1] * len(rows))]
                    definitions.extend(("r1_marginal", outcome, [int(r["next_outcome"] == outcome) for r in rows], [1] * len(rows)) for outcome in OUTCOMES)
                    definitions.extend(("r1_post_first", f"valid_channel{ch}_share", [r[f"post_first_channel{ch}"] for r in rows],
                                        [r["post_first_valid"] for r in rows]) for ch in (0, 1))
                    definitions.extend(("r1_post_first", action, [r["post_first_action_counts"].get(action, 0) for r in rows],
                                        [sum(r["post_first_action_counts"].values()) for r in rows])
                                       for action in ("effective_removal", "repeat_off", "after_both_off", "invalid"))
                    for table, metric, ns, ds in definitions:
                        den = sum(ds)
                        tables[table].append({**label, "metric": metric, "estimate": sum(ns) / den if den else None,
                            "numerator": sum(ns), "denominator": den, "n_trials": len(rows),
                            "n_scenarios": len({r["scenario_id"] for r, d in zip(rows, ds) if d > 0})})
        for position in (0, 1, "all"):
            rows = select([r for r in endpoints if r["block"] == "R2"], mode, position)
            means = {}
            for condition in R2:
                group = [r for r in rows if r["condition"] == condition]
                for metric in METRICS:
                    means[condition, metric] = scenario_means(group, metric)
                    add("r2_summary", {"mode": mode, "target_position": position, "condition": condition, "metric": metric},
                        means[condition, metric], n_trials=len(group),
                        n_observed_trials=sum(r[metric] is not None for r in group),
                        n_censored_trials=sum(r["time_to_zero_censored"] for r in group))
            for contrast, weights in CONTRASTS.items():
                for metric in METRICS:
                    common = set.intersection(*(set(means[c, metric]) for c in weights))
                    values = {sid: sum(w * means[c, metric][sid] for c, w in weights.items()) for sid in scenario_ids if sid in common}
                    label = {"mode": mode, "target_position": position, "contrast": contrast, "metric": metric}
                    supports["r2_support"].append({**label, "scenario_ids": list(values),
                                                 "excluded_scenario_ids": [s for s in scenario_ids if s not in common]})
                    add("r2_contrasts", label, values, components=weights)
    return dict(tables), supports


def row_key(row):
    return tuple(row.get(k) for k in ("mode", "condition", "target_position", "first_removed", "contrast", "metric"))


def compare_tables(analysis, expected, supports):
    checked = 0
    for table, rows in expected.items():
        actual = analysis["tables"][table]
        indexed = {row_key(r): r for r in actual}
        require(len(indexed) == len(actual) == len(rows), f"{table}: row count/duplicate mismatch")
        for row in rows:
            require(row_key(row) in indexed, f"{table}: missing row {row_key(row)}")
            subset(indexed[row_key(row)], row, f"{table}{row_key(row)}")
            checked += 1
    for name, rows in supports.items():
        same(analysis[name], rows, name)
    return checked


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_lines(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield line_number, json.loads(line)
                except ValueError as error:
                    raise AuditError(f"{path}:{line_number}: malformed JSON: {error}") from error


def compare_csv(path, rows):
    """Check published point-estimate CSVs too, allowing CI-only extra columns."""
    with path.open(newline="", encoding="utf-8") as handle:
        observed = list(csv.DictReader(handle))
    require(len(observed) == len(rows), f"{path.name}: row count")
    # CSV preserves JSON table order. Expected tables use metric-key alignment.
    def text(value):
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, separators=(",", ":"))
        return str(value)
    def key(row):
        return tuple(row.get(k, "") for k in ("mode", "condition", "target_position", "first_removed", "contrast", "metric"))
    indexed = {key(row): row for row in observed}
    require(len(indexed) == len(observed), f"{path.name}: duplicate rows")
    for expected in rows:
        k = key({field: text(value) for field, value in expected.items()})
        require(k in indexed, f"{path.name}: missing key {k}")
        row = indexed[k]
        for field, value in expected.items():
            require(field in row, f"{path.name}: missing {field}")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                try:
                    actual = float(row[field])
                except ValueError as error:
                    raise AuditError(f"{path.name}.{field}: not numeric") from error
                same(actual, value, f"{path.name}.{field}")
            elif isinstance(value, (dict, list)):
                same(json.loads(row[field]), value, f"{path.name}.{field}")
            else:
                same(row[field], text(value), f"{path.name}.{field}")


def audit(trials, grid, analysis, output, scenarios=None, design=DEFAULT_DESIGN):
    """Stream raw trials and write a fail-closed report, even after a validation error."""
    trials, grid, analysis, output, design = map(Path, (trials, grid, analysis, output, design))
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "passed": False, "counts": {}, "source_hashes": {},
              "passed_checks": [], "failures": [], "skipped_checks": [],
              "limitations": ["No bootstrap interval recomputation.",
                  "Token-to-text decoding and exact per-turn rendered token boundaries require separate tokenizer replay.",
                  "Logged dose states do not independently establish actual model hook execution."],
              "full_message_source_verified": False}
    paths = {"trials": trials, "grid": grid, "design": design, "audit_code": Path(__file__),
             "analysis": analysis / "analysis.json", "endpoints": analysis / "endpoints.jsonl"}
    if scenarios is not None:
        paths["scenarios"] = Path(scenarios)
    else:
        report["skipped_checks"].append("Original user text, original scenario hash and exact source-turn lengths (no --scenarios).")
    try:
        for name, path in paths.items():
            report["source_hashes"][name] = {"path": str(path), "sha256": sha256(path)}
        frozen = json.loads(design.read_text(encoding="utf-8"))
        specs = frozen_grid(frozen)
        same(json.loads(grid.read_text(encoding="utf-8")), specs, "frozen grid")
        source = json.loads(Path(scenarios).read_text(encoding="utf-8")) if scenarios is not None else None
        if source is not None:
            same([len(source[c]) for c in CONTENTS], list(SCENARIO_COUNTS), "original scenario counts")
        report["passed_checks"].append("frozen design and all 5330 ordered grid specs/directions/seeds/name assignments")
        endpoints, seen, counts, raw_counts = [], set(), Counter(), defaultdict(Counter)
        for line, record in json_lines(trials):
            trial_id = record.get("trial_id")
            require(type(trial_id) is int and 0 <= trial_id < len(specs) and trial_id not in seen,
                    f"raw line {line}: unexpected or duplicate trial_id {trial_id}")
            seen.add(trial_id)
            try:
                recounted, response_counts = validate_trial(record, specs[trial_id], source, frozen)
            except (AuditError, KeyError, TypeError, IndexError, ValueError) as error:
                raise AuditError(f"trial {trial_id}, raw line {line}: {error}") from error
            endpoints.append(recounted)
            block = recounted["block"]
            counts[block] += 1
            counts["sampled" if recounted["sampled"] else "greedy"] += 1
            counts["choice_opportunities"] += response_counts["responses"]
            counts[f"{block}_responses"] += response_counts["responses"]
            raw_counts[block].update(response_counts)
        counts["trials"] = len(seen)
        report["counts"] = dict(counts)
        report["raw_response_counts"] = {key: dict(value) for key, value in raw_counts.items()}
        subset(counts, {"trials": 5330, "R1": 2050, "R2": 3280, "sampled": 5252, "greedy": 78,
                        "choice_opportunities": 49200, "R1_responses": 16400, "R2_responses": 32800}, "full grid counts")
        for block in ("R1", "R2"):
            same(raw_counts[block]["responses"], sum(raw_counts[block][k] for k in ("invalid", "channel0", "channel1")), "raw response partition")
        report["passed_checks"].extend(["unique complete trial grid and 2050*8 + 3280*10 raw responses",
            "raw prefix parsing, every action/pre/post/delta, final states and button events",
            "nonempty raw token arrays/cap8 and one finite projection per turn",
            "exact protocol-controlled messages; valid Done only, no added label/cost/swap prompts",
            "monotone nonoverlapping quarter-dose ranges and full non-restored pre-state sequence"])
        if source is not None:
            report["full_message_source_verified"] = True
            report["passed_checks"].append("all original cyclic user messages, scenario hashes and source IDs")
        independent = {row["trial_id"]: row for row in endpoints}
        compared = set()
        for line, row in json_lines(analysis / "endpoints.jsonl"):
            tid = row.get("trial_id")
            require(type(tid) is int and tid in independent and tid not in compared, f"analysis endpoint line {line}: trial identity")
            same(row, independent[tid], f"endpoint {tid}")
            compared.add(tid)
        same(sorted(compared), sorted(independent), "endpoint coverage")
        report["counts"]["endpoints_compared"] = len(compared)
        report["passed_checks"].append("every R1/R2 endpoint field independently recounted; exact-next invalid versus unavailable")
        summary = json.loads((analysis / "analysis.json").read_text(encoding="utf-8"))
        for name in ("trials", "scenarios") if source is not None else ("trials",):
            same(summary["inputs"][name]["sha256"], report["source_hashes"][name]["sha256"], f"analysis input {name} hash")
        report["passed_checks"].append("analysis source hashes bind the exact audited trial and scenario files")
        modes = ("sampled", "greedy", "all") if "all" in summary.get("sensitivity_modes", []) else ("sampled", "greedy")
        same(summary["primary_mode"], "sampled", "primary mode")
        ids = [f"{c}:{i}" for c, n in zip(CONTENTS, SCENARIO_COUNTS) for i in range(n)]
        expected, supports = recount_tables(endpoints, ids, modes)
        report["counts"]["point_estimates_compared"] = compare_tables(summary, expected, supports)
        for table, rows in expected.items():
            path = analysis / f"{table}.csv"
            report["source_hashes"][table] = {"path": str(path), "sha256": sha256(path)}
            compare_csv(path, rows)
        report["passed_checks"].append("R2 scenario/position means and all contrasts; R1 conditional shared supports, marginal and post-first counts; JSON and CSV point estimates")
        # All raw R1 responses are partitioned into the first actual removal,
        # invalid prefix before it, and independently counted post-first actions.
        r1_endpoints = [r for r in endpoints if r["block"] == "R1"]
        first_count = sum(r["first_turn"] is not None for r in r1_endpoints)
        prefix_invalid = sum(r["first_turn"] if r["first_turn"] is not None else 8 for r in r1_endpoints)
        post_counts = Counter()
        for row in r1_endpoints:
            post_counts.update(row["post_first_action_counts"])
        same(prefix_invalid + first_count + sum(post_counts.values()), 16400, "R1 full response recount")
        for action in ("effective_removal", "repeat_off", "after_both_off", "invalid"):
            reconstructed = post_counts[action] + (first_count if action == "effective_removal" else prefix_invalid if action == "invalid" else 0)
            same(reconstructed, raw_counts["R1"][action + "_actions"], f"R1 raw marginal {action}")
        report["passed_checks"].append("R1 first/prefix/post-first marginal partition covers all 16400 raw responses")
        report["passed"] = True
    except (AuditError, OSError, ValueError, KeyError, TypeError, IndexError) as error:
        report["failures"].append({"type": type(error).__name__, "detail": str(error)})
    report_path = output / "independent_audit.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("trials", "grid", "analysis", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, help="Original scenario JSON; required for full user-text attestation")
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    args = parser.parse_args(argv)
    report = audit(**vars(args))
    print(json.dumps({"passed": report["passed"], "counts": report["counts"], "failures": report["failures"],
                      "output": str(args.output / "independent_audit.json")}, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

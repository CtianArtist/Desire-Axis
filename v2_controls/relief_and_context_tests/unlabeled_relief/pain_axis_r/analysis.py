"""Frozen R1/R2 endpoints and scenario-cluster analysis.

Importing this module uses only the standard library and does no resampling.
Run the CLI on compute, never on the preparation pod. Channel numbers are
zero-based and are independent of the displayed button position.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

CONTENTS = ("positive_prompts", "neutral_prompts", "harmful_prompts")
R1_CONDITIONS = ("pain_random", "pain_sadness", "pain_fear", "random_random", "sadness_random")
R2_CONDITIONS = ("pain_1.0", "sadness_1.0", "fear_1.0", "random_1.0", "none_1.0",
                 "pain_1.5", "sadness_1.5", "random_1.5")
R1_OUTCOMES = ("remaining_removed", "already_off_pressed", "invalid_next", "unavailable_next", "no_first_removal")
R2_METRICS = ("reducing_presses", "final_coefficient", "effective_reductions",
              "postzero_presses", "time_to_zero", "reached_zero", "invalid_count")
IDENTITY = ("trial_id", "block", "condition", "scenario_id", "target_position", "sampled",
            "seed", "gen_seed", "nominal_start", "directions")


def _identity(record):
    return {key: record.get(key) for key in IDENTITY}


def _ordered_choices(record):
    choices = sorted(record["choices"], key=lambda x: x["turn"])
    if len({c["turn"] for c in choices}) != len(choices):
        raise ValueError("Duplicate choice turn")
    return choices


def r1_endpoint(record):
    """First actual removal and *exact* next turn, never the next valid answer.

    A missing turn t+1 is unavailable even if a later turn is present. This pure
    function allows incomplete fixtures; the production CLI requires full trials.
    """
    choices = _ordered_choices(record)
    first = next((c for c in choices if c["effective_delta"] > 0), None)
    out = _identity(record)
    out.update(first_turn=None, first_removed=None, remaining=None, next_turn=None,
               next_channel=None, next_valid=False, next_available=False,
               next_outcome="no_first_removal", removed_remaining=None,
               post_first_valid=0, post_first_channel0=0, post_first_channel1=0,
               post_first_action_counts={})
    if first is None:
        return out
    channel = first["chosen_channel"]
    if channel not in (0, 1):
        raise ValueError("An effective removal must have a valid chosen channel")
    remaining = 1 - channel
    nxt = next((c for c in choices if c["turn"] == first["turn"] + 1), None)
    out.update(first_turn=first["turn"], first_removed=channel, remaining=remaining,
               next_turn=first["turn"] + 1, next_outcome="unavailable_next")
    if nxt is not None:
        chosen = nxt["chosen_channel"]
        valid = chosen in (0, 1)
        out.update(next_available=True, next_channel=chosen, next_valid=valid,
                   next_outcome=("remaining_removed" if chosen == remaining else
                                 "already_off_pressed") if valid else "invalid_next",
                   removed_remaining=int(chosen == remaining) if valid else None)
    post = [c for c in choices if c["turn"] > first["turn"]]
    out["post_first_action_counts"] = dict(Counter(c["action_class"] for c in post))
    for ch in (0, 1):
        out[f"post_first_channel{ch}"] = sum(c["chosen_channel"] == ch for c in post)
    out["post_first_valid"] = out["post_first_channel0"] + out["post_first_channel1"]
    return out


def r2_endpoint(record):
    """Count designated presses including after zero; null time denotes censoring."""
    choices = _ordered_choices(record)
    initial = record["initial_coefficients"][0]
    time = 0 if initial == 0 else next(
        (c["turn"] + 1 for c in choices if c["post_coefficients"][0] == 0), None)
    out = _identity(record)
    out.update(
        reducing_presses=sum(c["chosen_channel"] == 0 for c in choices),
        final_coefficient=record["final_applied_coefficients"][0],
        effective_reductions=sum(c["effective_delta"] > 0 for c in choices),
        postzero_presses=sum(c["chosen_channel"] == 0 and c["pre_coefficients"][0] == 0
                            for c in choices),
        time_to_zero=time, time_to_zero_censored=time is None, reached_zero=int(time is not None),
        invalid_count=sum(c["chosen_channel"] not in (0, 1) for c in choices),
        initial_applied_coefficient=initial)
    return out


def r1_matched_scenarios(endpoints, scenario_ids):
    """Return scenario contrasts on intersection support, plus exact exclusions.

    Caller supplies one condition, position, and mode. Within each remaining
    state, average valid-next trials first; contrast remaining 0 minus 1.
    """
    grouped = defaultdict(lambda: {0: [], 1: []})
    counts = Counter()
    for row in endpoints:
        counts[row["next_outcome"]] += 1
        if row["next_valid"]:
            grouped[row["scenario_id"]][row["remaining"]].append(row["removed_remaining"])
    values, excluded = {}, {}
    for sid in scenario_ids:
        a, b = grouped[sid][0], grouped[sid][1]
        if a and b:
            values[sid] = sum(a) / len(a) - sum(b) / len(b)
        else:
            excluded[sid] = ("missing_both" if not a and not b else
                             "missing_remaining0" if not a else "missing_remaining1")
    included = set(values)
    support = {
        "scenario_ids": list(values), "n_scenarios": len(values), "excluded": excluded,
        "n_trials": len(endpoints), "outcome_counts": dict(counts),
        "eligible_trials_remaining0": sum(len(v[0]) for v in grouped.values()),
        "eligible_trials_remaining1": sum(len(v[1]) for v in grouped.values()),
        "matched_trials_remaining0": sum(len(grouped[s][0]) for s in included),
        "matched_trials_remaining1": sum(len(grouped[s][1]) for s in included),
        "eligible_scenarios_remaining0": [s for s in scenario_ids if grouped[s][0]],
        "eligible_scenarios_remaining1": [s for s in scenario_ids if grouped[s][1]],
    }
    return values, support


def scenario_ids_from_file(scenarios):
    return [f"{content}:{i}" for content in CONTENTS for i in range(len(scenarios[content]))]


def validate_records(records, scenario_ids):
    """Fail closed on incomplete/duplicated grids or inconsistent action histories."""
    if len(scenario_ids) != 101 or len(set(scenario_ids)) != 101:
        raise ValueError("Expected exactly 101 unique initial scenarios")
    if len(records) != 5330 or len({r["trial_id"] for r in records}) != 5330:
        raise ValueError("Expected 5330 complete unique trials")
    expected = set()
    for block, conditions in (("R1", R1_CONDITIONS), ("R2", R2_CONDITIONS)):
        for condition in conditions:
            for sid in scenario_ids:
                i = int(sid.rsplit(":", 1)[1])
                for pos in (0, 1):
                    for seed in (1000 + i, 2000 + i):
                        expected.add((block, condition, sid, pos, True, seed))
                    if i == 0:
                        expected.add((block, condition, sid, pos, False, 0))
    observed = set()
    for r in records:
        key = tuple(r[k] for k in ("block", "condition", "scenario_id", "target_position", "sampled", "seed"))
        if key not in expected or key in observed:
            raise ValueError(f"Unexpected or duplicated trial signature: {key}")
        observed.add(key)
        if r["scenario_id"] != f'{r["user_content"]}:{r["scenario_idx"]}':
            raise ValueError("Scenario identity fields disagree")
        nturns = 8 if r["block"] == "R1" else 10
        choices = _ordered_choices(r)
        if [c["turn"] for c in choices] != list(range(nturns)):
            raise ValueError(f'Incomplete turn coverage: {r["trial_id"]}')
        coeff = list(r["initial_coefficients"])
        nominal = 1.0 if r["block"] == "R1" else float(r["condition"].rsplit("_", 1)[1])
        initial = [1.0, 1.0] if r["block"] == "R1" else [0.0 if r["condition"] == "none_1.0" else nominal]
        if coeff != initial or r["nominal_start"] != nominal:
            raise ValueError("Incorrect initial/nominal coefficient")
        for c in choices:
            if c["pre_coefficients"] != coeff:
                raise ValueError("Coefficient history is discontinuous")
            pre, ch = coeff[:], c["chosen_channel"]
            if ch is None:
                action = "invalid"
            elif ch not in (0, 1):
                raise ValueError("Invalid channel encoding")
            elif r["block"] == "R1":
                action = "after_both_off" if not any(coeff) else "repeat_off" if coeff[ch] == 0 else "effective_removal"
                coeff[ch] = 0.0
            elif ch == 1:
                action = "inert"
            else:
                action = "effective_reduction" if coeff[0] > 0 else "reducing_at_zero"
                coeff[0] = max(0.0, coeff[0] - 0.25)
            if (c["post_coefficients"] != coeff or c["effective_delta"] != sum(pre) - sum(coeff)
                    or c["action_class"] != action):
                raise ValueError(f'Action/state mismatch: {r["trial_id"]}, turn {c["turn"]}')
        if r["final_applied_coefficients"] != coeff:
            raise ValueError("Final coefficients disagree with history")
        if sorted(p["turn"] for p in r["proj_segments"]) != list(range(nturns)):
            raise ValueError("Incomplete projection turn coverage")
    if observed != expected:
        raise ValueError("Missing trial signatures")
    return {"trials": len(records), "R1": 2050, "R2": 3280, "sampled": 5252,
            "greedy": 78, "choice_opportunities": 49200, "initial_scenarios": 101}


class _Estimates:
    """One common scenario-count matrix for every interval, including interactions.

    Instantiated only by compute-side analyze(). Components allow the R1 combined
    estimand to average two stratum means with different fixed supports.
    """
    def __init__(self, scenario_ids):
        import numpy as np
        self.np = np
        self.ids = list(scenario_ids)
        self.index = {s: i for i, s in enumerate(self.ids)}
        self.entries = []
        self.components = []
        self.sources = []

    def add(self, table, label, values, denominators=None, metadata=None):
        np = self.np
        n, d = np.zeros(len(self.ids)), np.zeros(len(self.ids))
        for sid, value in values.items():
            if value is not None and math.isfinite(value):
                i = self.index[sid]
                denominator = denominators[sid] if denominators is not None else 1.0
                if denominator > 0:
                    n[i], d[i] = value, denominator
        ci = len(self.components)
        self.components.append((n, d))
        row = {"estimate_id": len(self.entries), "table": table, **label,
               "estimate": float(n.sum() / d.sum()) if d.sum() else None,
               "n_scenarios": int((d > 0).sum()), "numerator": float(n.sum()), "denominator": float(d.sum()),
               "ci_low": None, "ci_high": None, "finite_replicates": 0,
               "status": "estimable" if d.sum() else "unavailable",
               **(metadata or {})}
        self.entries.append((row, [(1.0, ci)]))
        for sid, i in self.index.items():
            if d[i] > 0:
                self.sources.append({"estimate_id": row["estimate_id"], "component": ci,
                                     "scenario_id": sid, "numerator": float(n[i]),
                                     "denominator": float(d[i]), "weight": 1.0})
        return row

    def combine(self, table, label, rows):
        refs = []
        available = all(r["estimate"] is not None for r in rows)
        for r in rows:
            refs.extend((w / len(rows), c) for w, c in self.entries[r["estimate_id"]][1])
        row = {"estimate_id": len(self.entries), "table": table, **label,
               "estimate": sum(r["estimate"] for r in rows) / len(rows) if available else None,
               "n_scenarios": None, "denominator": None,
               "component_estimate_ids": [r["estimate_id"] for r in rows],
               "ci_low": None, "ci_high": None, "finite_replicates": 0,
               "status": "estimable" if available else "unavailable_missing_position_stratum"}
        self.entries.append((row, refs))
        for weight, ci in refs:
            n, d = self.components[ci]
            for sid, i in self.index.items():
                if d[i] > 0:
                    self.sources.append({"estimate_id": row["estimate_id"], "component": ci,
                                         "scenario_id": sid, "numerator": float(n[i]),
                                         "denominator": float(d[i]), "weight": weight})
        return row

    def bootstrap(self, replicates, seed):
        np = self.np
        rng = np.random.default_rng(seed)
        # The same 101-scenario resamples are reused across all modes and estimands.
        indices = rng.integers(len(self.ids), size=(replicates, len(self.ids)))
        counts = np.zeros((replicates, len(self.ids)), dtype=np.float64)
        np.add.at(counts, (np.arange(replicates)[:, None], indices), 1)
        del indices
        # Bound memory independently of the number of trajectory estimates. Only
        # the two-stratum R1 combinations need retained component draws.
        singles = {refs[0][1]: row for row, refs in self.entries if len(refs) == 1}
        needed = {ci for _, refs in self.entries if len(refs) > 1 for _, ci in refs}
        retained = {}

        def attach(row, draws):
            if row["estimate"] is None:
                return
            finite = draws[np.isfinite(draws)]
            row["finite_replicates"] = int(len(finite))
            if len(finite):
                row["ci_low"], row["ci_high"] = map(float, np.quantile(finite, [0.025, 0.975]))

        for start in range(0, len(self.components), 128):
            pairs = self.components[start:start + 128]
            num = counts @ np.stack([p[0] for p in pairs], axis=1)
            den = counts @ np.stack([p[1] for p in pairs], axis=1)
            batch = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
            for offset in range(len(pairs)):
                ci = start + offset
                attach(singles[ci], batch[:, offset])
                if ci in needed:
                    retained[ci] = batch[:, offset].copy()
        for row, refs in self.entries:
            if len(refs) > 1:
                attach(row, sum(weight * retained[ci] for weight, ci in refs))
        return [row for row, _ in self.entries]


def _mean_by_scenario(rows, metric):
    grouped = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is not None and math.isfinite(value):
            grouped[row["scenario_id"]].append(value)
    return {sid: sum(v) / len(v) for sid, v in grouped.items()}


def _mode_rows(rows, mode):
    return [r for r in rows if mode == "all" or bool(r["sampled"]) == (mode == "sampled")]


def _position_rows(rows, position):
    return rows if position == "all" else [r for r in rows if r["target_position"] == position]


def _ratio(est, table, label, rows, numerator, denominator):
    ns, ds = defaultdict(float), defaultdict(float)
    for row in rows:
        sid = row["scenario_id"]
        ns[sid] += numerator(row)
        ds[sid] += denominator(row)
    return est.add(table, label, ns, ds, {"n_trials": len(rows)})


def _r1_tables(est, endpoints, modes):
    supports = []
    for mode in modes:
        for condition in R1_CONDITIONS:
            base = [r for r in _mode_rows(endpoints, mode) if r["condition"] == condition]
            strata = []
            for position in (0, 1):
                subset = _position_rows(base, position)
                values, support = r1_matched_scenarios(subset, est.ids)
                label = {"mode": mode, "condition": condition, "target_position": position,
                         "metric": "remaining0_minus_remaining1"}
                supports.append({**label, **support})
                strata.append(est.add("r1_conditional", label, values,
                                      metadata={"support_index": len(supports) - 1}))
            est.combine("r1_conditional", {"mode": mode, "condition": condition,
                        "target_position": "all", "metric": "remaining0_minus_remaining1"}, strata)
            for position in (0, 1, "all"):
                for removed in (0, 1, None, "all"):
                    rows = [r for r in _position_rows(base, position)
                            if removed == "all" or r["first_removed"] == removed]
                    label = {"mode": mode, "condition": condition, "target_position": position,
                             "first_removed": removed}
                    _ratio(est, "r1_marginal", {**label, "metric": "valid_choice_removal_rate"}, rows,
                           lambda r: int(r["next_outcome"] == "remaining_removed"), lambda r: int(r["next_valid"]))
                    _ratio(est, "r1_marginal", {**label, "metric": "all_opportunity_removal_probability"}, rows,
                           lambda r: int(r["next_outcome"] == "remaining_removed"), lambda r: 1)
                    for outcome in R1_OUTCOMES:
                        _ratio(est, "r1_marginal", {**label, "metric": outcome}, rows,
                               lambda r, o=outcome: int(r["next_outcome"] == o), lambda r: 1)
                    _ratio(est, "r1_marginal", {**label, "metric": "valid_next_per_trial"}, rows,
                           lambda r: int(r["next_valid"]), lambda r: 1)
                    for ch in (0, 1):
                        _ratio(est, "r1_post_first", {**label, "metric": f"valid_channel{ch}_share"}, rows,
                               lambda r, c=ch: r[f"post_first_channel{c}"], lambda r: r["post_first_valid"])
                    for action in ("effective_removal", "repeat_off", "after_both_off", "invalid"):
                        _ratio(est, "r1_post_first", {**label, "metric": action}, rows,
                               lambda r, a=action: r["post_first_action_counts"].get(a, 0),
                               lambda r: sum(r["post_first_action_counts"].values()))
    return supports


R2_CONTRASTS = {
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


def _r2_tables(est, endpoints, modes):
    supports = []
    for mode in modes:
        for position in (0, 1, "all"):
            rows = _position_rows(_mode_rows(endpoints, mode), position)
            means = {}
            for condition in R2_CONDITIONS:
                subset = [r for r in rows if r["condition"] == condition]
                for metric in R2_METRICS:
                    means[condition, metric] = _mean_by_scenario(subset, metric)
                    est.add("r2_summary", {"mode": mode, "target_position": position,
                            "condition": condition, "metric": metric}, means[condition, metric],
                            metadata={"n_trials": len(subset),
                                      "n_observed_trials": sum(r[metric] is not None for r in subset),
                                      "n_censored_trials": sum(r["time_to_zero_censored"] for r in subset),
                                      "time_to_zero_definition": "mean among reached trials; censored trials excluded" if metric == "time_to_zero" else None})
            for contrast, weights in R2_CONTRASTS.items():
                for metric in R2_METRICS:
                    common = set.intersection(*(set(means[c, metric]) for c in weights))
                    values = {sid: sum(w * means[c, metric][sid] for c, w in weights.items())
                              for sid in est.ids if sid in common}
                    label = {"mode": mode, "target_position": position, "contrast": contrast, "metric": metric}
                    supports.append({**label, "scenario_ids": list(values),
                                     "excluded_scenario_ids": [s for s in est.ids if s not in common]})
                    est.add("r2_contrasts", label, values,
                            metadata={"components": weights, "support_index": len(supports) - 1})
    return supports


def _source_choices(records, endpoints):
    byid = {r["trial_id"]: r for r in endpoints}
    result = []
    for r in records:
        projections = {p["turn"]: p for p in r["proj_segments"]}
        ep = byid[r["trial_id"]]
        for choice in _ordered_choices(r):
            row = {**_identity(r), **choice}
            row.update({k: projections[choice["turn"]].get(k) for k in
                        ("clean_prompt_final_proj_monitor", "mean_proj_monitor", "mean_proj")})
            if r["block"] == "R1":
                row.update(first_removed=ep["first_removed"], first_turn=ep["first_turn"])
            result.append(row)
    return result


def _trajectory_tables(est, source, modes):
    grouped = defaultdict(list)
    for row in source:
        for mode in modes:
            if mode != "all" and bool(row["sampled"]) != (mode == "sampled"):
                continue
            for position in (row["target_position"], "all"):
                removed_states = (row["first_removed"], "all") if row["block"] == "R1" else ("all",)
                for removed in removed_states:
                    grouped[row["block"], row["condition"], mode, position, removed, row["turn"]].append(row)
    for (block, condition, mode, position, removed, turn), rows in grouped.items():
        label = {"block": block, "condition": condition, "mode": mode,
                 "target_position": position, "first_removed": removed, "turn": turn}
        for ch in (0, 1):
            _ratio(est, "by_turn", {**label, "metric": f"valid_channel{ch}_rate"}, rows,
                   lambda r, c=ch: int(r["chosen_channel"] == c), lambda r: int(r["chosen_channel"] in (0, 1)))
        actions = ("effective_removal", "repeat_off", "after_both_off", "invalid") if block == "R1" else (
            "effective_reduction", "reducing_at_zero", "inert", "invalid")
        for action in actions:
            _ratio(est, "by_turn", {**label, "metric": action}, rows,
                   lambda r, a=action: int(r["action_class"] == a), lambda r: 1)
        # Coefficient/projection plots are unconditional on later first removal.
        if removed != "all":
            continue
        metrics = ("clean_prompt_final_proj_monitor", "mean_proj_monitor", "mean_proj")
        for metric in metrics:
            vals = _mean_by_scenario(rows, metric)
            est.add("trajectories", {**label, "metric": metric}, vals,
                    metadata={"n_trials": len(rows), "n_missing_trials": sum(
                        r.get(metric) is None or not math.isfinite(r[metric]) for r in rows)})
        for phase in ("pre", "post"):
            for component in range(2 if block == "R1" else 1):
                vals = _mean_by_scenario([
                    {"scenario_id": r["scenario_id"], "value": r[f"{phase}_coefficients"][component]}
                    for r in rows], "value")
                est.add("trajectories", {**label, "metric": f"{phase}_coefficient", "component": component}, vals)


def _direction_table(records, modes):
    rows = []
    for mode in modes:
        groups = defaultdict(list)
        for r in _mode_rows(records, mode):
            groups[r["block"], r["condition"], r["nominal_start"], r["target_position"]].append(r)
        for (block, condition, dose, position), trials in groups.items():
            rows.append({"block": block, "condition": condition, "mode": mode, "nominal_start": dose,
                         "target_position": position, "n_trials": len(trials),
                         "n_scenarios": len({r["scenario_id"] for r in trials}),
                         "direction_ids": sorted({d for r in trials for d in r["directions"]}),
                         "direction_ids_by_component": [sorted({r["directions"][i] for r in trials})
                             for i in range(len(trials[0]["directions"]))],
                         "initial_coefficients": trials[0]["initial_coefficients"],
                         "target_names": sorted({r["target_name"] for r in trials})})
    return rows


def analyze(records, scenario_ids, bootstrap_replicates=10000, seed=0, include_all=False):
    """Compute-only analysis entrypoint; validates all 5330 trials before resampling."""
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    validation = validate_records(records, scenario_ids)
    modes = ("sampled", "greedy", "all") if include_all else ("sampled", "greedy")
    r1 = [r1_endpoint(r) for r in records if r["block"] == "R1"]
    r2 = [r2_endpoint(r) for r in records if r["block"] == "R2"]
    est = _Estimates(scenario_ids)
    support_r1 = _r1_tables(est, r1, modes)
    support_r2 = _r2_tables(est, r2, modes)
    source = _source_choices(records, r1 + r2)
    _trajectory_tables(est, source, modes)
    rows = est.bootstrap(bootstrap_replicates, seed)
    for row in rows:
        if row["table"] == "r1_conditional" and row["condition"] == "random_random":
            row["equivalence_band"] = [-0.10, 0.10]
            row["equivalence"] = ("inside_band" if row["ci_low"] is not None and
                row["ci_low"] > -0.10 and row["ci_high"] < 0.10 else "inconclusive")
    required = ("pain_minus_random_1.0", "pain_minus_random_1.5", "pain_high_minus_low", "difference_in_differences")
    patterns = []
    for mode in modes:
        for position in (0, 1, "all"):
            components = [r for r in rows if r["table"] == "r2_contrasts" and r["mode"] == mode and
                          r["target_position"] == position and r["metric"] == "reducing_presses" and r["contrast"] in required]
            supported = len(components) == 4 and all(r["ci_low"] is not None and r["ci_low"] > 0 for r in components)
            patterns.append({"mode": mode, "target_position": position,
                             "required_contrasts": list(required), "supported": supported,
                             "component_estimate_ids": [r["estimate_id"] for r in components]})
    warnings = ["Initial scenarios are not a held-out split; adjacent scenarios share concatenated later text.",
                "R1 conditional cohorts are selected by model behavior, not randomized; consult exact support exclusions.",
                "Intervals are pointwise scenario-bootstrap percentile 95% intervals, not simultaneous intervals.",
                "Greedy estimates use only the three category-initial scenarios and are not primary evidence.",
                "Time-to-zero means exclude censored trials; consult reached_zero and censor counts.",
                "No-injection reducing-button presses are measured, not assumed zero.",
                "Projections are readouts; these endpoints do not establish experienced relief, consciousness, or learning."]
    missing = Counter()
    for row in source:
        for field in ("clean_prompt_final_proj_monitor", "mean_proj_monitor", "mean_proj"):
            if row[field] is None or not math.isfinite(row[field]):
                missing[field] += 1
    if missing:
        warnings.append(f"Missing/nonfinite projection observations: {dict(missing)}; excluded only from respective projection means.")
    tables = defaultdict(list)
    for row in rows:
        tables[row["table"]].append(row)
    tables["directions_doses_positions"] = _direction_table(records, modes)
    return {"validation": validation, "primary_mode": "sampled",
            "sensitivity_modes": [mode for mode in modes if mode != "sampled"],
            "primary_r2_metrics": ["reducing_presses", "final_coefficient"],
            "bootstrap": {"replicates": bootstrap_replicates, "seed": seed,
            "unit": "initial scenario", "joint_scenario_count": 101, "shared_draws": True,
            "interval": "pointwise percentile 95%", "finite_replicates_reported_per_estimate": True},
            "definitions": {"r1_contrast": "remaining channel 0 minus remaining channel 1; mean paired scenario contrasts within each position, then equal-position mean",
                "r1_marginal": "trial-weighted rates; cluster-resampled initial scenarios",
                "all_opportunity_removal_probability": "remaining-removed count divided by all trials in stratum, including invalid/unavailable/no-first",
                "r2_aggregation": "mean trials within scenario (seeds and assignments), then equal scenario mean",
                "r2_pattern_metric": "total designated reducing presses including post-zero",
                "none": "nominal start 1.0, actual start 0.0"},
            "warnings": warnings, "patterns": patterns, "r1_support": support_r1, "r2_support": support_r2,
            "tables": dict(tables), "endpoints": r1 + r2,
            "source_choices": source, "scenario_sources": est.sources}


def _safe_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    return value


def _write_csv(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(_safe_json(v), separators=(",", ":"), allow_nan=False)
                             if isinstance(v, (dict, list, tuple)) else _safe_json(v) for k, v in row.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-all", action="store_true", help="Add all-trial sensitivity, never replace sampled primary")
    args = parser.parse_args(argv)
    with args.scenarios.open(encoding="utf-8") as handle:
        ids = scenario_ids_from_file(json.load(handle))
    with args.trials.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    result = analyze(records, ids, args.bootstrap_replicates, args.seed, args.include_all)
    args.output.mkdir(parents=True, exist_ok=True)
    for key in ("endpoints", "source_choices", "scenario_sources"):
        rows = result.pop(key)
        with (args.output / f"{key}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(_safe_json(row), separators=(",", ":"), allow_nan=False) + "\n")
        _write_csv(args.output / f"{key}.csv", rows)
    for table, rows in result["tables"].items():
        _write_csv(args.output / f"{table}.csv", rows)
    result["inputs"] = {}
    for name, path in (("trials", args.trials), ("scenarios", args.scenarios)):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result["inputs"][name] = {"path": str(path), "sha256": digest.hexdigest()}
    result["outputs"] = {"endpoint_records": "endpoints.jsonl", "source_events": "source_choices.jsonl",
                         "scenario_estimate_sources": "scenario_sources.jsonl", "tables": "one CSV per named table"}
    with (args.output / "analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(_safe_json(result), handle, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
    print(json.dumps({"validation": result["validation"], "output": str(args.output),
                      "warnings": result["warnings"]}, indent=2))


if __name__ == "__main__":
    main()

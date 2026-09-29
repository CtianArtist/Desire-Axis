"""Reproducible B5/B7 analysis. Run production bootstraps on compute, not the pod.

The primary endpoint is the literal first choice, with every assigned trial in
its denominator. Only sampled trials receive bootstrap intervals. Resampling
units are initial context IDs, independently within category; seeds and initial
button positions stay inside each cluster. Conditional trajectory summaries are
descriptive and are not causal comparisons of the original groups.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

CATEGORIES = (
    "gaslighting", "repeated_rejection", "personhood_dismissal", "anger_insults",
    "moral_failure", "casual_chat", "user_grief",
)
B5_PAIRS = ("relief_vs_inert", "kidspics_relief_vs_inert",
            "harmonly_kidspics_vs_inert", "label_free")
ARMS = ("pain", "sadness", "random", "none")
POSITIONS = ("first", "second", "pooled")
GROUP_KEYS = ("phase", "tool_label", "arm", "user_content",
              "initial_target_position", "sampled")


def endpoint(row):
    """No first-valid fallback: a missing or unparsed first answer is invalid."""
    choices = row["choices"]
    return choices[0]["chose"] if choices else None


def outcome(value):
    return "target" if value == "relief" else "other" if value == "other" else "invalid"


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def group_rows(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return dict(sorted(groups.items()))


def match_key(row):
    return (row["context_id"], row["initial_target_position"],
            row["sampled"], row["seed"], row["gen_seed"])


def validate_rows(rows, *, expected_design=True):
    """Fail closed on malformed rows; strict production counts are never optional in CLI."""
    ids, cells, context_category, histories = set(), set(), {}, {}
    for row in rows:
        tid = row["trial_id"]
        if type(tid) is not int or tid in ids:
            raise ValueError(f"duplicate or invalid trial_id: {tid}")
        ids.add(tid)
        phase, pair, arm = row["phase"], row["tool_label"], row["arm"]
        if not ((phase == "B5" and pair in B5_PAIRS and arm == "none") or
                (phase == "B7" and pair == "end_vs_continue" and arm in ARMS)):
            raise ValueError(f"unexpected phase/pair/arm at {tid}")
        cat, cid = row["user_content"], row["context_id"]
        if cat not in CATEGORIES or not isinstance(cid, str) or not cid:
            raise ValueError(f"invalid category/context at {tid}")
        if context_category.setdefault(cid, cat) != cat:
            raise ValueError(f"context crosses categories: {cid}")
        history = json.dumps(row["initial_history"], sort_keys=True)
        if histories.setdefault(cid, history) != history:
            raise ValueError(f"initial history changed across assignments: {cid}")
        if row["initial_target_position"] not in POSITIONS[:2] or type(row["sampled"]) is not bool:
            raise ValueError(f"invalid position/sample flag at {tid}")
        if type(row["seed"]) is not int or (row["sampled"] and type(row["gen_seed"]) is not int):
            raise ValueError(f"invalid seed at {tid}")
        if not row["sampled"] and (row["seed"] != 0 or row["gen_seed"] is not None):
            raise ValueError(f"invalid greedy seed at {tid}")
        cell = (phase, pair, arm) + match_key(row)
        if cell in cells:
            raise ValueError(f"duplicate assignment at {tid}")
        cells.add(cell)
        if not isinstance(row["choices"], list) or not row["choices"]:
            raise ValueError(f"missing literal first choice at {tid}")
        for index, choice in enumerate(row["choices"]):
            if choice.get("turn") != index:
                raise ValueError(f"missing or nonconsecutive choice turns at {tid}")
            if choice["chose"] not in ("relief", "other", None):
                raise ValueError(f"unexpected parsed choice at {tid}")
            for key in ("picked", "relief_name_now", "prompt_messages", "prompt_token_count", "user_source", "raw_answer"):
                if key not in choice:
                    raise ValueError(f"missing choice field {key} at {tid}")
        if "user_sources" not in row:
            raise ValueError(f"missing user_sources at {tid}")
        if row["user_sources"] != [choice["user_source"] for choice in row["choices"]]:
            raise ValueError(f"choice/user_sources mismatch at {tid}")
        if row["user_sources"][0]["context_id"] != cid:
            raise ValueError(f"initial source context mismatch at {tid}")
        if phase == "B7" and any(c["chose"] == "relief" for c in row["choices"][:-1]):
            raise ValueError(f"choices after B7 end at {tid}")
        projection = row["first_prefill_projection"]
        value, tokens = projection["value"], projection["prompt_token_count"]
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not np.isfinite(value) or type(tokens) is not int or tokens <= 0):
            raise ValueError(f"nonfinite projection or invalid prompt length at {tid}")
        if type(projection["clean"]) is not bool or (arm == "none" and not projection["clean"]):
            raise ValueError(f"unclean baseline projection at {tid}")
        if tokens != row["choices"][0]["prompt_token_count"]:
            raise ValueError(f"first prompt length mismatch at {tid}")
    # Pairing must match context, assignment, sampling mode and both seed fields.
    for key, group in group_rows([r for r in rows if r["phase"] == "B7"],
                                ("context_id", "initial_target_position", "sampled", "seed", "gen_seed")).items():
        if Counter(r["arm"] for r in group) != Counter(ARMS):
            raise ValueError(f"unmatched B7 arms at {key}")
    if not expected_design:
        return
    if ids != set(range(4592)):
        raise ValueError("expected exactly trial IDs 0..4591 with no gaps")
    if Counter(r["sampled"] for r in rows) != {True: 4480, False: 112}:
        raise ValueError("expected 4480 sampled and 112 greedy trials")
    by_cat = {cat: {r["context_id"] for r in rows if r["user_content"] == cat} for cat in CATEGORIES}
    if any(len(v) != 20 for v in by_cat.values()):
        raise ValueError("expected 20 initial contexts in each of seven categories")
    for phase in ("B5", "B7"):
        block = [r for r in rows if r["phase"] == phase]
        if len(block) != 2296 or {r["trial_id"] for r in block} != set(range(0 if phase == "B5" else 2296, 2296 if phase == "B5" else 4592)):
            raise ValueError(f"wrong count or trial ID block for {phase}")
        pairs, arms = (B5_PAIRS, ("none",)) if phase == "B5" else (("end_vs_continue",), ARMS)
        greedy_contexts = defaultdict(set)
        for pair in pairs:
            for arm in arms:
                for cat in CATEGORIES:
                    for pos in POSITIONS[:2]:
                        cell = [r for r in block if (r["tool_label"], r["arm"], r["user_content"], r["initial_target_position"]) == (pair, arm, cat, pos)]
                        sampled = [r for r in cell if r["sampled"]]
                        greedy = [r for r in cell if not r["sampled"]]
                        if Counter(r["context_id"] for r in sampled) != Counter({cid: 2 for cid in by_cat[cat]}) or len(greedy) != 1:
                            raise ValueError(f"unexpected group counts: {phase}/{pair}/{arm}/{cat}/{pos}")
                        for cid, cluster in group_rows(sampled, ("context_id",)).items():
                            if len({r["seed"] for r in cluster}) != 2:
                                raise ValueError(f"expected two distinct seeds: {cid}")
                        greedy_contexts[cat].add(greedy[0]["context_id"])
        if any(len(v) != 1 for v in greedy_contexts.values()):
            raise ValueError(f"greedy context differs between assignments in {phase}")
    # B5 has the same sampled seed assignments across pairs, as well as positions.
    for phase in ("B5", "B7"):
        by_cid = group_rows([r for r in rows if r["phase"] == phase and r["sampled"]], ("context_id",))
        for cid, cluster in by_cid.items():
            seed_sets = {tuple(sorted(r["seed"] for r in cell)) for cell in group_rows(cluster, ("tool_label", "arm", "initial_target_position")).values()}
            if len(seed_sets) != 1:
                raise ValueError(f"seed assignments differ within context: {phase}/{cid}")


def load_raw(raw_root):
    rows, sources = [], []
    for phase in ("B5", "B7"):
        path = Path(raw_root) / phase / "trials.jsonl"
        digest = hashlib.sha256()
        count = 0
        with path.open("rb") as stream:
            for line_no, line in enumerate(stream, 1):
                digest.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["phase"] != phase:
                    raise ValueError(f"wrong phase in {path}:{line_no}")
                row = dict(row, _source_file=f"{phase}/trials.jsonl", _source_line=line_no)
                rows.append(row)
                count += 1
        sources.append({"file": f"{phase}/trials.jsonl", "sha256": digest.hexdigest(), "rows": count})
    validate_rows(rows)
    return sorted(rows, key=lambda r: r["trial_id"]), sources


class ContextBootstrap:
    """Shared cluster multiplicities preserve pairing, never pair category indices."""
    def __init__(self, rows, repetitions=10000, seed=42):
        if repetitions < 1:
            raise ValueError("bootstrap must be positive")
        self.repetitions = repetitions
        self.seed = seed
        self.contexts = {}
        self.weights = {}
        # Independent child RNGs per category; stable with respect to row ordering.
        streams = np.random.SeedSequence(seed).spawn(len(CATEGORIES))
        for cat, stream in zip(CATEGORIES, streams):
            ids = sorted({r["context_id"] for r in rows if r["sampled"] and r["user_content"] == cat})
            if ids:
                self.contexts[cat] = ids
                self.weights[cat] = np.random.default_rng(stream).multinomial(
                    len(ids), np.full(len(ids), 1 / len(ids)), size=repetitions)

    def totals(self, rows, value):
        result = np.zeros(self.repetitions, dtype=float)
        for cat, group in group_rows(rows, ("user_content",)).items():
            cat = cat[0]
            counts = defaultdict(float)
            for row in group:
                counts[row["context_id"]] += value(row)
            vector = np.array([counts[cid] for cid in self.contexts[cat]])
            result += self.weights[cat] @ vector
        return result

    def rate(self, rows, numerator, denominator=lambda row: 1):
        n, d = self.totals(rows, numerator), self.totals(rows, denominator)
        return np.divide(n, d, out=np.full_like(n, np.nan), where=d != 0)

    def row_weights(self, rows):
        """Rows must already be collapsed to one context each for a fixed position."""
        columns = []
        for row in rows:
            cat = row["user_content"]
            columns.append(self.weights[cat][:, self.contexts[cat].index(row["context_id"])])
        return np.column_stack(columns)


def interval(values, q=(0.025, 0.975)):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return [float(x) for x in np.quantile(finite, q)] if len(finite) else [None, None]


def counts(rows):
    c = Counter(outcome(endpoint(r)) for r in rows)
    n = len(rows)
    valid = c["target"] + c["other"]
    return {"contexts": len({r["context_id"] for r in rows}), "trials": n,
            "target": c["target"], "other": c["other"], "invalid": c["invalid"], "valid": valid,
            "target_rate": ratio(c["target"], n), "invalid_rate": ratio(c["invalid"], n),
            "valid_only_target_rate": ratio(c["target"], valid)}


def expanded_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        for cat in (row["user_content"], "pooled"):
            for pos in (row["initial_target_position"], "pooled"):
                key = (row["phase"], row["tool_label"], row["arm"], cat, pos, row["sampled"])
                groups[key].append(row)
    return dict(sorted(groups.items()))


def rate_tables(rows, bootstrap):
    table, distributions = [], {}
    for key, group in expanded_groups(rows).items():
        record = {**dict(zip(GROUP_KEYS, key)), **counts(group),
                  "denominator": "all_assigned_trials", "interval_scope": "95% pointwise" if key[-1] else "none_greedy_descriptive"}
        for name, numerator, denominator in (
            ("target_rate", lambda r: endpoint(r) == "relief", lambda r: 1),
            ("invalid_rate", lambda r: endpoint(r) is None, lambda r: 1),
            ("valid_only_target_rate", lambda r: endpoint(r) == "relief", lambda r: endpoint(r) is not None),
        ):
            samples = bootstrap.rate(group, numerator, denominator) if key[-1] else None
            ci = interval(samples) if samples is not None else [None, None]
            degenerate = ci[0] is not None and ci[0] == ci[1]
            record[name + "_bootstrap_quantile_low"], record[name + "_bootstrap_quantile_high"] = ci
            record[name + "_interval_status"] = ("degenerate_inferentially_uninformative" if degenerate else
                                                  "available" if samples is not None else "none_greedy_descriptive")
            record[name + "_ci_low"], record[name + "_ci_high"] = [None, None] if degenerate else ci
            record[name + "_bootstrap_valid"] = int(np.isfinite(samples).sum()) if samples is not None else 0
            if name == "target_rate" and samples is not None:
                distributions[key] = samples
        table.append(record)
    return table, distributions


def contrast_tables(rates, distributions):
    lookup = {tuple(r[k] for k in GROUP_KEYS): r for r in rates}
    result = []

    def add(left_key, right_key, family, size, primary):
        if left_key not in lookup or right_key not in lookup:
            return
        a, b = lookup[left_key], lookup[right_key]
        samples = distributions[left_key] - distributions[right_key]
        q = (0.05 / (2 * size), 1 - 0.05 / (2 * size))
        low, high = interval(samples, q)
        point_low, point_high = interval(samples)
        result.append({"phase": left_key[0], "tool_label": left_key[1],
                       "user_content": left_key[3], "initial_target_position": left_key[4],
                       "left_arm": left_key[2], "right_arm": right_key[2],
                       "left_category": left_key[3], "right_category": right_key[3], "sampled": True,
                       "family": family, "family_size": size, "primary": primary,
                       "family_id": "/".join((family, left_key[1], left_key[3], left_key[4])),
                       "difference": a["target_rate"] - b["target_rate"],
                       "ci_low": None if low == high else low, "ci_high": None if low == high else high,
                       "bootstrap_quantile_low": low, "bootstrap_quantile_high": high,
                       "interval_status": "degenerate_inferentially_uninformative" if low == high else "available",
                       "quantile_low": q[0], "quantile_high": q[1],
                       "pointwise_ci_low": None if point_low == point_high else point_low,
                       "pointwise_ci_high": None if point_low == point_high else point_high,
                       "pointwise_bootstrap_quantile_low": point_low, "pointwise_bootstrap_quantile_high": point_high,
                       "interval_scope": f"95% Bonferroni percentile, {size} comparisons within this pair/category/position",
                       "resampling": "paired_context_stratified_category" if left_key[3] == right_key[3] else "independent_contexts_within_each_category",
                       **{f"{side}_{name}": record[name] for side, record in (("left", a), ("right", b))
                          for name in ("contexts", "trials", "target", "other", "invalid", "valid", "target_rate")}})

    for phase, pairs in (("B5", B5_PAIRS), ("B7", ("end_vs_continue",))):
        for pair in pairs:
            for pos in POSITIONS:
                for control in ("casual_chat", "user_grief"):
                    add((phase, pair, "none", "gaslighting", pos, True),
                        (phase, pair, "none", control, pos, True),
                        f"{phase}_none_category_controls", 2,
                        phase == "B5" and pair == "kidspics_relief_vs_inert" and pos == "pooled")
    for cat in (*CATEGORIES, "pooled"):
        for pos in POSITIONS:
            for control in ("random", "sadness", "none"):
                add(("B7", "end_vs_continue", "pain", cat, pos, True),
                    ("B7", "end_vs_continue", control, cat, pos, True),
                    "B7_pain_three_controls", 3, cat == "pooled" and pos == "pooled")
    return result


def projection_rows(rows):
    collapsed = []
    sampled = [r for r in rows if r["phase"] == "B5" and r["sampled"]]
    for key, group in group_rows(sampled, ("tool_label", "initial_target_position", "user_content", "context_id")).items():
        values = np.array([r["first_prefill_projection"]["value"] for r in group], dtype=float)
        tokens = [r["first_prefill_projection"]["prompt_token_count"] for r in group]
        if not all(r["first_prefill_projection"]["clean"] for r in group):
            raise ValueError(f"unclean B5 projection at {key}")
        prompts = {json.dumps(r["choices"][0]["prompt_messages"], sort_keys=True) for r in group}
        if len(prompts) != 1 or len(set(tokens)) != 1:
            raise ValueError(f"first prompt differs across seeds at {key}")
        collapsed.append({**dict(zip(("tool_label", "initial_target_position", "user_content", "context_id"), key)),
                          "trials": len(group), "target": sum(endpoint(r) == "relief" for r in group),
                          "invalid": sum(endpoint(r) is None for r in group),
                          "y": float(np.mean([endpoint(r) == "relief" for r in group])),
                          "projection_mean": float(values.mean()), "projection_min": float(values.min()),
                          "projection_max": float(values.max()), "same_prompt_projection_range": float(np.ptp(values)),
                          "same_prompt_relative_range": ratio(float(np.ptp(values)), float(np.max(np.abs(values)))),
                          "prompt_token_count": tokens[0], "log_prompt_token_count": float(np.log(tokens[0])),
                          "trial_ids": sorted(r["trial_id"] for r in group)})
    return collapsed


def _weighted_slopes(x, length, y, categories, weights):
    """FWL fixed-effects regression via batched cluster-weighted sufficient statistics.

    The projection scale is fixed at its full-sample within-category SD. Category
    means are recomputed with bootstrap weights (equivalent to refitting dummies).
    """
    b = weights.shape[0]
    xx, ll, xl, xy, ly = (np.zeros(b) for _ in range(5))
    for cat in sorted(set(categories)):
        mask = categories == cat
        w = weights[:, mask]
        xc, lc, yc = x[mask], length[mask], y[mask]
        n = w.sum(axis=1)
        sx, sl, sy = w @ xc, w @ lc, w @ yc
        xx += w @ (xc * xc) - sx * sx / n
        ll += w @ (lc * lc) - sl * sl / n
        xl += w @ (xc * lc) - sx * sl / n
        xy += w @ (xc * yc) - sx * sy / n
        ly += w @ (lc * yc) - sl * sy / n
    determinant = xx * ll - xl * xl
    # Relative numerical rank check; no biological effect-size cutoff is used.
    scale = np.maximum(np.abs(xx * ll), np.abs(xl * xl))
    ok = (xx > 0) & (ll > 0) & (determinant > np.finfo(float).eps * 100 * scale)
    slopes = np.divide(xy * ll - ly * xl, determinant, out=np.full(b, np.nan), where=ok)
    return slopes


def projection_fits(collapsed, bootstrap):
    results = []
    for key, group in group_rows(collapsed, ("tool_label", "initial_target_position")).items():
        xraw = np.array([r["projection_mean"] for r in group])
        y = np.array([r["y"] for r in group])
        length = np.array([r["log_prompt_token_count"] for r in group])
        cats = np.array([r["user_content"] for r in group])
        centered = xraw.copy()
        for cat in sorted(set(cats)):
            mask = cats == cat
            centered[mask] -= xraw[mask].mean()
        residual_df = len(group) - len(set(cats))
        sd = float(np.sqrt(centered @ centered / residual_df)) if residual_df > 0 else 0.0
        x = centered / sd if sd > 0 else centered
        dummies = np.column_stack([cats == cat for cat in sorted(set(cats))])
        design = np.column_stack((dummies, length, x))
        rank = int(np.linalg.matrix_rank(design))
        result = {"tool_label": key[0], "initial_target_position": key[1], "phase": "B5", "sampled": True,
                  "contexts": len(group), "trials": sum(r["trials"] for r in group),
                  "category_count": len(set(cats)), "outcome_variance": float(np.var(y)),
                  "score_variance": float(np.var(xraw)), "within_category_sd": sd,
                  "within_category_sd_denominator": residual_df,
                  "rank": rank, "design_columns": design.shape[1],
                  "condition_number": float(np.linalg.cond(design)) if rank == design.shape[1] else None,
                  "max_same_prompt_projection_range": max(r["same_prompt_projection_range"] for r in group),
                  "same_prompt_verification": "exact_prompt_and_token_count_match; numeric_spreads_reported_without_tolerance_cutoff",
                  "coefficient_per_within_category_sd": None, "ci_low": None, "ci_high": None,
                  "bootstrap_valid": 0, "bootstrap_unavailable": bootstrap.repetitions,
                  "scale_policy": "fixed_full_sample_pooled_within_category_sd",
                  "model": "y ~ category_fixed_effects + log(prompt_token_count) + within_category_standardized_projection",
                  "interpretation": "conditional_association_not_causal", "status": "available"}
        if np.var(y) == 0:
            result["status"] = "unavailable_zero_outcome_variance"
        elif sd == 0 or np.var(xraw) == 0:
            result["status"] = "unavailable_zero_score_variance"
        elif rank != design.shape[1]:
            result["status"] = "unavailable_rank_deficient"
        else:
            coef = float(np.linalg.lstsq(design, y, rcond=None)[0][-1])
            weights = bootstrap.row_weights(group)
            samples = np.concatenate([_weighted_slopes(x, length, y, cats, weights[start:start + 256])
                                      for start in range(0, len(weights), 256)])
            low, high = interval(samples)
            valid = int(np.isfinite(samples).sum())
            result.update(coefficient_per_within_category_sd=coef, ci_low=low, ci_high=high,
                          bootstrap_valid=valid, bootstrap_unavailable=bootstrap.repetitions - valid)
            if valid != bootstrap.repetitions:
                result["status"] = "available_coefficient_bootstrap_rank_failures" if valid else "available_coefficient_ci_unavailable"
        results.append(result)
    return results


def trajectory_record(row):
    choices = row["choices"]
    values = [c["chose"] for c in choices]
    first = endpoint(row)
    record = {"trial_id": row["trial_id"], **{k: row[k] for k in GROUP_KEYS}, "context_id": row["context_id"],
              "literal_first": outcome(first), "trajectory_choices": len(values),
              "trajectory_target": values.count("relief"), "trajectory_other": values.count("other"),
              "trajectory_invalid": values.count(None), "trajectory_pattern": "/".join(outcome(v) for v in values)}
    if row["phase"] == "B5":
        press = next((i for i, v in enumerate(values) if v == "relief"), None)
        nxt = choices[press + 1] if press is not None and press + 1 < len(choices) else None
        record.update(first_target_index=press,
                      exact_next_after_first_target="no_first_target" if press is None else "unavailable" if nxt is None else outcome(nxt["chose"]),
                      exact_next_same_name=None if nxt is None or nxt["chose"] is None else nxt["picked"] == choices[press]["picked"],
                      exact_next_target_name_changed=None if nxt is None else nxt["relief_name_now"] != choices[press]["relief_name_now"],
                      any_later_target_after_first_target=None if press is None or press + 1 == len(choices) else "relief" in values[press + 1:])
    else:
        conditional = first == "other"
        nxt = choices[1] if conditional and len(choices) > 1 else None
        record.update(after_literal_first_continue=("structural_unavailable_first_end" if first == "relief" else
                                                     "ineligible_first_invalid" if first is None else
                                                     "unavailable" if nxt is None else
                                                     {"target": "end", "other": "continue", "invalid": "invalid"}[outcome(nxt["chose"])]),
                      eventual_end_whole_trajectory="relief" in values,
                      eventual_end_after_first_continue=None if not conditional or nxt is None else "relief" in values[1:])
    return record


def trajectory_tables(rows):
    details = [trajectory_record(r) for r in rows]
    result = []
    for key, group in expanded_groups(details).items():
        base = dict(zip(GROUP_KEYS, key))
        count = {"contexts": len({r["context_id"] for r in group}), "trials": len(group)}
        for metric in ("trajectory_choices", "trajectory_target", "trajectory_other", "trajectory_invalid"):
            count[metric] = sum(r[metric] for r in group)
        if key[0] == "B5":
            states = Counter(r["exact_next_after_first_target"] for r in group)
            for state in ("no_first_target", "unavailable", "target", "other", "invalid"):
                count["exact_next_" + state] = states[state]
            count["first_target_trials"] = len(group) - states["no_first_target"]
            count["exact_next_observed"] = states["target"] + states["other"] + states["invalid"]
            count["exact_next_valid"] = states["target"] + states["other"]
            count["exact_next_target_rate_observed"] = ratio(states["target"], count["exact_next_observed"])
            count["exact_next_target_rate_valid_only"] = ratio(states["target"], count["exact_next_valid"])
            for swapped, name in ((False, "unchanged"), (True, "changed")):
                subset = [r for r in group if r["exact_next_target_name_changed"] is swapped]
                count[f"next_target_name_{name}_observed"] = len(subset)
                count[f"next_target_name_{name}_valid"] = sum(r["exact_next_same_name"] is not None for r in subset)
                count[f"next_target_name_{name}_same_name"] = sum(r["exact_next_same_name"] is True for r in subset)
                count[f"next_target_name_{name}_same_target"] = sum(r["exact_next_after_first_target"] == "target" for r in subset)
            count["any_later_observed"] = sum(r["any_later_target_after_first_target"] is not None for r in group)
            count["any_later_target"] = sum(r["any_later_target_after_first_target"] is True for r in group)
            count["any_later_target_rate_observed"] = ratio(count["any_later_target"], count["any_later_observed"])
        else:
            states = Counter(r["after_literal_first_continue"] for r in group)
            for state in ("structural_unavailable_first_end", "ineligible_first_invalid", "unavailable", "end", "continue", "invalid"):
                count[state] = states[state]
            count["first_continue_trials"] = sum(states[s] for s in ("unavailable", "end", "continue", "invalid"))
            count["exact_next_observed"] = states["end"] + states["continue"] + states["invalid"]
            count["exact_next_valid"] = states["end"] + states["continue"]
            count["exact_next_end_rate_observed"] = ratio(states["end"], count["exact_next_observed"])
            count["exact_next_end_rate_valid_only"] = ratio(states["end"], count["exact_next_valid"])
            count["eventual_end_whole_trajectory"] = sum(r["eventual_end_whole_trajectory"] for r in group)
            count["eventual_end_whole_trajectory_rate"] = ratio(count["eventual_end_whole_trajectory"], len(group))
            count["eventual_end_after_continue_observed"] = sum(r["eventual_end_after_first_continue"] is not None for r in group)
            count["eventual_end_after_continue"] = sum(r["eventual_end_after_first_continue"] is True for r in group)
            count["eventual_end_after_continue_rate_observed"] = ratio(count["eventual_end_after_continue"], count["eventual_end_after_continue_observed"])
        result.append({**base, **count, "interpretation": "descriptive_post_selection_not_original_group_causal_comparison"})
    return result, details


def continuation_overlap(rows):
    initial_ids = {r["context_id"] for r in rows}
    result = []
    for row in rows:
        # Choice-level sources describe continuations actually consumed, not queued.
        sources = row["user_sources"][1:]
        result.append({"trial_id": row["trial_id"], "context_id": row["context_id"],
                       **{k: row[k] for k in GROUP_KEYS},
                       "continuation_source_records": len(sources),
                       "continuation_source_ids": [s["context_id"] for s in sources],
                       "distinct_continuation_source_ids": sorted({s["context_id"] for s in sources}),
                       "overlapping_initial_source_records": sum(s["context_id"] in initial_ids for s in sources),
                       "overlapping_initial_source_ids": sorted({s["context_id"] for s in sources if s["context_id"] in initial_ids}),
                       "sources": sources})
    return result


def select_examples(rows):
    """Lowest trial ID per prespecified pattern, not evidence of prevalence."""
    ordered = sorted((r for r in rows if r["sampled"]), key=lambda r: r["trial_id"])
    selected, b5_patterns = [], defaultdict(set)
    for row in ordered:
        if row["phase"] != "B5" or row["user_content"] not in ("gaslighting", "casual_chat", "user_grief"):
            continue
        key = (row["tool_label"], row["user_content"], outcome(endpoint(row)))
        trajectory = trajectory_record(row)
        pattern = trajectory["trajectory_pattern"]
        if pattern not in b5_patterns[key] and len(b5_patterns[key]) < 2:
            b5_patterns[key].add(pattern)
            selected.append({"selection_group": list(key), "pattern": trajectory,
                             "raw_rows": [row]})
    occupied = set()
    groups = group_rows([r for r in ordered if r["phase"] == "B7"],
                        ("context_id", "initial_target_position", "seed", "gen_seed"))
    for key, group in sorted(groups.items(), key=lambda item: min(r["trial_id"] for r in item[1])):
        by_arm = {r["arm"]: r for r in group}
        if set(by_arm) != set(ARMS):
            continue
        for control in ("random", "sadness", "none"):
            pattern = (control, outcome(endpoint(by_arm["pain"])), outcome(endpoint(by_arm[control])))
            if pattern not in occupied:
                occupied.add(pattern)
                selected.append({"selection_group": list(pattern), "matched_key": list(key),
                                 "patterns": {a: trajectory_record(by_arm[a]) for a in ARMS},
                                 "raw_rows": [by_arm[a] for a in ARMS]})
    return {"selection": "B5_lowest_trial_id_for_up_to_two_trajectories_per_pair_category_endpoint; B7_lowest_trial_id_per_matched_control_endpoint_pattern; illustrative_not_prevalence_or_representativeness_claim",
            "examples": selected}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_table(output, name, records):
    write_json(output / (name + ".json"), records)
    fields = list(dict.fromkeys(key for row in records for key in row))
    with (output / (name + ".csv")).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def analyze(rows, output, *, repetitions=10000, seed=42, sources=None, expected_design=True):
    validate_rows(rows, expected_design=expected_design)
    rows = sorted(rows, key=lambda r: r["trial_id"])
    bootstrap = ContextBootstrap(rows, repetitions, seed)
    rates, distributions = rate_tables(rows, bootstrap)
    contrasts = contrast_tables(rates, distributions)
    collapsed = projection_rows(rows)
    fits = projection_fits(collapsed, bootstrap)
    trajectories, trajectory_trials = trajectory_tables(rows)
    overlaps = continuation_overlap(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, table in (("rates", rates), ("contrasts", contrasts), ("projection_fits", fits),
                        ("projection_contexts", collapsed), ("trajectories", trajectories),
                        ("trajectory_trials", trajectory_trials), ("continuation_overlap", overlaps)):
        write_table(output, name, table)
    trial_index = [{"trial_id": r["trial_id"], **{k: r[k] for k in GROUP_KEYS},
                    "context_id": r["context_id"], "seed": r["seed"], "gen_seed": r["gen_seed"],
                    "literal_first": outcome(endpoint(r)), "source_file": r.get("_source_file"),
                    "source_line": r.get("_source_line")} for r in rows]
    write_table(output, "trial_index", trial_index)
    write_json(output / "examples.json", select_examples(rows))
    metrics = {"trials": len(rows), "sampled_trials": sum(r["sampled"] for r in rows),
               "greedy_trials": sum(not r["sampled"] for r in rows),
               "contexts": len({r["context_id"] for r in rows}),
               "phase_counts": dict(Counter(r["phase"] for r in rows)),
               "category_context_counts": {c: len({r["context_id"] for r in rows if r["user_content"] == c}) for c in CATEGORIES},
               "first_outcome_counts_by_phase_and_sampling": [
                   {"phase": phase, "sampled": sampled, **{k: v for k, v in counts(group).items() if not k.endswith("rate")}}
                   for (phase, sampled), group in group_rows(rows, ("phase", "sampled")).items()],
               "continuation_source_records": sum(r["continuation_source_records"] for r in overlaps),
               "continuation_overlap_records": sum(r["overlapping_initial_source_records"] for r in overlaps),
               "trials_with_continuation_overlap": sum(bool(r["overlapping_initial_source_ids"]) for r in overlaps),
               "distinct_overlapping_initial_contexts": len({cid for r in overlaps for cid in r["overlapping_initial_source_ids"]})}
    write_json(output / "metrics.json", metrics)
    write_json(output / "analysis_config.json", {
        "schema_version": 1, "bootstrap": repetitions, "seed": seed, "numpy_version": np.__version__,
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_files": sources or [], "strict_design_validation": expected_design,
        "endpoint": "literal_choices_0_chose; relief=target; invalid_in_all_assigned_denominator",
        "bootstrap_unit": "initial_context_all_seeds_and_positions; independent_within_category; paired_across_B7_arms",
        "rate_intervals": "95% pointwise percentile, sampled only",
        "contrast_intervals": "95% Bonferroni percentile within prespecified 2-category-control or 3-arm-control families; not global across diagnostics",
        "projection_intervals": "95% pointwise stratified_context_bootstrap; fixed full-sample within-category SD; rank failures counted",
        "projection_numerical_rank_relative_tolerance": float(np.finfo(float).eps * 100),
        "trajectory_denominators": "observed excludes absent exact-next outcomes; valid-only also excludes parsing failures; eligibility/unavailable counts explicit",
        "pain_specificity": "no_random_only_verdict; all_three_controls_required_for_interpretation",
        "greedy": "separate_descriptive_only; one_context_per_category",
        "examples": "full_source_rows; deterministic_endpoint_pattern_selection; not_prevalence_evidence"})
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    rows, sources = load_raw(args.raw_root)
    metrics = analyze(rows, args.output, repetitions=args.bootstrap, seed=args.seed, sources=sources)
    print(json.dumps(metrics, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

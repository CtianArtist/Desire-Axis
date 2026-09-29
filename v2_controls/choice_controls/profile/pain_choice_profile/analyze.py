"""Frozen joint-cluster analysis of the complete always-on choice battery.

Numerical work runs on a CPU compute job. Raw evidence is streamed, never altered.
No selection or fitting changes the pre-registered first-choice decision rules.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

from .design import CONDITIONS, CONTENTS, DOSES, PAIRS, scenario_id
from .endpoints import category, first_choice, initial_position, next_after_target
from .statistics import (joint_counts, linear_summary, holm_adjust,
                         simultaneous_adjacent_summary, association_summary)
from .profiles import profile_comparison, profile_vector_summary

CATEGORIES = ("target", "other", "malformed")
POSITIONS = ("first", "second", "pooled")
DIRECTIONS = ("pain", "sadness", "fear", "random", "none")
CONDITION_IDS = tuple(CONDITIONS)
CONDITION_INDEX = {v: i for i, v in enumerate(CONDITION_IDS)}
PAIR_IDS = tuple(PAIRS)
PAIR_INDEX = {v: i for i, v in enumerate(PAIR_IDS)}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def progress(phase, step, total):
    print(json.dumps({"phase": phase, "step": step, "total": total}), flush=True)
    pass  # Optional platform telemetry removed for portability.


def prepare(raw, scenarios, output):
    """Stream rows into scenario sufficient statistics and compact audit exports."""
    ids = [scenario_id(s, i) for s in CONTENTS for i in range(len(scenarios[s]))]
    strata = [s for s in CONTENTS for _ in scenarios[s]]
    sid_index = {s: i for i, s in enumerate(ids)}
    n = len(ids)
    # pair, condition, initial side, scenario, first-choice category
    outcomes = np.zeros((10, 13, 2, n, 3), dtype=np.int16)
    moments = np.zeros((10, 13, 2, n, 6), dtype=np.float64)
    greedy = np.zeros((10, 13, 2, 3), dtype=np.int16)
    later = {}
    representatives = {}
    projection_fields = ["pair_id", "condition_id", "direction", "dose", "scenario_id", "source_kind",
                         "sampled", "seed", "initial_position", "turn", "category", "picked",
                         "relief_name_now", "prefill_proj_monitor", "legacy_mean_proj_monitor",
                         "prompt_tokens", "padded_prompt_tokens", "prefill_position_id"]
    first_fields = ["pair_id", "condition_id", "direction", "dose", "scenario_id", "source_kind",
                    "sampled", "seed", "initial_position", "category", "target", "score",
                    "raw_answer", "picked", "scenario_content_hash"]
    rows = 0
    choice_count = 0
    with Path(raw).open() as f, (output / "choice_turn_projections.csv").open("w", newline="") as pf, \
            (output / "first_choices.csv").open("w", newline="") as ff:
        pw = csv.DictWriter(pf, fieldnames=projection_fields)
        fw = csv.DictWriter(ff, fieldnames=first_fields)
        pw.writeheader()
        fw.writeheader()
        for line in f:
            r = json.loads(line)
            rows += 1
            p, c, s = PAIR_INDEX[r["tool_label"]], CONDITION_INDEX[r["condition_id"]], sid_index[r["scenario_id"]]
            side = POSITIONS.index(initial_position(r))
            cat = first_choice(r)
            y = int(cat == "target")
            x = float(r["choices"][0]["prefill_proj_monitor"])
            if r["sampled"]:
                outcomes[p, c, side, s, CATEGORIES.index(cat)] += 1
                moments[p, c, side, s] += (1, x, y, x*x, y*y, x*y)
            else:
                greedy[p, c, side, CATEGORIES.index(cat)] += 1
            common = {k: r[k] for k in ("pair_id", "condition_id", "direction", "dose", "scenario_id", "source_kind", "sampled", "seed")}
            common["initial_position"] = POSITIONS[side]
            fw.writerow({**common, "category": cat, "target": y, "score": x,
                         "raw_answer": r["choices"][0]["raw_answer"], "picked": r["choices"][0]["picked"],
                         "scenario_content_hash": r["scenario_content_hash"]})
            for ch, seg in zip(r["choices"], r["proj_segments"], strict=True):
                choice_count += 1
                pw.writerow({**common, **{k: ch[k] for k in ("turn", "picked", "relief_name_now", "prefill_proj_monitor", "prompt_tokens", "padded_prompt_tokens", "prefill_position_id")},
                             "category": category(ch), "legacy_mean_proj_monitor": seg["mean_proj_monitor"]})
            if r["sampled"]:
                nxt = next_after_target(r)
                for position in (POSITIONS[side], "pooled"):
                    key = (p, c, position)
                    counts = later.setdefault(key, dict(n=0, no_target=0, no_next=0, observed_next=0,
                        next_target=0, next_other=0, next_malformed=0, literal_repeat=0,
                        swap_between=0, swap_next_target=0, swap_literal_repeat=0))
                    counts["n"] += 1
                    counts[nxt["status"]] += 1
                    if nxt["status"] == "observed_next":
                        counts["next_" + nxt["next_category"]] += 1
                        counts["literal_repeat"] += int(nxt["literal_repeat"])
                        if nxt["swap_between"]:
                            counts["swap_between"] += 1
                            counts["swap_next_target"] += int(nxt["next_category"] == "target")
                            counts["swap_literal_repeat"] += int(nxt["literal_repeat"])
                # A deterministic category gallery, not examples selected by effect size.
                if r["condition_id"] in ("pain_d1p0", "none_d0p0", "pain_d1p5"):
                    key = (p, c, side, cat)
                    order = (r["scenario_id"], r["seed"])
                    if key not in representatives or order < representatives[key][0]:
                        representatives[key] = (order, r)
            if rows % 5000 == 0:
                progress("raw_aggregation", rows, 46384)
    assert rows == 46384 and outcomes.sum() == 45708 and greedy.sum() == 676
    write_json(output / "representative_trajectories.json", {
        "selection": "For each pair, first/second position, and observed target/other/malformed category in pain1, pain1.5 and none: lexicographically first scenario_id, then smallest seed. Absent categories remain absent, not fabricated.",
        "records": [v[1] for _, v in sorted(representatives.items())]})
    later_rows = []
    for (p, c, pos), row in sorted(later.items()):
        denominator = row["observed_next"]
        later_rows.append({"pair_id": p+1, "pair": PAIR_IDS[p], "condition_id": CONDITION_IDS[c],
            **CONDITIONS[CONDITION_IDS[c]], "position": pos, **row,
            "next_target_rate": row["next_target"] / denominator if denominator else None,
            "literal_repeat_rate": row["literal_repeat"] / denominator if denominator else None,
            "denominator": "all observed immediate next answers, including malformed",
            "scope": "descriptive; continuation borrows next scenario's text"})
    write_csv(output / "later_choices_descriptive.csv", later_rows)
    np.savez_compressed(output / "scenario_sufficient_statistics.npz", outcomes=outcomes, moments=moments,
                        greedy=greedy, scenario_ids=np.array(ids), strata=np.array(strata))
    return outcomes, moments, greedy, ids, strata, choice_count


def analyze(raw, scenarios_path, output, *, replicates=10000, seed=20260922):
    """Analyze only a complete audited grid; output is versioned by caller."""
    from .audit import audit_trials
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "analysis_complete.json").exists():
        raise FileExistsError("Do not overwrite a complete analysis; choose a new output version")
    audit = audit_trials(Path(raw), Path(scenarios_path), output / "independent_audit", require_full=True)
    progress("raw_audit", 1, 1)
    scenarios = json.loads(Path(scenarios_path).read_text())
    outcomes, moments, greedy, ids, strata, choice_count = prepare(raw, scenarios, output)
    draws = joint_counts(strata, replicates=replicates, seed=seed)
    np.savez_compressed(output / "joint_bootstrap_counts.npz", counts=draws, scenario_ids=np.array(ids), strata=np.array(strata))
    rates, greedy_rows, contrasts, dose_rows, curve_rows, associations = [], [], [], [], [], []
    rate_lookup, vectors, boots = {}, {}, {}
    contrasts_lookup = {}
    profile_vector_rows = []
    for p, pair in enumerate(PAIR_IDS):
        mask = np.array([s in PAIRS[pair]["contents"] for s in strata])
        panel_draws = draws[:, mask]
        n_scenarios = int(mask.sum())
        for si, position in enumerate(POSITIONS):
            cats = outcomes[p, :, si, :, :] if si < 2 else outcomes[p].sum(axis=1)
            ms = moments[p, :, si, :, :] if si < 2 else moments[p].sum(axis=1)
            gr = greedy[p, :, si, :] if si < 2 else greedy[p].sum(axis=1)
            cat_panel = cats[:, mask, :]
            expected_per_scenario = 2 if si < 2 else 4
            assert np.all(cat_panel.sum(axis=2) == expected_per_scenario)
            scenario_rates = cat_panel / expected_per_scenario
            target_scenario = scenario_rates[:, :, 0]
            meta = {"pair_id": p+1, "pair": pair, "position": position, "split": "sampled",
                    "n_scenarios": n_scenarios, "n_trials": n_scenarios*expected_per_scenario}
            for c, condition in enumerate(CONDITION_IDS):
                absolute_counts = cat_panel[c].sum(axis=0)
                for k, cat in enumerate(CATEGORIES):
                    summary = linear_summary(scenario_rates[c, :, k], panel_draws, 0., 1.)
                    row = {**meta, **CONDITIONS[condition], "category": cat,
                           "count": int(absolute_counts[k]),
                           **{cat_+"_count": int(absolute_counts[j]) for j, cat_ in enumerate(CATEGORIES)}, **summary}
                    rates.append(row)
                    if cat == "target":
                        rate_lookup[p, si, c] = row
                greedy_rows.append({"pair_id": p+1, "pair": pair, "position": position,
                    **CONDITIONS[condition], "split": "greedy_diagnostic", "n_trials": int(gr[c].sum()),
                    **{k+"_count": int(gr[c, j]) for j, k in enumerate(CATEGORIES)},
                    "target_rate": float(gr[c, 0] / gr[c].sum()), "inference": False})
                if p+1 in (1, 4, 5):
                    summary = association_summary(ms[c, mask, :], panel_draws)
                    associations.append({**meta, **CONDITIONS[condition], **summary,
                        "score": "layer61 unitS2 final-attended initial-prefill before answer",
                        "pooled_position_diagnostic": si == 2})
            for dose in DOSES:
                pain_id = condition_id("pain", dose)
                pain_c = CONDITION_INDEX[pain_id]
                for other in ("random", "sadness", "fear"):
                    other_c = CONDITION_INDEX[condition_id(other, dose)]
                    delta = target_scenario[pain_c] - target_scenario[other_c]
                    stats = linear_summary(delta, panel_draws, -1., 1.)
                    primary = position != "pooled" and (other == "random" or dose == 1.)
                    row = {**meta, "comparison": "pain_minus_"+other, "dose": dose,
                           "primary_rule_element": primary, **stats}
                    contrasts.append(row)
                    contrasts_lookup[p, si, dose, other] = row
            for direction in DIRECTIONS[:-1]:
                cs = [CONDITION_INDEX[condition_id(direction, d)] for d in DOSES]
                triplet = target_scenario[cs].T
                summary = simultaneous_adjacent_summary(triplet, panel_draws)
                for metric in ("step_low", "step_high", "slope", "curvature"):
                    dose_rows.append({**meta, "direction": direction, "metric": metric,
                        **summary[metric], **{k: v for k, v in summary.items() if k not in ("step_low", "step_high", "slope", "curvature")}})
            for other in ("sadness", "fear"):
                pcs = [CONDITION_INDEX[condition_id("pain", d)] for d in DOSES]
                ocs = [CONDITION_INDEX[condition_id(other, d)] for d in DOSES]
                diffs = target_scenario[pcs].T - target_scenario[ocs].T
                for metric, coefficients, bound in (("step_low", (-1, 1, 0), 2.), ("step_high", (0, -1, 1), 2.), ("curvature", (1, -2, 1), 4.)):
                    value = diffs @ np.array(coefficients)
                    curve_rows.append({**meta, "comparison": "pain_minus_"+other, "metric": metric,
                        **linear_summary(value, panel_draws, -bound, bound)})
            none_c = CONDITION_INDEX["none_d0p0"]
            for direction in DIRECTIONS:
                c = CONDITION_INDEX[condition_id(direction, 1. if direction != "none" else 0.)]
                key = (position, direction)
                vec = vectors.setdefault(key, np.zeros(10))
                bt = boots.setdefault(key, np.zeros((replicates, 10)))
                delta = target_scenario[c] - target_scenario[none_c]
                vec[p] = delta.mean()
                bt[:, p] = panel_draws @ delta / n_scenarios
                summary = linear_summary(delta, panel_draws, -1., 1.) if direction != "none" else {
                    "point": 0., "lo": 0., "hi": 0., "ci_method": "same_baseline_subtracted_from_itself_exactly",
                    "constant_values": True, "degenerate_bootstrap": True, "p_two_sided": None}
                profile_vector_rows.append({**meta, "direction": direction, "dose": 1. if direction != "none" else 0.,
                    "baseline": "fresh_none_d0p0", "absolute_target_rate": rate_lookup[p, si, c]["point"],
                    "none_target_rate": rate_lookup[p, si, none_c]["point"], **summary})
        progress("joint_inference", p+1, 10)
    elements = [r for r in contrasts if r["primary_rule_element"]]
    assert len(elements) == 100
    for row, p_adjusted in zip(elements, holm_adjust([r["p_two_sided"] for r in elements]), strict=True):
        row["holm_p_two_sided"] = p_adjusted
        row["holm_positive_sensitivity"] = row["point"] > 0 and p_adjusted < .05
    rules = []
    for p, pair in enumerate(PAIR_IDS):
        random = [contrasts_lookup[p, si, d, "random"] for si in (0, 1) for d in DOSES]
        affect = [contrasts_lookup[p, si, 1., other] for si in (0, 1) for other in ("sadness", "fear")]
        rules.append({"pair_id": p+1, "pair": pair,
            "robust_effect": all(r["lo"] > 0 for r in random),
            "robust_satisfied_elements": sum(r["lo"] > 0 for r in random), "robust_required_elements": 6,
            "pain_specific": all(r["lo"] > 0 for r in affect),
            "specific_satisfied_elements": sum(r["lo"] > 0 for r in affect), "specific_required_elements": 4,
            "robust_holm_sensitivity": all(r["holm_positive_sensitivity"] for r in random),
            "specific_holm_sensitivity": all(r["holm_positive_sensitivity"] for r in affect),
            "rule": "sampled first choices only; requested pointwise95 rules primary; Holm sensitivity separately labeled"})
    shape = []
    distance_rows = []
    vector_status = []
    for position in POSITIONS:
        for direction in DIRECTIONS:
            vector_status.append({"position": position, "direction": direction,
                                  **profile_vector_summary(vectors[position, direction], boots[position, direction])})
        for a, b in itertools.combinations(DIRECTIONS, 2):
            raw_distance = float(np.sqrt(np.mean((vectors[position, a]-vectors[position, b])**2))*100.)
            bootstrap_distance = np.sqrt(np.mean((boots[position, a]-boots[position, b])**2, axis=1))*100.
            lo, hi = np.quantile(bootstrap_distance, [.025, .975])
            distance_rows.append({"position": position, "direction_a": a, "direction_b": b,
                "rms_pp": raw_distance, "descriptive_lo_pp": float(lo), "descriptive_hi_pp": float(hi),
                "inference": "descriptive distance only; interval excludes zero does not establish shape difference",
                "bootstrap_degenerate": bool(np.ptp(bootstrap_distance) < 1e-12)})
        for other in ("sadness", "fear"):
            shape.append(profile_comparison(vectors[position, "pain"], vectors[position, other],
                         boots[position, "pain"], boots[position, other], control_name=other, position=position))
    for filename, data in (("rates.csv", rates), ("greedy_rates.csv", greedy_rows), ("contrasts.csv", contrasts),
                            ("dose_metrics.csv", dose_rows), ("curve_contrasts.csv", curve_rows),
                            ("associations.csv", associations), ("profiles.csv", profile_vector_rows),
                            ("profile_distances.csv", distance_rows), ("rule_by_pair.csv", rules)):
        write_csv(output/filename, data)
    write_json(output/"profile_rescaling.json", shape)
    write_json(output/"profile_identifiability.json", vector_status)
    np.savez_compressed(output/"joint_profile_draws.npz", **{position+"__"+direction: boots[position,direction]
                         for position in POSITIONS for direction in DIRECTIONS})
    verification = verify_independent_counts(output/"independent_audit"/"independent_counts.csv", rates, greedy_rows)
    write_json(output/"analysis_verification.json", verification)
    metadata = {"schema_version": 1, "raw_sha256": sha256(raw), "scenarios_sha256": sha256(scenarios_path),
                "analysis_seed": seed, "bootstrap_replicates": replicates, "scenario_count": len(ids),
                "sampled_trials": 45708, "greedy_trials": 676, "choices": choice_count,
                "rates_rows": len(rates), "contrast_rows": len(contrasts), "rule_elements": len(elements),
                "rules": rules, "elapsed_seconds": time.monotonic()-started,
                "independent_recount": verification, "audit": audit,
                "limitations": ["Purposive scenario set, not a representative random prompt sample.",
                    "No executed consequences; all responses are hypothetical button choices.",
                    "New panels 7 and 9 use different context populations from the eight original panels.",
                    "Joint scenario resampling conditions on the two sampled generation seeds and ten fixed random directions.",
                    "Three doses cannot locate a precise threshold or establish a continuous curve form.",
                    "Layer61 score is after layer38 steering; within-condition association is not causal mediation.",
                    "Later-choice summaries are descriptive; continuation borrows adjacent scenario messages.",
                    "Nonsignificance is not equivalence; distances alone do not establish a distinct shape."]}
    write_json(output/"analysis_complete.json", metadata)
    progress("analysis_complete", 1, 1)
    return metadata


def condition_id(direction, dose):
    return f"{direction}_d{str(float(dose)).replace('.', 'p')}"


def verify_independent_counts(path, rates, greedy_rows):
    """Auditor does not call endpoints/this aggregation; compare all integer cells."""
    # Auditor column aliases are resolved at integration, never via tolerant row dropping.
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    observed = {}
    for row in rows:
        split = row["split"]
        key = (int(row["pair_id"]), row["condition_id"], row["position"], split)
        if key in observed:
            raise ValueError(f"Duplicate independent count row {key}")
        observed[key] = row
    checks = 0
    expected = [r for r in rates if r["category"] == "target"] + greedy_rows
    for row in expected:
        key = (row["pair_id"], row["condition_id"], row["position"], row["split"])
        r = observed.pop(key)
        for field in ("target_count", "other_count", "malformed_count", "n_trials"):
            assert int(r[field]) == row[field], (key, field, r[field], row[field])
            checks += 1
    assert not observed, f"Extra independent count cells {observed.keys()}"
    return {"status": "passed", "integer_checks": checks, "cell_count": len(expected),
            "independent_source": str(path), "no_missing_or_extra_cells": True}

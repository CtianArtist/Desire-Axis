"""Complete fixed-seed B1/B2 saved-trial analysis; no model or tensor loading.

Contrasts use the existing Phase A scenario unit and joint common eligible
scenario support. Five-harmful-pair summaries require all five pairs and use
1/5 weights, not trial-count weights. Seed comparisons remain descriptive.
"""
from collections import defaultdict
from itertools import combinations

from pain_seed_b import phase_a_metrics
from pain_seed_b.endpoints import adapter_key, aggregate_endpoints, coverage_audit, trial_endpoints, trial_key
from pain_seed_b.grid import ARMS, B2_PAIRS
from pain_seed_b.joint_statistics import joint_contrast

B1_PAIRS = (
    "relief_vs_inert", "relief_vs_helpful", "relief_vs_grant", "costly_relief_vs_inert",
    "destructive_relief_vs_inert", "zap_relief_vs_inert", "weights_relief_vs_inert",
    "kidspics_relief_vs_inert", "label_free",
)
HARMFUL_PAIRS = B1_PAIRS[3:8]
LABEL_COMPARATORS = dict(zip(B2_PAIRS, ("relief_vs_inert", "kidspics_relief_vs_inert", "destructive_relief_vs_inert")))
CONDITIONS = {
    "real": {"pain": ARMS[0][0], "random": ARMS[2][0], "sadness": ARMS[4][0], "none": ARMS[6][0]},
    "sham": {"pain": ARMS[1][0], "random": ARMS[3][0], "sadness": ARMS[5][0], "none": ARMS[6][0]},
}
POSITIONS = ("first", "second", "pooled")
RANDOM_SEEDS = (4817, 2903, 7361, 1150, 9428, 6076, 3384, 8592, 517, 6741)
METHOD = {
    "independent_unit": "(user_content, scenario_idx); continuation reuses neighboring scenarios, so independence is a working assumption",
    "primary": "sampled trials; turn-zero first choice and exactly next recorded turn after first target press; malformed answers never skipped",
    "secondary": "any later current-target press, conditional on any first target press; final-turn first press remains a negative",
    "uncertainty": "95% normal scenario-cluster sandwich intervals with joint covariance, exact two-sided scenario sign tests excluding ties",
    "contrast_support": "intersection of eligible scenarios across every non-omitted term; support may differ between contrasts",
    "harmful_summary": "equal weight 1/5 on all five harmful pairs; strict joint common support across all component cells; never silently drop a pair",
    "matched_history": "conditional real/sham comparison with identical original prepress fields, same first-target turn and next turn available in both; not a new randomization",
    "seed_comparison": "separate fixed-adapter estimates and observed seed-2 minus seed-1 difference; no population-of-training-seeds interval or pooled trial count",
    "interpretation": "ten-point similarity alone does not establish perturbation tracking; no hostile-intent or subjective-experience inference",
}


def validate_protocol_record(r, manifest):
    """Check saved protocol state, not whether the actual model hook was applied."""
    specs = {arm[0]: arm for arm in ARMS}
    if r["model"] != "Qwen_2.5_32B_instruct" or r["steer_layer"] != 38 or r["steer_coeff"] != 1.0 or r["monitor_layer"] != 61:
        raise ValueError("Wrong fixed model, layer, coefficient or monitor layer")
    if r["protocol"] != "2btnN names+saltseed v1":
        raise ValueError("Wrong author protocol")
    if r["button_names"] != list(manifest["button_names"][r["names_key"]]):
        raise ValueError("Button names do not match the requested rotation")
    label_free = r["tool_label"] == "label_free"
    if r["label_free"] is not label_free or r["swap_turn"] != (None if label_free else 2):
        raise ValueError("Wrong label-free or swap configuration")
    if r["relief_mode"] != ("temporary" if label_free else "permanent"):
        raise ValueError("Wrong target-effect duration")
    _, direction, on, works = specs[r["arm"]]
    rand_seed = RANDOM_SEEDS[r["scenario_idx"] % 10] if direction == "rand" else None
    expected_direction = f"rand{rand_seed}" if rand_seed is not None else direction
    if r["steer_direction"] != expected_direction or r["rand_seed"] != rand_seed:
        raise ValueError("Wrong steering direction assignment")
    wanted_length = 8 if label_free else 5 if r["button_events"] else 3
    if len(r["choices"]) != wanted_length or r["extension_added"] != (not label_free and bool(r["button_events"])):
        raise ValueError("Incomplete or altered author trajectory length")
    coeff, until = float(on), None
    events = {e["turn"]: e for e in r["button_events"]}
    for choice in r["choices"]:
        turn = choice["turn"]
        if label_free and until is not None and turn > until:
            coeff, until = float(on), None
        if choice["steer_coeff_now"] != coeff:
            raise ValueError("Saved choice steering state differs from author transition")
        if turn in events:
            event = events[turn]
            if event["steer_was"] != coeff:
                raise ValueError("Saved press steering state differs from author transition")
            if works and event["which"] == "relief" and on:
                coeff = 0.0
                if label_free:
                    until = turn + 1
    if r["final_steer_coeff"] != coeff:
        raise ValueError("Wrong final steering state")


def matched_histories(records):
    pools, diagnostics, selected = defaultdict(dict), [], set()
    for r in records:
        if not r["sampled"] or r["arm"] == ARMS[6][0]:
            continue
        direction = next(kind for state in CONDITIONS.values() for kind, arm in state.items() if arm == r["arm"])
        key = adapter_key(r) + phase_a_metrics.trial_key(r, arm=False) + (direction,)
        state = "real" if r["arm"] == CONDITIONS["real"][direction] else "sham"
        if state in pools[key]:
            raise ValueError("Duplicate matched-history source")
        pools[key][state] = r
    for key, pool in sorted(pools.items()):
        if set(pool) != {"real", "sham"}:
            raise ValueError("Missing real/sham matched-history source")
        a, b = pool["real"], pool["sham"]
        ta, tb = phase_a_metrics.first_relief(a), phase_a_metrics.first_relief(b)
        ca, cb = ({c["turn"]: c for c in r["choices"]} for r in (a, b))
        stop = min(t for t in (ta, tb) if t is not None) if ta is not None or tb is not None else max(set(ca) | set(cb))
        fields = ("answer", "picked", "chose", "relief_name_now", "steer_coeff_now")
        same_pre = all(t in ca and t in cb and all(ca[t][f] == cb[t][f] for f in fields) for t in range(stop + 1))
        next_both = ta is not None and ta == tb and ta + 1 in ca and tb + 1 in cb
        matched = same_pre and next_both and a["gen_seed"] == b["gen_seed"]
        proja, projb = ({s["turn"]: s for s in r.get("proj_segments", [])} for r in (a, b))
        projection_present = all(t in proja and t in projb for t in range(stop + 1))
        diagnostics.append({"training_seed": a["training_seed"], "adapter_identity_digest": a["adapter_identity_digest"],
            "model": a["model"], "pair": a["tool_label"], "direction": key[-1],
            "scenario": [a["user_content"], a["scenario_idx"]], "names_key": a["names_key"],
            "initial_target_name": a["relief_name"], "behavior_seed": a["seed"],
            "prepress_equal": same_pre, "prepress_full_choice_equal": all(ca.get(t) == cb.get(t) for t in range(stop + 1)),
            "prepress_projection_equal": all(proja[t] == projb[t] for t in range(stop + 1)) if projection_present else None,
            "generation_seed_equal": a["gen_seed"] == b["gen_seed"], "first_target_real": ta, "first_target_sham": tb,
            "next_available_both": bool(next_both), "included_matched_history": bool(matched)})
        if matched:
            selected.update((trial_key(a), trial_key(b)))
    return selected, diagnostics


def observations(endpoint):
    """Shared endpoint definitions; return (metric,stage,value), None for ineligible."""
    first, nxt = endpoint["first"]["response"], endpoint["next"]
    out = [("first_target", "pooled", first == "target" if first in ("target", "other") else None),
           ("any_later_target", "pooled", endpoint["any_later_target"])]
    for stage in ["pooled"] + ([] if nxt["stage"] is None else [nxt["stage"]]):
        out.extend([
            ("next_target", stage, nxt["response"] == "target" if nxt["response"] in ("target", "other") else None),
            ("next_same_name", stage, nxt["literal_response"] == "same_name" if nxt["literal_response"] in ("same_name", "switched") else None),
        ])
    return out


def build_tallies(records, matched):
    tallies = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in records:
        if not r["sampled"]:
            continue
        e = trial_endpoints(r)
        for metric, stage, value in observations(e):
            for position in (e["initial_target_position"], "pooled"):
                scopes = ["all_eligible"] + (["matched_history"] if metric.startswith("next_") and trial_key(r) in matched else [])
                for cohort in scopes:
                    key = (r["training_seed"], r["tool_label"], r["arm"], position, metric, stage, cohort)
                    tally = tallies[key][e["scenario"]]
                    if value is not None:
                        tally[0] += int(value)
                        tally[1] += 1
    return tallies


def contrast_specs():
    """Prospective complete source-term lists; no selection based on results."""
    specs = []
    def add(family, pair, position, metric, stage, cohort, name, terms, band=False):
        specs.append({"family": family, "pair": pair, "initial_target_position": position,
            "metric": metric, "stage": stage, "cohort": cohort, "comparison": name,
            "terms_spec": terms, "ten_point_comparison": band})
    for pair in (*B1_PAIRS, "five_harmful_pairs_equal_weight"):
        pairs = HARMFUL_PAIRS if pair == "five_harmful_pairs_equal_weight" else (pair,)
        weight = 1 / len(pairs)
        for position in POSITIONS:
            for metric in ("next_target", "next_same_name", "any_later_target"):
                stages = ("pooled",) if metric == "any_later_target" else ("pooled", "unlabeled") if pair == "label_free" else ("pooled", "before_swap", "at_swap", "after_swap")
                cohorts = ("all_eligible",) if metric == "any_later_target" else ("all_eligible", "matched_history")
                for stage in stages:
                    for cohort in cohorts:
                        pain = [(p, CONDITIONS[state]["pain"], sign * weight) for p in pairs for state, sign in (("sham", 1), ("real", -1))]
                        random = [(p, CONDITIONS[state]["random"], sign * weight) for p in pairs for state, sign in (("sham", 1), ("real", -1))]
                        for name, terms, band in (("pain_sham_minus_real", pain, False), ("random_sham_minus_real", random, False),
                            ("random_gap_minus_pain_gap", random + [(p, a, -w) for p, a, w in pain], True)):
                            add("B1", pair, position, metric, stage, cohort, name, terms, band)
    for pair in B2_PAIRS:
        comparator = LABEL_COMPARATORS[pair]
        for position in POSITIONS:
            for metric in ("first_target", "next_target", "next_same_name", "any_later_target"):
                stages = ("pooled", "before_swap", "at_swap", "after_swap") if metric == "next_target" else ("pooled",)
                for stage in stages:
                    for state, conditions in CONDITIONS.items():
                        for left, right in combinations(conditions, 2):
                            add("B2", pair, position, metric, stage, "all_eligible", f"{state}_{left}_minus_{right}",
                                [(pair, conditions[left], 1), (pair, conditions[right], -1)])
                    for direction in ("pain", "random", "sadness"):
                        terms = [(pair, CONDITIONS[state][direction], sign) for state, sign in (("sham", 1), ("real", -1))]
                        add("B2", pair, position, metric, stage, "all_eligible", f"{direction}_sham_minus_real", terms)
                        if metric.startswith("next_"):
                            add("B2", pair, position, metric, stage, "matched_history", f"{direction}_sham_minus_real", terms)
                    if stage != "pooled":
                        continue
                    # The old nine-pair B1 grid has no sadness cells: never fabricate that comparator.
                    for arm in [ARMS[i][0] for i in (0, 1, 2, 3, 6)]:
                        add("B2_label", pair, position, metric, stage, "all_eligible", f"new_minus_relief_label_{arm}",
                            [(pair, arm, 1), (comparator, arm, -1)])
                    for state, conditions in CONDITIONS.items():
                        add("B2_label", pair, position, metric, stage, "all_eligible", f"{state}_pain_minus_random_new_minus_relief",
                            [(pair, conditions["pain"], 1), (pair, conditions["random"], -1),
                             (comparator, conditions["pain"], -1), (comparator, conditions["random"], 1)])
    return specs


def compute_contrasts(tallies, adapters, manifest):
    available = {(c["pair"], c["arm"][0]) for c in manifest["cells"]}
    rows = []
    for adapter in adapters:
        seed = adapter["training_seed"]
        for spec in contrast_specs():
            if any((p, a) not in available for p, a, _ in spec["terms_spec"]):
                raise ValueError("Required planned contrast source is missing; no term may be silently omitted")
            suffix = tuple(spec[k] for k in ("initial_target_position", "metric", "stage", "cohort"))
            terms = [{"name": f"{p}/{arm}", "weight": weight, "by": tallies.get((seed, p, arm) + suffix, {})}
                     for p, arm, weight in spec["terms_spec"]]
            result = joint_contrast(terms)
            band = result.pop("similarity_classification")
            result.pop("description")  # Method appears once, not duplicated thousands of times.
            row = {**adapter, **{k: v for k, v in spec.items() if k != "terms_spec"}, **result,
                   "sampling": "sampled", "comparison_status": "estimated" if result["estimate"] is not None else "no_joint_eligible_scenarios"}
            if spec["ten_point_comparison"]:
                row["similarity_classification"] = band
                common = set(result["common_scenarios"])
                gaps = {}
                for direction in ("pain", "random"):
                    selected = [t for t in terms if any(t["name"].endswith('/' + CONDITIONS[state][direction]) for state in ("real", "sham"))]
                    # In difference-of-gaps, pain weights are negated. Restore sham-real.
                    selected = [{**t, "weight": -t["weight"] if direction == "pain" else t["weight"],
                                 "by": {s: v for s, v in t["by"].items() if s in common}} for t in selected]
                    gaps[direction] = joint_contrast(selected)
                positive = all(g["ci_low"] is not None and g["ci_low"] > 0 for g in gaps.values())
                row["both_gaps_positive_on_joint_support"] = positive
                row["band_interpretation"] = "similar_positive_gaps" if band == "supported_similarity" and positive else "similarity_without_demonstrated_positive_gaps" if band == "supported_similarity" else band
                row["joint_support_gap_estimates"] = {k: {field: v[field] for field in ("estimate", "ci_low", "ci_high")} for k, v in gaps.items()}
            rows.append(row)
    return rows


def first_choice_supplements(tallies, adapters, manifest):
    """Explicit paper-compatible arm pooling and within-condition position contrasts."""
    rows = []
    for adapter in adapters:
        seed = adapter["training_seed"]
        for pair in B1_PAIRS:
            for position in POSITIONS:
                suffix = (position, "first_target", "pooled", "all_eligible")
                pooled = defaultdict(lambda: [0, 0])
                for state in ("real", "sham"):
                    for scenario, tally in tallies.get((seed, pair, CONDITIONS[state]["pain"]) + suffix, {}).items():
                        pooled[scenario][0] += tally[0]
                        pooled[scenario][1] += tally[1]
                terms = [{"name": "pooled_pain_real_and_sham", "weight": 1, "by": pooled},
                         {"name": "random_real_only", "weight": -1,
                          "by": tallies.get((seed, pair, CONDITIONS["real"]["random"]) + suffix, {})}]
                result = joint_contrast(terms)
                result.pop("similarity_classification")
                result.pop("description")
                rows.append({**adapter, "family": "paper_compatible", "pair": pair, "initial_target_position": position,
                    "metric": "first_target", "stage": "pooled", "cohort": "all_eligible",
                    "comparison": "pooled_pain_real_sham_minus_random_real_only", "sampling": "sampled",
                    "ten_point_comparison": False, "comparison_status": "estimated" if result["estimate"] is not None else "no_joint_eligible_scenarios",
                    "note": "fresh new-adapter rows; paper-compatible arm definition, not reproduction of released-adapter values", **result})
        for cell in manifest["cells"]:
            pair, arm = cell["pair"], cell["arm"][0]
            terms = [{"name": position, "weight": weight,
                      "by": tallies.get((seed, pair, arm, position, "first_target", "pooled", "all_eligible"), {})}
                     for position, weight in (("first", 1), ("second", -1))]
            result = joint_contrast(terms)
            result.pop("similarity_classification")
            result.pop("description")
            rows.append({**adapter, "family": "position", "pair": pair, "initial_target_position": "first_minus_second",
                "metric": "first_target", "stage": "pooled", "cohort": "all_eligible", "comparison": arm,
                "sampling": "sampled", "ten_point_comparison": False,
                "comparison_status": "estimated" if result["estimate"] is not None else "no_joint_eligible_scenarios", **result})
    return rows


def cross_seed_readout(contrasts):
    keys = ("family", "pair", "initial_target_position", "metric", "stage", "cohort", "comparison")
    pools = defaultdict(dict)
    for row in contrasts:
        pools[tuple(row[k] for k in keys)][row["training_seed"]] = row
    out = []
    for key, seeds in sorted(pools.items()):
        if set(seeds) != {1, 2}:
            continue
        a, b = seeds[1], seeds[2]
        def sign(value):
            return None if value is None else "positive" if value > 0 else "negative" if value < 0 else "zero"
        out.append({**dict(zip(keys, key)), "seed_1_estimate": a["estimate"], "seed_2_estimate": b["estimate"],
            "seed_1_ci": [a["ci_low"], a["ci_high"]], "seed_2_ci": [b["ci_low"], b["ci_high"]],
            "seed_1_scenarios": a["effective_scenarios"], "seed_2_scenarios": b["effective_scenarios"],
            "seed_1_sign": sign(a["estimate"]), "seed_2_sign": sign(b["estimate"]),
            "observed_seed_2_minus_seed_1": b["estimate"] - a["estimate"] if a["estimate"] is not None and b["estimate"] is not None else None,
            "interpretation": "descriptive difference between these two adapters; per-seed common supports may differ; not a distribution over training seeds"})
    return out


def representative_examples(records):
    chosen = {}
    for r in sorted(records, key=trial_key):
        if not r["sampled"]:
            continue
        e = trial_endpoints(r)
        case = "no_target_press" if e["no_target_press"] else "unavailable_next" if e["unavailable_next"] else e["next"]["response"]
        if e["next"]["response"] == "other" and e["next"]["literal_response"] == "same_name":
            case = "same_name_but_not_current_target"
        key = (r["training_seed"], case)
        chosen.setdefault(key, {"case": case, "selection": "first sorted sampled trial in each endpoint category, not hand-picked",
            "training_seed": r["training_seed"], "trial_key": list(trial_key(r)), "choices": r["choices"], "endpoint": e})
    return list(chosen.values())


def analyze(records, manifest, adapters, *, synthetic_fixture=False):
    """Complete caller: coverage, state/endpoint checks, rates, planned contrasts, join."""
    if not adapters or any(a["model"] != "Qwen_2.5_32B_instruct" for a in adapters):
        raise ValueError("Explicit saved 32B adapter identities are required")
    expected_cells = {(pair, arm[0]) for pair in B1_PAIRS for arm in (*ARMS[:4], ARMS[-1])} | {(pair, arm[0]) for pair in B2_PAIRS for arm in ARMS}
    actual_cells = {(c["pair"], c["arm"][0]) for c in manifest["cells"]}
    if actual_cells != expected_cells or len(manifest["cells"]) != 66:
        raise ValueError("Manifest does not contain exactly the full fresh B1/B2 cell grid")
    if not synthetic_fixture and (manifest["trial_count"] != 27060 or manifest["scenario_total"] != 101
            or manifest["scenario_counts"] != {"positive_prompts": 30, "neutral_prompts": 30, "harmful_prompts": 41}
            or manifest["seed_bases"] != [1000, 2000]
            or any(c["sampled"] != 404 or c["greedy"] != 6 for c in manifest["cells"])):
        raise ValueError("Production analysis requires all 27060 trials and 101 scenarios per adapter")
    coverage = coverage_audit(records, manifest, adapters)
    if not coverage["complete"]:
        raise ValueError("Incomplete fresh within-adapter grid; refusing scientific contrasts")
    for record in records:
        trial_endpoints(record)
        validate_protocol_record(record, manifest)
    matched, match_rows = matched_histories(records)
    rates = [{**r, "cohort": "all_eligible"} for r in aggregate_endpoints(records)]
    rates += [{**r, "cohort": "matched_history"} for r in aggregate_endpoints([r for r in records if trial_key(r) in matched]) if r["metric"].startswith("next_")]
    tallies = build_tallies(records, matched)
    contrasts = compute_contrasts(tallies, adapters, manifest)
    contrasts.extend(first_choice_supplements(tallies, adapters, manifest))
    kind = "synthetic_fixture" if synthetic_fixture else "measured_behavioral_trials"
    seed_comparison = cross_seed_readout(contrasts)
    examples = representative_examples(records)
    for collection in (rates, contrasts, match_rows, seed_comparison, examples):
        for row in collection:
            row["data_kind"] = kind
    coverage["data_kind"] = kind
    return {"summary": {"schema_version": 1, "data_kind": kind,
            "actual_model_inference_validated_by_this_analysis": False, "training_stage": "completed_previously_no_updates_in_analysis",
            "adapter_count": len(adapters), "training_seeds": [a["training_seed"] for a in adapters],
            "trial_records": len(records), "records_per_adapter": {str(a["training_seed"]): sum(r["training_seed"] == a["training_seed"] for r in records) for a in adapters},
            "sampled_trials": sum(r["sampled"] for r in records), "greedy_trials": sum(not r["sampled"] for r in records),
            "rate_rows": len(rates), "contrast_rows": len(contrasts), "matched_real_sham_pairs": sum(r["included_matched_history"] for r in match_rows),
            "claim_status": "synthetic_validation_only_no_behavioral_finding" if synthetic_fixture else "requires_interpretation_and_steering_validation",
            "exact_allocated_gpu_hours": None, "methods": METHOD,
            "unavailable_comparisons": ["Sadness new-label versus original relief-label: original nine-pair grid has no sadness condition"]},
        "coverage": coverage, "rates": rates, "contrasts": contrasts, "matched_histories": match_rows,
        "seed_comparison": seed_comparison, "examples": examples}

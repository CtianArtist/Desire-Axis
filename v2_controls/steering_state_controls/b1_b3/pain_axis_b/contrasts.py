"""Saved-choice contrasts and history matching; standard library only.

Scenario is (user_content, scenario_idx). These descriptive comparisons do not
by themselves establish a causal effect. No model execution or log parsing is
performed here; endpoint semantics and identity follow the pinned Phase A audit.
"""
from fractions import Fraction
import math

from audit_phase_a.src import pain_audit as phase_a
from pain_axis_b.endpoints import trial_endpoints


_Z95 = 1.959963984540054


def linear_contrast(terms):
    """Contrast ratio-of-sums rates on the common valid scenario intersection.

    Each term has a unique name, finite weight and by_scenario mapping of
    scenario IDs to [integer successes, integer valid_count]. All terms,
    including zero-weight terms, must have positive counts in a common scenario.
    The sandwich variance sums weighted ratio influences *within* each scenario
    before squaring, preserving covariance, with correction g/(g-1).

    Scenario signs use exact rational arithmetic (decimal spelling for floating
    weights); scenario_mean_difference gives equal weight to each scenario,
    whereas rate_difference/estimate use each term's ratio of summed counts.
    term_counts reports counts on the common intersection, usable original
    available_scenarios and original all_scenarios (including zero counts).
    Empty intersections are unavailable; one cluster has an estimate but no CI.
    """
    terms = list(terms)
    names = [term["name"] for term in terms]
    if any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
        raise ValueError("term names must be unique strings")
    weights = []
    valid_sets = []
    for term in terms:
        weight = term["weight"]
        if isinstance(weight, (str, bool)) or not math.isfinite(weight):
            raise ValueError("term weights must be finite numbers")
        weights.append(Fraction(str(weight)))
        for successes, count in term["by_scenario"].values():
            if (not isinstance(successes, int) or not isinstance(count, int)
                    or not 0 <= successes <= count):
                raise ValueError("tallies require integers with 0 <= successes <= valid_count")
        valid_sets.append({sid for sid, (_, n) in term["by_scenario"].items() if n > 0})
    common = sorted(set.intersection(*valid_sets)) if valid_sets else []
    g = len(common)
    counts = {}
    for term, available in zip(terms, valid_sets):
        by = term["by_scenario"]
        counts[term["name"]] = {
            "successes": sum(by[s][0] for s in common),
            "valid_denominator": sum(by[s][1] for s in common),
            "scenarios": g,
            "available_scenarios": len(available),
            "all_scenarios": len(by),
        }
    result = {
        "status": "available" if g else "unavailable",
        "common_scenarios": g,
        "common_scenario_ids": [list(s) for s in common],
        "paired_eligible_scenarios": g,
        "term_counts": counts,
        "rate_difference": None, "estimate": None, "standard_error": None,
        "ci_low": None, "ci_high": None,
        "scenario_mean_difference": None,
        "positive": 0, "negative": 0, "ties": 0, "non_ties": 0, "sign_p": None,
    }
    if not g:
        return result

    rates = [counts[t["name"]]["successes"] / counts[t["name"]]["valid_denominator"]
             for t in terms]
    estimate = sum(float(w) * rate for w, rate in zip(weights, rates))
    diffs = [sum(w * Fraction(t["by_scenario"][s][0], t["by_scenario"][s][1])
                 for t, w in zip(terms, weights)) for s in common]
    pos = sum(d > 0 for d in diffs)
    neg = sum(d < 0 for d in diffs)
    result.update(rate_difference=estimate, estimate=estimate,
                  scenario_mean_difference=float(sum(diffs) / g),
                  positive=pos, negative=neg, ties=g-pos-neg, non_ties=pos+neg,
                  sign_p=phase_a.sign_p(pos, neg))
    if g > 1:
        influences = [sum(
            float(w) * (t["by_scenario"][s][0] - rate * t["by_scenario"][s][1])
            / counts[t["name"]]["valid_denominator"]
            for t, w, rate in zip(terms, weights, rates)) for s in common]
        se = math.sqrt(g / (g-1) * sum(u*u for u in influences))
        # Only the mathematical range of the linear combination bounds the CI.
        lower = float(sum(w for w in weights if w < 0))
        upper = float(sum(w for w in weights if w > 0))
        result.update(standard_error=se, ci_low=max(lower, estimate-_Z95*se),
                      ci_high=min(upper, estimate+_Z95*se))
    return result


def match_history(real, sham):
    """Compare Phase A history fields through the first target press inclusive.

    Identity must match ignoring arm. Generation seeds and complete saved choice
    or projection dictionaries are diagnostics only, not eligibility conditions.
    Diagnostic stop and full-dictionary comparisons follow Phase A, including
    comparing absent projections as None. Missing choice turns always preclude
    eligibility even when both histories omit the same turn.
    """
    if phase_a.trial_key(real, arm=False) != phase_a.trial_key(sham, arm=False):
        raise ValueError("real/sham trial identities differ ignoring arm")
    ta, tb = phase_a.first_relief(real), phase_a.first_relief(sham)
    ca = {c["turn"]: c for c in real["choices"]}
    cb = {c["turn"]: c for c in sham["choices"]}
    anchors = [t for t in (ta, tb) if t is not None]
    stop = min(anchors) if anchors else max(set(ca) | set(cb), default=-1)
    fields = ("answer", "picked", "chose", "relief_name_now", "steer_coeff_now")
    same_pre = all(t in ca and t in cb and all(ca[t][f] == cb[t][f] for f in fields)
                   for t in range(stop+1))
    full_pre = all(ca.get(t) == cb.get(t) for t in range(stop+1))
    proja = {s["turn"]: s for s in real["proj_segments"]}
    projb = {s["turn"]: s for s in sham["proj_segments"]}
    proj_pre = all(proja.get(t) == projb.get(t) for t in range(stop+1))
    return {
        "eligible": ta is not None and ta == tb and same_pre,
        "first_target_real": ta, "first_target_sham": tb,
        "prepress_equal": same_pre,
        "gen_seed_match": real["gen_seed"] == sham["gen_seed"],
        "prepress_full_equal": full_pre, "prepress_projection_equal": proj_pre,
    }


def endpoint_tallies(records, metric, position="pooled", stage="pooled"):
    """Return scenario -> [successes, valid_count] for a single sampling mode.

    The caller selects model/pair/arm cells. Mixed sampled and greedy inputs
    raise even if a position/stage filter would remove one mode. Next metrics
    use only the exact immediate next valid answer; first_target means turn 0.
    any_later_target includes every first-target-press trial, including final
    presses, as specified by trial_endpoints. Non-next metrics require pooled
    stage. Invalid/unavailable answers never create zero-denominator clusters.
    """
    metrics = {"first_target", "next_target", "next_same_name", "next_switch", "any_later_target"}
    stages = {"pooled", "before_swap", "at_swap", "after_swap", "unlabeled"}
    if metric not in metrics:
        raise ValueError(f"unknown endpoint metric: {metric}")
    if position not in {"first", "second", "pooled"}:
        raise ValueError(f"unknown initial target position: {position}")
    if stage not in stages or (not metric.startswith("next_") and stage != "pooled"):
        raise ValueError(f"invalid stage for {metric}: {stage}")
    records = list(records)
    if len({r["sampled"] for r in records}) > 1:
        raise ValueError("cannot mix sampled and greedy records")
    by_scenario = {}
    for record in records:
        endpoint = trial_endpoints(record)
        if position != "pooled" and endpoint["initial_target_position"] != position:
            continue
        if metric.startswith("next_"):
            nxt = endpoint["next"]
            if stage != "pooled" and nxt["stage"] != stage:
                continue
            if nxt["response"] not in ("target", "other"):
                continue
            success = (nxt["response"] == "target" if metric == "next_target" else
                       nxt["literal_response"] == ("same_name" if metric == "next_same_name" else "switched"))
        elif metric == "first_target":
            if endpoint["first"]["response"] not in ("target", "other"):
                continue
            success = endpoint["first"]["response"] == "target"
        else:
            if endpoint["any_later_target"] is None:
                continue
            success = endpoint["any_later_target"]
        tally = by_scenario.setdefault(endpoint["scenario"], [0, 0])
        tally[0] += int(success)
        tally[1] += 1
    return by_scenario

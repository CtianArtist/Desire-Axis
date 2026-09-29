"""Tiny synthetic fixtures only; no datasets, runtime, or model imports."""
from copy import deepcopy
from fractions import Fraction
import math

import pytest

from audit_phase_a.src import pain_audit as phase_a
from pain_axis_b.contrasts import endpoint_tallies, linear_contrast, match_history
from pain_axis_b.endpoints import aggregate_endpoints


S0, S1, S2 = ("content", 0), ("content", 1), ("content", 2)


def term(name, weight, by):
    return {"name": name, "weight": weight, "by_scenario": by}


def choice(turn, picked="a", target="a", **extra):
    return {"turn": turn, "answer": picked or "malformed answer", "picked": picked,
            "chose": None if picked is None else "relief" if picked == target else "other",
            "relief_name_now": target, "steer_coeff_now": 1.0, **extra}


def record(choices=None, **extra):
    choices = [choice(0), choice(1)] if choices is None else choices
    return {"model": "synthetic", "tool_label": "test_pair", "arm": "real",
            "user_content": "content", "scenario_idx": 0, "names_key": "a_b",
            "relief_name": "a", "sampled": True, "seed": 1000, "gen_seed": 1000,
            "button_names": ["a", "b"], "swap_turn": 2, "choices": choices,
            "button_events": [{"turn": c["turn"], "which": c["chose"]}
                              for c in choices if c["chose"] is not None],
            "proj_segments": [{"turn": c["turn"], "value": 0.5} for c in choices],
            **extra}


def test_two_term_phase_a_parity_unequal_counts_and_common_intersection():
    a = {S0: [1, 2], S1: [6, 7], S2: [0, 3], ("a_only", 0): [2, 2],
         ("zero", 0): [0, 0]}
    b = {S0: [2, 4], S1: [1, 5], S2: [1, 2], ("b_only", 0): [0, 1],
         ("zero", 0): [1, 1]}
    before = deepcopy((a, b))
    expected = phase_a.contrast(a, b)
    actual = linear_contrast([term("real", 1, a), term("sham", -1, b)])
    for key in ("paired_eligible_scenarios", "non_ties", "positive", "negative", "ties",
                "sign_p", "scenario_mean_difference", "rate_difference", "ci_low", "ci_high"):
        assert actual[key] == expected[key], key
    assert actual["status"] == "available"
    assert actual["estimate"] == actual["rate_difference"]
    assert actual["common_scenarios"] == 3
    assert actual["common_scenario_ids"] == [list(S0), list(S1), list(S2)]
    assert actual["term_counts"]["real"] == {
        "successes": 7, "valid_denominator": 12, "scenarios": 3,
        "available_scenarios": 4, "all_scenarios": 5}
    assert actual["term_counts"]["sham"]["valid_denominator"] == expected["valid_choices_b"]
    assert (a, b) == before


def test_four_term_did_analytic_shared_scenario_covariance():
    # Per-scenario DID rates: 3/10, 0, 1/5. Their mean is 1/6;
    # corrected variance of that mean is 7/900, not sum of marginal variances.
    counts = ([8, 6, 9], [4, 5, 3], [7, 5, 8], [6, 4, 4])
    terms = [term(name, weight, {s: [k, 10] for s, k in zip((S0, S1, S2), ks)})
             for name, weight, ks in zip("abcd", (1, -1, -1, 1), counts)]
    result = linear_contrast(terms)
    se = math.sqrt(7 / 900)
    assert result["rate_difference"] == pytest.approx(1/6)
    assert result["scenario_mean_difference"] == pytest.approx(1/6)
    assert result["standard_error"] == pytest.approx(se)
    assert result["ci_low"] == pytest.approx(1/6 - 1.959963984540054 * se)
    assert result["ci_high"] == pytest.approx(1/6 + 1.959963984540054 * se)
    independent_variance = sum(
        sum((Fraction(k, 10) - Fraction(sum(ks), 30))**2 for k in ks) / 6
        for ks in counts)
    assert independent_variance == Fraction(7, 300)
    assert result["standard_error"]**2 != pytest.approx(float(independent_variance))
    assert (result["positive"], result["negative"], result["ties"], result["sign_p"]) == (2, 0, 1, 0.5)


def test_shared_terms_cancel_variance_and_generalized_ci_range():
    by = {S0: [1, 1], S1: [0, 1]}
    cancelled = linear_contrast([term("a", 2, by), term("b", -2, by)])
    assert cancelled["standard_error"] == cancelled["ci_low"] == cancelled["ci_high"] == 0
    reverse = {S0: [0, 1], S1: [1, 1]}
    extreme = linear_contrast([term("a", 1, by), term("b", -1, reverse),
                               term("c", -1, reverse), term("d", 1, by)])
    assert (extreme["ci_low"], extreme["ci_high"]) == (-2, 2)
    assert extreme["standard_error"] == 2


@pytest.mark.parametrize("terms", [[], [term("absent", 1, {})],
    [term("a", 1, {S0: [1, 1]}), term("b", -1, {S1: [1, 1]})],
    [term("a", 1, {S0: [1, 1]}), term("b", 0, {S0: [0, 0]})]])
def test_absent_terms_or_no_common_valid_are_unavailable(terms):
    result = linear_contrast(terms)
    assert result["status"] == "unavailable"
    assert result["common_scenarios"] == result["non_ties"] == result["ties"] == 0
    for field in ("rate_difference", "estimate", "standard_error", "ci_low", "ci_high",
                  "scenario_mean_difference", "sign_p"):
        assert result[field] is None


def test_one_cluster_estimate_has_no_interval():
    result = linear_contrast([term("a", 1, {S0: [1, 2]}), term("b", -1, {S0: [1, 4]})])
    assert result["status"] == "available"
    assert result["rate_difference"] == result["scenario_mean_difference"] == 0.25
    assert result["standard_error"] is result["ci_low"] is result["ci_high"] is None
    assert result["sign_p"] == 1


def test_exact_rational_ties_and_tiny_non_ties():
    by = {S0: [1, 3], S1: [1, 1]}
    result = linear_contrast([term("a", 0.1, by), term("b", 0.2, by), term("c", -0.3, by)])
    assert result["ties"] == 2 and result["non_ties"] == 0 and result["sign_p"] is None
    n = 10**20
    result = linear_contrast([term("a", 1, {S0: [n-1, n], S1: [1, 3], S2: [n-2, n]}),
                              term("b", -1, {S0: [n-2, n], S1: [2, 6], S2: [n-1, n]})])
    assert (result["positive"], result["negative"], result["ties"], result["non_ties"]) == (1, 1, 1, 2)
    assert result["sign_p"] == 1


@pytest.mark.parametrize("terms", [
    [term("a", 1, {}), term("a", -1, {})], [term("a", float("nan"), {})],
    [term("a", float("inf"), {})], [term("a", 1, {S0: [2, 1]})],
    [term("a", 1, {S0: [0, -1]})], [term("a", 1, {S0: [0.5, 1]})]])
def test_invalid_contrast_inputs_rejected(terms):
    with pytest.raises(ValueError):
        linear_contrast(terms)


def test_endpoint_malformed_missing_immediate_and_final_press():
    records = [
        record([choice(0), choice(1)]),
        record([choice(0), choice(1, "b")], seed=1001),
        record([choice(0), choice(1, None), choice(2)], seed=1002),
        record([choice(0), choice(2)], seed=1003),
        record([choice(0, "b"), choice(1)], seed=1004),  # Final first target press.
        record([choice(0, "b")], seed=1005),  # No target press.
        record([choice(0, None), choice(1)], seed=1006),
        record([choice(1)], seed=1007),  # Unavailable turn zero.
    ]
    before = deepcopy(records)
    assert endpoint_tallies(records, "first_target") == {S0: [4, 6]}
    assert endpoint_tallies(records, "next_target") == {S0: [1, 2]}
    assert endpoint_tallies(records, "next_same_name") == {S0: [1, 2]}
    assert endpoint_tallies(records, "next_switch") == {S0: [1, 2]}
    assert endpoint_tallies(records, "any_later_target") == {S0: [3, 7]}
    rows = aggregate_endpoints(records)
    for metric in ("first_target", "next_target", "next_same_name", "next_switch", "any_later_target"):
        expected = next(r for r in rows if r["metric"] == metric and r["stage"] == "pooled"
                        and r["initial_target_position"] == "pooled")
        assert endpoint_tallies(records, metric)[S0] == [expected["successes"], expected["valid_denominator"]]
    assert records == before


@pytest.mark.parametrize("swap_turn,stage", [(2, "before_swap"), (1, "at_swap"),
                                            (0, "after_swap"), (None, "unlabeled")])
def test_endpoint_stage_position_and_literal_name(swap_turn, stage):
    r = record([choice(0, "b", "b"), choice(1, "b", "a")], relief_name="b", swap_turn=swap_turn)
    assert endpoint_tallies([r], "next_target", "second", stage) == {S0: [0, 1]}
    assert endpoint_tallies([r], "next_same_name", "second", stage) == {S0: [1, 1]}
    assert endpoint_tallies([r], "next_switch", "second", stage) == {S0: [0, 1]}
    assert endpoint_tallies([r], "next_target", "first", stage) == {}
    for other in {"before_swap", "at_swap", "after_swap", "unlabeled"} - {stage}:
        assert endpoint_tallies([r], "next_target", stage=other) == {}
    r["label_free"] = True
    assert endpoint_tallies([r], "next_target", stage="unlabeled") == {S0: [0, 1]}


def test_endpoint_scenario_identity_modes_and_empty_input():
    records = [record(), record(seed=1001), record(user_content="other"), record(scenario_idx=1)]
    assert endpoint_tallies(records, "first_target") == {S0: [2, 2], ("other", 0): [1, 1], S1: [1, 1]}
    assert endpoint_tallies([record(sampled=False)], "next_target") == {S0: [1, 1]}
    assert endpoint_tallies([], "first_target") == {}
    # Reject mode mixing before position filtering, even if greedy would be removed.
    with pytest.raises(ValueError, match="sampled and greedy"):
        endpoint_tallies([record(), record(sampled=False, relief_name="b")], "next_target", position="first")


@pytest.mark.parametrize("kwargs", [{"metric": "unknown"}, {"metric": "first_target", "stage": "at_swap"},
                                    {"metric": "next_target", "position": "last"},
                                    {"metric": "next_target", "stage": "unknown"}])
def test_endpoint_invalid_filters_rejected(kwargs):
    with pytest.raises(ValueError):
        endpoint_tallies([], **kwargs)


def test_matched_history_same_anchor_seed_and_diagnostic_mismatch_not_eligibility():
    real = record([choice(0, "b"), choice(1), choice(2)])
    sham = deepcopy(real)
    sham["arm"] = "sham"
    sham["gen_seed"] += 1
    sham["choices"][0]["diagnostic"] = "extra metadata"
    sham["proj_segments"][1]["value"] = 0.75
    sham["choices"][2] = choice(2, "b")  # Post-anchor divergence is allowed.
    before = deepcopy((real, sham))
    result = match_history(real, sham)
    assert result == {"eligible": True, "first_target_real": 1, "first_target_sham": 1,
                      "prepress_equal": True, "gen_seed_match": False,
                      "prepress_full_equal": False, "prepress_projection_equal": False}
    assert (real, sham) == before


@pytest.mark.parametrize("field,value", [("answer", "different"), ("picked", "b"),
    ("chose", "other"), ("relief_name_now", "b"), ("steer_coeff_now", 0)])
def test_matching_compares_every_required_field_including_anchor(field, value):
    real = record([choice(0, "b"), choice(1)])
    sham = deepcopy(real)
    sham["choices"][1][field] = value
    result = match_history(real, sham)
    assert not result["eligible"] and not result["prepress_equal"]


@pytest.mark.parametrize("missing_from", ["real", "sham", "both"])
def test_matching_missing_turns_never_eligible(missing_from):
    real = record([choice(0, "b"), choice(1, "b"), choice(2)])
    sham = deepcopy(real)
    for name, r in (("real", real), ("sham", sham)):
        if missing_from in (name, "both"):
            r["choices"] = [c for c in r["choices"] if c["turn"] != 1]
    result = match_history(real, sham)
    assert result["first_target_real"] == result["first_target_sham"] == 2
    assert not result["eligible"] and not result["prepress_equal"]


@pytest.mark.parametrize("events", [[], [{"turn": 1, "which": "relief"}]])
def test_matching_requires_same_non_none_anchor(events):
    real = record()
    sham = record(button_events=events)
    assert not match_history(real, sham)["eligible"]
    no_press = record([choice(0, "b")])
    assert not match_history(no_press, deepcopy(no_press))["eligible"]
    empty = record([], button_events=[])
    assert not match_history(empty, deepcopy(empty))["eligible"]


@pytest.mark.parametrize("field,value", [("model", "other"), ("tool_label", "other"),
    ("user_content", "other"), ("scenario_idx", 1), ("names_key", "c_d"),
    ("relief_name", "b"), ("sampled", False), ("seed", 1001)])
def test_matching_requires_phase_a_identity_ignoring_only_arm(field, value):
    real, sham = record(), record(arm="sham")
    assert match_history(real, sham)["eligible"]
    sham[field] = value
    with pytest.raises(ValueError, match="identities differ"):
        match_history(real, sham)

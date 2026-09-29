"""Tiny synthetic tests only; no saved scientific data or model access."""

from copy import deepcopy
from fractions import Fraction
import math

import pytest

from pain_seed_b.joint_statistics import classify_similarity, joint_contrast
from pain_seed_b.phase_a_metrics import contrast, sign_p


def term(name, weight, counts):
    return {"name": name, "weight": weight, "by": counts}


def by(*counts):
    return {(i,): list(pair) for i, pair in enumerate(counts)}


@pytest.mark.parametrize("a,b", [
    (by((1, 2), (4, 5), (0, 3)), by((0, 2), (2, 5), (3, 3))),
    (by((0, 0), (1, 3), (2, 2)), by((0, 1), (1, 1))),
    (by((1, 2), (1, 2)), by((1, 2), (1, 2))),
    (by((1, 2)), by((0, 3))),
    ({}, by((1, 2))),
])
def test_two_terms_match_phase_a_exactly(a, b):
    expected = contrast(a, b)
    actual = joint_contrast([term("a", 1, a), term("b", -1, b)])
    for old, new in [
        ("rate_difference", "estimate"),
        ("paired_eligible_scenarios", "effective_scenarios"),
        ("ci_low", "ci_low"), ("ci_high", "ci_high"),
        ("positive", "positive"), ("negative", "negative"), ("ties", "ties"),
        ("non_ties", "non_ties"), ("sign_p", "sign_p"),
        ("scenario_mean_difference", "scenario_mean_difference"),
    ]:
        assert actual[new] == expected[old]
    for name, summary in zip(("a", "b"), actual["terms"]):
        assert summary["successes"] == expected[f"successes_{name}"]
        assert summary["denominator"] == expected[f"valid_choices_{name}"]
        assert actual["available_scenarios"][name] == expected[f"available_scenarios_{name}"]


def test_four_term_joint_covariance_not_independent_se_sum():
    varying = by((0, 1), (1, 1), (1, 1), (0, 1))
    zero = by((0, 1), (0, 1), (0, 1), (0, 1))
    first_gap = joint_contrast([term("a", 1, varying), term("b", -1, zero)])
    second_gap = joint_contrast([term("c", 1, varying), term("d", -1, zero)])
    joint = joint_contrast([
        term("a", 1, varying), term("b", -1, zero),
        term("c", -1, varying), term("d", 1, zero),
    ])
    independent_se = math.hypot(first_gap["se"], second_gap["se"])
    assert independent_se > 0.4
    assert joint["se"] == 0
    assert joint["estimate"] == joint["ci_low"] == joint["ci_high"] == 0
    assert joint["ties"] == 4
    assert joint["sign_p"] is None


def test_weighted_five_harm_pair_average_twenty_terms():
    terms = []
    pair_estimates = []
    exact_diffs = [Fraction(0) for _ in range(4)]
    for pair in range(5):
        pair_terms = []
        for source, sign in enumerate((1, -1, -1, 1)):
            counts = by(*[((pair + source + scenario) % 7, 8) for scenario in range(4)])
            terms.append(term(f"pair{pair}_source{source}", sign * 0.2, counts))
            pair_terms.append(term(str(source), sign, counts))
            for scenario in range(4):
                exact_diffs[scenario] += Fraction(sign, 5) * Fraction(*counts[(scenario,)])
        pair_estimates.append(joint_contrast(pair_terms)["estimate"])
    result = joint_contrast(terms)
    assert len(result["terms"]) == 20
    assert result["effective_scenarios"] == 4
    assert result["estimate"] == pytest.approx(sum(pair_estimates) / 5)
    # Equal denominators let the independent hand calculation use the SE of
    # the four exact scenario-level contrasts, not a sum of marginal SEs.
    mean = sum(exact_diffs) / 4
    expected_se = math.sqrt(float(sum((d - mean) ** 2 for d in exact_diffs) / 12))
    assert result["se"] == pytest.approx(expected_se)
    assert result["scenario_mean_difference"] == float(mean)
    assert result["positive"] == sum(d > 0 for d in exact_diffs)
    assert result["negative"] == sum(d < 0 for d in exact_diffs)
    assert result["ties"] == sum(d == 0 for d in exact_diffs)
    assert result["theoretical_bounds"] == pytest.approx([-2, 2])


def test_unequal_denominators_and_joint_eligibility_including_zero_weight():
    terms = [
        term("a", 1, by((1, 1), (0, 9), (8, 8), (0, 0))),
        term("b", -1, by((0, 9), (1, 1), (0, 8), (1, 1))),
        term("required", 0, by((1, 1), (1, 1), (0, 0))),
    ]
    original = deepcopy(terms)
    result = joint_contrast(terms)
    assert terms == original
    assert result["common_scenarios"] == [(0,), (1,)]
    assert result["effective_scenarios"] == 2
    assert result["available_scenarios"] == {"a": 4, "b": 4, "required": 3}
    assert [t["eligible_scenarios"] for t in result["terms"]] == [3, 4, 2]
    assert [t["successes"] for t in result["terms"]] == [1, 1, 2]
    assert [t["denominator"] for t in result["terms"]] == [10, 10, 2]
    assert result["estimate"] == 0
    assert result["se"] == pytest.approx(0.36)
    assert (result["positive"], result["negative"], result["ties"]) == (1, 1, 0)
    assert result["sign_p"] == 1


def test_ratio_of_sums_is_not_scenario_mean():
    result = joint_contrast([
        term("a", 1, by((1, 1), (0, 9))),
        term("b", -1, by((0, 1), (0, 1))),
    ])
    assert result["estimate"] == 0.1
    assert result["scenario_mean_difference"] == 0.5


@pytest.mark.parametrize("weights", [
    [0.1, 0.2, -0.3],
    [Fraction(1, 3), Fraction(2, 3), -1],
])
def test_exact_rational_ties_not_float_residuals(weights):
    result = joint_contrast([term(str(i), w, by((1, 1), (1, 2)))
                             for i, w in enumerate(weights)])
    assert result["positive"] == result["negative"] == result["non_ties"] == 0
    assert result["ties"] == 2
    assert result["sign_p"] is None
    assert result["scenario_mean_difference"] == 0


def test_exact_signs_and_nontrivial_sign_p_despite_rounded_float_rates():
    n = 10 ** 18
    a = by(*([(n, n)] * 6 + [(n - 1, n), (n, n)]))
    b = by(*([(n - 1, n)] * 6 + [(n, n), (n, n)]))
    result = joint_contrast([term("a", 0.2, a), term("b", -0.2, b)])
    assert result["positive"] == 6
    assert result["negative"] == result["ties"] == 1
    assert result["sign_p"] == sign_p(6, 1) == 0.125
    assert result["scenario_mean_difference"] == float(Fraction(1, 8 * n))


@pytest.mark.parametrize("weight", [1e-200, 1e200])
def test_finite_extreme_weights_do_not_overflow_or_underflow_se(weight):
    result = joint_contrast([term("a", weight, by((0, 1), (1, 1)))])
    assert math.isfinite(result["se"])
    assert result["se"] / weight == pytest.approx(0.5)
    assert result["estimate"] / weight == pytest.approx(0.5)


def test_one_cluster_has_point_and_sign_but_no_uncertainty():
    result = joint_contrast([term("a", 1, by((1, 1))), term("b", -1, by((0, 1)))])
    assert result["estimate"] == 1
    assert result["effective_scenarios"] == result["positive"] == 1
    assert result["sign_p"] == sign_p(1, 0)
    assert result["se"] is result["ci_low"] is result["ci_high"] is None
    assert result["similarity_classification"] == "unavailable"


@pytest.mark.parametrize("empty_source", [{}, {(9,): [1, 1]}, {(0,): [0, 0]}])
def test_empty_intersection(empty_source):
    result = joint_contrast([
        term("a", 1, by((1, 1))), term("b", -1, empty_source),
    ])
    assert result["common_scenarios"] == []
    assert result["effective_scenarios"] == 0
    assert result["estimate"] is result["scenario_mean_difference"] is None
    assert result["se"] is result["ci_low"] is result["ci_high"] is None
    assert result["sign_p"] is None
    assert result["positive"] == result["negative"] == result["ties"] == 0
    assert all(t["successes"] == t["denominator"] == 0 for t in result["terms"])
    assert all(t["rate"] is None for t in result["terms"])


def test_ci_clipped_to_linear_contrast_bounds():
    result = joint_contrast([
        term("a", 2, by((1, 1), (0, 1))),
        term("b", -3, by((0, 1), (1, 1))),
    ])
    assert result["theoretical_bounds"] == [-3, 2]
    assert result["estimate"] == -0.5
    assert result["se"] == 2.5
    assert result["ci_low"] == -3
    assert result["ci_high"] == 2


def test_zero_gaps_do_not_claim_perturbation_tracking_or_report_verdict():
    result = joint_contrast([term(str(i), w, by((0, 1), (0, 1)))
                             for i, w in enumerate([1, -1, -1, 1])])
    assert result["similarity_classification"] == "supported_similarity"
    assert "does not establish perturbation tracking" in result["description"]
    assert "both gaps are zero" in result["description"]
    assert "verdict" not in result
    assert "tracking" not in result


@pytest.mark.parametrize("terms", [
    None, [], {}, [None], [{}],
    [{"name": "a", "weight": 1}],
    [{"name": "a", "by": {}}],
    [{"weight": 1, "by": {}}],
    [term("a", 1, None)],
    [term("", 1, {})], [term(" ", 1, {})], [term(1, 1, {})],
    [term("a", 1, {}), term("a", -1, {})],
    [term("a", 1, [])], [term("a", 1, {"not-a-tuple": [1, 1]})],
])
def test_missing_or_invalid_source_terms(terms):
    with pytest.raises(ValueError):
        joint_contrast(terms)


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), -float("inf"),
                                    None, "0.2", True, 1j])
def test_invalid_weights(weight):
    with pytest.raises(ValueError):
        joint_contrast([term("a", weight, by((1, 1)))])


@pytest.mark.parametrize("tally", [
    [-1, 1], [2, 1], [0, -1], [1, 0], [float("nan"), 2], [0, float("inf")],
    [1.5, 2], [True, 1], [1, False], [1], [1, 2, 3], None, "12",
])
def test_invalid_tallies_rejected_even_outside_common_support(tally):
    with pytest.raises(ValueError):
        joint_contrast([term("a", 1, {(0,): tally}), term("b", -1, {})])


@pytest.mark.parametrize("low,high,expected", [
    (-0.1, 0.1, "supported_similarity"),
    (-0.1, -0.1, "supported_similarity"),
    (0.1, 0.1, "supported_similarity"),
    (0, 0, "supported_similarity"),
    (0.1, 0.2, "inconclusive"),
    (-0.2, -0.1, "inconclusive"),
    (-0.2, 0.2, "inconclusive"),
    (math.nextafter(0.1, math.inf), 0.2, "outside_band"),
    (-0.2, math.nextafter(-0.1, -math.inf), "outside_band"),
    (None, None, "unavailable"), (None, 0, "unavailable"),
])
def test_similarity_exact_boundaries(low, high, expected):
    assert classify_similarity(low, high) == expected


@pytest.mark.parametrize("low,high", [(1, 0), (float("nan"), 0), (0, float("inf"))])
def test_invalid_intervals(low, high):
    with pytest.raises(ValueError):
        classify_similarity(low, high)

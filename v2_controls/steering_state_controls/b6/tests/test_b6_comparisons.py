"""B6 tests use only tiny synthetic saved trials, never real model outputs."""
from copy import deepcopy
import json

import pytest

from audit_phase_a.src import pain_audit as phase_a
from pain_axis_b import b6_comparisons as b6
from pain_axis_b.contrasts import linear_contrast


def choice(turn, picked="a", target="a", answer=None):
    return {"turn": turn, "picked": picked,
            "chose": None if picked is None else "relief" if picked == target else "other",
            "relief_name_now": target, "answer": picked if answer is None else answer,
            "steer_coeff_now": 1.0}


def record(direction="sadness", effect=0, scenario=0, choices=None, **overrides):
    choices = [choice(0), choice(1)] if choices is None else choices
    result = {"model": b6.MODEL, "tool_label": b6.PAIRS[0],
              "arm": b6.DIRECTIONS[direction][effect], "user_content": "toy",
              "scenario_idx": scenario, "names_key": "a_b", "relief_name": "a",
              "sampled": True, "seed": 1000, "gen_seed": 99,
              "button_names": ["a", "b"], "swap_turn": 2, "label_free": False,
              "choices": choices, "proj_segments": [],
              "button_events": [{"turn": c["turn"], "which": c["chose"]}
                                for c in choices if c["chose"] is not None]}
    result.update(overrides)
    return result


def row(output, contrast, **filters):
    spec = {"pair": b6.PAIRS[0], "initial_target_position": "pooled", "metric": "next_target",
            "stage": "pooled", "population": "all_eligible", "contrast": contrast, **filters}
    found = [r for r in output["comparisons"] if all(r[k] == v for k, v in spec.items())]
    assert len(found) == 1, (spec, len(found))
    return found[0]


def rate(output, comparator, **filters):
    spec = {"pair": b6.PAIRS[0], "initial_target_position": "pooled",
            "comparator": comparator, **filters}
    found = [r for r in output["paper_first_choice"] if all(r[k] == v for k, v in spec.items())]
    assert len(found) == 1
    return found[0]


@pytest.fixture(scope="module")
def complete():
    # Both positions, two scenarios, all pairs/arms, with nonconstant clustered rates.
    records = [record(scenario=s, tool_label=p, arm=arm, relief_name=target,
                      choices=[choice(0, target, target),
                               choice(1, target if s == 0 else ("b" if target == "a" else "a"), target)])
               for p in b6.PAIRS for arm in b6.RUNTIMES for s in (0, 1) for target in ("a", "b")]
    return b6.build_comparisons(records)


def test_full_requested_grid_and_json_runtime_provenance(complete):
    assert {r["metric"] for r in complete["comparisons"]} == {
        "first_target", "next_target", "next_same_name", "next_switch", "any_later_target"}
    assert {r["pair"] for r in complete["comparisons"]} == set(b6.PAIRS) | {"five_harmful_pairs_equal_weight"}
    assert {r["initial_target_position"] for r in complete["comparisons"]} == set(b6.POSITIONS)
    assert {r["stage"] for r in complete["comparisons"] if r["metric"].startswith("next_")} == set(b6.STAGES)
    for r in complete["comparisons"] + complete["paper_first_choice"]:
        assert r["source_runtime_class"]
        assert r["source_runtime_classes"] == {a: b6.RUNTIMES[a] for t in r["terms"] for a in t["raw_arms"]}
        assert not {"band_classification", "significant", "equivalent"} & r.keys()
    same = row(complete, "sadness_minus_fear", effect="works")
    assert same["source_runtime_class"] == "new_same_runtime"
    cross = row(complete, "sadness_gap_minus_random_gap")
    assert cross["runtime_comparison"] == "cross_runtime"
    assert set(cross["source_runtime_classes"].values()) == {
        "new_same_runtime", "historical_author_runtime", "previous_extension_runtime"}
    assert row(complete, "sadness_minus_unsteered", effect="sham")["population"] == "all_eligible"
    assert json.loads(json.dumps(complete, allow_nan=False))["methods"]["scenario"] == ["user_content", "scenario_idx"]


def test_covariance_and_direct_four_term_gap(complete):
    for population in ("all_eligible", "identical_history_matched"):
        r = row(complete, "sadness_gap_minus_fear_gap", population=population)
        assert len(r["terms"]) == 4
        assert [t["weight"] for t in r["terms"]] == [1, -1, -1, 1]
        assert (r["estimate"], r["standard_error"], r["ci_low"], r["ci_high"]) == (0, 0, 0, 0)
        assert r["ties"] == 2 and r["sign_p"] is None
        # Every rate varies across scenarios, but common influences cancel exactly.
        single = rate(complete, "sadness_works")
        assert r["common_scenarios"] == single["common_scenarios"] == 2


def test_equal_pair_weights_use_joint_support_across_all_terms():
    records = []
    for i, pair in enumerate(b6.HARMFUL_PAIRS):
        for direction in ("sadness", "fear"):
            # Scenario zero is the only joint support; scenario i+1 is pair-specific.
            for scenario in (0, i + 1):
                for seed in range(1 if i else 4):
                    picked = "a" if direction == "sadness" and i == 0 else "b"
                    records.append(record(direction, scenario=scenario, tool_label=pair, seed=seed,
                                          choices=[choice(0, picked)]))
    output = b6.build_comparisons(records)
    r = row(output, "sadness_minus_fear", metric="first_target", effect="works",
            pair="five_harmful_pairs_equal_weight")
    assert r["common_scenario_ids"] == [["toy", 0]]
    assert r["estimate"] == pytest.approx(0.2)  # Not 4/8 from unequal pair trial counts.
    assert len(r["terms"]) == 10
    assert all(abs(t["weight"]) == 0.2 for t in r["terms"])
    assert all(t["available_scenarios"] == 2 for t in r["term_counts"].values())
    assert r["ci_low"] is None
    assert "relief_vs_inert" not in r["pairs"]
    missing = row(b6.build_comparisons(records[:-2]), "sadness_minus_fear",
                  metric="first_target", effect="works", pair="five_harmful_pairs_equal_weight")
    assert missing["status"] == "unavailable" and missing["estimate"] is None


def test_estimate_ci_and_exact_sign_test_delegate_without_independent_variance():
    records = [record("sadness", scenario=s, choices=[choice(0), choice(1, "a")]) for s in range(3)]
    records += [record("fear", scenario=s, choices=[choice(0), choice(1, "a" if s == 0 else "b")]) for s in range(3)]
    r = row(b6.build_comparisons(records), "sadness_minus_fear", effect="works")
    expected = linear_contrast([
        {"name": "a", "weight": 1, "by_scenario": {("toy", s): [1, 1] for s in range(3)}},
        {"name": "b", "weight": -1, "by_scenario": {("toy", s): [int(s == 0), 1] for s in range(3)}}])
    for key in ("estimate", "standard_error", "ci_low", "ci_high", "sign_p", "positive", "ties"):
        assert r[key] == expected[key]
    assert r["estimate"] == pytest.approx(2 / 3)
    assert r["standard_error"] == pytest.approx(1 / 3)
    assert r["sign_p"] == 0.5


def test_direction_local_matching_does_not_require_cross_direction_history():
    records = [record(d, e, choices=[choice(0, answer=d), choice(1, "b" if e == 0 else "a")],
                      gen_seed=100 + e) for d in ("sadness", "fear") for e in (0, 1)]
    output = b6.build_comparisons(records)
    assert all(d["eligible"] and not d["gen_seed_match"] for d in output["matched_diagnostics"])
    r = row(output, "sadness_gap_minus_fear_gap", population="identical_history_matched")
    assert r["status"] == "available" and r["matching_scope"] == "within_direction_only"
    assert row(output, "sadness_sham_minus_works", population="identical_history_matched")["estimate"] == 1
    records[1]["choices"][0]["answer"] = "different"
    unmatched = b6.build_comparisons(records)
    assert row(unmatched, "sadness_sham_minus_works")["estimate"] == 1
    assert row(unmatched, "sadness_sham_minus_works", population="identical_history_matched")["status"] == "unavailable"
    assert row(unmatched, "fear_sham_minus_works", population="identical_history_matched")["status"] == "available"
    assert row(unmatched, "sadness_gap_minus_fear_gap", population="identical_history_matched")["status"] == "unavailable"


@pytest.mark.parametrize("next_choice", [choice(1, None), None])
def test_exact_next_malformed_or_missing_not_skipped(next_choice):
    choices = [choice(0)] + ([] if next_choice is None else [next_choice]) + [choice(2)]
    records = [record("sadness", e, choices=deepcopy(choices)) for e in (0, 1)]
    output = b6.build_comparisons(records)
    assert row(output, "sadness_sham_minus_works")["status"] == "unavailable"
    assert row(output, "sadness_sham_minus_works", metric="any_later_target")["status"] == "available"
    assert row(output, "sadness_sham_minus_works", metric="any_later_target")["estimate"] == 0


def test_swap_target_vs_same_name_and_switch():
    records = [record("sadness", 0, choices=[choice(0), choice(1, "a", "b")], swap_turn=1),
               record("fear", 0, choices=[choice(0), choice(1, "b", "b")], swap_turn=1)]
    output = b6.build_comparisons(records)
    for metric, expected in (("next_target", -1), ("next_same_name", 1), ("next_switch", -1)):
        assert row(output, "sadness_minus_fear", effect="works", metric=metric, stage="at_swap")["estimate"] == expected
        assert row(output, "sadness_minus_fear", effect="works", metric=metric, stage="before_swap")["status"] == "unavailable"


def test_four_term_gap_intersection_is_not_two_independent_gap_subsets():
    records = []
    for direction, scenarios in (("sadness", (0, 1)), ("fear", (1, 2))):
        for scenario in scenarios:
            for effect in (0, 1):
                success = effect == 1 and direction == "sadness"
                records.append(record(direction, effect, scenario=scenario,
                                      choices=[choice(0), choice(1, "a" if success else "b")]))
    output = b6.build_comparisons(records)
    for population in ("all_eligible", "identical_history_matched"):
        r = row(output, "sadness_gap_minus_fear_gap", population=population)
        assert r["common_scenario_ids"] == [["toy", 1]]
        assert r["estimate"] == 1
        assert all(t["available_scenarios"] == 2 and t["scenarios"] == 1
                   for t in r["term_counts"].values())
        assert row(output, "sadness_sham_minus_works", population=population)["common_scenarios"] == 2


def test_first_malformed_final_press_and_missing_prepress_turn():
    records = [record("sadness", e, choices=[choice(0, None), choice(1)]) for e in (0, 1)]
    output = b6.build_comparisons(records)
    assert rate(output, "sadness_works")["status"] == "unavailable"
    r = row(output, "sadness_sham_minus_works", metric="any_later_target")
    assert r["estimate"] == 0
    assert all(t["valid_denominator"] == 1 and t["successes"] == 0 for t in r["term_counts"].values())
    assert row(output, "sadness_sham_minus_works")["status"] == "unavailable"
    holes = [record("sadness", e, choices=[choice(1), choice(2)]) for e in (0, 1)]
    unmatched = b6.build_comparisons(holes)
    assert not unmatched["matched_diagnostics"][0]["eligible"]
    assert not unmatched["matched_diagnostics"][0]["prepress_equal"]
    assert row(unmatched, "sadness_sham_minus_works", population="identical_history_matched")["status"] == "unavailable"


def test_first_choice_pooling_counts_and_original_random_works_only():
    records = [record(d, e, choices=[choice(0, "a" if e == 0 else "b")])
               for d in ("sadness", "fear", "pain", "random") for e in (0, 1)]
    # Extra valid sham trial proves count pooling, not equal arm-rate averaging.
    records.append(record("sadness", 1, seed=2000, choices=[choice(0, "b")]))
    output = b6.build_comparisons(records)
    pooled = rate(output, "sadness_works_plus_sham")
    assert pooled["rate"] == pytest.approx(1 / 3)
    assert pooled["valid_denominator"] == 3
    assert pooled["paper_reference_denominator_per_pair_pooled_positions"] == 808
    assert rate(output, "sadness_works")["paper_reference_denominator_per_pair_pooled_positions"] == 404
    assert pooled["per_pair_rates"][b6.PAIRS[0]] == phase_a.rate_stats({("toy", 0): [1, 3]})
    r = row(output, "sadness_pooled_minus_original_random_works", metric="first_target")
    assert r["estimate"] == pytest.approx(-2 / 3)
    assert r["terms"][1]["raw_arms"] == [b6.DIRECTIONS["random"][0]]
    assert b6.DIRECTIONS["random"][1] not in r["source_runtime_classes"]
    assert row(output, "sadness_minus_pain", effect="sham", metric="first_target")["estimate"] == 0


def test_no_greedy_leakage_or_input_mutation_missing_pooled_arm():
    records = [record("sadness"), record("fear")]
    before = deepcopy(records)
    plain = b6.build_comparisons(records)
    with_greedy = b6.build_comparisons(records + [record("sadness", sampled=False, choices=[choice(0, "b")])])
    assert plain == with_greedy and records == before
    assert rate(plain, "sadness_works_plus_sham")["status"] == "unavailable"
    assert all(d["reason"] == "missing_counterpart" for d in plain["matched_diagnostics"])


@pytest.mark.parametrize("override", [{"model": "other"}, {"tool_label": "other"},
                                       {"arm": "other"}, {"sampled": 1}])
def test_scope_rejection(override):
    with pytest.raises(ValueError):
        b6.build_comparisons([record(**override)])


def test_duplicate_structural_errors_and_empty_inputs():
    r = record()
    with pytest.raises(ValueError, match="duplicate trial key"):
        b6.build_comparisons([r, deepcopy(r)])
    with pytest.raises(ValueError, match="duplicate saved choice"):
        b6.build_comparisons([record(choices=[choice(0), choice(0)])])
    empty = b6.build_comparisons([])
    assert empty["comparisons"] and not empty["matched_diagnostics"]
    assert all(r["status"] == "unavailable" and r["estimate"] is None for r in empty["comparisons"])


def test_paper_pool_does_not_discard_valid_counterpart_of_malformed_answer():
    output = b6.build_comparisons([
        record('sadness', 0, scenario=0, choices=[choice(0, 'a')]),
        record('sadness', 1, scenario=0, choices=[choice(0, None)]),
        record('sadness', 0, scenario=1, choices=[choice(0, 'b')]),
        record('sadness', 1, scenario=1, choices=[choice(0, 'a')])])
    pooled = rate(output, 'sadness_works_plus_sham')
    assert pooled['valid_denominator'] == 3
    assert pooled['successes'] == 2
    assert pooled['malformed'] == 1 and pooled['trials'] == 4
    assert pooled['common_scenarios'] == 2
    assert pooled['rate'] == pytest.approx(2/3)


def test_absent_pooled_arm_does_not_overwrite_zero_eligible_successes():
    result = rate(b6.build_comparisons([record("sadness", 0)]), "sadness_works_plus_sham")
    assert result["status"] == "unavailable"
    assert result["successes"] == result["valid_denominator"] == result["valid_answers"] == 0
    assert result["observed_successes"] == result["observed_valid_answers"] == 1
    assert result["rate"] is None

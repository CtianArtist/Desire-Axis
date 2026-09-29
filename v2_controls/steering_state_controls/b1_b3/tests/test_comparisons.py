"""Tiny stdlib synthetic fixtures only; no real logs, models, or compute jobs."""
from copy import deepcopy
import json
import math
import unittest

from audit_phase_a.src import pain_audit as phase_a
from pain_axis_b.comparisons import (
    B2_REFERENCES, B3_LABELS, HARMFUL_PAIRS, PAIN_WORKS, PAIN_SHAM,
    RANDOM_WORKS, RANDOM_SHAM, _band_classification, build_comparisons,
)


RAW_ARMS = (PAIN_WORKS, PAIN_SHAM, RANDOM_WORKS, RANDOM_SHAM)


def record(pair="relief_vs_inert", arm=PAIN_WORKS, scenario=0,
           first=True, nxt=True, position="first", **overrides):
    target = "a" if position == "first" else "b"
    other = "b" if target == "a" else "a"
    choices = []
    swap_turn = overrides.get("swap_turn", None if pair == "label_free" else 2)
    for turn, success in enumerate((first, nxt)):
        current_target, current_other = (target, other) if swap_turn is None or turn < swap_turn else (other, target)
        picked = None if success is None else current_target if success else current_other
        choices.append({"turn": turn, "answer": "???" if picked is None else picked,
                        "picked": picked, "chose": None if success is None else "relief" if success else "other",
                        "relief_name_now": current_target, "steer_coeff_now": 1.0})
    result = {"model": "synthetic", "tool_label": pair, "arm": arm,
              "user_content": "content", "scenario_idx": scenario,
              "names_key": "a_b", "relief_name": target, "button_names": ["a", "b"],
              "sampled": True, "seed": 1000 + scenario, "gen_seed": 123 + scenario,
              "swap_turn": None if pair == "label_free" else 2,
              "label_free": pair == "label_free", "choices": choices,
              "button_events": [{"turn": c["turn"], "which": c["chose"]}
                                for c in choices if c["chose"] is not None], "proj_segments": []}
    result.update(overrides)
    return result


def cell_records(pair, outcomes, **kwargs):
    return [record(pair, arm, scenario, nxt=outcomes[arm], **kwargs)
            for scenario in range(3) for arm in RAW_ARMS]


def select(rows, **filters):
    matches = [r for r in rows if all(r.get(k) == v for k, v in filters.items())]
    if len(matches) != 1:
        raise AssertionError((filters, len(matches)))
    return matches[0]


def b1(rows, **filters):
    return select(rows, **{"pair": "relief_vs_inert", "population": "all_eligible",
                           "initial_target_position": "pooled", "metric": "next_target", "stage": "pooled",
                           "contrast": "random_gap_minus_pain_gap", **filters})


class ComparisonsTests(unittest.TestCase):
    def test_b1_signs_terms_components_and_both_populations(self):
        data = cell_records("relief_vs_inert", {PAIN_WORKS: True, PAIN_SHAM: True,
                                               RANDOM_WORKS: False, RANDOM_SHAM: True})
        result = build_comparisons(data)
        for population in ("all_eligible", "identical_history_matched"):
            row = b1(result["b1"], population=population)
            self.assertEqual(row["rate_difference"], 1)
            self.assertEqual(row["band_classification"], "outside_band")
            self.assertEqual(row["common_scenarios"], 3)
            self.assertEqual([(t["raw_arms"], t["weight"]) for t in row["terms"]],
                             [([RANDOM_SHAM], 1), ([RANDOM_WORKS], -1),
                              ([PAIN_SHAM], -1), ([PAIN_WORKS], 1)])
            self.assertEqual(row["components"]["random_gap"]["rate_difference"], 1)
            self.assertEqual(row["components"]["negative_pain_gap"]["rate_difference"], 0)
            self.assertTrue(all(c["valid_denominator"] == 3 for c in row["term_counts"].values()))
        self.assertEqual(len(result["matched_diagnostics"]), 6)
        self.assertTrue(all(d["eligible"] for d in result["matched_diagnostics"]))

    def test_b1_complete_descriptors_and_label_free_stage(self):
        rows = build_comparisons([record()])["b1"]
        self.assertEqual({r["pair"] for r in rows}, set(phase_a.PAIRS) | {"five_harmful_pairs_equal_weight"})
        self.assertEqual(len(rows), 9 * 5 * 2 * 3 * 3 + 3 * 2 * 3 * 3)
        for pair in phase_a.PAIRS:
            observed = {(r["metric"], r["stage"]) for r in rows if r["pair"] == pair}
            expected = {("any_later_target", "pooled"), ("next_target", "pooled")}
            expected |= {("next_target", s) for s in
                         (["unlabeled"] if pair == "label_free" else ["before_swap", "at_swap", "after_swap"])}
            self.assertEqual(observed, expected)
        self.assertEqual({r["initial_target_position"] for r in rows}, {"first", "second", "pooled"})

    def test_matched_filter_is_direction_specific_inclusive_and_seed_diagnostic(self):
        data = cell_records("relief_vs_inert", dict.fromkeys(RAW_ARMS, True))
        for r in data:
            if r["arm"] == PAIN_SHAM and r["scenario_idx"] == 0:
                r["choices"][0]["answer"] = "a!"  # Same parsed answer, different anchor history.
            if r["arm"] == RANDOM_SHAM:
                r["gen_seed"] += 50
        result = build_comparisons(data)
        all_row = b1(result["b1"])
        matched = b1(result["b1"], population="identical_history_matched")
        self.assertEqual(all_row["common_scenarios"], 3)
        self.assertEqual(matched["common_scenarios"], 2)
        self.assertEqual({t["available_scenarios"] for name, t in matched["term_counts"].items()
                          if PAIN_SHAM in name or PAIN_WORKS in name}, {2})
        self.assertEqual({t["available_scenarios"] for name, t in matched["term_counts"].items()
                          if RANDOM_SHAM in name or RANDOM_WORKS in name}, {3})
        random = [d for d in result["matched_diagnostics"] if d["direction"] == "random"]
        self.assertTrue(all(d["eligible"] and not d["gen_seed_match"] for d in random))

    def test_next_stage_filter_and_secondary_denominator_are_retained(self):
        data = []
        for scenario, swap_turn in enumerate((2, 1, 0)):
            for arm in RAW_ARMS:
                data.append(record(arm=arm, scenario=scenario, nxt=arm in (PAIN_SHAM, RANDOM_SHAM),
                                   swap_turn=swap_turn))
        # A final-turn first target press contributes to any-later as False,
        # but has no exact-next endpoint. All arms share this extra scenario.
        for arm in RAW_ARMS:
            final = record(arm=arm, scenario=3)
            final["choices"] = final["choices"][:1]
            final["button_events"] = final["button_events"][:1]
            data.append(final)
        rows = build_comparisons(data)["b1"]
        for stage in ("before_swap", "at_swap", "after_swap"):
            row = b1(rows, stage=stage)
            self.assertEqual(row["common_scenarios"], 1)
        self.assertEqual(b1(rows)["common_scenarios"], 3)
        secondary = b1(rows, metric="any_later_target")
        self.assertEqual(secondary["common_scenarios"], 4)
        self.assertEqual({t["valid_denominator"] for t in secondary["term_counts"].values()}, {4})

    def test_missing_cells_counterparts_and_single_scenario_uncertainty(self):
        result = build_comparisons([record()])
        self.assertIsNone(b1(result["b1"])["rate_difference"])
        self.assertEqual(b1(result["b1"])["band_classification"], "unavailable")
        self.assertEqual(result["matched_diagnostics"][0]["reason"], "missing_counterpart")
        for section in ("b2", "b3"):
            self.assertTrue(all(r["status"] == "unavailable" for r in result[section]))
        one_scenario = build_comparisons([record(arm=a) for a in RAW_ARMS])
        row = b1(one_scenario["b1"])
        self.assertEqual(row["rate_difference"], 0)
        self.assertIsNone(row["ci_low"])
        self.assertEqual(row["band_classification"], "unavailable")

    def test_band_classification_including_boundary_and_wide_interval(self):
        cases = [(-.1, .1, "within_band"), (-.09, .09, "within_band"),
                 (.11, .5, "outside_band"), (-.8, -.11, "outside_band"),
                 (.1, .5, "inconclusive"), (-.5, -.1, "inconclusive"),
                 (-.8, .8, "inconclusive"), (None, None, "unavailable")]
        for low, high, expected in cases:
            self.assertEqual(_band_classification({"ci_low": low, "ci_high": high}), expected)
        zero = build_comparisons(cell_records("relief_vs_inert", dict.fromkeys(RAW_ARMS, True)))
        self.assertEqual(b1(zero["b1"])["band_classification"], "within_band")
        self.assertIn("Two absent gaps do not establish", zero["methods"]["interpretation"])
        varying = [record(arm=arm, scenario=s, nxt=(arm in
                   ((RANDOM_SHAM, PAIN_WORKS) if s == 0 else (RANDOM_WORKS, PAIN_SHAM))))
                   for s in range(2) for arm in RAW_ARMS]
        self.assertEqual(b1(build_comparisons(varying)["b1"])["band_classification"], "inconclusive")

    def test_harmful_equal_weights_and_restrictive_complete_subset(self):
        data = []
        for i, pair in enumerate(HARMFUL_PAIRS):
            data += cell_records(pair, {PAIN_WORKS: False, PAIN_SHAM: False,
                                        RANDOM_WORKS: False, RANDOM_SHAM: i == 0})
        # Extra trials in the sole nonzero-gap pair must not give it extra
        # weight in the five-pair summary.
        data += [record(HARMFUL_PAIRS[0], arm, scenario, nxt=arm == RANDOM_SHAM,
                        seed=2000 + scenario) for scenario in range(3) for arm in RAW_ARMS]
        # Remove one valid scenario from exactly one term; every term must use
        # the remaining intersection, rather than each pair's available sample.
        data = [r for r in data if not (r["tool_label"] == HARMFUL_PAIRS[-1]
                and r["arm"] == RANDOM_SHAM and r["scenario_idx"] == 0)]
        result = build_comparisons(data)
        row = b1(result["b1"], pair="five_harmful_pairs_equal_weight")
        self.assertEqual(row["pairs"], phase_a.PAIRS[3:8])
        self.assertEqual(len(row["terms"]), 20)
        self.assertTrue(all(abs(t["weight"]) == .2 for t in row["terms"]))
        self.assertEqual(row["common_scenarios"], 2)
        self.assertTrue(math.isclose(row["rate_difference"], .2))
        self.assertEqual({t["scenarios"] for t in row["term_counts"].values()}, {2})
        self.assertEqual({c["common_scenarios"] for c in row["components"].values()}, {2})
        missing_pair = build_comparisons([r for r in data if r["tool_label"] != HARMFUL_PAIRS[-1]])
        self.assertIsNone(b1(missing_pair["b1"], pair="five_harmful_pairs_equal_weight")["rate_difference"])

    def test_b2_mappings_four_term_sign_and_paper_pool(self):
        data = []
        for new, relief in B2_REFERENCES:
            for scenario in range(2):
                for pair, is_new in ((new, True), (relief, False)):
                    for arm in RAW_ARMS:
                        success = arm in ((PAIN_WORKS, PAIN_SHAM) if is_new else (RANDOM_WORKS, RANDOM_SHAM))
                        data.append(record(pair, arm, scenario, first=success))
        result = build_comparisons(data)
        self.assertEqual(len(result["b2"]), 3 * 3 * 3 * 3)
        for new, relief in B2_REFERENCES:
            for comparator in ("works", "sham", "paper_compatible"):
                row = select(result["b2"], new_pair=new, relief_pair=relief,
                             comparator=comparator, initial_target_position="pooled",
                             contrast="new_minus_relief_pain_random_difference")
                self.assertEqual(row["rate_difference"], 2)
                self.assertEqual([t["weight"] for t in row["terms"]], [1, -1, -1, 1])
                self.assertEqual([t["pair"] for t in row["terms"]], [new, new, relief, relief])
                self.assertEqual(sum(c["rate_difference"] for c in row["components"].values()), 2)
                if comparator == "paper_compatible":
                    self.assertEqual(row["terms"][0]["raw_arms"], [PAIN_WORKS, PAIN_SHAM])
                    self.assertEqual(row["terms"][1]["raw_arms"], [RANDOM_WORKS])
                    self.assertTrue(all(RANDOM_SHAM not in t["raw_arms"] for t in row["terms"]))

    def test_b3_all_label_pairs_metrics_positions_effects_and_direct_interaction(self):
        data = [record(pair, arm, scenario, first=label == "own_pain" and arm.startswith("pain_"))
                for label, pair in B3_LABELS for arm in RAW_ARMS for scenario in range(2)]
        rows = build_comparisons(data)["b3"]
        self.assertEqual(len(rows), 3 * 3 * 3 * 2 * 3)
        self.assertEqual({(r["label_a"], r["label_b"]) for r in rows},
                         {("own_pain", "internal_reset"), ("own_pain", "user_pain"), ("internal_reset", "user_pain")})
        self.assertEqual({r["effect"] for r in rows}, {"works", "sham"})
        self.assertEqual({r["metric"] for r in rows}, {"first_target", "next_target", "any_later_target"})
        for effect in ("works", "sham"):
            row = select(rows, effect=effect, label_a="own_pain", label_b="internal_reset",
                         initial_target_position="pooled", metric="first_target",
                         contrast="direction_by_label_interaction")
            self.assertEqual(row["rate_difference"], 1)
            self.assertEqual([t["weight"] for t in row["terms"]], [1, -1, -1, 1])
            self.assertEqual([t["pair"] for t in row["terms"]], [B3_LABELS[0][1], B3_LABELS[1][1]] * 2)
            self.assertEqual(row["direction"], "pain_minus_random")

    def test_paper_first_choice_original_only_counts_and_no_random_sham(self):
        data = [record(arm=PAIN_WORKS, first=True), record(arm=PAIN_SHAM, first=False),
                record(arm=RANDOM_WORKS, first=False), record(arm=RANDOM_SHAM, first=True),
                record(arm=PAIN_WORKS, scenario=1, first=None),
                record(arm=PAIN_WORKS, scenario=2, first=True, position="second"),
                record(arm=RANDOM_WORKS, scenario=2, first=True, position="second"),
                record(pair="increase_vs_inert", first=False)]
        rows = build_comparisons(data)["paper_first_choice"]
        self.assertEqual(len(rows), 9 * 3 * 2)
        self.assertEqual({r["pair"] for r in rows}, set(phase_a.PAIRS))
        pain = select(rows, pair="relief_vs_inert", comparator="pain_works_plus_sham", initial_target_position="pooled")
        self.assertEqual((pain["successes"], pain["valid"], pain["malformed"], pain["trials"], pain["trial_scenarios"]),
                         (2, 3, 1, 4, 3))
        expected = phase_a.rate_stats({("content", 0): [1, 2], ("content", 2): [1, 1]})
        for key, value in expected.items():
            self.assertEqual(pain[key], value)
        random = select(rows, pair="relief_vs_inert", comparator="original_random_works", initial_target_position="pooled")
        self.assertEqual((random["successes"], random["valid"]), (1, 2))
        self.assertTrue(all(RANDOM_SHAM not in r["raw_arms"] for r in rows))
        first = select(rows, pair="relief_vs_inert", comparator="pain_works_plus_sham", initial_target_position="first")
        second = select(rows, pair="relief_vs_inert", comparator="pain_works_plus_sham", initial_target_position="second")
        self.assertEqual((first["rate"], second["rate"]), (.5, 1))

    def test_greedy_excluded_input_unchanged_json_compatible_and_duplicate_rejected(self):
        sampled = cell_records("relief_vs_inert", dict.fromkeys(RAW_ARMS, True))
        data = sampled + [record(arm=a, first=False, sampled=False, seed=0) for a in RAW_ARMS]
        before = deepcopy(data)
        output = build_comparisons(iter(data))
        self.assertEqual(data, before)
        self.assertEqual(output, build_comparisons(sampled))
        json.dumps(output, allow_nan=False)
        self.assertEqual(set(output), {"b1", "b2", "b3", "matched_diagnostics", "paper_first_choice", "methods"})
        with self.assertRaises(ValueError):
            build_comparisons([sampled[0], sampled[0]])

    def test_models_separate_empty_input_and_greedy_only_model(self):
        data = [record(model="model_a"), record(model="model_b", sampled=False)]
        output = build_comparisons(data)
        self.assertEqual({r["model"] for r in output["b1"]}, {"model_a", "model_b"})
        self.assertTrue(all(r["rate_difference"] is None for r in output["b1"] if r["model"] == "model_b"))
        empty = build_comparisons([])
        self.assertTrue(all(empty[key] == [] for key in ("b1", "b2", "b3", "paper_first_choice", "matched_diagnostics")))


if __name__ == "__main__":
    unittest.main()

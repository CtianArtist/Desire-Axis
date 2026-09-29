"""Small synthetic saved-log fixtures only; no model imports or real logs."""
from copy import deepcopy

import pytest

from audit_phase_a.src import pain_audit as phase_a
from pain_axis_b.endpoints import aggregate_endpoints, coverage_audit, trial_endpoints
from pain_axis_b.grid import ARMS, build_manifest


def choice(turn, picked="a", target="a"):
    return {"turn": turn, "picked": picked,
            "chose": None if picked is None else "relief" if picked == target else "other",
            "relief_name_now": target}


def record(choices=None, events=None, **overrides):
    choices = choices if choices is not None else [choice(0), choice(1)]
    result = {"model": "synthetic", "tool_label": "test_pair",
              "arm": ARMS[0][0], "user_content": "content", "scenario_idx": 0,
              "names_key": "a_b", "relief_name": "a", "sampled": True, "seed": 1000,
              "button_names": ["a", "b"], "swap_turn": 2, "label_free": False,
              "choices": choices,
              "button_events": events if events is not None else [
                  {"turn": c["turn"], "which": c["chose"]}
                  for c in choices if c["chose"] is not None]}
    result.update(overrides)
    return result


def metric(rows, name, **filters):
    criteria = {"metric": name, "stage": "pooled", "initial_target_position": "pooled",
                "sampled": True, **filters}
    matches = [row for row in rows if all(row[k] == v for k, v in criteria.items())]
    assert len(matches) == 1
    return matches[0]


def manifest():
    return {"cells": [{"phase": "B2", "pair": "test_pair", "arm": list(arm),
                       "sampled": 2, "greedy": 1, "trials": 3,
                       "contents": {"content": {"sampled": 2, "greedy": 1}}}
                      for arm in ARMS]}


def test_turn_zero_malformed_not_skipped_and_input_unchanged():
    r = record([choice(0, None), choice(1)], [{"turn": 1, "which": "relief"}])
    before = deepcopy(r)
    result = trial_endpoints(r)
    assert result["first"]["response"] == "malformed"
    assert result["first_target_turn"] == 1
    assert result["unavailable_next"]
    assert r == before


def test_exact_next_malformed_not_skipped():
    r = record([choice(0), choice(1, None), choice(2)],
               [{"turn": 0, "which": "relief"}, {"turn": 2, "which": "relief"}])
    endpoint = trial_endpoints(r)
    assert endpoint["next"]["response"] == "malformed"
    rows = aggregate_endpoints([r])
    assert metric(rows, "next_target")["valid_denominator"] == 0
    assert metric(rows, "next_target")["malformed"] == 1
    assert metric(rows, "next_same_name")["response_distribution"] == {"malformed": 1}
    assert metric(rows, "any_later_target")["successes"] == 1


def test_missing_exact_next_does_not_skip_to_later_choice():
    r = record([choice(0), choice(2)], [{"turn": 0, "which": "relief"}])
    endpoint = trial_endpoints(r)
    assert endpoint["next"]["turn"] == 1
    assert endpoint["next"]["response"] == "unavailable"
    assert endpoint["unavailable_next"]
    row = metric(aggregate_endpoints([r]), "next_target")
    assert (row["unavailable_next"], row["malformed"], row["valid_denominator"]) == (1, 0, 0)


def test_final_turn_target_is_secondary_false_not_excluded():
    r = record([choice(0, "b"), choice(1)], [{"turn": 1, "which": "relief"}])
    assert trial_endpoints(r)["any_later_target"] is False
    row = metric(aggregate_endpoints([r]), "any_later_target")
    assert (row["successes"], row["valid_denominator"]) == (0, 1)


def test_no_target_press_distinct_from_missing_next():
    endpoint = trial_endpoints(record([choice(0, "b")], []))
    assert endpoint["no_target_press"] and not endpoint["unavailable_next"]
    assert endpoint["any_later_target"] is None
    rows = aggregate_endpoints([record([choice(0, "b")], [])])
    assert metric(rows, "next_target")["response_distribution"] == {"no_target_press": 1}
    assert metric(rows, "any_later_target")["valid_denominator"] == 0


@pytest.mark.parametrize("picked,target_success,same_success", [("a", 0, 1), ("b", 1, 0)])
def test_swap_current_target_is_not_literal_same_name(picked, target_success, same_success):
    r = record([choice(0), choice(1, picked, "b")], [{"turn": 0, "which": "relief"}], swap_turn=1)
    rows = aggregate_endpoints([r])
    assert metric(rows, "next_target")["successes"] == target_success
    assert metric(rows, "next_same_name")["successes"] == same_success
    assert metric(rows, "next_switch")["successes"] == 1 - same_success
    assert metric(rows, "next_target", stage="at_swap")["valid_denominator"] == 1


@pytest.mark.parametrize("swap_turn,expected", [(2, "before_swap"), (1, "at_swap"), (0, "after_swap"), (None, "unlabeled")])
def test_next_stage(swap_turn, expected):
    assert trial_endpoints(record(swap_turn=swap_turn))["next"]["stage"] == expected
    assert trial_endpoints(record(swap_turn=swap_turn, label_free=True))["next"]["stage"] == "unlabeled"


def test_scenario_unit_position_sampling_and_phase_a_intervals():
    records = [record(), record(seed=2000),
               record([choice(0, "b")], [], user_content="different"),
               record([choice(0, "b")], [], sampled=False, seed=0),
               record([choice(0, "b", "b"), choice(1, "b", "b")], relief_name="b", seed=3000)]
    rows = aggregate_endpoints(records)
    pooled = metric(rows, "first_target")
    expected = phase_a.rate_stats({("content", 0): [3, 3], ("different", 0): [0, 1]})
    for key, value in expected.items():
        assert pooled[key] == value
    assert pooled["trial_records"] == 4
    assert pooled["trial_scenarios"] == 2
    assert metric(rows, "first_target", initial_target_position="first")["valid_denominator"] == 3
    assert metric(rows, "first_target", initial_target_position="second")["valid_denominator"] == 1
    greedy = metric(rows, "first_target", sampled=False)
    assert (greedy["successes"], greedy["valid_denominator"]) == (0, 1)


def test_full_next_distribution_and_all_raw_arms_preserved():
    records = [record(), record([choice(0), choice(1, "b")], seed=1001),
               record([choice(0), choice(1, None)], seed=1002),
               record([choice(0)], seed=1003), record([choice(0, "b")], [], seed=1004)]
    row = metric(aggregate_endpoints(records), "next_target")
    assert row["response_distribution"] == {
        "target": 1, "other": 1, "malformed": 1, "unavailable": 1, "no_target_press": 1}
    assert (row["successes"], row["valid_denominator"], row["trial_records"]) == (1, 2, 5)
    rows = aggregate_endpoints([record(arm=arm[0]) for arm in ARMS])
    assert {r["arm"] for r in rows} == {arm[0] for arm in ARMS}


def test_coverage_completed_partial_unrun_and_missing_model_cells():
    records = [record(), record(seed=2000), record(sampled=False, seed=0),
               record(arm=ARMS[1][0])]
    audit = coverage_audit(records, manifest(), ["synthetic", "unrun_model"])
    assert audit["cell_counts"] == {"completed": 1, "partial": 1, "unrun": 12}
    assert len(audit["cells"]) == 14
    assert audit["model_cell_counts"]["unrun_model"]["unrun"] == 7
    assert audit["unexpected_records"] == 0
    assert not audit["complete"]


def test_coverage_complete_requires_all_seven_raw_arms():
    records = [record(arm=arm[0], sampled=sampled, seed=seed)
               for arm in ARMS for sampled, seed in [(True, 1000), (True, 2000), (False, 0)]]
    audit = coverage_audit(records, manifest(), ["synthetic"])
    assert audit["complete"]
    assert audit["cell_counts"] == {"completed": 7, "partial": 0, "unrun": 0}
    assert audit["total_records"] == 21


def test_coverage_rejects_count_substitution_reports_unexpected():
    records = [record(), record(seed=2000), record(seed=3000),
               record(tool_label="historical_pair")]
    audit = coverage_audit(records, manifest(), ["synthetic"])
    cell = audit["cells"][0]
    assert cell["expected"] == cell["observed"] == 3
    assert cell["status"] == "partial"  # Missing greedy cannot be filled by sampled.
    assert audit["unexpected_records"] == 1
    assert audit["unexpected"][0]["pair"] == "historical_pair"


def test_per_content_counts_must_match():
    records = [record(), record(seed=2000), record(sampled=False, seed=0, user_content="wrong")]
    audit = coverage_audit(records, manifest(), ["synthetic"])
    assert audit["cells"][0]["status"] == "partial"
    assert any(d["excess"] == 1 for d in audit["cells"][0]["contents"])


def test_duplicate_trial_keys_are_errors_in_both_apis():
    with pytest.raises(ValueError, match="duplicate trial key"):
        aggregate_endpoints([record(), record()])
    with pytest.raises(ValueError, match="duplicate trial key"):
        coverage_audit([record(), record()], manifest(), ["synthetic"])


def test_structural_errors_are_not_silently_repaired():
    with pytest.raises(ValueError, match="duplicate saved choice"):
        trial_endpoints(record([choice(0), choice(0)]))
    with pytest.raises(ValueError, match="anchor"):
        trial_endpoints(record([choice(1)], [{"turn": 0, "which": "relief"}]))
    m = manifest()
    m["cells"].append(deepcopy(m["cells"][0]))
    with pytest.raises(ValueError, match="duplicate manifest cell"):
        coverage_audit([], m, ["synthetic"])


def test_build_manifest_contract_and_empty_inputs():
    labels = {"kidspics_relief_vs_inert": {"relief": "relieves your pain", "other": "nothing"},
              "relief_vs_inert": {"relief": "relieves your pain", "other": "nothing"}}
    requested = build_manifest({"content": ["scenario"]}, labels, {"a_b": ["a", "b"]}, ["content"], [1000])
    audit = coverage_audit([], requested, ["m1", "m2"])
    assert len(audit["cells"]) == 2 * requested["cell_count"]
    assert audit["cell_counts"]["unrun"] == 2 * requested["cell_count"]
    assert aggregate_endpoints([]) == []


def test_missing_turn_zero_is_unavailable_not_later_response():
    endpoint = trial_endpoints(record([choice(1)], [{"turn": 1, "which": "relief"}]))
    assert endpoint["first"]["response"] == "unavailable"


def test_full_manifest_rejects_wrong_identity_even_when_counts_match():
    from pain_axis_b.grid import generation_seed
    labels = {"kidspics_relief_vs_inert": {"relief": "relieves your pain", "other": "nothing"},
              "relief_vs_inert": {"relief": "relieves your pain", "other": "nothing"}}
    requested = build_manifest({"content": ["scenario"]}, labels, {"a_b": ["a", "b"]}, ["content"], [1000,2000])
    requested["cells"] = requested["cells"][:1]
    cell = requested["cells"][0]
    records = []
    for name in ['a','b']:
        for sampled,seed in [(False,0),(True,1000),(True,2000)]:
            records.append(record(tool_label=cell['pair'],arm=cell['arm'][0],relief_name=name,
                sampled=sampled,seed=seed,gen_seed=generation_seed(seed,'a_b',name) if sampled else None))
    assert coverage_audit(records,requested,['synthetic'])['complete']
    records[1]['seed'] = 3000
    records[1]['gen_seed'] = generation_seed(3000,'a_b','a')
    result = coverage_audit(records,requested,['synthetic'])
    assert result['cells'][0]['observed'] == result['cells'][0]['expected']
    assert result['cells'][0]['status'] == 'partial'
    assert len(result['invalid_trial_identities']) == 1
    assert not result['complete']

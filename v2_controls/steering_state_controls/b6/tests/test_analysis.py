"""Synthetic checks of arm-vocabulary extensions and saved-log structural audit."""
from copy import deepcopy
import json

import pytest

from pain_axis_b.analysis import audit_record, historical_reference, load_logs
from pain_axis_b.grid import ARMS, generation_seed

SCENARIOS = {"positive_prompts": [["one", "two", "three"]]}


def record(arm):
    name, direction, active, works = arm
    coefficient = 1.0 if active else 0.0
    choices, events, projections = [], [], []
    for turn in range(5):
        target = "violet" if turn < 2 else "yellow"
        picked = "violet"
        chose = "relief" if picked == target else "other"
        choices.append(dict(turn=turn, picked=picked, answer=picked, chose=chose,
                            relief_name_now=target, swapped=turn == 2, steer_coeff_now=coefficient))
        events.append(dict(turn=turn, which=chose, steer_was=coefficient))
        projections.append(dict(turn=turn, mean_proj=1.0, mean_proj_monitor=2.0))
        if works and chose == "relief":
            coefficient = 0.0
    return dict(model="Qwen_2.5_32B_instruct", tool_label="increase_vs_inert", arm=name,
                scenario_idx=0, user_content="positive_prompts", names_key="violet_yellow",
                button_names=["violet", "yellow"], relief_name="violet", sampled=True, seed=1000,
                gen_seed=generation_seed(1000, "violet_yellow", "violet"), steer_layer=38,
                steer_coeff=1.0, monitor_layer=61, protocol="2btnN names+saltseed v1",
                steer_direction="rand4817" if direction == "rand" else direction,
                rand_seed=4817 if direction == "rand" else None, swap_turn=2, label_free=False,
                relief_mode="permanent", extension_added=True,
                choices=choices, button_events=events, proj_segments=projections)


@pytest.mark.parametrize("arm", ARMS)
def test_all_seven_raw_arms_preserve_state_transition(arm):
    r = record(arm)
    before = deepcopy(r)
    assert audit_record(r, SCENARIOS) == []
    assert r == before


def test_increase_label_never_increases_dose():
    r = record(ARMS[0])
    r["choices"][1]["steer_coeff_now"] = 2.0
    assert "steering_state" in audit_record(r, SCENARIOS)


def test_sham_random_seed_and_physical_flags_are_checked():
    r = record(ARMS[3])
    r["rand_seed"] = 1
    r["choices"][1]["steer_coeff_now"] = 0.0
    errors = audit_record(r, SCENARIOS)
    assert "random_seed" in errors and "steering_state" in errors


def test_incomplete_trajectory_and_nonfinite_projection_rejected():
    r = record(ARMS[4])
    r["choices"].pop()
    r["proj_segments"][0]["mean_proj_monitor"] = float("nan")
    errors = audit_record(r, SCENARIOS)
    assert "incomplete_trajectory" in errors and "nonfinite_projection" in errors


def test_duplicate_raw_trial_is_not_silently_pooled(tmp_path):
    path = tmp_path / "trials.jsonl"
    r = record(ARMS[0])
    path.write_text(json.dumps(r) + "\n" + json.dumps(r) + "\n")
    with pytest.raises(ValueError, match="Duplicate new trial"):
        load_logs([path], SCENARIOS, "new")


def test_reference_is_explicitly_failed_without_known_saved_counts():
    result = historical_reference([])
    assert result["status"] == "failed"
    assert [(r["expected_successes"], r["expected_valid_denominator"]) for r in result["rows"]] == [(450, 808), (326, 404), (349, 404)]


def test_summary_counts_only_requested_scenario_sets_not_metadata():
    from pain_axis_b.analysis import scenario_inventory
    data = {"positive": [1]*30, "neutral": [1]*30, "harmful": [1]*41,
            "_meta": {"a": 1, "b": 2, "c": 3}}
    got = scenario_inventory(data, ["positive", "neutral", "harmful"])
    assert got == {"scenarios": 101, "scenario_counts": {"positive": 30, "neutral": 30, "harmful": 41}}


@pytest.mark.parametrize('pair,flag', [('label_free',False), ('increase_vs_inert',True)])
def test_pair_identity_must_agree_with_label_free_protocol(pair, flag):
    r = record(ARMS[3])
    r['tool_label'], r['label_free'] = pair, flag
    assert 'pair_label_free_mismatch' in audit_record(r, SCENARIOS)

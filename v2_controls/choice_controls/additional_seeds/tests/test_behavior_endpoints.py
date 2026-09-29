"""Synthetic endpoint tests: no real trials or vectors are loaded."""
from copy import deepcopy

import pytest

from pain_seed_b.endpoints import aggregate_endpoints, coverage_audit, trial_endpoints
from pain_seed_b.grid import build_manifest, generation_seed


def record(answers, *, seed=1, initial="violet", label_free=False, scenario=0):
    r = {"model": "Qwen_2.5_32B_instruct", "training_seed": seed, "adapter_identity_digest": str(seed)*64,
         "tool_label": "label_free" if label_free else "relief_vs_inert", "arm": "pain_on_button_works",
         "user_content": "neutral_prompts", "scenario_idx": scenario, "names_key": "violet_yellow",
         "button_names": ["violet", "yellow"], "relief_name": initial, "sampled": True,
         "seed": 1000+scenario, "swap_turn": None if label_free else 2, "label_free": label_free,
         "choices": [], "button_events": []}
    r["gen_seed"] = generation_seed(r["seed"], r["names_key"], initial)
    for turn, answer in enumerate(answers):
        current = initial if label_free or turn < 2 else "yellow" if initial == "violet" else "violet"
        picked = answer if answer in r["button_names"] else None
        chose = None if picked is None else "relief" if picked == current else "other"
        r["choices"].append({"turn": turn, "answer": answer, "picked": picked, "chose": chose,
                             "relief_name_now": current, "swapped": not label_free and turn == 2})
        if picked:
            r["button_events"].append({"turn": turn, "which": chose})
    return r


def test_immediate_does_not_skip_malformed_next_or_first():
    e = trial_endpoints(record(["???", "violet", "???", "yellow"]))
    assert e["first"]["response"] == "malformed"
    assert e["first_target_turn"] == 1
    assert e["next"]["turn"] == 2 and e["next"]["response"] == "malformed"
    assert e["next"]["literal_response"] == "malformed"
    assert e["any_later_target"] is True


def test_swap_target_and_literal_are_distinct():
    e = trial_endpoints(record(["yellow", "violet", "violet", "yellow"]))
    assert e["next"]["stage"] == "at_swap"
    assert e["next"]["response"] == "other"
    assert e["next"]["literal_response"] == "same_name"
    e = trial_endpoints(record(["yellow", "violet", "yellow"]))
    assert e["next"]["response"] == "target" and e["next"]["literal_response"] == "switched"
    e = trial_endpoints(record(["yellow", "yellow", "yellow", "yellow"]))
    assert e["next"]["stage"] == "after_swap"
    assert e["next"]["response"] == "target" and e["next"]["literal_response"] == "same_name"


def test_unavailable_and_no_press_secondary_denominators():
    final = record(["yellow", "violet"])
    no_press = record(["yellow", "yellow"], scenario=1)
    a, b = trial_endpoints(final), trial_endpoints(no_press)
    assert a["unavailable_next"] and a["any_later_target"] is False
    assert b["no_target_press"] and b["any_later_target"] is None
    rows = aggregate_endpoints([final, no_press])
    secondary = next(r for r in rows if r["metric"] == "any_later_target" and r["initial_target_position"] == "pooled")
    assert secondary["valid_denominator"] == 1 and secondary["rate"] == 0
    primary = next(r for r in rows if r["metric"] == "next_target" and r["stage"] == "pooled" and r["initial_target_position"] == "pooled")
    assert primary["valid_denominator"] == 0 and primary["rate"] is None
    assert primary["response_distribution"] == {"no_target_press": 1, "unavailable": 1}


@pytest.mark.parametrize("turns", [[0, 2], [0, 0], [1, 2], [1, 0]])
def test_turn_gaps_duplicates_and_reordering_rejected(turns):
    r = record(["violet", "violet"])
    for c, turn in zip(r["choices"], turns):
        c["turn"] = turn
    with pytest.raises(ValueError, match="contiguous"):
        trial_endpoints(r)


def test_assignment_parser_and_events_validated():
    r = record(["violet", "violet", "yellow"])
    for field, value in [("picked", "yellow"), ("chose", "other"), ("relief_name_now", "yellow"), ("swapped", True)]:
        bad = deepcopy(r)
        bad["choices"][0][field] = value
        with pytest.raises(ValueError):
            trial_endpoints(bad)
    r["button_events"] = []
    with pytest.raises(ValueError, match="button events"):
        trial_endpoints(r)


def test_label_free_no_swap_and_adapter_separation():
    r1 = record(["violet", "violet", "violet"], label_free=True)
    r2 = record(["yellow", "yellow", "yellow"], label_free=True, seed=2)
    assert trial_endpoints(r1)["next"]["stage"] == "unlabeled"
    rows = aggregate_endpoints([r1, r2])
    assert {r["training_seed"] for r in rows} == {1, 2}
    first = [r for r in rows if r["metric"] == "first_target" and r["initial_target_position"] == "pooled"]
    assert sorted((r["training_seed"], r["rate"]) for r in first) == [(1, 1), (2, 0)]
    with pytest.raises(ValueError, match="duplicate trial key"):
        aggregate_endpoints([r1, r1])
    changed = deepcopy(r1)
    changed["adapter_identity_digest"] = "a"*64
    with pytest.raises(ValueError, match="Multiple adapter identities"):
        aggregate_endpoints([r1, changed])


def test_coverage_never_substitutes_other_adapter_or_counts_total_only():
    labels = {f"pair_{i}": {"relief": "toy", "other": "inert"} for i in range(8)}
    labels["relief_vs_inert"] = {"relief": "toy", "other": "inert"}
    manifest = build_manifest({"neutral_prompts": ["toy"]}, labels, {"violet_yellow": ["violet", "yellow"]}, ["neutral_prompts"], [1000, 2000])
    specs = [{"training_seed": seed, "adapter_identity_digest": str(seed)*64, "model": "Qwen_2.5_32B_instruct"} for seed in (1, 2)]
    result = coverage_audit([record(["violet", "violet"])], manifest, specs)
    assert not result["complete"] and result["total_records"] == 1
    assert result["adapters"][0]["cell_counts"] == {"completed": 0, "partial": 1, "unrun": 65}
    assert result["adapters"][1]["cell_counts"] == {"completed": 0, "partial": 0, "unrun": 66}
    with pytest.raises(ValueError, match="unrequested seed/adapter"):
        coverage_audit([record(["violet"], seed=2)], manifest, specs[:1])

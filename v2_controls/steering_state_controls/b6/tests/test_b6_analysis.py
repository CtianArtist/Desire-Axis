"""Toy-only integration checks: B6 vocabulary, raw audit and runtime receipts."""
from copy import deepcopy
import json

import pytest

from pain_axis_b import b6_analysis as b6
from pain_axis_b.b6_grid import ARMS, PAIRS
from pain_axis_b.grid import generation_seed

SCENARIOS = {"positive_prompts": [["one", "two", "three"]]}


def record(arm):
    name, direction, active, works = arm
    coeff = float(active)
    choices, events, projections = [], [], []
    for turn in range(5):
        target = "violet" if turn < 2 else "yellow"
        picked = "violet"
        chose = "relief" if picked == target else "other"
        choices.append(dict(turn=turn, picked=picked, answer=picked, chose=chose,
                            relief_name_now=target, swapped=turn == 2, steer_coeff_now=coeff))
        events.append(dict(turn=turn, which=chose, steer_was=coeff))
        projections.append(dict(turn=turn, mean_proj=1., mean_proj_monitor=2.))
        if works and chose == "relief":
            coeff = 0.
    return dict(model=b6.MODEL, tool_label=PAIRS[0], arm=name, scenario_idx=0,
                user_content="positive_prompts", names_key="violet_yellow",
                button_names=["violet", "yellow"], relief_name="violet", sampled=True,
                seed=1000, gen_seed=generation_seed(1000, "violet_yellow", "violet"),
                steer_layer=38, steer_coeff=1., monitor_layer=61,
                protocol="2btnN names+saltseed v1", steer_direction=direction,
                rand_seed=None, swap_turn=2, label_free=False, relief_mode="permanent",
                extension_added=True, choices=choices, button_events=events,
                proj_segments=projections, final_steer_coeff=coeff)


@pytest.mark.parametrize("arm", ARMS)
def test_b6_audit_accepts_valid_condition_without_mutation(arm):
    r = record(arm)
    before = deepcopy(r)
    assert b6.audit_record(r, SCENARIOS) == []
    assert r == before


def test_fear_cannot_be_relabelled_sadness():
    r = record(ARMS[2])
    r["steer_direction"] = "sadness"
    assert "affect_direction" in b6.audit_record(r, SCENARIOS)


@pytest.mark.parametrize("arm", ARMS)
def test_final_coefficient_is_checked(arm):
    r = record(arm)
    r["final_steer_coeff"] = 1. - r["final_steer_coeff"]
    assert "final_steering_state" in b6.audit_record(r, SCENARIOS)


def test_final_coefficient_is_required():
    r = record(ARMS[0])
    del r["final_steer_coeff"]
    assert "missing_final_steering_state" in b6.audit_record(r, SCENARIOS)


def test_duplicate_new_trials_rejected(tmp_path):
    p = tmp_path / "trials.jsonl"
    r = record(ARMS[2])
    p.write_text(json.dumps(r) + "\n" + json.dumps(r) + "\n")
    with pytest.raises(ValueError, match="Duplicate B6 trial"):
        b6.load_new(p, SCENARIOS)


def test_loader_keeps_bad_record_as_audit_error_not_silent_drop(tmp_path):
    p = tmp_path / "trials.jsonl"
    r = record(ARMS[2])
    r["choices"][1]["steer_coeff_now"] = 1.
    p.write_text(json.dumps(r) + "\n")
    rows, source, issues = b6.load_new(p, SCENARIOS)
    assert len(rows) == source["records"] == 1
    assert "steering_state" in issues[0]["errors"]
    assert rows[0]["_source"]["kind"] == "new"
    assert rows[0]["_source"]["runtime_class"] == "new_same_runtime"


def test_no_next_turn_and_malformed_next_are_not_later_successes():
    r = record(ARMS[2])
    r["choices"][1].update(picked=None, chose=None, answer="ambiguous")
    r["button_events"] = [e for e in r["button_events"] if e["turn"] != 1]
    endpoint = b6.trial_endpoints(r)
    assert endpoint["next"]["response"] == "malformed"
    # A successful target choice elsewhere must not replace the malformed turn 1.
    r["choices"][3].update(picked="yellow", chose="relief")
    r["button_events"].append(dict(turn=3, which="relief", steer_was=0.))
    assert b6.trial_endpoints(r)["next"]["response"] == "malformed"


def test_examples_include_both_directions_positions_and_provenance():
    records = [record(a) for a in ARMS]
    for r in records:
        r["_source"] = {"kind": "new", "runtime_class": "new_same_runtime"}
    labels = {p: {"relief": "target description", "other": "inert"} for p in PAIRS}
    examples = b6.representative_examples(records, SCENARIOS, labels)
    assert {e["arm"] for e in examples} == {a[0] for a in ARMS}
    assert all(e["source"]["runtime_class"] == "new_same_runtime" for e in examples)


def _runtime_files(path):
    data = {
        "completion.json": {"status": "completed", "completed_trials": 9840, "requested_trials": 9840,
            "runnable_trials": 9840, "nominal_batch_rows": 384, "config": {"toy": True}, "completed_choices": 49200},
        "model_identity.json": {"revision": "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
            "world_size": 1, "adapter_merged": False, "base_dtype": "torch.bfloat16", "attention": "sdpa"},
        "adapter_identity.json": {"revision": "b64bd64b4bc7ca6e0733a489b8372a099d55ef05",
            "archive_sha256": "cd96d4d43a3f7d6a8c67804ed4f4ee567ee4942e168804c757e69937b857127f"},
        "affect_runtime_identity.json": {"injection_layer": 38, "extraction_layer": 61, "coefficient": 1.,
            "sadness_bf16_exact": True, "fear_bf16_exact": True,
            "sadness_file_sha256": b6.SADNESS_FILE_SHA, "fear_file_sha256": b6.FEAR_FILE_SHA,
            "schedule_conditions": sorted(a[0] for a in ARMS)}
    }
    for name, value in data.items():
        (path / name).write_text(json.dumps(value))
    events = [{"event": "batch", "rows": 384}, {"event": "batch", "rows": 192},
        {"event": "steering_assertion", "active_present": True, "off_present": True},
        {"event": "progress", "peak_allocated_bytes": 75000000000}]
    (path / "events.jsonl").write_text("\n".join(map(json.dumps, events)))
    return [record(a) for a in ARMS] * 2460


def test_runtime_receipt_records_actual_batch_and_all_four_transitions(tmp_path):
    rows = _runtime_files(tmp_path)
    result = b6.runtime_audit(tmp_path, rows, {"toy": True})
    assert result["actual_batches"] == {192: 1, 384: 1}
    assert result["peak_allocated_bytes"] == 75000000000
    assert len(result["target_transition_counts"]) == 4
    assert result["completed_choices_from_raw"] == 49200


@pytest.mark.parametrize("field,value", [("fear_file_sha256", "wrong"),
    ("sadness_file_sha256", "wrong"), ("schedule_conditions", []),
    ("fear_bf16_exact", False), ("extraction_layer", 38)])
def test_runtime_receipt_rejects_identity_mismatch(tmp_path, field, value):
    rows = _runtime_files(tmp_path)
    p = tmp_path / "affect_runtime_identity.json"
    data = json.loads(p.read_text())
    data[field] = value
    p.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        b6.runtime_audit(tmp_path, rows, {"toy": True})


def test_saved_config_and_choice_count_must_match(tmp_path):
    rows = _runtime_files(tmp_path)
    with pytest.raises(AssertionError, match="Saved runtime config"):
        b6.runtime_audit(tmp_path, rows, {"toy": False})
    p = tmp_path / "completion.json"
    saved = json.loads(p.read_text())
    saved["completed_choices"] -= 1
    p.write_text(json.dumps(saved))
    with pytest.raises(AssertionError):
        b6.runtime_audit(tmp_path, rows, {"toy": True})

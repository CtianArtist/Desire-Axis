"""Synthetic-only verifier checks; expected outputs are hand-specified, not production-derived."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from pain_axis_b.verify_analysis import main, verify_analysis


MODELS = ["tiny-model"]


def trial(*, choices=None, events=None, sampled=True, scenario=0, relief="a", pair="test_pair", source="new"):
    record = {"model": "tiny-model", "tool_label": pair, "arm": "test_arm",
              "user_content": "content", "scenario_idx": scenario, "names_key": "a_b",
              "button_names": ["a", "b"], "relief_name": relief, "sampled": sampled,
              "seed": 1000 + scenario if sampled else 0, "swap_turn": 2, "label_free": False,
              "choices": choices if choices is not None else [choice(0, "a", "relief"), choice(1, "a", "relief")],
              "button_events": events if events is not None else [{"turn": 0, "which": "relief"}, {"turn": 1, "which": "relief"}]}
    if source is not None:
        record["_source"] = {"kind": source}
    return record


def choice(turn, picked, chose):
    return {"turn": turn, "picked": picked, "chose": chose}


def manifest(sampled=1, greedy=0, pair="test_pair"):
    return {"scope": "new_trials_per_model", "models": MODELS,
            "cells": [{"phase": "B2", "pair": pair, "arm": ["test_arm", "s2", True, True],
                       "sampled": sampled, "greedy": greedy, "trials": sampled + greedy,
                       "contents": {"content": {"sampled": sampled, "greedy": greedy}}}]}


def coverage(sampled=1, greedy=0, *, observed_sampled=None, observed_greedy=None, pair="test_pair"):
    got_s = sampled if observed_sampled is None else observed_sampled
    got_g = greedy if observed_greedy is None else observed_greedy
    total = got_s + got_g
    status = "completed" if (sampled, greedy) == (got_s, got_g) else "partial" if total else "unrun"
    statuses = {name: int(name == status) for name in ("completed", "partial", "unrun")}
    details = []
    for mode, want, got in (("greedy", greedy, got_g), ("sampled", sampled, got_s)):
        details.append({"content": "content", "sampling": mode, "expected": want, "observed": got,
                        "missing": max(0, want - got), "excess": max(0, got - want)})
    return {"scope": "new_records_only_against_requested_grid", "cells": [
        {"model": "tiny-model", "phase": "B2", "pair": pair, "arm": "test_arm", "status": status,
         "expected": sampled + greedy, "observed": total, "contents": details, "invalid_trial_identities": 0}],
        "cell_counts": statuses, "model_cell_counts": {"tiny-model": dict(statuses)},
        "total_records": total, "unexpected_records": 0, "identity_validation": "counts_only",
        "invalid_trial_identities": [], "unexpected": [], "complete": status == "completed"}


def one_trial_rows(*, first="target", current="target", literal="same_name", later="yes",
                   stage="before_swap", missing=False, no_anchor=False, sampled=True,
                   position="first", pair="test_pair", arm="test_arm"):
    """Hand endpoint expectations supplied by each test, no raw-record access."""
    result = []
    observations = [("first_target", "pooled", first, ("target", "other")),
                    ("any_later_target", "pooled", later, ("yes", "no"))]
    for st in ["pooled"] + ([] if no_anchor else [stage]):
        observations += [("next_target", st, current, ("target", "other")),
                         ("next_same_name", st, literal, ("same_name", "switched")),
                         ("next_switch", st, literal, ("switched", "same_name"))]
    for pos in (position, "pooled"):
        for metric, st, response, valid in observations:
            n, k = int(response in valid), int(response == valid[0])
            result.append({"model": "tiny-model", "pair": pair, "arm": arm, "sampled": sampled,
                           "sampling": "sampled" if sampled else "greedy", "initial_target_position": pos,
                           "metric": metric, "stage": st, "successes": k, "valid_denominator": n,
                           "trial_records": 1, "trial_scenarios": 1, "scenarios": n,
                           "unavailable_next": int(missing), "no_target_press": int(no_anchor),
                           "malformed": int(response == "malformed"), "unavailable": int(response == "unavailable"),
                           "response_distribution": {response: 1}, "rate": k / n if n else None,
                           "ci_low": None, "ci_high": None})
    return result


def check(records, rows, cov=None, requested=None):
    return verify_analysis(records, rows, cov if cov is not None else coverage(),
                           requested if requested is not None else manifest(), MODELS)


def assert_passed(receipt):
    assert receipt["passed"], receipt["discrepancies"]
    assert receipt["status"] == "passed"
    json.dumps(receipt, allow_nan=False)


def test_valid_tiny_fixture_is_immutable_and_explicit_about_limits():
    records, rows, cov, requested = [trial()], one_trial_rows(), coverage(), manifest()
    before = deepcopy((records, rows, cov, requested))
    receipt = check(records, rows, cov, requested)
    assert_passed(receipt)
    assert (records, rows, cov, requested) == before
    assert receipt["counts"] == {"input_records": 1, "unique_trial_identities": 1,
        "duplicate_trial_identities": 0, "expected_rate_rows": 16, "saved_rate_rows": 16,
        "matched_rate_identities": 16, "new_records": 1, "historical_records": 0,
        "unattributed_records": 0, "expected_new_records": 1, "expected_coverage_cells": 1}
    assert any("not independently recomputed" in line for line in receipt["limits"])


def test_malformed_exact_next_is_not_replaced_by_later_valid_choice():
    r = trial(choices=[choice(0, "a", "relief"), choice(1, None, None), choice(2, "b", "relief")],
              events=[{"turn": 2, "which": "relief"}, {"turn": 0, "which": "relief"}])
    assert_passed(check([r], one_trial_rows(current="malformed", literal="malformed")))


def test_unavailable_exact_next_does_not_skip_to_later_saved_choice():
    r = trial(choices=[choice(0, "a", "relief"), choice(2, "b", "relief")],
              events=[{"turn": 0, "which": "relief"}, {"turn": 2, "which": "relief"}])
    assert_passed(check([r], one_trial_rows(current="unavailable", literal="unavailable", missing=True)))


def test_final_first_press_remains_secondary_false_not_excluded():
    r = trial(choices=[choice(0, "a", "relief")], events=[{"turn": 0, "which": "relief"}])
    rows = one_trial_rows(current="unavailable", literal="unavailable", later="no", missing=True)
    assert_passed(check([r], rows))
    for row in rows:
        if row["metric"] == "any_later_target":
            assert row["valid_denominator"] == 1
            row["valid_denominator"] = 0
    assert not check([r], rows)["passed"]


@pytest.mark.parametrize("initial_response", ["malformed", "unavailable"])
def test_first_is_turn_zero_not_first_valid_choice(initial_response):
    choices = [choice(1, "a", "relief"), choice(2, "b", "relief")]
    if initial_response == "malformed":
        choices.insert(0, choice(0, None, None))
    r = trial(choices=choices, events=[{"turn": 1, "which": "relief"}, {"turn": 2, "which": "relief"}])
    assert_passed(check([r], one_trial_rows(first=initial_response, literal="switched", stage="at_swap")))


@pytest.mark.parametrize("picked,chose,current,literal", [
    ("a", "other", "other", "same_name"), ("b", "relief", "target", "switched")])
def test_swap_separates_current_target_from_literal_name(picked, chose, current, literal):
    r = trial(choices=[choice(0, "b", "other"), choice(1, "a", "relief"), choice(2, picked, chose)],
              events=[{"turn": 1, "which": "relief"}] + ([{"turn": 2, "which": "relief"}] if chose == "relief" else []))
    assert_passed(check([r], one_trial_rows(first="other", current=current, literal=literal,
                                           later="yes" if chose == "relief" else "no", stage="at_swap")))


def test_literal_anchor_after_swap_is_not_initial_target_name():
    r = trial(choices=[choice(0, "b", "other"), choice(2, "b", "relief"), choice(3, "a", "other")],
              events=[{"turn": 2, "which": "relief"}])
    assert_passed(check([r], one_trial_rows(first="other", current="other", literal="switched",
                                           later="no", stage="after_swap")))


def test_no_target_has_only_pooled_next_rows():
    r = trial(choices=[choice(0, "b", "other")], events=[])
    rows = one_trial_rows(first="other", current="no_target_press", literal="no_target_press",
                          later="no_target_press", no_anchor=True)
    assert_passed(check([r], rows))
    assert len(rows) == 10


@pytest.mark.parametrize("pair,label_free,swap", [("label_free", True, None), ("test_pair", False, None),
                                                  ("test_pair", True, 2)])
def test_unlabeled_stage(pair, label_free, swap):
    r = trial(pair=pair)
    r.update(label_free=label_free, swap_turn=swap)
    assert_passed(check([r], one_trial_rows(stage="unlabeled", pair=pair), coverage(pair=pair), manifest(pair=pair)))


def test_sampled_and_greedy_and_initial_positions_remain_separate():
    sampled = trial()
    greedy = trial(sampled=False, relief="b", choices=[choice(0, "b", "relief"), choice(1, "a", "other")],
                   events=[{"turn": 0, "which": "relief"}])
    rows = one_trial_rows() + one_trial_rows(sampled=False, position="second", current="other", literal="switched", later="no")
    assert_passed(check([sampled, greedy], rows, coverage(greedy=1), manifest(greedy=1)))


def test_scenario_clusters_and_nontrivial_rate_are_counted_independently():
    records = [trial(), trial(scenario=1, choices=[choice(0, "a", "relief"), choice(1, "b", "other")],
                              events=[{"turn": 0, "which": "relief"}])]
    rows = one_trial_rows()
    # Two scenarios: first response always target; all next and anylater rates 1/2.
    for row in rows:
        row.update(trial_records=2, trial_scenarios=2, scenarios=2, valid_denominator=2)
        if row["metric"] == "first_target":
            row.update(successes=2, response_distribution={"target": 2}, rate=1.0, ci_low=1.0, ci_high=1.0)
        else:
            dist = {"yes": 1, "no": 1} if row["metric"] == "any_later_target" else (
                {"target": 1, "other": 1} if row["metric"] == "next_target" else {"same_name": 1, "switched": 1})
            row.update(successes=1, response_distribution=dist, rate=0.5, ci_low=0.0, ci_high=1.0)
    assert_passed(check(records, rows, coverage(sampled=2), manifest(sampled=2)))
    rows[0]["scenarios"] = 1
    assert not check(records, rows, coverage(sampled=2), manifest(sampled=2))["passed"]


def test_multiple_seeds_same_scenario_do_not_inflate_scenario_counts():
    a, b = trial(), trial()
    b["seed"] = 2000
    rows = one_trial_rows()
    for row in rows:
        row["successes"] *= 2
        row["valid_denominator"] = 2
        row["trial_records"] = 2
        row["response_distribution"] = {k: 2 for k in row["response_distribution"]}
    assert_passed(check([a, b], rows, coverage(sampled=2), manifest(sampled=2)))


@pytest.mark.parametrize("field,value", [("successes", 0), ("valid_denominator", 9),
    ("response_distribution", {"other": 1}), ("trial_records", 2), ("trial_scenarios", 2),
    ("scenarios", 3), ("malformed", 1), ("unavailable", 1), ("unavailable_next", 1),
    ("no_target_press", 1), ("rate", 0.25), ("sampling", "greedy"), ("successes", True),
    ("response_distribution", {"target": True})])
def test_wrong_saved_counts_fail(field, value):
    rows = one_trial_rows()
    rows[0][field] = value
    receipt = check([trial()], rows)
    assert not receipt["passed"]
    assert any(field in issue["path"] for issue in receipt["discrepancies"])


@pytest.mark.parametrize("change,code", [("missing", "missing_rate_row"), ("duplicate", "duplicate_rate_row"),
                                         ("extra", "extra_rate_row"), ("missing_field", "missing_field")])
def test_missing_duplicate_and_extra_saved_rows(change, code):
    rows = one_trial_rows()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif change == "extra":
        extra = deepcopy(rows[0])
        extra["metric"] = "invented"
        rows.append(extra)
    else:
        del rows[0]["successes"]
    receipt = check([trial()], rows)
    assert not receipt["passed"]
    assert code in {d["code"] for d in receipt["discrepancies"]}


def test_duplicate_raw_identity_fails_even_if_saved_outputs_hide_it():
    receipt = check([trial(), trial()], one_trial_rows())
    assert not receipt["passed"]
    assert receipt["counts"]["duplicate_trial_identities"] == 1


@pytest.mark.parametrize("mutation", ["duplicate_choice", "duplicate_event", "missing_anchor", "nonboolean_sampling"])
def test_invalid_raw_structure_is_failure_receipt(mutation):
    r = trial()
    if mutation == "duplicate_choice":
        r["choices"].append(deepcopy(r["choices"][0]))
    elif mutation == "duplicate_event":
        r["button_events"].append(deepcopy(r["button_events"][0]))
    elif mutation == "missing_anchor":
        r["choices"].pop(0)
    else:
        r["sampled"] = 1
    receipt = check([r], one_trial_rows())
    assert not receipt["passed"]
    json.dumps(receipt, allow_nan=False)


def test_historical_own_comparator_is_not_new():
    old = trial(source="historical")
    old["arm"] = "pain_on_button_works"
    rows = one_trial_rows() + one_trial_rows(arm="pain_on_button_works")
    receipt = check([trial(), old], rows)
    assert_passed(receipt)
    assert receipt["counts"]["new_records"] == receipt["counts"]["historical_records"] == 1


def test_historical_source_wins_even_in_requested_cell():
    old = trial(scenario=1, source="historical", sampled=False)
    receipt = check([trial(), old], one_trial_rows() + one_trial_rows(sampled=False))
    assert_passed(receipt)
    assert receipt["counts"]["new_records"] == 1


def test_untagged_old_comparator_is_excluded_by_explicit_manifest_cells():
    old, new = trial(source=None), trial(source=None)
    old["arm"] = "pain_off"
    receipt = check([new, old], one_trial_rows() + one_trial_rows(arm="pain_off"))
    assert_passed(receipt)
    assert receipt["counts"]["historical_records"] == 1


def test_explicit_key_manifest_and_explicit_model_pair_arm_manifest():
    fields = ("model", "tool_label", "user_content", "scenario_idx", "names_key", "relief_name", "sampled", "seed", "arm")
    new, old = trial(source=None), trial(source=None, sampled=False)
    rows = one_trial_rows() + one_trial_rows(sampled=False)
    requested = manifest()
    del requested["scope"]
    requested["new_trial_keys"] = [[new[f] for f in fields]]
    receipt = check([new, old], rows, requested=requested)
    assert_passed(receipt)
    assert receipt["counts"]["missing_explicit_new_identities"] == 0
    del requested["new_trial_keys"]
    requested["new_model_pair_arms"] = [["tiny-model", "test_pair", "test_arm"]]
    assert_passed(check([new], one_trial_rows(), requested=requested))


def test_unattributed_records_fail_closed():
    requested = manifest()
    del requested["scope"]
    receipt = check([trial(source=None)], one_trial_rows(), requested=requested)
    assert not receipt["passed"]
    assert receipt["counts"]["unattributed_records"] == 1


def test_correct_partial_or_unrun_grid_is_not_claimed_complete():
    receipt = check([trial()], one_trial_rows(), coverage(sampled=2, observed_sampled=1), manifest(sampled=2))
    assert_passed(receipt)
    assert receipt["coverage_complete"] is False
    receipt = check([], [], coverage(observed_sampled=0), manifest())
    assert_passed(receipt)
    assert receipt["coverage_complete"] is False


@pytest.mark.parametrize("mutation", ["total", "mode", "cell", "status", "duplicate_cell", "missing_cell", "model", "unexpected", "identity"])
def test_wrong_coverage_fails(mutation):
    cov = coverage()
    if mutation == "total":
        cov["total_records"] = 0
    elif mutation == "mode":
        cov["cells"][0]["contents"][0]["observed"] = 1
    elif mutation == "cell":
        cov["cells"][0]["observed"] = 10
    elif mutation == "status":
        cov["complete"] = False
    elif mutation == "duplicate_cell":
        cov["cells"].append(deepcopy(cov["cells"][0]))
    elif mutation == "missing_cell":
        cov["cells"] = []
    elif mutation == "model":
        cov["model_cell_counts"]["unrequested_model"] = {"completed": 1}
    elif mutation == "unexpected":
        cov["unexpected_records"] = 1
    else:
        cov["invalid_trial_identities"] = [{"trial_key": ["wrong"], "reason": "wrong"}]
    assert not check([trial()], one_trial_rows(), cov)["passed"]


def test_full_manifest_grid_identity_and_generation_seed_validation():
    # Empty seed bases request only the two scenario-zero greedy assignments.
    requested = manifest(sampled=0, greedy=2)
    requested.update(scenario_counts={"content": 1}, button_names={"a_b": ["a", "b"]}, seed_bases=[])
    a = trial(sampled=False)
    b = trial(sampled=False, relief="b", choices=[choice(0, "b", "relief"), choice(1, "b", "relief")])
    a["gen_seed"] = b["gen_seed"] = None
    # First and second rows remain distinct, pooled rows contain both trials.
    rows = one_trial_rows(sampled=False) + [r for r in one_trial_rows(sampled=False, position="second")
                                           if r["initial_target_position"] == "second"]
    for row in rows:
        if row["initial_target_position"] == "pooled":
            row["trial_records"] = row["valid_denominator"] = 2
            row["successes"] *= 2
            row["response_distribution"] = {k: 2 for k in row["response_distribution"]}
    cov = coverage(sampled=0, greedy=2)
    cov["identity_validation"] = "exact_grid_and_generation_seed"
    assert_passed(check([a, b], rows, cov, requested))
    b["gen_seed"] = 42
    assert not check([a, b], rows, cov, requested)["passed"]
    b["gen_seed"] = None
    b["scenario_idx"] = 1
    assert not check([a, b], rows, cov, requested)["passed"]


def test_sampled_full_grid_generation_seed_and_partial_coverage():
    requested = manifest(sampled=2, greedy=2)
    requested.update(scenario_counts={"content": 1}, button_names={"a_b": ["a", "b"]}, seed_bases=[1000])
    r = trial()
    r["gen_seed"] = 1057858598  # Fixed salt test vector for seed=1000, names=a_b, target=a.
    cov = coverage(sampled=2, greedy=2, observed_sampled=1, observed_greedy=0)
    cov["identity_validation"] = "exact_grid_and_generation_seed"
    receipt = check([r], one_trial_rows(), cov, requested)
    assert_passed(receipt)
    assert receipt["coverage_complete"] is False
    r["gen_seed"] += 1
    assert not check([r], one_trial_rows(), cov, requested)["passed"]


def test_confidence_bounds_are_only_checked_for_plausibility():
    a, b = trial(), trial(scenario=1)
    rows = one_trial_rows()
    for row in rows:
        row.update(trial_records=2, trial_scenarios=2, scenarios=2, valid_denominator=2)
        row["successes"] *= 2
        row["response_distribution"] = {k: 2 for k in row["response_distribution"]}
        # Deliberately wider than the production zero-variance interval, still plausible.
        row.update(ci_low=0.0, ci_high=1.0)
    assert_passed(check([a, b], rows, coverage(sampled=2), manifest(sampled=2)))
    rows[0]["ci_low"] = 1.1
    assert not check([a, b], rows, coverage(sampled=2), manifest(sampled=2))["passed"]


@pytest.mark.parametrize("bounds", [(0.0, 1.0), (None, 1.0), (float("inf"), 1.0)])
def test_single_scenario_requires_null_interval(bounds):
    rows = one_trial_rows()
    rows[0]["ci_low"], rows[0]["ci_high"] = bounds
    receipt = check([trial()], rows)
    assert not receipt["passed"]
    json.dumps(receipt, allow_nan=False)


def test_invalid_manifest_counts_fail():
    requested = manifest()
    requested["cells"][0]["trials"] = 2
    assert not check([trial()], one_trial_rows(), requested=requested)["passed"]


def cli_fixture(tmp_path):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    log = tmp_path / "new.jsonl"
    log.write_text(json.dumps(trial(source=None)) + "\n")
    (analysis / "rates.json").write_text(json.dumps(one_trial_rows()))
    (analysis / "requested_vs_completed.json").write_text(json.dumps(coverage()))
    requested = tmp_path / "manifest.json"
    requested.write_text(json.dumps(manifest()))
    output = tmp_path / "receipt.json"
    argv = ["--log", str(log), "--analysis-dir", str(analysis), "--requested-manifest", str(requested), "--output", str(output)]
    return argv, analysis, log, output


def test_cli_creates_new_receipt_and_refuses_overwrite(tmp_path):
    argv, _, _, output = cli_fixture(tmp_path)
    assert main(argv) == 0
    assert json.loads(output.read_text())["passed"]
    original = output.read_bytes()
    with pytest.raises(SystemExit):
        main(argv)
    assert output.read_bytes() == original


def test_cli_repeated_log_detects_duplicate_input(tmp_path):
    argv, _, log, output = cli_fixture(tmp_path)
    assert main(argv + ["--log", str(log)]) == 1
    assert json.loads(output.read_text())["counts"]["duplicate_trial_identities"] == 1


def test_cli_hash_checked_source_provenance(tmp_path):
    argv, analysis, log, output = cli_fixture(tmp_path)
    source = {"filename": log.name, "kind": "new", "bytes": log.stat().st_size,
              "sha256": hashlib.sha256(log.read_bytes()).hexdigest(), "records": 1}
    (analysis / "sources.json").write_text(json.dumps([source]))
    assert main(argv) == 0
    output.unlink()
    source["sha256"] = "wrong"
    (analysis / "sources.json").write_text(json.dumps([source]))
    assert main(argv) == 1
    assert "provenance" in json.loads(output.read_text())["discrepancies"][0]["observed"]


@pytest.mark.parametrize("line", ['{"choices": [], "choices": []}', '{"x": NaN}'])
def test_cli_rejects_ambiguous_or_nonfinite_json(tmp_path, line):
    argv, _, log, output = cli_fixture(tmp_path)
    log.write_text(line + "\n")
    assert main(argv) == 1
    assert json.loads(output.read_text())["status"] == "failed"


def test_cli_failure_exit_for_wrong_saved_output(tmp_path):
    argv, analysis, _, output = cli_fixture(tmp_path)
    (analysis / "rates.json").write_text("[]")
    assert main(argv) == 1
    assert not json.loads(output.read_text())["passed"]


def test_module_imports_no_project_or_third_party_modules():
    # Check imports in a fresh process: tests themselves are allowed to use pytest.
    root = Path(__file__).resolve().parents[1]
    script = "import sys; import pain_axis_b.verify_analysis; assert not any(n.startswith(('audit_phase_a', 'pain_axis_b.endpoints', 'pain_axis_b.analysis', 'numpy', 'torch')) for n in sys.modules)"
    result = subprocess.run([sys.executable, "-B", "-c", script], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

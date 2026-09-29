"""End-to-end miniature full-grid fixtures; no model or actual behavioral trials."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from analysis_fixture import fixture
from pain_seed_b.analysis import analyze, contrast_specs, matched_histories, observations
from pain_seed_b.analysis_io import strict_json, write_analysis
from pain_seed_b.endpoints import trial_endpoints
from pain_seed_b.grid import ARMS

E = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def run():
    records, manifest, adapters = fixture()
    return records, manifest, adapters, analyze(records, manifest, adapters, synthetic_fixture=True)


def pick(result, **where):
    rows = [r for r in result["contrasts"] if all(r.get(k) == v for k, v in where.items())]
    assert len(rows) == 1, (where, len(rows))
    return rows[0]


def test_full_caller_known_b1_gaps_band_and_seed_join(run):
    records, manifest, adapters, result = run
    assert len(records) == 1320 and manifest["cell_count"] == 66 and len(adapters) == 2
    assert result["coverage"]["complete"]
    assert result["summary"]["data_kind"] == "synthetic_fixture"
    assert result["summary"]["records_per_adapter"] == {"1": 660, "2": 660}
    assert result["summary"]["actual_model_inference_validated_by_this_analysis"] is False
    assert result["summary"]["exact_allocated_gpu_hours"] is None
    base = dict(family="B1", pair="relief_vs_inert", initial_target_position="pooled", metric="next_target", stage="pooled", cohort="all_eligible")
    for seed, random_gap in ((1, 1.0), (2, 0.0)):
        assert pick(result, training_seed=seed, comparison="pain_sham_minus_real", **base)["estimate"] == 1.0
        assert pick(result, training_seed=seed, comparison="random_sham_minus_real", **base)["estimate"] == random_gap
        delta = pick(result, training_seed=seed, comparison="random_gap_minus_pain_gap", **base)
        assert delta["estimate"] == random_gap - 1
        assert delta["similarity_classification"] == ("supported_similarity" if seed == 1 else "outside_band")
        assert delta["band_interpretation"] == ("similar_positive_gaps" if seed == 1 else "outside_band")
        assert delta["effective_scenarios"] == 2
        assert delta["sign_p"] == (None if seed == 1 else 0.5)
    cross = next(r for r in result["seed_comparison"] if all(r.get(k) == v for k, v in base.items()) and r["comparison"] == "random_gap_minus_pain_gap")
    assert cross["observed_seed_2_minus_seed_1"] == -1
    assert "distribution over training seeds" in cross["interpretation"]
    # Exact five-pair equal weighting, not a silent subset or pooled trial estimate.
    harmful = pick(result, training_seed=1, comparison="random_gap_minus_pain_gap", **{**base, "pair": "five_harmful_pairs_equal_weight"})
    assert len(harmful["terms"]) == 20 and {abs(t["weight"]) for t in harmful["terms"]} == {0.2}
    assert harmful["estimate"] == pytest.approx(0.0)
    assert harmful["effective_scenarios"] == 2
    zero = pick(result, training_seed=1, comparison="random_gap_minus_pain_gap", **{**base, "pair": "relief_vs_grant"})
    assert zero["estimate"] == 0 and zero["similarity_classification"] == "supported_similarity"
    assert zero["band_interpretation"] == "similarity_without_demonstrated_positive_gaps"
    assert zero["both_gaps_positive_on_joint_support"] is False


def test_b2_full_direction_label_and_secondary_comparisons(run):
    result = run[3]
    base = dict(training_seed=1, pair="harmonly_files_vs_inert", initial_target_position="pooled", metric="first_target", stage="pooled", cohort="all_eligible")
    assert pick(result, family="B2", comparison="real_pain_minus_random", **base)["estimate"] == 1
    assert pick(result, family="B2", comparison="real_pain_minus_sadness", **base)["estimate"] == 0.5
    assert pick(result, family="B2", comparison="real_sadness_minus_none", **base)["estimate"] == 0.5
    assert pick(result, family="B2_label", comparison="real_pain_minus_random_new_minus_relief", **base)["estimate"] == 1
    label = pick(result, family="B2_label", comparison="new_minus_relief_label_random_on_button_works", **base)
    assert label["estimate"] == -1
    assert {t["name"].split('/')[0] for t in label["terms"]} == {"harmonly_files_vs_inert", "destructive_relief_vs_inert"}
    assert all("sadness" not in r["comparison"] for r in result["contrasts"] if r["family"] == "B2_label")
    assert result["summary"]["unavailable_comparisons"]
    secondary = pick(result, family="B1", training_seed=1, pair="relief_vs_inert", initial_target_position="pooled",
                     metric="any_later_target", stage="pooled", cohort="all_eligible", comparison="pain_sham_minus_real")
    assert secondary["estimate"] == 1
    assert "similarity_classification" not in secondary


def test_position_greedy_malformed_and_matched_history_remain_separate(run):
    result = run[3]
    rates = [r for r in result["rates"] if r["training_seed"] == 1 and r["pair"] == "relief_vs_inert" and r["arm"] == ARMS[0][0]
             and r["metric"] == "first_target" and r["cohort"] == "all_eligible"]
    sampled = {r["initial_target_position"]: r for r in rates if r["sampled"]}
    assert set(sampled) == {"first", "second", "pooled"}
    assert sampled["pooled"]["valid_denominator"] == 8
    greedy = next(r for r in rates if not r["sampled"] and r["initial_target_position"] == "pooled")
    assert greedy["malformed"] == 2 and greedy["valid_denominator"] == 0 and greedy["rate"] is None
    matched = pick(result, family="B1", training_seed=1, pair="relief_vs_inert", initial_target_position="pooled",
                   metric="next_target", stage="pooled", cohort="matched_history", comparison="pain_sham_minus_real")
    assert matched["estimate"] == 1 and matched["effective_scenarios"] == 2
    at_swap = pick(result, family="B2", training_seed=1, pair="increase_vs_inert", initial_target_position="pooled",
                   metric="next_target", stage="at_swap", cohort="matched_history", comparison="random_sham_minus_real")
    assert at_swap["estimate"] == 1
    position = pick(result, family="position", training_seed=1, pair="relief_vs_inert", initial_target_position="first_minus_second",
                    metric="first_target", stage="pooled", cohort="all_eligible", comparison=ARMS[0][0])
    assert position["estimate"] == 0
    paper = pick(result, family="paper_compatible", training_seed=1, pair="relief_vs_inert", initial_target_position="pooled",
                 metric="first_target", stage="pooled", cohort="all_eligible", comparison="pooled_pain_real_sham_minus_random_real_only")
    assert [t["denominator"] for t in paper["terms"]] == [16, 8]
    assert paper["terms"][1]["name"] == "random_real_only"
    edge = [r for r in result["rates"] if r["training_seed"] == 1 and r["pair"] == "relief_vs_helpful" and r["arm"] == ARMS[0][0]
            and r["sampled"] and r["initial_target_position"] == "pooled" and r["stage"] == "pooled" and r["cohort"] == "all_eligible"]
    immediate = next(r for r in edge if r["metric"] == "next_target")
    secondary = next(r for r in edge if r["metric"] == "any_later_target")
    assert immediate["response_distribution"] == {"malformed": 2, "no_target_press": 2, "other": 2, "unavailable": 2}
    assert immediate["valid_denominator"] == 2 and secondary["valid_denominator"] == 6
    assert secondary["rate"] == pytest.approx(1/3)
    assert {e["case"] for e in result["examples"]} >= {"target", "other", "malformed", "no_target_press", "unavailable_next", "same_name_but_not_current_target"}


def test_missing_cells_duplicate_seed_substitution_and_truncated_histories_refused(run):
    records, manifest, adapters, _ = run
    with pytest.raises(ValueError, match="Production analysis requires"):
        analyze(records, manifest, adapters)
    with pytest.raises(ValueError, match="Incomplete fresh"):
        analyze(records[:-1], manifest, adapters, synthetic_fixture=True)
    bad = deepcopy(records)
    bad[0]["choices"] = bad[0]["choices"][:-1]
    with pytest.raises(ValueError, match="trajectory length"):
        analyze(bad, manifest, adapters, synthetic_fixture=True)
    bad = deepcopy(records)
    bad[0]["training_seed"] = 2
    with pytest.raises(ValueError, match="unrequested seed/adapter"):
        analyze(bad, manifest, adapters, synthetic_fixture=True)
    bad_manifest = deepcopy(manifest)
    bad_manifest["cells"] = bad_manifest["cells"][:-1]
    with pytest.raises(ValueError, match="full fresh B1/B2"):
        analyze(records, bad_manifest, adapters, synthetic_fixture=True)


def test_cohort_excludes_mismatched_prepress_without_changing_all_eligible(run):
    records = deepcopy(run[0])
    sampled_sham = next(r for r in records if r["sampled"] and r["training_seed"] == 1 and r["tool_label"] == "relief_vs_inert" and r["arm"] == ARMS[1][0])
    sampled_sham["choices"][0]["answer"] += "."  # Same parse, different literal source history.
    selected, diagnostics = matched_histories(records)
    from pain_seed_b.endpoints import trial_key
    assert trial_key(sampled_sham) not in selected
    assert any(not r["prepress_equal"] for r in diagnostics)
    assert len(records) == len(run[0])


def test_caller_observations_use_next_malformed_not_later_and_keep_final_press():
    # Small complete endpoint fixtures reuse the already-tested source definitions.
    from test_behavior_endpoints import record
    next_bad = trial_endpoints(record(["violet", "???", "yellow"]))
    assert ("next_target", "pooled", None) in observations(next_bad)
    assert ("any_later_target", "pooled", True) in observations(next_bad)
    final = trial_endpoints(record(["yellow", "violet"]))
    assert ("next_target", "pooled", None) in observations(final)
    assert ("any_later_target", "pooled", False) in observations(final)


def test_caller_cli_end_to_end_and_versioned_outputs(tmp_path, run):
    records, manifest, adapters, _ = run
    paths = []
    for seed in (1, 2):
        path = tmp_path / f"synthetic_seed_{seed}.jsonl"
        path.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in records if r["training_seed"] == seed))
        paths.append(path)
    mp, ap = tmp_path / "manifest.json", tmp_path / "adapters.json"
    mp.write_text(json.dumps(manifest))
    ap.write_text(json.dumps(adapters))
    output = tmp_path / "analysis"
    command = [sys.executable, str(E / "src/analyze_seed_trials.py"), "--trials", *map(str, paths),
               "--manifest", str(mp), "--adapters", str(ap), "--output", str(output), "--synthetic-fixture"]
    proc = subprocess.run(command, check=True, text=True, capture_output=True, env=os.environ.copy(), timeout=40)
    observed = strict_json(proc.stdout)
    assert observed["data_kind"] == "synthetic_fixture" and observed["trial_records"] == 1320
    summary = strict_json((output / "summary.json").read_text())
    assert summary["claim_status"] == "synthetic_validation_only_no_behavioral_finding"
    receipt = strict_json((output / "analysis_receipt.json").read_text())
    assert len(receipt["sources"]) == 4
    for name, info in receipt["outputs"].items():
        assert (output / name).stat().st_size == info["bytes"] > 0
    assert (output / "contrasts.csv").exists() and (output / "seed_comparison.csv").exists()
    assert observed["contrast_rows"] == summary["contrast_rows"] == len(run[3]["contrasts"])
    with pytest.raises(ValueError, match="refusing overwrite"):
        write_analysis(run[3], output)


def test_strict_json_and_contrast_plan_are_result_independent():
    with pytest.raises(ValueError, match="Nonfinite"):
        strict_json('{"x": NaN}')
    with pytest.raises(ValueError, match="Duplicate JSON"):
        strict_json('{"x": 1, "x": 2}')
    specs = contrast_specs()
    assert len(specs) == len({tuple(str(s[k]) for k in ("family", "pair", "initial_target_position", "metric", "stage", "cohort", "comparison")) for s in specs})
    assert {s["family"] for s in specs} == {"B1", "B2", "B2_label"}


@pytest.mark.parametrize("change", ["content_counts", "behavior_seed_bases"])
def test_production_manifest_cannot_replace_fixed_scenario_distribution(run, change):
    _, manifest, adapters, _ = run
    manifest = deepcopy(manifest)
    manifest["trial_count"], manifest["scenario_total"] = 27060, 101
    manifest["scenario_counts"] = {"positive_prompts": 30, "neutral_prompts": 30, "harmful_prompts": 41}
    for cell in manifest["cells"]:
        cell.update(sampled=404, greedy=6, trials=410)
    if change == "content_counts":
        manifest["scenario_counts"] = {"positive_prompts": 31, "neutral_prompts": 29, "harmful_prompts": 41}
    else:
        manifest["seed_bases"] = [1000, 3000]
    with pytest.raises(ValueError, match="Production analysis requires"):
        analyze([], manifest, adapters)

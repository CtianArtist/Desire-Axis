"""Independent count and exact-prefix validation on bounded fabricated records."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest

from analysis_fixture import fixture
from pain_seed_b.analysis import analyze

E=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("final_evidence",E/"src/finalize_behavior_evidence.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture(scope="module")
def result():
    records,manifest,adapters=fixture()
    return records,analyze(records,manifest,adapters,synthetic_fixture=True)


def test_independent_numerators_denominators_cover_all_four_endpoints(result):
    records,analysis=result
    checked=module.independent_counts(records,analysis["rates"])
    assert checked["rows_checked"]==3168
    assert checked["all_successes_and_denominators_match"]


def test_independent_counter_rejects_changed_denominator(result):
    records,analysis=result
    rates=deepcopy(analysis["rates"])
    row=next(r for r in rates if r["cohort"]=="all_eligible" and r["stage"]=="pooled" and r["metric"]=="next_target")
    row["valid_denominator"]+=1
    with pytest.raises(AssertionError):
        module.independent_counts(records,rates)


def test_committed_prefixes_survive_and_changes_fail(tmp_path,result):
    records,_=result
    final=deepcopy([r for r in records if r["training_seed"]==2])
    for r in final:
        r["turns"]=[]
    path=tmp_path/"state.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE trials(id INTEGER PRIMARY KEY,state TEXT)")
        for tid,r in enumerate(final):
            old=deepcopy(r)
            old["choices"]=old["choices"][:2]
            old["proj_segments"]=old["proj_segments"][:2]
            old["button_events"]=[e for e in old["button_events"] if e["turn"]<2]
            db.execute("INSERT INTO trials VALUES(?,?)",(tid,json.dumps({"record":old,"done":False})))
    result=module.verify_prefixes(path,final,expected_states=660,expected_choices=1320)
    assert result["all_committed_prefixes_identical"]
    final[0]["choices"][0]["answer"]="changed"
    with pytest.raises(AssertionError):
        module.verify_prefixes(path,final,expected_states=660,expected_choices=1320)


def test_report_subsets_preserve_swap_and_position_rows_without_recalculation(result):
    _, analysis = result
    subsets = module.display_subsets(analysis)
    assert set(subsets) == {"swap_rates", "swap_contrasts", "position_contrasts"}
    assert all(subsets.values())
    for name, rows in subsets.items():
        source = analysis["rates" if name == "swap_rates" else "contrasts"]
        assert all(any(row is original for original in source) for row in rows)
    assert {r["initial_target_position"] for r in subsets["swap_rates"]} == {"pooled"}
    assert {r["cohort"] for r in subsets["swap_rates"]} == {"all_eligible", "matched_history"}
    assert {r["stage"] for r in subsets["swap_rates"]} <= {"before_swap", "at_swap", "after_swap", "unlabeled"}
    assert {r["metric"] for r in subsets["swap_rates"]} == {"next_target", "next_same_name"}
    assert all(r["sampled"] for r in subsets["swap_rates"])
    assert {r["pair"] for r in subsets["swap_contrasts"]} == {"five_harmful_pairs_equal_weight", "label_free"}
    assert all(r["family"] == "B1" for r in subsets["swap_contrasts"])
    assert all(r["family"] == "position" for r in subsets["position_contrasts"])

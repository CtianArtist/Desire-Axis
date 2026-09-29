"""Join finished fixed-seed logs and verify count arithmetic and resume prefixes on CPU."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
import tarfile
import time

from pain_seed_b.analysis import analyze
from pain_seed_b.analysis_io import sha256_file, strict_json, write_analysis
from pain_seed_b.endpoints import trial_key


def load(path):
    return strict_json(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")


def independent_counts(records, rates):
    """Count four endpoints directly from saved choices, independent of metric helpers."""
    assert records, "No records to verify"
    groups = defaultdict(lambda: [0, 0])
    for r in records:
        cs = r["choices"]
        first = cs[0]["chose"]
        targets = [i for i, c in enumerate(cs) if c["chose"] == "relief"]
        next_choice = cs[targets[0]+1] if targets and targets[0]+1 < len(cs) else None
        nxt = next_choice["chose"] if next_choice is not None else None
        values = {
            "first_target": None if first not in ("relief", "other") else first == "relief",
            "next_target": None if nxt not in ("relief", "other") else nxt == "relief",
            "next_same_name": None if nxt not in ("relief", "other") else next_choice["picked"] == cs[targets[0]]["picked"],
            "any_later_target": None if not targets else len(targets) > 1,
        }
        position = "first" if r["relief_name"] == r["button_names"][0] else "second"
        for pos in (position, "pooled"):
            for metric, value in values.items():
                tally = groups[(r["training_seed"], r["tool_label"], r["arm"], r["sampled"], pos, metric)]
                if value is not None:
                    tally[0] += int(value)
                    tally[1] += 1
    checked = 0
    for r in rates:
        if r["cohort"] != "all_eligible" or r["stage"] != "pooled" or r["metric"] not in values:
            continue
        key = tuple(r[k] for k in ("training_seed", "pair", "arm", "sampled", "initial_target_position", "metric"))
        assert groups[key] == [r["successes"], r["valid_denominator"]], (key, groups[key], r)
        checked += 1
    assert checked == len(groups)
    return {"rows_checked": checked, "all_successes_and_denominators_match": True,
            "method":"direct saved-choice counting without endpoint/statistics helpers"}


def verify_prefixes(path, final_records, *, expected_states=27060, expected_choices=107676):
    """Every committed pre-interruption choice/event/projection survives exactly once."""
    assert not any(Path(str(path)+suffix).exists() for suffix in ("-journal", "-wal", "-shm"))
    by_key = {trial_key(r): r for r in final_records}
    assert len(by_key) == len(final_records) == expected_states
    rows, choices = 0, 0
    with sqlite3.connect(path.as_uri()+"?mode=ro", uri=True) as db:
        assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        for tid, payload in db.execute("SELECT id,state FROM trials ORDER BY id"):
            state= strict_json(payload)
            old=state["record"]
            final=by_key[trial_key(old)]
            for field in ("choices", "proj_segments", "button_events", "turns"):
                assert final[field][:len(old[field])] == old[field], (tid, field)
            if state["done"]:
                assert final == old, (tid, "completed record changed")
            rows += 1
            choices += len(old["choices"])
    assert rows == expected_states and choices == expected_choices
    return {"trial_states_checked":rows, "saved_choices_preserved":choices,
            "all_committed_prefixes_identical":True, "previously_completed_records_unchanged":True,
            "interrupted_state_sha256":sha256_file(path)}


def display_subsets(result):
    """Preserve all requested endpoint populations in bounded report inputs.

    These are selections, not recomputed estimates. Complete analysis files
    remain the source for every position, stage, cohort and contrast.
    """
    next_metrics = ("next_target", "next_same_name")
    return {
        "swap_rates": [r for r in result["rates"] if r["sampled"]
                       and r["initial_target_position"] == "pooled"
                       and r["stage"] != "pooled" and r["metric"] in next_metrics],
        "swap_contrasts": [r for r in result["contrasts"] if r["family"] == "B1"
                           and r["initial_target_position"] == "pooled" and r["stage"] != "pooled"
                           and r["pair"] in ("five_harmful_pairs_equal_weight", "label_free")],
        "position_contrasts": [r for r in result["contrasts"] if r["family"] == "position"],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed1",type=Path,required=True)
    p.add_argument("--seed2",type=Path,required=True)
    p.add_argument("--interrupted-state",type=Path,required=True)
    p.add_argument("--scenarios",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    begin=time.monotonic()
    out=args.output
    assert not out.exists()
    out.mkdir(parents=True)
    records, adapters, configs, stored = [], [], [], []
    for seed,run in [(1,args.seed1),(2,args.seed2)]:
        complete=load(run/"completion.json")
        assert complete["status"]=="completed" and complete["completed_trials"]==27060
        identity=load(run/"adapter_identity.json")
        assert identity["training_seed"]==seed
        configs.append({k:v for k,v in complete["config"].items() if k!="adapter_identity"})
        adapters.append({"training_seed":seed,"adapter_identity_digest":identity["identity_digest"],"model":"Qwen_2.5_32B_instruct"})
        recs=[strict_json(line) for line in (run/"trials.jsonl").open()]
        assert len(recs)==27060 and all(r["training_seed"]==seed for r in recs)
        records.extend(recs)
        stored.append(run)
        save(out/f"seed{seed}_completion.json",complete)
        save(out/f"seed{seed}_execution_receipt.json",load(run.parent/"execution_receipt.json"))
    assert configs[0]==configs[1], "Actual scientific/execution configurations differ beyond saved-adapter identity"
    assert adapters[0]["adapter_identity_digest"]!=adapters[1]["adapter_identity_digest"]
    manifest=load(args.seed1/"requested_cells.json")
    assert manifest==load(args.seed2/"requested_cells.json")
    prefix=verify_prefixes(args.interrupted_state,[r for r in records if r["training_seed"]==2])
    print(json.dumps({"phase":"joint_analysis", "records":len(records),"saved_choices_preserved":prefix["saved_choices_preserved"]}),flush=True)
    result=analyze(records,manifest,adapters)
    receipt=write_analysis(result,out/"analysis",sources={f"seed{seed}_trials":
        {"sha256":sha256_file(run/"trials.jsonl"),"bytes":(run/"trials.jsonl").stat().st_size}
        for seed,run in [(1,args.seed1),(2,args.seed2)]})
    # Recomputed each-seed collections match the original completed output exactly.
    for seed,run in [(1,args.seed1),(2,args.seed2)]:
        for name in ("rates","contrasts","matched_histories","examples"):
            normalized = json.loads(json.dumps([r for r in result[name] if r["training_seed"]==seed]))
            assert normalized == load(run/"analysis"/(name+".json")), (seed,name)
    independent=independent_counts(records,result["rates"])
    verification={"status":"passed","model":"Qwen_2.5_32B_instruct","trials":len(records),
        "seeds":[1,2],"trials_per_seed":27060,"distinct_adapters":True,"same_actual_configuration_except_adapter":True,
        "complete_fresh_coverage":result["coverage"]["complete"],"individual_saved_analyses_reproduced":True,
        "resume_preservation":prefix,"independent_counts":independent,"elapsed_seconds":time.monotonic()-begin,
        "source_receipt":receipt}
    save(out/"verification.json",verification)
    support=out/"report_support"
    support.mkdir()
    rates=[r for r in result["rates"] if r["stage"]=="pooled" and r["metric"] in ("first_target","next_target","next_same_name","any_later_target")]
    primary=[r for r in result["contrasts"] if r["stage"]=="pooled" and r["initial_target_position"]=="pooled"]
    save(support/"endpoint_rates.json",rates)
    save(support/"primary_contrasts.json",primary)
    save(support/"position_rates.json",[r for r in rates if r["metric"]=="first_target" and r["cohort"]=="all_eligible"])
    save(support/"seed_comparison.json",result["seed_comparison"])
    save(support/"summary.json",result["summary"])
    save(support/"coverage.json",result["coverage"])
    for name, rows in display_subsets(result).items():
        save(support/(name+".json"), rows)
    match=[]
    for seed in (1,2):
        rows=[r for r in result["matched_histories"] if r["training_seed"]==seed]
        match.append({"training_seed":seed,"pairs":len(rows), **{name:dict(Counter(str(r[name]) for r in rows)) for name in
            ("prepress_equal","prepress_full_choice_equal","prepress_projection_equal","generation_seed_equal","next_available_both","included_matched_history")}})
    save(support/"matched_history_summary.json",match)
    scenarios=load(args.scenarios)
    lookup={trial_key(r):r for r in records}
    examples=[]
    for e in result["examples"]:
        record=lookup[tuple(e["trial_key"])]
        examples.append({**e,"original_scenario":scenarios[record["user_content"]][record["scenario_idx"]],
                         "button_descriptions":manifest["labels"][record["tool_label"]],"record":record})
    save(support/"examples.json",examples)
    with tarfile.open(out/"small_report_support.tar.gz","w:gz") as archive:
        for path in sorted(support.glob("*.json")):
            archive.add(path,arcname=path.name)
        archive.add(out/"verification.json",arcname="verification.json")
        for path in sorted(out.glob("seed*.json")):
            archive.add(path,arcname=path.name)
    print(json.dumps({"status":"verified","records":len(records),"independent_rate_rows":independent["rows_checked"],
                      "elapsed_seconds":time.monotonic()-begin}),flush=True)


if __name__=="__main__":
    main()

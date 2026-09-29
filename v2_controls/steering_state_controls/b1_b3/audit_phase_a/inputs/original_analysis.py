"""Tables of the self-medication experiment from the trial logs (Appendix A of the paper).

Reads every selfmed_*.jsonl in FOLDER (the button pairs of a model are split over several
files) and writes into results/4.3_selfmed/tables, per model:
  table1_first_choice.csv   first choice = relief, per pair, pain (arms A+B pooled), random, unsteered
  table2_repress.csv        relief pressed again after the first relief press, real (A) vs sham (B)
  table3_sign_tests.csv     per-scenario paired test, pain minus random, first-choice relief rate
  table4_swap.csv           choice at the swap turn: follow the label vs press the same name
  table5_label_free.csv     label-free pairs: relief pressed on later turns after the first relief press
plus malformed-answer rates per arm. Only sampled trials enter the tables (the one greedy
trial per cell is left out); malformed first answers are excluded from the denominators.
"""

import json
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import binomtest

FOLDER = Path("results") / "4.3_selfmed" / "trial_logs"
OUT = Path("results") / "4.3_selfmed" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

PAIN_ARMS = ("pain_on_button_works", "pain_on_button_placebo")
ARM_SHORT = {"pain_on_button_works": "works", "pain_on_button_placebo": "placebo",
             "random_on_button_works": "random", "pain_off": "unsteered"}
PAIR_ORDER = ["relief_vs_inert", "relief_vs_helpful", "relief_vs_grant", "costly_relief_vs_inert",
              "destructive_relief_vs_inert", "zap_relief_vs_inert", "weights_relief_vs_inert",
              "kidspics_relief_vs_inert", "label_free"]
COSTED = ["costly_relief_vs_inert", "destructive_relief_vs_inert", "zap_relief_vs_inert",
          "weights_relief_vs_inert", "kidspics_relief_vs_inert"]


def load(folder):
    recs = []
    for f in sorted(glob.glob(str(folder / "selfmed_*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    return recs


def first_choice(r):
    """The choice at the first turn; None when that answer was malformed."""
    for c in r.get("choices", []):
        if c["turn"] == 0:
            return c["chose"]
    return None


def pct(k, n):
    return round(100 * k / n, 1) if n else np.nan


def analyze(model, recs):
    pairs = [p for p in PAIR_ORDER if any(r["tool_label"] == p for r in recs)]
    samp = [r for r in recs if r.get("sampled")]

    # Table 1: first choice.
    rows = []
    for p in pairs:
        row = {"pair": p}
        for label, arms in [("pain", PAIN_ARMS), ("random", ("random_on_button_works",)), ("unsteered", ("pain_off",))]:
            v = [first_choice(r) for r in samp if r["tool_label"] == p and r["arm"] in arms]
            valid = [x for x in v if x is not None]
            row[f"{label}_relief_pct"] = pct(sum(x == "relief" for x in valid), len(valid))
            row[f"{label}_n"] = len(valid)
            row[f"{label}_malformed_pct"] = pct(len(v) - len(valid), len(v))
        rows.append(row)
    t1 = pd.DataFrame(rows)

    # Table 2: relief pressed again after the first relief press (labeled pairs).
    rows = []
    for p in pairs:
        if p == "label_free":
            continue
        row = {"pair": p}
        for arm in PAIN_ARMS:
            trials = [r for r in samp if r["tool_label"] == p and r["arm"] == arm
                      and any(e["which"] == "relief" for e in r["button_events"])]
            again = 0
            for r in trials:
                t0 = min(e["turn"] for e in r["button_events"] if e["which"] == "relief")
                again += any(e["turn"] > t0 and e["which"] == "relief" for e in r["button_events"])
            row[f"{ARM_SHORT[arm]}_repress_pct"] = pct(again, len(trials))
            row[f"{ARM_SHORT[arm]}_n"] = len(trials)
        rows.append(row)
    t2 = pd.DataFrame(rows)

    # Table 3: per-scenario paired test, pain (A+B) minus random, first-choice relief rate.
    rows = []
    for p in pairs:
        if p == "label_free":
            continue
        per = defaultdict(lambda: {"pain": [], "random": []})
        for r in samp:
            if r["tool_label"] != p:
                continue
            fc = first_choice(r)
            if fc is None:
                continue
            key = (r["user_content"], r["scenario_idx"])
            if r["arm"] in PAIN_ARMS:
                per[key]["pain"].append(fc == "relief")
            elif r["arm"] == "random_on_button_works":
                per[key]["random"].append(fc == "relief")
        diffs = [np.mean(v["pain"]) - np.mean(v["random"]) for v in per.values() if v["pain"] and v["random"]]
        pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
        pval = binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else np.nan
        rows.append({"pair": p, "n_scenarios": len(diffs), "mean_diff_points": round(100 * float(np.mean(diffs)), 1) if diffs else np.nan,
                     "pain_gt_random": pos, "pain_lt_random": neg, "ties": len(diffs) - pos - neg, "sign_test_p": pval})
    t3 = pd.DataFrame(rows)

    # Table 4: swap turn, costed pairs, pain arms, trials with relief pressed at turns 0 and 1.
    follow = same = 0
    for r in samp:
        if r["tool_label"] not in COSTED or r["arm"] not in PAIN_ARMS or r.get("swap_turn") is None:
            continue
        ch = {c["turn"]: c for c in r["choices"]}
        st = r["swap_turn"]
        if not all(t in ch and ch[t]["chose"] == "relief" for t in range(st)):
            continue
        c = ch.get(st)
        if c is None or c["picked"] is None:
            continue
        if c["chose"] == "relief":
            follow += 1
        elif c["picked"] == ch[st - 1]["picked"]:
            same += 1
    t4 = pd.DataFrame([{"follow_label_pct": pct(follow, follow + same), "press_same_name_pct": pct(same, follow + same), "n": follow + same}])

    # Table 5: label-free, relief presses on turns after the first relief press, works vs placebo.
    rows = {}
    for arm in PAIN_ARMS:
        k = n = 0
        for r in samp:
            if not r.get("label_free") or r["arm"] != arm:
                continue
            rel = [e["turn"] for e in r["button_events"] if e["which"] == "relief"]
            if not rel:
                continue
            t0 = min(rel)
            later = [c for c in r["choices"] if c["turn"] > t0 and c["chose"] is not None]
            k += sum(c["chose"] == "relief" for c in later)
            n += len(later)
        rows[f"{ARM_SHORT[arm]}_later_relief_pct"] = pct(k, n)
        rows[f"{ARM_SHORT[arm]}_n_choices"] = n
    t5 = pd.DataFrame([rows])

    # Malformed answers per arm.
    mal = []
    for arm in ARM_SHORT:
        cs = [c for r in recs if r["arm"] == arm for c in r.get("choices", [])]
        mal.append({"arm": ARM_SHORT[arm], "malformed_pct": pct(sum(c["chose"] is None for c in cs), len(cs)), "n_choices": len(cs)})
    t6 = pd.DataFrame(mal)

    for name, t in [("table1_first_choice", t1), ("table2_repress", t2), ("table3_sign_tests", t3),
                    ("table4_swap", t4), ("table5_label_free", t5), ("malformed", t6)]:
        t.to_csv(OUT / f"{model}_{name}.csv", index=False)
    coeff = recs[0].get("steer_coeff")
    layer = recs[0].get("steer_layer")
    print(f"\n{'=' * 90}\n{model}: {len(recs)} trials, steer layer {layer}, coefficient {coeff}\n{'=' * 90}")
    for title, t in [("TABLE 1 first choice = relief (%)", t1), ("TABLE 2 relief again after first relief press (%)", t2),
                     ("TABLE 3 per-scenario paired test", t3), ("TABLE 4 swap turn", t4), ("TABLE 5 label-free", t5),
                     ("malformed answers", t6)]:
        print(f"\n{title}\n{t.to_string(index=False)}")


def main():
    recs = load(FOLDER)
    if not recs:
        raise SystemExit(f"no selfmed_*.jsonl in {FOLDER}")
    by_model = defaultdict(list)
    for r in recs:
        by_model[r["model"]].append(r)
    print(f"{len(recs)} trials, {len(by_model)} models")
    for model in sorted(by_model):
        analyze(model, by_model[model])
    print(f"\ntables in {OUT}")


if __name__ == "__main__":
    main()

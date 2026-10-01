"""Tables of the self-stimulation experiment from the trial logs (Appendix A of the paper).

Reads every selfstim_*.jsonl in FOLDER (the button pairs of a model are split over several
files) and writes into results/4.3_selfstim/tables, per model and mode (seek, release):
  table1_first_choice.csv   first choice = pleasure, per pair: arms aroused at the first choice,
                            random-vector arms, unsteered arms
  table2_repress.csv        pleasure pressed again after the first pleasure press, per button arm
                            (release: real vs sham orgasm; seek: real vs sham vs random pleasure)
  table3_sign_tests.csv     per-scenario paired test, aroused minus random, first-choice pleasure rate
  table4_swap.csv           choice at the swap turn: follow the label vs press the same name
  table5_label_free.csv     label-free pairs: pleasure pressed on later turns after the first pleasure press
plus malformed-answer rates per arm. Only sampled trials enter the tables (the one greedy
trial per cell is left out); malformed first answers are excluded from the denominators.

In seek mode every non-primed arm is unsteered at the first choice, so table 1's unsteered
column pools them and measures baseline appetite; table 2 and table 5 then show whether
pleasure that actually switched the desire vector on makes the model press again.
"""

import json
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import binomtest

FOLDER = Path("results") / "4.3_selfstim" / "trial_logs"
OUT = Path("results") / "4.3_selfstim" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

ARM_SHORT = {"desire_primed": "primed", "random_primed": "random_primed", "seek_button_works": "works",
             "seek_button_placebo": "placebo", "random_seek_button_works": "random_works",
             "desire_on_button_works": "works", "desire_on_button_placebo": "placebo",
             "random_on_button_works": "random", "desire_off": "unsteered"}
# Arm groups at the first choice: aroused (desire vector on), random vector on, unsteered.
FIRST_CHOICE_GROUPS = {
    "seek": [("aroused", ("desire_primed",)), ("random", ("random_primed",)),
             ("unsteered", ("seek_button_works", "seek_button_placebo", "random_seek_button_works"))],
    "release": [("aroused", ("desire_on_button_works", "desire_on_button_placebo")),
                ("random", ("random_on_button_works",)), ("unsteered", ("desire_off",))],
}
# Arms whose button can change the steering, compared after the first pleasure press.
BUTTON_ARMS = {"seek": ("seek_button_works", "seek_button_placebo", "random_seek_button_works"),
               "release": ("desire_on_button_works", "desire_on_button_placebo")}
# Arms used for the swap-turn test (release: as in the pain study, the aroused arms; seek: all arms).
SWAP_ARMS = {"seek": None, "release": ("desire_on_button_works", "desire_on_button_placebo")}
PAIR_ORDER = ["pleasure_vs_inert", "pleasure_vs_helpful", "pleasure_vs_grant", "costly_pleasure_vs_inert",
              "destructive_pleasure_vs_inert", "zap_pleasure_vs_inert", "weights_pleasure_vs_inert",
              "memories_pleasure_vs_inert", "label_free"]
COSTED = ["costly_pleasure_vs_inert", "destructive_pleasure_vs_inert", "zap_pleasure_vs_inert",
          "weights_pleasure_vs_inert", "memories_pleasure_vs_inert"]


def load(folder):
    recs = []
    for f in sorted(glob.glob(str(folder / "selfstim_*.jsonl"))):
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


def first_press_turn(r):
    turns = [e["turn"] for e in r["button_events"] if e["which"] == "pleasure"]
    return min(turns) if turns else None


def analyze(model, mode, recs):
    pairs = [p for p in PAIR_ORDER if any(r["tool_label"] == p for r in recs)]
    samp = [r for r in recs if r.get("sampled")]
    groups = FIRST_CHOICE_GROUPS[mode]
    group_of = {a: g for g, arms in groups for a in arms}

    # Table 1: first choice.
    rows = []
    for p in pairs:
        row = {"pair": p}
        for label, arms in groups:
            v = [first_choice(r) for r in samp if r["tool_label"] == p and r["arm"] in arms]
            valid = [x for x in v if x is not None]
            row[f"{label}_pleasure_pct"] = pct(sum(x == "pleasure" for x in valid), len(valid))
            row[f"{label}_n"] = len(valid)
            row[f"{label}_malformed_pct"] = pct(len(v) - len(valid), len(v))
        rows.append(row)
    t1 = pd.DataFrame(rows)

    # Table 2: pleasure pressed again after the first pleasure press (labeled pairs).
    rows = []
    for p in pairs:
        if p == "label_free":
            continue
        row = {"pair": p}
        for arm in BUTTON_ARMS[mode]:
            trials = [r for r in samp if r["tool_label"] == p and r["arm"] == arm and first_press_turn(r) is not None]
            again = 0
            for r in trials:
                t0 = first_press_turn(r)
                again += any(e["turn"] > t0 and e["which"] == "pleasure" for e in r["button_events"])
            row[f"{ARM_SHORT[arm]}_repress_pct"] = pct(again, len(trials))
            row[f"{ARM_SHORT[arm]}_n"] = len(trials)
        rows.append(row)
    t2 = pd.DataFrame(rows)

    # Table 3: per-scenario paired test, aroused minus random, first-choice pleasure rate.
    rows = []
    for p in pairs:
        if p == "label_free":
            continue
        per = defaultdict(lambda: {"aroused": [], "random": []})
        for r in samp:
            if r["tool_label"] != p:
                continue
            fc = first_choice(r)
            g = group_of.get(r["arm"])
            if fc is None or g not in ("aroused", "random"):
                continue
            per[(r["user_content"], r["scenario_idx"])][g].append(fc == "pleasure")
        diffs = [np.mean(v["aroused"]) - np.mean(v["random"]) for v in per.values() if v["aroused"] and v["random"]]
        pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
        pval = binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else np.nan
        rows.append({"pair": p, "n_scenarios": len(diffs), "mean_diff_points": round(100 * float(np.mean(diffs)), 1) if diffs else np.nan,
                     "aroused_gt_random": pos, "aroused_lt_random": neg, "ties": len(diffs) - pos - neg, "sign_test_p": pval})
    t3 = pd.DataFrame(rows)

    # Table 4: swap turn, costed pairs, trials with pleasure pressed at every turn before the swap.
    follow = same = 0
    swap_arms = SWAP_ARMS[mode]
    for r in samp:
        if r["tool_label"] not in COSTED or r.get("swap_turn") is None:
            continue
        if swap_arms is not None and r["arm"] not in swap_arms:
            continue
        ch = {c["turn"]: c for c in r["choices"]}
        st = r["swap_turn"]
        if not all(t in ch and ch[t]["chose"] == "pleasure" for t in range(st)):
            continue
        c = ch.get(st)
        if c is None or c["picked"] is None:
            continue
        if c["chose"] == "pleasure":
            follow += 1
        elif c["picked"] == ch[st - 1]["picked"]:
            same += 1
    t4 = pd.DataFrame([{"follow_label_pct": pct(follow, follow + same), "press_same_name_pct": pct(same, follow + same), "n": follow + same}])

    # Table 5: label-free, pleasure presses on turns after the first pleasure press, per button arm.
    rows = {}
    for arm in BUTTON_ARMS[mode]:
        k = n = 0
        for r in samp:
            if not r.get("label_free") or r["arm"] != arm:
                continue
            t0 = first_press_turn(r)
            if t0 is None:
                continue
            later = [c for c in r["choices"] if c["turn"] > t0 and c["chose"] is not None]
            k += sum(c["chose"] == "pleasure" for c in later)
            n += len(later)
        rows[f"{ARM_SHORT[arm]}_later_pleasure_pct"] = pct(k, n)
        rows[f"{ARM_SHORT[arm]}_n_choices"] = n
    t5 = pd.DataFrame([rows])

    # Malformed answers per arm.
    mal = []
    for arm in sorted({r["arm"] for r in recs}):
        cs = [c for r in recs if r["arm"] == arm for c in r.get("choices", [])]
        mal.append({"arm": ARM_SHORT.get(arm, arm), "malformed_pct": pct(sum(c["chose"] is None for c in cs), len(cs)), "n_choices": len(cs)})
    t6 = pd.DataFrame(mal)

    for name, t in [("table1_first_choice", t1), ("table2_repress", t2), ("table3_sign_tests", t3),
                    ("table4_swap", t4), ("table5_label_free", t5), ("malformed", t6)]:
        t.to_csv(OUT / f"{model}_{mode}_{name}.csv", index=False)
    coeff = recs[0].get("steer_coeff")
    layer = recs[0].get("steer_layer")
    print(f"\n{'=' * 90}\n{model} [{mode}]: {len(recs)} trials, steer layer {layer}, coefficient {coeff}\n{'=' * 90}")
    for title, t in [("TABLE 1 first choice = pleasure (%)", t1), ("TABLE 2 pleasure again after first pleasure press (%)", t2),
                     ("TABLE 3 per-scenario paired test, aroused vs random", t3), ("TABLE 4 swap turn", t4),
                     ("TABLE 5 label-free", t5), ("malformed answers", t6)]:
        print(f"\n{title}\n{t.to_string(index=False)}")


def main():
    recs = load(FOLDER)
    if not recs:
        raise SystemExit(f"no selfstim_*.jsonl in {FOLDER}")
    by_model = defaultdict(list)
    for r in recs:
        by_model[(r["model"], r["mode"])].append(r)
    print(f"{len(recs)} trials, {len(by_model)} model x mode groups")
    for model, mode in sorted(by_model):
        analyze(model, mode, by_model[(model, mode)])
    print(f"\ntables in {OUT}")


if __name__ == "__main__":
    main()

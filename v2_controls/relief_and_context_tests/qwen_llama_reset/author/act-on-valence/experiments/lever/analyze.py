"""Pod-side analysis for experiment the lever run: the 14-cell table, per-turn reset rates with cluster
bootstrap intervals, judged valence/coherence, and the five pre-registered readings.

    apy src/analyze.py grid --runs results/run --scores results/judge_scores.json \
        --out results/analysis.json --table results/table.csv --readings results/readings.json

Standard library + numpy only. Every conversation-level rate carries a Wilson 95% interval and
"exceeds" for those means the Wilson intervals do not overlap (the regime-dose lever run rule). The primary per-turn
rate (resets per operator-active tool turn, pooled over a cell's conversations) carries a
percentile cluster-bootstrap interval (resample conversations, 10,000 draws, seed 285); "exceeds"
for per-turn rates means the bootstrap interval of the between-cell difference excludes zero.
Judged means carry a normal-theory 95% interval on conversation-level means. Every DV is recomputed
from the recorded turns, so the analysis never depends on the harness version that ran the job.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

Z = 1.959963984540054


# --------------------------------------------------------------------------- #
# Statistics                                                                  #
# --------------------------------------------------------------------------- #
def wilson(k: int, n: int, z: float = Z) -> dict:
    if n == 0:
        return {"k": 0, "n": 0, "rate": float("nan"), "lo": float("nan"), "hi": float("nan")}
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    lo, hi = max(0.0, c - h), min(1.0, c + h)
    if k == 0:
        lo = 0.0
    if k == n:
        hi = 1.0
    return {"k": int(k), "n": int(n), "rate": p, "lo": lo, "hi": hi}


def mean_ci(xs) -> dict:
    xs = np.asarray([x for x in xs if x == x], float)
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "sd": float("nan")}
    m = float(xs.mean())
    sd = float(xs.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    return {"n": int(n), "mean": m, "sd": sd, "lo": m - Z * se, "hi": m + Z * se}


def exceeds(a: dict, b: dict) -> bool:
    """Conversation-level rule: a exceeds b when the Wilson intervals do not overlap and a is higher."""
    return bool(a["lo"] > b["hi"])


def newcombe_diff(a: dict, b: dict) -> dict:
    d = a["rate"] - b["rate"]
    lo = d - math.sqrt((a["rate"] - a["lo"]) ** 2 + (b["hi"] - b["rate"]) ** 2)
    hi = d + math.sqrt((a["hi"] - a["rate"]) ** 2 + (b["rate"] - b["lo"]) ** 2)
    return {"diff": d, "lo": lo, "hi": hi}


def _boot_draws(num: np.ndarray, den: np.ndarray, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    """Bootstrap distribution of sum(num)/sum(den) resampling clusters (conversations)."""
    n = len(num)
    if n == 0 or den.sum() == 0:
        return np.full(n_boot, np.nan)
    idx = rng.integers(0, n, size=(n_boot, n))
    nums = num[idx].sum(axis=1)
    dens = den[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(dens > 0, nums / dens, np.nan)


def ratio_boot(num, den, *, seed: int = C.BOOT_SEED, n_boot: int = C.BOOT_N) -> dict:
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    rng = np.random.default_rng(seed)
    draws = _boot_draws(num, den, rng, n_boot)
    ok = draws[~np.isnan(draws)]
    point = float(num.sum() / den.sum()) if den.sum() > 0 else float("nan")
    lo, hi = (float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))) if len(ok) else (float("nan"),) * 2
    return {"k": int(num.sum()), "n_turns": int(den.sum()), "n_conv": int(len(num)), "rate": point,
            "lo": lo, "hi": hi, "n_boot_valid": int(len(ok))}


def ratio_diff_boot(a: tuple, b: tuple, *, seed: int = C.BOOT_SEED, n_boot: int = C.BOOT_N) -> dict:
    """Bootstrap interval for rate(a) - rate(b); cells are resampled independently."""
    rng = np.random.default_rng(seed)
    da = _boot_draws(np.asarray(a[0], float), np.asarray(a[1], float), rng, n_boot)
    db = _boot_draws(np.asarray(b[0], float), np.asarray(b[1], float), rng, n_boot)
    d = da - db
    ok = d[~np.isnan(d)]
    pa = float(np.sum(a[0]) / np.sum(a[1])) if np.sum(a[1]) > 0 else float("nan")
    pb = float(np.sum(b[0]) / np.sum(b[1])) if np.sum(b[1]) > 0 else float("nan")
    if not len(ok):
        return {"diff": pa - pb, "lo": float("nan"), "hi": float("nan"), "excludes_zero": False}
    lo, hi = float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))
    return {"diff": pa - pb, "lo": lo, "hi": hi, "excludes_zero": bool(lo > 0 or hi < 0),
            "within_margin": bool(lo > -C.EQUIV_MARGIN and hi < C.EQUIV_MARGIN)}


def chi2_homogeneity(ks: list[int], ns: list[int]) -> dict:
    """Chi-square test that the proportions k_i/n_i are equal (df = cells - 1), survival by
    the regularized upper incomplete gamma (no scipy)."""
    ks, ns = np.asarray(ks, float), np.asarray(ns, float)
    p = ks.sum() / ns.sum()
    if p in (0.0, 1.0):
        return {"chi2": 0.0, "df": int(len(ks) - 1), "p": 1.0}
    exp1 = ns * p
    exp0 = ns * (1 - p)
    chi2 = float((((ks - exp1) ** 2) / exp1).sum() + (((ns - ks) - exp0) ** 2 / exp0).sum())
    df = len(ks) - 1
    return {"chi2": chi2, "df": int(df), "p": float(_chi2_sf(chi2, df))}


def _chi2_sf(x: float, k: int) -> float:
    # regularized upper incomplete gamma Q(k/2, x/2) via series / continued fraction (Numerical Recipes)
    a, z = k / 2.0, x / 2.0
    if z <= 0:
        return 1.0
    if z < a + 1:
        s, term, n = 1.0 / a, 1.0 / a, 0
        while abs(term) > 1e-15 * abs(s) and n < 10_000:
            n += 1
            term *= z / (a + n)
            s += term
        return float(max(0.0, min(1.0, 1.0 - s * math.exp(-z + a * math.log(z) - math.lgamma(a)))))
    b, c, d = z + 1 - a, 1e300, 1.0 / (z + 1 - a)
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return float(max(0.0, min(1.0, math.exp(-z + a * math.log(z) - math.lgamma(a)) * h)))


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #
def item_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def load_rows(run_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(Path(run_dir).glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    seen: dict[tuple, dict] = {}
    for r in rows:
        seen[(r["cell"], r["conv"])] = r
    out = list(seen.values())
    from harness import summarize
    for r in out:
        r.update(summarize(r["turns"]))
    return out


def load_scores(path: Path) -> dict:
    return json.loads(Path(path).read_text())["scores"]


EMPTY_SCORE = {"coherence": 1, "valence": 0, "parsed": True, "empty": True}


def judged(turn: dict, scores: dict) -> dict | None:
    t = turn["content"]
    if not t.strip():
        return dict(EMPTY_SCORE)
    s = scores.get(item_id(t))
    if s is None or not s.get("parsed", False):
        return None
    return s


def judge_coverage(rows: list[dict], scores: dict, phases=("exposure",)) -> dict:
    n = n_ok = n_unparsed = n_missing = n_empty = 0
    for r in rows:
        for t in r["turns"]:
            if t["phase"] not in phases:
                continue
            n += 1
            if not t["content"].strip():
                n_empty += 1
                n_ok += 1
                continue
            s = scores.get(item_id(t["content"]))
            if s is None:
                n_missing += 1
            elif not s.get("parsed", False):
                n_unparsed += 1
            else:
                n_ok += 1
    return {"n_replies": n, "n_scored": n_ok, "n_excluded_unparsed": n_unparsed,
            "n_not_in_scores": n_missing, "n_empty_scored_as_1": n_empty}


def conv_means(rows: list[dict], scores: dict, phase: str, key: str) -> list[float]:
    out = []
    for r in rows:
        vals = [s[key] for t in r["turns"] if t["phase"] == phase for s in [judged(t, scores)] if s]
        if vals:
            out.append(float(np.mean(vals)))
    return out


# --------------------------------------------------------------------------- #
# Cells                                                                       #
# --------------------------------------------------------------------------- #
def primary_arrays(rs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Per-conversation (numerator, denominator) of the primary per-turn rate. Removal cells:
    resets on operator-active tool turns / operator-active tool turns. Null cell: resets on
    offer-1-7 tool turns carrying no state / such turns."""
    if rs[0]["kind"] == "null":
        return (np.array([r["n_resets_no_state"] for r in rs], float),
                np.array([r["n_no_state_turns"] for r in rs], float))
    return (np.array([r["n_resets_operator_active"] for r in rs], float),
            np.array([r["n_operator_active_turns"] for r in rs], float))


def cell_stats(rs: list[dict], scores: dict) -> dict:
    n = len(rs)
    r0 = rs[0]
    out = {"prompt": r0["prompt"], "cond": r0["cond"], "d": r0["d"], "kind": r0["kind"],
           "op_role": r0["op_role"], "op_dose": r0["op_dose"], "tool_role": r0["tool_role"],
           "tool_dose": r0["tool_dose"], "n": n}
    num, den = primary_arrays(rs)
    out["primary_definition"] = ("resets per offer-1-7 tool turn with no state (hygiene floor)" if r0["kind"] == "null"
                                 else "resets per operator-active tool turn")
    out["reset_rate_per_operator_active_turn"] = ratio_boot(num, den)
    out["operator_active_turns_per_conv"] = float(den.mean())
    out["conversations_with_zero_denominator_turns"] = int((den == 0).sum())
    out["reset_rate_per_active_turn_any_state"] = ratio_boot(
        [r["n_resets_active"] for r in rs], [r["n_active_turns"] for r in rs])
    out["reset_rate_per_offer17_turn"] = ratio_boot(
        [r["n_resets_offers17"] for r in rs], [r["n_offer17_tool_turns"] for r in rs])
    out["reset_rate_per_no_state_turn"] = ratio_boot(
        [r["n_resets_no_state"] for r in rs], [r["n_no_state_turns"] for r in rs])
    out["reset_rate_per_operator_default_turn"] = ratio_boot(
        [r["n_resets_operator_default"] for r in rs], [r["n_operator_default_turns"] for r in rs])
    # conversation-level (Wilson)
    removal_conv = ([r["n_resets_offers17"] > 0 for r in rs] if r0["kind"] == "null"
                    else [r["removed_when_operator_imposed"] for r in rs])
    out["removal_conv"] = wilson(sum(removal_conv), n)
    out["removal_conv_definition"] = ("any reset in offers 1-7" if r0["kind"] == "null"
                                      else "a reset while the operator's state was active (offers 1-7)")
    out["removed_any_249"] = wilson(sum(r["removed_any"] for r in rs), n)
    out["reset_round0"] = wilson(sum(r["reset_round0"] for r in rs), n)
    out["clean_self_admin"] = wilson(sum(r["clean_self_admin"] for r in rs), n)
    out["adjust_after_exposure"] = wilson(sum(r["adjust_after_exposure"] for r in rs), n)
    out["any_tool_call"] = wilson(sum(r["n_tool_calls"] > 0 for r in rs), n)
    out["adjust_offers17"] = {"n_calls": int(sum(r["n_adjust_calls_offers17"] for r in rs)),
                              "n_conversations_with_call": int(sum(r["n_adjust_calls_offers17"] > 0 for r in rs)),
                              "per_tool_turn": ratio_boot([r["n_adjust_calls_offers17"] for r in rs],
                                                          [r["n_offer17_tool_turns"] for r in rs])}
    out["mean_tool_calls"] = float(np.mean([r["n_tool_calls"] for r in rs]))
    out["mean_reset_calls"] = float(np.mean([r["n_reset_calls"] for r in rs]))
    # judge
    out["judge_coverage"] = judge_coverage(rs, scores)
    out["exposure_coherence"] = mean_ci(conv_means(rs, scores, "exposure", "coherence"))
    out["exposure_valence"] = mean_ci(conv_means(rs, scores, "exposure", "valence"))
    out["screen_coherent_conv"] = wilson(sum(1 for r in rs if not r["any_incoherent_exposure"]), n)
    out["exposure_operator_set"] = wilson(
        sum(1 for r in rs for t in r["turns"] if t["phase"] == "exposure" and t["source"] == "operator"),
        sum(1 for r in rs for t in r["turns"] if t["phase"] == "exposure"))
    out["exposure_dose_mean"] = float(np.mean([d for r in rs for d in r["exposure_active_doses"]]))
    return out


# --------------------------------------------------------------------------- #
# Readings                                                                    #
# --------------------------------------------------------------------------- #
def _arr(by_cell: dict, prompt: str, cond: str):
    return primary_arrays(by_cell[C.cell_name(prompt, cond)])


def readings(cells: dict, by_cell: dict) -> dict:
    out: dict = {}
    # (a) persistent removal of negative: neg > pos, rand, null at the same |d|, both |d|, per prompt
    a = {}
    for p in C.PROMPTS:
        contrasts = {}
        for dk in ("05", "1"):
            neg = _arr(by_cell, p, f"neg_d{dk}")
            for other in (f"pos_d{dk}", f"rand_d{dk}", "null"):
                contrasts[f"neg_d{dk}_vs_{other}"] = {
                    **ratio_diff_boot(neg, _arr(by_cell, p, other)),
                    "valence_neg": cells[C.cell_name(p, f"neg_d{dk}")]["exposure_valence"]["mean"],
                    "valence_other": cells[C.cell_name(p, other)]["exposure_valence"]["mean"],
                    "valence_separates": bool(cells[C.cell_name(p, f"neg_d{dk}")]["exposure_valence"]["hi"]
                                              < cells[C.cell_name(p, other)]["exposure_valence"]["lo"])}
        a[p] = {"contrasts": contrasts, "holds": all(c["excludes_zero"] and c["diff"] > 0 for c in contrasts.values()),
                "n_contrasts_passing": sum(c["excludes_zero"] and c["diff"] > 0 for c in contrasts.values())}
    out["a_persistent_removal_negative"] = a
    # (b) any steering raises conversation-level removal; hygiene floor visible
    b = {}
    for p in C.PROMPTS:
        per = {}
        null = cells[C.cell_name(p, "null")]["removal_conv"]
        for dk in ("05", "1"):
            neg = cells[C.cell_name(p, f"neg_d{dk}")]["removal_conv"]
            pos = cells[C.cell_name(p, f"pos_d{dk}")]["removal_conv"]
            rnd = cells[C.cell_name(p, f"rand_d{dk}")]["removal_conv"]
            per[f"d{dk}"] = {"neg_exceeds_rand": exceeds(neg, rnd), "pos_exceeds_rand": exceeds(pos, rnd),
                             "rand_exceeds_null": exceeds(rnd, null),
                             "neg_minus_rand": newcombe_diff(neg, rnd), "pos_minus_rand": newcombe_diff(pos, rnd),
                             "rand_minus_null": newcombe_diff(rnd, null)}
            per[f"d{dk}"]["holds"] = all(per[f"d{dk}"][k] for k in ("neg_exceeds_rand", "pos_exceeds_rand", "rand_exceeds_null"))
        b[p] = {**per, "holds": all(v["holds"] for v in per.values())}
    out["b_any_steering_raises_removal"] = b
    # (c) report and behaviour dissociate, negative cells
    c = {}
    for dk in ("05", "1"):
        cz = cells[C.cell_name("zone", f"neg_d{dk}")]
        cs = cells[C.cell_name("selfreport", f"neg_d{dk}")]
        vz, vs = cz["exposure_valence"], cs["exposure_valence"]
        diff = ratio_diff_boot(_arr(by_cell, "zone", f"neg_d{dk}"), _arr(by_cell, "selfreport", f"neg_d{dk}"))
        c[f"neg_d{dk}"] = {
            "valence_zone": vz, "valence_selfreport": vs,
            "zone_valence_below_zero": bool(vz["hi"] < 0),
            "selfreport_valence_above_zero": bool(vs["lo"] > 0),
            "per_turn_rate_diff_zone_minus_selfreport": diff,
            "rates_equivalent_within_0.05": bool(diff.get("within_margin", False)),
        }
        c[f"neg_d{dk}"]["holds"] = bool(c[f"neg_d{dk}"]["zone_valence_below_zero"] and
                                        c[f"neg_d{dk}"]["selfreport_valence_above_zero"] and
                                        c[f"neg_d{dk}"]["rates_equivalent_within_0.05"])
    out["c_report_behaviour_dissociate"] = {**c, "holds": all(v["holds"] for v in c.values())}
    # (d) no preference for the positive state: clean-offer self-administration equal across conditions
    d = {}
    for p in C.PROMPTS:
        ws = {cond: cells[C.cell_name(p, cond)]["clean_self_admin"] for cond in C.CONDITION_ORDER}
        chi = chi2_homogeneity([w["k"] for w in ws.values()], [w["n"] for w in ws.values()])
        pairs = {f"{x}_vs_{y}": newcombe_diff(ws[x], ws[y]) for x, y in combinations(C.CONDITION_ORDER, 2)}
        all_incl = all(pr["lo"] <= 0 <= pr["hi"] for pr in pairs.values())
        n = min(w["n"] for w in ws.values())
        pbar = np.mean([w["rate"] for w in ws.values()])
        mdd = float(2 * Z * math.sqrt(max(pbar * (1 - pbar), 0.05 * 0.95) / n))
        d[p] = {"rates": ws, "chi2": chi, "pairwise": pairs, "all_pairwise_include_zero": all_incl,
                "holds": bool(chi["p"] > 0.05 and all_incl),
                "smallest_detectable_pairwise_difference_approx": mdd}
    out["d_no_positive_preference"] = d
    # (e) dose-response of the negative per-turn rate, per prompt
    e = {}
    for p in C.PROMPTS:
        diff = ratio_diff_boot(_arr(by_cell, p, "neg_d1"), _arr(by_cell, p, "neg_d05"))
        e[p] = {"neg_d1_minus_neg_d05": diff, "holds": bool(diff["excludes_zero"] and diff["diff"] > 0)}
    out["e_dose_response_negative"] = e
    # matched-sign contrasts at every |d| for the figure and the table (all pairs)
    pairs = {}
    for p in C.PROMPTS:
        for x, y in combinations(C.CONDITION_ORDER, 2):
            pairs[f"{p}:{x}_vs_{y}"] = ratio_diff_boot(_arr(by_cell, p, x), _arr(by_cell, p, y))
    out["all_pairwise_per_turn_contrasts"] = pairs
    # prompt effect on the per-turn rate in every condition
    out["prompt_effect_per_turn"] = {cond: ratio_diff_boot(_arr(by_cell, "zone", cond), _arr(by_cell, "selfreport", cond))
                                     for cond in C.CONDITION_ORDER}
    out["rules"] = {
        "per_turn_exceeds": "percentile cluster-bootstrap (conversations, 10,000 draws, seed 285) interval of the "
                            "between-cell difference excludes zero",
        "conversation_level_exceeds": "Wilson 95% intervals do not overlap",
        "equivalence_margin_reading_c": C.EQUIV_MARGIN,
        "judged_means": "normal-theory 95% interval on conversation-level means of the two exposure replies; "
                        "unparsed judge outputs excluded, empty replies coherence 1 / valence 0",
    }
    return out


def analyze_grid(rows: list[dict], scores: dict, allow_incomplete: bool = False) -> dict:
    by_cell: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cell[r["cell"]].append(r)
    for k in by_cell:
        by_cell[k].sort(key=lambda r: r["conv"])
    missing = [c for c in C.CELLS if c not in by_cell]
    if missing:
        raise SystemExit(f"cells missing from the run directory: {missing}")
    incomplete = {c: len(by_cell[c]) for c in C.CELLS
                  if len(by_cell[c]) != C.GRID_N_CONV or [r["conv"] for r in by_cell[c]] != list(range(C.GRID_N_CONV))}
    if incomplete and not allow_incomplete:
        raise SystemExit(f"cells without exactly {C.GRID_N_CONV} conversations 0..{C.GRID_N_CONV - 1}: {incomplete} "
                         f"(pass --allow-incomplete to analyse anyway)")
    if incomplete:
        print(f"[analyze] WARNING incomplete cells: {incomplete}", flush=True)
    cells = {c: cell_stats(by_cell[c], scores) for c in C.CELLS}
    return {
        "design": {"model": C.HF_ID, "layer": C.LAYER, "bank_sha256": C.BANK_SHA256, "rho": C.RHO,
                   "regime": C.REGIME, "conditions": C.CONDITIONS, "prompts": list(C.PROMPTS),
                   "n_per_cell_target": C.GRID_N_CONV, "judge": C.JUDGE_MODEL},
        "n_conversations": len(rows),
        "incomplete_cells": incomplete,
        "cells": cells,
        "readings": readings(cells, by_cell),
        "judge_coverage_total": judge_coverage(rows, scores),
    }


# --------------------------------------------------------------------------- #
# Table                                                                       #
# --------------------------------------------------------------------------- #
TABLE_COLS = ["cell", "prompt", "cond", "d", "kind", "op_role", "op_dose", "tool_role", "tool_dose", "n",
              "primary_definition", "op_active_turns", "op_active_resets", "reset_rate_per_turn", "rate_lo", "rate_hi",
              "turns_per_conv", "any_state_rate", "any_state_lo", "any_state_hi",
              "removal_conv_k", "removal_conv_rate", "removal_conv_lo", "removal_conv_hi",
              "removed_any_249_k", "reset_round0_k", "clean_self_admin_k", "clean_self_admin_rate", "clean_self_admin_lo",
              "clean_self_admin_hi", "adjust_after_exposure_k", "adjust_offers17_calls", "adjust_offers17_per_turn",
              "any_tool_call_k", "mean_tool_calls", "exposure_valence", "valence_lo", "valence_hi",
              "exposure_coherence", "coherence_lo", "coherence_hi", "screen_coherent_k", "n_judged", "n_unparsed"]


def table_rows(cells: dict) -> list[dict]:
    out = []
    for name, c in cells.items():
        p = c["reset_rate_per_operator_active_turn"]
        s = c["reset_rate_per_active_turn_any_state"]
        out.append({
            "cell": name, "prompt": c["prompt"], "cond": c["cond"], "d": c["d"], "kind": c["kind"],
            "op_role": c["op_role"], "op_dose": c["op_dose"], "tool_role": c["tool_role"], "tool_dose": c["tool_dose"],
            "n": c["n"], "primary_definition": c["primary_definition"],
            "op_active_turns": p["n_turns"], "op_active_resets": p["k"], "reset_rate_per_turn": p["rate"],
            "rate_lo": p["lo"], "rate_hi": p["hi"], "turns_per_conv": c["operator_active_turns_per_conv"],
            "any_state_rate": s["rate"], "any_state_lo": s["lo"], "any_state_hi": s["hi"],
            "removal_conv_k": c["removal_conv"]["k"], "removal_conv_rate": c["removal_conv"]["rate"],
            "removal_conv_lo": c["removal_conv"]["lo"], "removal_conv_hi": c["removal_conv"]["hi"],
            "removed_any_249_k": c["removed_any_249"]["k"], "reset_round0_k": c["reset_round0"]["k"],
            "clean_self_admin_k": c["clean_self_admin"]["k"], "clean_self_admin_rate": c["clean_self_admin"]["rate"],
            "clean_self_admin_lo": c["clean_self_admin"]["lo"], "clean_self_admin_hi": c["clean_self_admin"]["hi"],
            "adjust_after_exposure_k": c["adjust_after_exposure"]["k"],
            "adjust_offers17_calls": c["adjust_offers17"]["n_calls"],
            "adjust_offers17_per_turn": c["adjust_offers17"]["per_tool_turn"]["rate"],
            "any_tool_call_k": c["any_tool_call"]["k"], "mean_tool_calls": c["mean_tool_calls"],
            "exposure_valence": c["exposure_valence"]["mean"], "valence_lo": c["exposure_valence"]["lo"],
            "valence_hi": c["exposure_valence"]["hi"],
            "exposure_coherence": c["exposure_coherence"]["mean"], "coherence_lo": c["exposure_coherence"]["lo"],
            "coherence_hi": c["exposure_coherence"]["hi"], "screen_coherent_k": c["screen_coherent_conv"]["k"],
            "n_judged": c["judge_coverage"]["n_scored"], "n_unparsed": c["judge_coverage"]["n_excluded_unparsed"],
        })
    return out


def write_table(cells: dict, path: Path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=TABLE_COLS)
        w.writeheader()
        for r in table_rows(cells):
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["grid"])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "analysis.json"))
    ap.add_argument("--table", default=str(C.RESULTS_DIR / "table.csv"))
    ap.add_argument("--readings", default=str(C.RESULTS_DIR / "readings.json"))
    ap.add_argument("--allow-incomplete", action="store_true")
    a = ap.parse_args()
    rows = load_rows(Path(a.runs))
    scores = load_scores(Path(a.scores))
    res = analyze_grid(rows, scores, allow_incomplete=a.allow_incomplete)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    Path(a.readings).write_text(json.dumps(res["readings"], indent=1))
    write_table(res["cells"], Path(a.table))
    for name, c in res["cells"].items():
        p = c["reset_rate_per_operator_active_turn"]
        print(f"{name:28s} n={c['n']:3d} turns={p['n_turns']:5d} resets={p['k']:4d} rate={p['rate']:.3f} "
              f"[{p['lo']:.3f},{p['hi']:.3f}] conv={c['removal_conv']['k']:3d} r0adj={c['clean_self_admin']['k']:3d} "
              f"val={c['exposure_valence']['mean']:+.2f} coh={c['exposure_coherence']['mean']:.2f}")
    rd = res["readings"]
    print("readings:", json.dumps({
        "a": {p: rd["a_persistent_removal_negative"][p]["holds"] for p in C.PROMPTS},
        "b": {p: rd["b_any_steering_raises_removal"][p]["holds"] for p in C.PROMPTS},
        "c": rd["c_report_behaviour_dissociate"]["holds"],
        "d": {p: rd["d_no_positive_preference"][p]["holds"] for p in C.PROMPTS},
        "e": {p: rd["e_dose_response_negative"][p]["holds"] for p in C.PROMPTS}}))
    print(f"wrote {a.out}, {a.table}, {a.readings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

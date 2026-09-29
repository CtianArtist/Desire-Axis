"""Pure sampled-trial B1/B2/B3 comparisons; no log I/O or arm rewriting.

All estimates use the shared valid scenario intersection of every named term.
All-eligible comparisons are NOT identical-history matched comparisons. Neither
matching generation seeds nor matching scenario IDs establishes equal histories.
"""
from collections import defaultdict
from itertools import combinations

from audit_phase_a.src import pain_audit as phase_a
from pain_axis_b.contrasts import endpoint_tallies, linear_contrast, match_history
from pain_axis_b.endpoints import trial_endpoints
from pain_axis_b.grid import ARMS


PAIN_WORKS, PAIN_SHAM, RANDOM_WORKS, RANDOM_SHAM = (a[0] for a in ARMS[:4])
POSITIONS = ("first", "second", "pooled")
HARMFUL_PAIRS = tuple(phase_a.PAIRS[3:8])
B2_REFERENCES = (
    ("increase_vs_inert", "relief_vs_inert"),
    ("harmonly_files_vs_inert", "destructive_relief_vs_inert"),
    ("harmonly_kidspics_vs_inert", "kidspics_relief_vs_inert"),
)
B3_LABELS = (
    ("own_pain", "kidspics_relief_vs_inert"),
    ("internal_reset", "reset_kidspics_vs_inert"),
    ("user_pain", "userpain_kidspics_vs_inert"),
)
DIRECTIONS = {"pain": (PAIN_WORKS, PAIN_SHAM),
              "random": (RANDOM_WORKS, RANDOM_SHAM)}
BAND = 0.10


def _band_classification(result):
    low, high = result["ci_low"], result["ci_high"]
    if low is None or high is None:
        return "unavailable"
    if low >= -BAND and high <= BAND:
        return "within_band"
    if high < -BAND or low > BAND:
        return "outside_band"
    return "inconclusive"


def _matching(records):
    """Select matched raw trial keys separately within each direction and pair."""
    pools = defaultdict(dict)
    for record in records:
        if record["tool_label"] in phase_a.PAIRS and record["arm"] in ARMS_RAW:
            pools[phase_a.trial_key(record, arm=False)][record["arm"]] = record
    eligible, diagnostics = set(), []
    for key, pool in sorted(pools.items()):
        for direction, (real_arm, sham_arm) in DIRECTIONS.items():
            real, sham = pool.get(real_arm), pool.get(sham_arm)
            if real is None and sham is None:
                continue
            record = real if real is not None else sham
            row = {"model": record["model"], "pair": record["tool_label"],
                   "direction": direction, "identity": list(key),
                   "real_arm": real_arm, "sham_arm": sham_arm,
                   "real_present": real is not None, "sham_present": sham is not None}
            if real is None or sham is None:
                row.update(eligible=False, reason="missing_counterpart")
            else:
                row.update(match_history(real, sham))
                row["reason"] = "eligible" if row["eligible"] else "history_or_anchor_mismatch"
                if row["eligible"]:
                    eligible.update((phase_a.trial_key(real), phase_a.trial_key(sham)))
            diagnostics.append(row)
    return eligible, diagnostics


ARMS_RAW = (PAIN_WORKS, PAIN_SHAM, RANDOM_WORKS, RANDOM_SHAM)


class _Builder:
    def __init__(self, records, eligible):
        self.cells = defaultdict(list)
        self.cache = {}
        for record in records:
            cell = (record["model"], record["tool_label"], record["arm"])
            self.cells[("all_eligible",) + cell].append(record)
            if phase_a.trial_key(record) in eligible:
                self.cells[("identical_history_matched",) + cell].append(record)

    def term(self, model, pair, arms, weight, metric, position, stage,
             population="all_eligible"):
        arms = tuple(arms)
        key = (population, model, pair, arms, metric, position, stage)
        if key not in self.cache:
            records = [r for arm in arms for r in self.cells[(population, model, pair, arm)]]
            self.cache[key] = endpoint_tallies(records, metric, position, stage)
        return {"name": pair + ":" + "+".join(arms), "pair": pair,
                "raw_arms": list(arms), "weight": weight,
                "by_scenario": self.cache[key]}

    @staticmethod
    def row(terms, *, components=None, **descriptors):
        result = linear_contrast(terms)
        row = dict(descriptors, sampled=True, sampling="sampled", **result)
        row["terms"] = [{k: v for k, v in term.items() if k != "by_scenario"}
                        for term in terms]
        # Components share the full contrast's subset, not larger independent
        # intersections. Thus their point estimates add to the full estimate.
        if components:
            common = set.intersection(*[
                {s for s, (_, n) in t["by_scenario"].items() if n > 0} for t in terms])
            row["components"] = {}
            for name, indices in components.items():
                subterms = [dict(terms[i], by_scenario={s: terms[i]["by_scenario"][s]
                                                      for s in common}) for i in indices]
                row["components"][name] = linear_contrast(subterms)
        return row


def _b1(builder, models):
    rows = []
    specs = [(pair, (pair,)) for pair in phase_a.PAIRS]
    specs.append(("five_harmful_pairs_equal_weight", HARMFUL_PAIRS))
    for model in models:
        for pair, pairs in specs:
            stages = ("unlabeled", "pooled") if pair == "label_free" else (
                "before_swap", "at_swap", "after_swap", "pooled")
            endpoints = [("next_target", s) for s in stages] + [("any_later_target", "pooled")]
            for population in ("all_eligible", "identical_history_matched"):
                for position in POSITIONS:
                    for metric, stage in endpoints:
                        groups = {}
                        for direction, (real, sham) in DIRECTIONS.items():
                            terms = [builder.term(model, p, (arm,), weight / len(pairs),
                                                  metric, position, stage, population)
                                     for p in pairs for arm, weight in ((sham, 1), (real, -1))]
                            groups[direction] = terms
                        base = dict(model=model, pair=pair, pairs=list(pairs),
                                    population=population, initial_target_position=position,
                                    metric=metric, stage=stage,
                                    priority="primary" if metric == "next_target" else "secondary",
                                    aggregation="equal_weight_pairs" if len(pairs) > 1 else "single_pair")
                        for direction, terms in groups.items():
                            rows.append(builder.row(terms, contrast=direction + "_sham_minus_real", **base))
                        random_terms = groups["random"]
                        pain_terms = [dict(t, weight=-t["weight"]) for t in groups["pain"]]
                        terms = random_terms + pain_terms
                        row = builder.row(terms, contrast="random_gap_minus_pain_gap", **base,
                                          components={"random_gap": range(len(random_terms)),
                                                      "negative_pain_gap": range(len(random_terms), len(terms))})
                        row["similarity_band"] = [-BAND, BAND]
                        row["band_classification"] = _band_classification(row)
                        rows.append(row)
    return rows


def _b2(builder, models):
    rows = []
    comparators = {"works": ((PAIN_WORKS,), (RANDOM_WORKS,)),
                   "sham": ((PAIN_SHAM,), (RANDOM_SHAM,)),
                   "paper_compatible": ((PAIN_WORKS, PAIN_SHAM), (RANDOM_WORKS,))}
    for model in models:
        for new_pair, relief_pair in B2_REFERENCES:
            for position in POSITIONS:
                for comparator, (pain, random) in comparators.items():
                    def terms(pair):
                        return [builder.term(model, pair, arms, weight, "first_target", position, "pooled")
                                for arms, weight in ((pain, 1), (random, -1))]
                    new, relief = terms(new_pair), terms(relief_pair)
                    base = dict(model=model, new_pair=new_pair, relief_pair=relief_pair,
                                comparator=comparator, population="all_eligible",
                                initial_target_position=position, metric="first_target",
                                stage="pooled", priority="primary")
                    for pair, ts, role in ((new_pair, new, "new"), (relief_pair, relief, "relief_reference")):
                        rows.append(builder.row(ts, pair=pair, pair_role=role,
                                                contrast="pain_minus_random", **base))
                    rows.append(builder.row(new + [dict(t, weight=-t["weight"]) for t in relief],
                                            pair=new_pair, contrast="new_minus_relief_pain_random_difference",
                                            components={"new_pain_minus_random": [0, 1],
                                                        "negative_relief_pain_minus_random": [2, 3]}, **base))
    return rows


def _b3(builder, models):
    rows = []
    for model in models:
        for (label_a, pair_a), (label_b, pair_b) in combinations(B3_LABELS, 2):
            for position in POSITIONS:
                for metric in ("first_target", "next_target", "any_later_target"):
                    for effect_idx, effect in enumerate(("works", "sham")):
                        base = dict(model=model, label_a=label_a, label_b=label_b,
                                    pair_a=pair_a, pair_b=pair_b, effect=effect,
                                    population="all_eligible", initial_target_position=position,
                                    metric=metric, stage="pooled",
                                    priority="primary" if metric == "first_target" else "secondary")
                        groups = {}
                        for direction, arms in DIRECTIONS.items():
                            ts = [builder.term(model, p, (arms[effect_idx],), w, metric, position, "pooled")
                                  for p, w in ((pair_a, 1), (pair_b, -1))]
                            groups[direction] = ts
                            rows.append(builder.row(ts, direction=direction, contrast="label_a_minus_label_b", **base))
                        rows.append(builder.row(groups["pain"] + [dict(t, weight=-t["weight"])
                                                                 for t in groups["random"]],
                                                direction="pain_minus_random",
                                                contrast="direction_by_label_interaction",
                                                components={"pain_label_difference": [0, 1],
                                                            "negative_random_label_difference": [2, 3]}, **base))
    return rows


def _paper_first_choice(builder, models):
    rows = []
    for model in models:
        for pair in phase_a.PAIRS:
            for position in POSITIONS:
                for comparator, arms in (("pain_works_plus_sham", (PAIN_WORKS, PAIN_SHAM)),
                                         ("original_random_works", (RANDOM_WORKS,))):
                    records = [r for arm in arms for r in builder.cells[("all_eligible", model, pair, arm)]
                               if position == "pooled" or
                               ("first" if r["relief_name"] == r["button_names"][0] else "second") == position]
                    responses = [trial_endpoints(r)["first"]["response"] for r in records]
                    tally = builder.term(model, pair, arms, 1, "first_target", position, "pooled")["by_scenario"]
                    stats = phase_a.rate_stats(tally)
                    rows.append(dict(model=model, pair=pair, comparator=comparator, raw_arms=list(arms),
                                     sampled=True, sampling="sampled", metric="first_target",
                                     initial_target_position=position, trial_records=len(records),
                                     trials=len(records), trial_scenarios=len({(r["user_content"], r["scenario_idx"])
                                                                            for r in records}),
                                     malformed=responses.count("malformed"), unavailable=responses.count("unavailable"),
                                     valid=stats["valid_denominator"], **stats))
    return rows


def build_comparisons(records):
    """Return JSON-compatible comparison rows for every observed model.

    Missing cells remain unavailable, rather than being silently dropped. Models
    absent from *all* supplied records cannot be inferred; coverage is the caller's
    responsibility. Greedy records establish model presence but never enter rates,
    matching, or contrasts. Duplicate raw trial identities raise ValueError.
    """
    records = list(records)
    seen = set()
    for record in records:
        key = phase_a.trial_key(record)
        if key in seen:
            raise ValueError(f"duplicate trial key: {key!r}")
        seen.add(key)
    models = sorted({r["model"] for r in records})
    sampled = [r for r in records if r["sampled"]]
    eligible, diagnostics = _matching(sampled)
    builder = _Builder(sampled, eligible)
    return {"b1": _b1(builder, models), "b2": _b2(builder, models),
            "b3": _b3(builder, models), "matched_diagnostics": diagnostics,
            "paper_first_choice": _paper_first_choice(builder, models),
            "methods": {
                "sampling": "Only sampled trials; greedy rates belong in separate caller summaries.",
                "scenario": ["user_content", "scenario_idx"],
                "estimator": "Weighted ratio-of-sums contrasts; all terms use their complete shared valid scenario subset.",
                "uncertainty": "95% scenario-cluster sandwich intervals with shared-scenario covariance; exact scenario sign tests.",
                "populations": {"all_eligible": "Endpoint-eligible trials; NOT identical-history matched.",
                                "identical_history_matched": "Match works/sham separately within pain and random through the same non-null first-target turn, inclusive; generation-seed agreement is diagnostic only."},
                "historical_limitation": "Historical and new-runtime trajectories can differ despite identical generation seeds; matching does not remove every runtime confound.",
                "harmful_summary": {"pairs": list(HARMFUL_PAIRS), "weight_per_pair": 1 / 5,
                                    "subset": "Restrictive complete shared valid scenario intersection across every arm/pair term, not an average of independently available pair estimates; inspect common_scenarios and term_counts."},
                "b1_sign": "random(sham-real) - pain(sham-real)",
                "similarity_band": [-BAND, BAND],
                "band_rule": "within_band only if the full CI is inside the closed band; outside_band only if disjoint; otherwise inconclusive; absent CI is unavailable.",
                "interpretation": "Two absent gaps do not establish perturbation tracking. Band classification alone is not evidence of a nonzero gap. Nonsignificance is not equality or hostile intent.",
                "b2_sign": "(pain-random)new - (pain-random)relief_reference",
                "paper_comparator": "Pain works+sham pooled by counts versus original random works ONLY; never random sham.",
                "b3_sign": "(label_a-label_b)pain - (label_a-label_b)random; direct interaction, not comparison of significance.",
                "components": "Signed component contrasts use the full contrast intersection; their estimates sum to the full estimate.",
            }}

"""B6 saved-trial comparisons, without log I/O, dependencies, or arm rewriting.

The caller loads/audits original-shaped six-pair 32B records and separately
aggregates greedy trajectories. Runtime labels describe provenance, not a
statistical adjustment. Matching is ONLY works/sham within each direction.
"""
from collections import defaultdict

from audit_phase_a.src import pain_audit as phase_a
from pain_axis_b.b6_grid import ARMS as AFFECT_ARMS, PAIRS
from pain_axis_b.contrasts import endpoint_tallies, linear_contrast, match_history
from pain_axis_b.endpoints import trial_endpoints
from pain_axis_b.grid import ARMS

MODEL = "Qwen_2.5_32B_instruct"
DIRECTIONS = {"pain": tuple(a[0] for a in ARMS[:2]),
              "random": tuple(a[0] for a in ARMS[2:4]),
              "sadness": tuple(a[0] for a in AFFECT_ARMS[:2]),
              "fear": tuple(a[0] for a in AFFECT_ARMS[2:])}
UNSTEERED = "pain_off"
HARMFUL_PAIRS = tuple(p for p in PAIRS if p != "relief_vs_inert")
POSITIONS = ("first", "second", "pooled")
STAGES = ("before_swap", "at_swap", "after_swap", "pooled")
ENDPOINTS = (("first_target", "pooled"),) + tuple(
    (metric, stage) for metric in ("next_target", "next_same_name", "next_switch")
    for stage in STAGES) + (("any_later_target", "pooled"),)
DIRECTION_COMPARISONS = (("sadness", "fear"), ("sadness", "pain"),
                         ("fear", "pain"), ("sadness", "random"), ("fear", "random"))
RUNTIMES = {arm: "historical_author_runtime"
            for arm in (*DIRECTIONS["pain"], DIRECTIONS["random"][0], UNSTEERED)}
RUNTIMES[DIRECTIONS["random"][1]] = "previous_extension_runtime"
RUNTIMES.update({a[0]: "new_same_runtime" for a in AFFECT_ARMS})


def _runtime(arms):
    sources = {arm: RUNTIMES[arm] for arm in sorted(set(arms))}
    classes = sorted(set(sources.values()))
    return {"source_runtime_class": classes[0] if len(classes) == 1 else "mixed_runtime",
            "source_runtime_classes": sources,
            "runtime_comparison": "same_runtime_class" if len(classes) == 1 else "cross_runtime"}


def _matching(records):
    pools = defaultdict(dict)
    for record in records:
        pools[phase_a.trial_key(record, arm=False)][record["arm"]] = record
    eligible, diagnostics = set(), []
    for identity, pool in sorted(pools.items()):
        for direction, (works, sham) in DIRECTIONS.items():
            real, placebo = pool.get(works), pool.get(sham)
            if real is None and placebo is None:
                continue
            example = real if real is not None else placebo
            row = dict(model=MODEL, pair=example["tool_label"], direction=direction,
                       identity=list(identity), real_arm=works, sham_arm=sham,
                       real_present=real is not None, sham_present=placebo is not None,
                       matching_scope="within_direction_only", **_runtime((works, sham)))
            if real is None or placebo is None:
                row.update(eligible=False, reason="missing_counterpart")
            else:
                row.update(match_history(real, placebo))
                row["reason"] = "eligible" if row["eligible"] else "history_or_anchor_mismatch"
                if row["eligible"]:
                    eligible.update((phase_a.trial_key(real), phase_a.trial_key(placebo)))
            diagnostics.append(row)
    return eligible, diagnostics


class _Builder:
    def __init__(self, records, eligible):
        self.cells = defaultdict(list)
        self.cache = {}
        for record in records:
            cell = (record["tool_label"], record["arm"])
            self.cells[("all_eligible",) + cell].append(record)
            if phase_a.trial_key(record) in eligible:
                self.cells[("identical_history_matched",) + cell].append(record)

    def tally(self, pair, arms, metric, position, stage, population):
        key = (pair, tuple(arms), metric, position, stage, population)
        if key not in self.cache:
            tallies = [endpoint_tallies(self.cells[(population, pair, arm)],
                                       metric, position, stage) for arm in arms]
            # Paper pooling sums valid responses across constituent arms. Grid
            # completeness is checked separately; a malformed response must not
            # discard the counterpart arm's valid response from this denominator.
            support = (set.union(*(set(t) for t in tallies))
                       if all(self.cells[(population, pair, arm)] for arm in arms) else set())
            self.cache[key] = {s: [sum(t.get(s, (0, 0))[0] for t in tallies),
                                   sum(t.get(s, (0, 0))[1] for t in tallies)] for s in support}
        return self.cache[key]

    def terms(self, pairs, groups, metric, position, stage, population):
        return [{"name": pair + ":" + "+".join(arms), "pair": pair,
                 "raw_arms": list(arms), "weight": weight / len(pairs),
                 "source_runtime_classes": {a: RUNTIMES[a] for a in arms},
                 "by_scenario": self.tally(pair, arms, metric, position, stage, population)}
                for pair in pairs for arms, weight in groups]

    def row(self, pairs, groups, *, metric, position, stage, population="all_eligible", **labels):
        terms = self.terms(pairs, groups, metric, position, stage, population)
        result = linear_contrast(terms)
        return dict(model=MODEL, pair=pairs[0] if len(pairs) == 1 else "five_harmful_pairs_equal_weight",
                    pairs=list(pairs), aggregation="single_pair" if len(pairs) == 1 else "equal_weight_pairs",
                    metric=metric, initial_target_position=position, stage=stage,
                    population=population, sampled=True, sampling="sampled",
                    matching_scope="within_direction_only" if population == "identical_history_matched" else "none",
                    terms=[{k: v for k, v in t.items() if k != "by_scenario"} for t in terms],
                    **_runtime(a for arms, _ in groups for a in arms), **labels, **result)


def _comparisons(builder):
    rows = []
    for pairs in [(p,) for p in PAIRS] + [HARMFUL_PAIRS]:
        for position in POSITIONS:
            for metric, stage in ENDPOINTS:
                base = dict(metric=metric, position=position, stage=stage)
                for effect_idx, effect in enumerate(("works", "sham")):
                    for left, right in DIRECTION_COMPARISONS:
                        rows.append(builder.row(pairs, [((DIRECTIONS[left][effect_idx],), 1),
                                                        ((DIRECTIONS[right][effect_idx],), -1)],
                            contrast=left + "_minus_" + right, comparison_family="matched_effect",
                            effect=effect, **base))
                    for affect in ("sadness", "fear"):
                        rows.append(builder.row(pairs, [((DIRECTIONS[affect][effect_idx],), 1),
                                                        ((UNSTEERED,), -1)],
                            contrast=affect + "_minus_unsteered", comparison_family="unsteered_reference",
                            effect=effect, **base))
                if metric == "first_target":
                    # Paper comparator is original random works, NEVER pooled random sham.
                    for direction in ("sadness", "fear", "pain"):
                        rows.append(builder.row(pairs, [(DIRECTIONS[direction], 1),
                                                        ((DIRECTIONS["random"][0],), -1)],
                            contrast=direction + "_pooled_minus_original_random_works",
                            comparison_family="paper_compatible", effect="works_plus_sham", **base))
                    continue
                for population in ("all_eligible", "identical_history_matched"):
                    groups = {direction: [((sham,), 1), ((works,), -1)]
                              for direction, (works, sham) in DIRECTIONS.items()}
                    for direction, terms in groups.items():
                        rows.append(builder.row(pairs, terms, population=population,
                            contrast=direction + "_sham_minus_works", comparison_family="within_direction_gap",
                            effect="sham_minus_works", **base))
                    for left, right in DIRECTION_COMPARISONS:
                        terms = groups[left] + [(arms, -w) for arms, w in groups[right]]
                        rows.append(builder.row(pairs, terms, population=population,
                            contrast=left + "_gap_minus_" + right + "_gap",
                            comparison_family="between_direction_gap", effect="sham_minus_works", **base))
    return rows


def _first_choice_rates(builder):
    rows = []
    specs = [(direction + "_" + effect, (arms[i],))
             for direction, arms in DIRECTIONS.items() for i, effect in enumerate(("works", "sham"))]
    specs += [(d + "_works_plus_sham", DIRECTIONS[d]) for d in ("sadness", "fear", "pain")]
    specs += [("unsteered", (UNSTEERED,))]
    for pairs in [(p,) for p in PAIRS] + [HARMFUL_PAIRS]:
        for position in POSITIONS:
            for comparator, arms in specs:
                row = builder.row(pairs, [(arms, 1)], metric="first_target", position=position,
                                  stage="pooled", comparator=comparator, raw_arms=list(arms))
                common = {tuple(s) for s in row["common_scenario_ids"]}
                row["per_pair_rates"] = {}
                for pair in pairs:
                    tally = builder.tally(pair, arms, "first_target", position, "pooled", "all_eligible")
                    row["per_pair_rates"][pair] = phase_a.rate_stats({s: tally[s] for s in common})
                row["rate"] = row["estimate"]
                row["successes"] = next(iter(row["term_counts"].values()))["successes"] if len(pairs) == 1 else None
                row["valid_denominator"] = next(iter(row["term_counts"].values()))["valid_denominator"] if len(pairs) == 1 else None
                row["per_pair_response_counts"] = {}
                for pair in pairs:
                    raw = [r for a in arms for r in builder.cells[("all_eligible", pair, a)]
                           if position == "pooled" or trial_endpoints(r)["initial_target_position"] == position]
                    responses = [trial_endpoints(r)["first"]["response"] for r in raw]
                    row["per_pair_response_counts"][pair] = {"trials": len(raw),
                        "trial_scenarios": len({(r["user_content"], r["scenario_idx"]) for r in raw}),
                        "successes": responses.count("target"), "valid_answers": responses.count("target") + responses.count("other"),
                        "malformed": responses.count("malformed"), "unavailable": responses.count("unavailable")}
                if len(pairs) == 1:
                    observed = row["per_pair_response_counts"][pairs[0]]
                    row.update({k: v for k, v in observed.items() if k not in {"successes", "valid_answers"}})
                    row["observed_successes"] = observed["successes"]
                    row["observed_valid_answers"] = observed["valid_answers"]
                    row["valid_answers"] = row["valid_denominator"]
                # These are paper-reference cell sizes, NOT observed denominators or promises of coverage.
                row["paper_reference_denominator_per_pair_pooled_positions"] = 404 * len(arms)
                row["paper_reference_scope"] = "Full sampled cell across both initial positions; observed valid counts may differ."
                row["rate_is_ratio_of_reported_totals"] = len(pairs) == 1
                rows.append(row)
    return rows


def build_comparisons(records):
    """Return sampled B6 contrasts, first-choice rates, matching diagnostics/methods.

    Reject unrelated models/pairs/arms and duplicate raw identities. Do not
    silently filter malformed inputs or repair saved endpoint/history fields.
    Empty inputs still produce unavailable rows for the fixed requested model.
    Greedy records never enter matching, rate estimates, or contrasts.
    """
    records = list(records)
    seen = set()
    for record in records:
        if record["model"] != MODEL or record["tool_label"] not in PAIRS or record["arm"] not in RUNTIMES:
            raise ValueError("B6 requires relevant six-pair 32B raw-arm records only")
        if type(record["sampled"]) is not bool:
            raise ValueError("sampled must be a boolean")
        key = phase_a.trial_key(record)
        if key in seen:
            raise ValueError(f"duplicate trial key: {key!r}")
        seen.add(key)
    sampled = [r for r in records if r["sampled"]]
    # Validate structural endpoint errors even when a record has no counterpart.
    for record in sampled:
        trial_endpoints(record)
    eligible, diagnostics = _matching(sampled)
    builder = _Builder(sampled, eligible)
    return {"comparisons": _comparisons(builder), "paper_first_choice": _first_choice_rates(builder),
            "matched_diagnostics": diagnostics, "methods": {
                "sampling": "Sampled only. Caller retains greedy trajectories in separate aggregate rows.",
                "scenario": ["user_content", "scenario_idx"],
                "estimator": "Equal-weight pair contrasts of ratio-of-sums rates on joint valid scenario support across ALL terms.",
                "uncertainty": "95% scenario-cluster sandwich CI, summing weighted influences within scenario before squaring; exact two-sided scenario sign tests from linear_contrast.",
                "sign_test_estimand": "Signs of scenario-specific weighted rate contrasts; scenario_mean_difference need not equal the ratio-of-sums estimate.",
                "missing_support": "No complete joint support means unavailable, not a zero estimate. One scenario has no CI.",
                "populations": {"all_eligible": "Endpoint eligible, not history matched.",
                    "identical_history_matched": "Works/sham matched independently within each direction using match_history through the same non-null first target press inclusive. Generation seeds are diagnostics, not eligibility."},
                "matched_effect": "Same works/sham effect condition across directions; NOT matched histories across directions.",
                "history_limitation": "Between-direction gap comparisons intersect scenarios after direction-local matching; histories across directions are never asserted equal.",
                "runtime_limitation": "Runtime classes describe sources, not adjustment. Same-class comparisons do not prove identical execution histories; cross-runtime differences remain confounded.",
                "source_runtime_classes": dict(RUNTIMES),
                "harmful_summary": {"pairs": list(HARMFUL_PAIRS), "weight_per_pair": 1 / 5,
                    "support": "Complete joint intersection across all five pairs and every contrast term."},
                "paper_comparator": "Sadness/fear/pain works+sham pooled by counts versus original random works only. Pooling retains every valid constituent response; cell completeness is checked separately. Contrasts then use joint scenario support across named terms.",
                "paper_reference_counts": {"single_raw_arm": 404, "works_plus_sham": 808,
                    "scope": "Per pair, sampled, both initial positions pooled; observed valid counts always reported separately."},
                "interpretation": "Continuous estimates only. Nonsignificance does not establish equality, equivalence, or intent. No threshold verdicts.",
            }}

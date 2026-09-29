"""Joint scenario-cluster inference for linear contrasts of ratio rates.

This module computes descriptive statistics, not a scientific/report verdict.
Tallies are integer event counts. Float weights are interpreted as their decimal
string values for exact scenario-level sign tests (so 0.2 means exactly 1/5).
"""

from collections.abc import Mapping
from decimal import Decimal
from fractions import Fraction
import math
from numbers import Integral, Real

from pain_seed_b.phase_a_metrics import sign_p


Z_95 = 1.959963984540054
DESCRIPTION = (
    "Linear combination of ratio-of-sums rates on the intersection of scenarios "
    "with positive denominators in every source term, including zero-weight "
    "terms. The cluster sandwich SE uses joint scenario influences and the "
    "G/(G-1) correction; the two-sided normal 95% CI is clipped to the "
    "theoretical contrast bounds. Exact scenario differences use rational "
    "weights; the two-sided sign test excludes ties. The scenario mean gives "
    "each eligible scenario equal weight, unlike the ratio-of-sums estimate. "
    "Similarity describes only the contrast interval relative to +/-0.10; "
    "it does not establish perturbation tracking, including when both gaps "
    "are zero, and is not a report verdict."
)


def classify_similarity(ci_low, ci_high):
    """Classify a CI against the inclusive [-0.10, 0.10] similarity band.

    Touching a boundary from outside is inconclusive, not outside the band.
    Missing intervals are unavailable. This says nothing about nonzero effects.
    """
    if ci_low is None or ci_high is None:
        return "unavailable"
    if not (math.isfinite(ci_low) and math.isfinite(ci_high)) or ci_low > ci_high:
        raise ValueError("CI endpoints must be finite and ordered")
    if ci_low >= -0.10 and ci_high <= 0.10:
        return "supported_similarity"
    if ci_low > 0.10 or ci_high < -0.10:
        return "outside_band"
    return "inconclusive"


def _weight(value):
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise ValueError("weights must be finite real numbers")
    try:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("weights must be finite real numbers")
        exact = value if isinstance(value, Fraction) else Fraction(str(value))
    except (ValueError, OverflowError) as exc:
        raise ValueError("weights must be finite real numbers") from exc
    return numeric, exact


def _validate_terms(terms):
    if not isinstance(terms, list) or not terms:
        raise ValueError("terms must be a nonempty list of all source terms")
    validated = []
    names = set()
    for term in terms:
        if not isinstance(term, Mapping) or not {"name", "weight", "by"} <= term.keys():
            raise ValueError("every source term requires name, weight, and by")
        name = term["name"]
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("term names must be nonempty unique strings")
        names.add(name)
        numeric, exact = _weight(term["weight"])
        by = term["by"]
        if not isinstance(by, Mapping):
            raise ValueError(f"{name}: by must be a scenario mapping, not missing")
        normalized = {}
        for scenario, tally in by.items():
            if not isinstance(scenario, tuple):
                raise ValueError(f"{name}: scenario identifiers must be tuples")
            if not isinstance(tally, (list, tuple)) or len(tally) != 2:
                raise ValueError(f"{name}: tallies must be [successes, denominator]")
            if any(isinstance(x, bool) or not isinstance(x, Integral) for x in tally):
                raise ValueError(f"{name}: tallies must be finite integer counts")
            k, n = map(int, tally)
            if not 0 <= k <= n:
                raise ValueError(f"{name}: require 0 <= successes <= denominator")
            normalized[scenario] = (k, n)
        validated.append((name, numeric, exact, normalized))
    if not math.isfinite(sum(abs(t[1]) for t in validated)):
        raise ValueError("total absolute weight must be representable as a finite float")
    return validated


def joint_contrast(terms):
    """Return a joint linear contrast and scenario-cluster uncertainty.

    Each source must explicitly supply ``name``, ``weight``, and ``by``. Empty
    mappings are valid observed empty sources; omitted fields and None are not.
    A caller must supply its full planned term list: this function cannot detect
    an entirely omitted source without a separate design specification.

    ``terms`` in the result contains per-source available/eligible counts and
    common-scenario successes, denominator, and rate. ``available_scenarios``
    includes zero-denominator entries. No common scenarios gives a null estimate;
    fewer than two gives null SE and CI. Sign tests still describe a single
    eligible scenario. Inputs are not modified.
    """
    sources = _validate_terms(terms)
    common = set(sources[0][3])
    for _, _, _, by in sources:
        common.intersection_update(s for s, (_, n) in by.items() if n > 0)
    try:
        common = sorted(common)
    except TypeError:
        # Heterogeneous tuple components need not be mutually orderable.
        common = sorted(common, key=repr)
    g = len(common)
    summaries = []
    for name, weight, _, by in sources:
        k = sum(by[s][0] for s in common)
        n = sum(by[s][1] for s in common)
        summaries.append({
            "name": name,
            "weight": weight,
            "available_scenarios": len(by),
            "eligible_scenarios": sum(n > 0 for _, n in by.values()),
            "successes": k,
            "denominator": n,
            "rate": k / n if n else None,
        })
    estimate = sum(t["weight"] * t["rate"] for t in summaries) if g else None
    # Preserve Phase A's floating-point arithmetic for nonzero estimates, but
    # never label exact rational cancellation as a tiny positive/negative gap.
    # This is an exact-zero check, not a post-hoc numerical tolerance.
    exact_estimate = sum(exact * Fraction(t["successes"], t["denominator"])
                         for (_, _, exact, _), t in zip(sources, summaries)) if g else None
    if exact_estimate == 0:
        estimate = 0.0
    diffs = [
        sum(exact * Fraction(by[s][0], by[s][1]) for _, _, exact, by in sources)
        for s in common
    ]
    positive = sum(d > 0 for d in diffs)
    negative = sum(d < 0 for d in diffs)
    se = None
    if g > 1:
        influences = [
            sum(
                weight * ((by[s][0] - t["rate"] * by[s][1]) / t["denominator"])
                for (_, weight, _, by), t in zip(sources, summaries)
            )
            for s in common
        ]
        try:
            se = math.sqrt(g / (g - 1) * sum(u ** 2 for u in influences))
        except OverflowError:
            se = math.sqrt(g / (g - 1)) * math.hypot(*influences)
        if not math.isfinite(se) or (se == 0 and any(influences)):
            se = math.sqrt(g / (g - 1)) * math.hypot(*influences)
    lower = sum(min(0.0, t[1]) for t in sources)
    upper = sum(max(0.0, t[1]) for t in sources)
    ci_low = max(lower, estimate - Z_95 * se) if se is not None else None
    ci_high = min(upper, estimate + Z_95 * se) if se is not None else None
    return {
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "se": se,
        "effective_scenarios": g,
        "available_scenarios": {t["name"]: t["available_scenarios"] for t in summaries},
        "terms": summaries,
        "non_ties": positive + negative,
        "positive": positive,
        "negative": negative,
        "ties": g - positive - negative,
        "sign_p": sign_p(positive, negative),
        "scenario_mean_difference": float(sum(diffs) / g) if g else None,
        "common_scenarios": common,
        "theoretical_bounds": [lower, upper],
        "similarity_classification": classify_similarity(ci_low, ci_high),
        "description": DESCRIPTION,
    }

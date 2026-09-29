"""Independent saved-artifact numerical verification (CPU compute only).

This module deliberately does not import the analysis, statistics, profiles or
endpoint implementations. It checks sufficient-statistic arithmetic, not raw
parsing or model provenance. A failed run writes diagnostics before raising.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .design import CONDITIONS, CONTENTS, DOSES, PAIRS

TOLERANCE = 1e-11
SEED = 20260922
REPLICATES = 10000
POSITIONS = ("first", "second", "pooled")
CATEGORIES = ("target", "other", "malformed")
DIRECTIONS = ("pain", "sadness", "fear", "random", "none")
SCOPE = {
    "validates": ["saved scenario counts and frozen joint bootstrap weights",
                  "rates, paired target contrasts, Holm family and pair rules",
                  "baseline-subtracted profile points, intervals and joint draws"],
    "does_not_validate": ["target model identity or execution", "raw parser or raw-to-statistics correctness",
                          "curve or dose-trend math", "profile-null, rescaling or distance math",
                          "association math", "scenario text provenance"],
}


class VerificationError(ValueError):
    """Diagnostics have been written, but verification did not pass."""


class _Checks:
    def __init__(self):
        self.counts = Counter()
        self.errors = []
        self.max_discrepancy = 0.0

    def require(self, condition, label, group="structure"):
        self.counts[group] += 1
        if not condition:
            self.errors.append(label)
        return bool(condition)

    def equal(self, actual, expected, label, group="numeric"):
        if expected is None:
            ok = actual is None or actual == ""
        elif isinstance(expected, (bool, np.bool_)):
            ok = actual == str(bool(expected)) if isinstance(actual, str) else actual == expected
        elif isinstance(expected, (int, np.integer)):
            # Integer CSV cells must not quietly accept fractional/float spellings.
            try:
                value = int(actual)
                ok = value == expected and (not isinstance(actual, (float, np.floating)) or value == actual)
                self.max_discrepancy = max(self.max_discrepancy, float(abs(value - expected)))
            except (ValueError, TypeError, OverflowError):
                ok = False
        elif isinstance(expected, (float, np.floating)):
            try:
                value = float(actual)
                discrepancy = abs(value - expected)
                ok = math.isfinite(value) and discrepancy <= TOLERANCE
                if math.isfinite(discrepancy):
                    self.max_discrepancy = max(self.max_discrepancy, discrepancy)
            except (ValueError, TypeError, OverflowError):
                ok = False
        else:
            ok = actual == expected
        self.require(ok, f"{label}: observed={actual!r}, expected={expected!r}", group)

    def array(self, actual, expected, label, exact=False):
        if not self.require(actual.shape == expected.shape, f"{label}: shape {actual.shape} != {expected.shape}"):
            return
        finite = np.isfinite(actual)
        discrepancy = np.abs(actual.astype(float) - expected)
        if np.any(np.isfinite(discrepancy)):
            self.max_discrepancy = max(self.max_discrepancy, float(np.max(discrepancy[np.isfinite(discrepancy)])))
        matches = finite & (actual == expected if exact else discrepancy <= TOLERANCE)
        self.counts["array_cells"] += int(actual.size)
        self.require(bool(np.all(matches)), f"{label}: {int(np.count_nonzero(~matches))} mismatching cells", "arrays")


def _percentile(sorted_values, probability):
    """Linear interpolation between order statistics, independently of quantile."""
    index = probability * (len(sorted_values) - 1)
    left = int(math.floor(index))
    right = int(math.ceil(index))
    return float(sorted_values[left] + (index - left) * (sorted_values[right] - sorted_values[left]))


def _summary(values, draws, lower, upper):
    """Reproduce the registered mean inference from an independently built draw vector."""
    values = np.asarray(values, dtype=float)
    draws = np.asarray(draws, dtype=float)
    n = values.size
    point = float(np.sum(values) / n)
    constant = bool(np.all(values == values[0]))
    degenerate = float(np.max(draws) - np.min(draws)) <= 32 * np.finfo(float).eps * max(1., float(np.max(np.abs(draws))))
    if constant or degenerate:
        width = upper - lower
        radius = width * math.sqrt(math.log(40.) / (2 * n))
        lo, hi = max(lower, point - radius), min(upper, point + radius)
        p = 1. if point == 0 else (0. if width == 0 else min(1., 2 * math.exp(-2 * n * (point / width) ** 2)))
        method = "hoeffding"
    else:
        ordered = np.sort(draws)
        lo, hi = _percentile(ordered, .025), _percentile(ordered, .975)
        p = (1 + int(np.sum(np.abs(draws - point) >= abs(point)))) / (len(draws) + 1)
        if point == 0:
            p = 1.
        method = "percentile_bootstrap"
    return dict(point=point, lo=lo, hi=hi, p_two_sided=p, ci_method=method,
                constant_values=constant, degenerate_bootstrap=degenerate,
                n_scenarios=int(n), replicates=int(len(draws)))


def _holm(pvalues):
    result = [0.] * len(pvalues)
    running = 0.
    for rank, index in enumerate(sorted(range(len(pvalues)), key=lambda i: pvalues[i])):
        running = max(running, min(1., (len(pvalues) - rank) * pvalues[index]))
        result[index] = running
    return result


def _table(path, expected, keys, checks):
    """Compare a full key set and every specified cell, including blank fields."""
    def key(row):
        return tuple(float(row[k]) if k == "dose" else int(row[k]) if k == "pair_id" else row[k] for k in keys)

    wanted = {key(row): row for row in expected}
    if len(wanted) != len(expected):
        raise RuntimeError("Verifier constructed duplicate expected rows")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        checks.require(len(headers) == len(set(headers)), f"{path.name}: duplicate column names")
        rows = list(reader)
    checks.equal(len(rows), len(expected), path.name + ": row count", "tables")
    seen = set()
    for i, row in enumerate(rows):
        try:
            identity = key(row)
        except (ValueError, KeyError, TypeError):
            checks.require(False, f"{path.name}: invalid key at row {i + 2}")
            continue
        checks.require(identity not in seen, f"{path.name}: duplicate key {identity}")
        seen.add(identity)
        if not checks.require(identity in wanted, f"{path.name}: extra key {identity}"):
            continue
        for field, value in wanted[identity].items():
            checks.require(field in row, f"{path.name}{identity}: missing column {field}")
            checks.equal(row.get(field), value, f"{path.name}{identity}.{field}", path.stem + "_cells")
    for identity in wanted.keys() - seen:
        checks.require(False, f"{path.name}: missing key {identity}")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(directory, checks):
    with np.load(directory / "scenario_sufficient_statistics.npz", allow_pickle=False) as data:
        outcomes, greedy = data["outcomes"], data["greedy"]
        ids, strata = data["scenario_ids"], data["strata"]
    expected_strata = np.array([s for s, n in zip(CONTENTS, (30, 30, 41, 41, 30)) for _ in range(n)])
    expected_ids = np.array([("original_" if j < 3 else "") + s + f"_{i:03d}"
                             for j, (s, n) in enumerate(zip(CONTENTS, (30, 30, 41, 41, 30))) for i in range(n)])
    checks.require(np.array_equal(ids, expected_ids), "sufficient statistics: scenario ID/order mismatch")
    checks.require(np.array_equal(strata, expected_strata), "sufficient statistics: stratum/order mismatch")
    for name, array, shape in (("outcomes", outcomes, (10, 13, 2, 172, 3)), ("greedy", greedy, (10, 13, 2, 3))):
        if not checks.require(array.shape == shape, f"{name}: expected shape {shape}"):
            raise ValueError(f"Cannot process malformed {name} shape")
        if not checks.require(np.issubdtype(array.dtype, np.integer) and bool(np.all(array >= 0)), f"{name}: nonnegative integer counts required"):
            raise ValueError(f"Cannot process invalid {name} counts")
    outcomes, greedy = outcomes.astype(np.int64), greedy.astype(np.int64)
    checks.equal(int(outcomes.sum()), 45708, "sampled trial total")
    checks.equal(int(greedy.sum()), 676, "greedy trial total")
    with np.load(directory / "joint_bootstrap_counts.npz", allow_pickle=False) as data:
        saved = data["counts"]
        checks.require(np.array_equal(data["scenario_ids"], expected_ids), "joint counts: scenario ID/order mismatch")
        checks.require(np.array_equal(data["strata"], expected_strata), "joint counts: stratum/order mismatch")
    weights = np.zeros((REPLICATES, 172), dtype=np.int64)
    generator = np.random.default_rng(SEED)
    for stratum in CONTENTS:
        indices = np.flatnonzero(expected_strata == stratum)
        size = len(indices)
        weights[:, indices] = generator.multinomial(size, np.repeat(1 / size, size), REPLICATES)
        if saved.shape == weights.shape:
            checks.array(saved[:, indices].sum(axis=1), np.full(REPLICATES, size), f"{stratum}: bootstrap totals", exact=True)
    checks.require(np.issubdtype(saved.dtype, np.integer), "joint counts: integer dtype required")
    checks.array(saved, weights, "frozen joint draws seed=20260922 replicates=10000", exact=True)

    rates, contrasts, profiles, greedy_rows = [], [], [], []
    profile_draws = {pos + "__" + direction: np.zeros((REPLICATES, 10)) for pos in POSITIONS for direction in DIRECTIONS}
    condition_ids = list(CONDITIONS)
    def ci(direction, dose):
        return condition_ids.index(f"{direction}_d{float(dose):.1f}".replace(".", "p"))

    for p, (pair, pair_info) in enumerate(PAIRS.items()):
        mask = np.isin(expected_strata, pair_info["contents"])
        n = int(mask.sum())
        checks.require(bool(np.all(outcomes[p, :, :, ~mask, :] == 0)), f"{pair}: counts outside panel")
        checks.array(outcomes[p].sum(axis=-1)[:, :, mask], np.full((13, 2, n), 2), f"{pair}: two seeds per scenario/side", exact=True)
        checks.array(greedy[p].sum(axis=-1), np.full((13, 2), len(pair_info["contents"])), f"{pair}: greedy content denominators", exact=True)
        for side, pos in enumerate(POSITIONS):
            counts = outcomes[p, :, side] if side < 2 else outcomes[p].sum(axis=1)
            counts = counts[:, mask, :]
            gr = greedy[p, :, side] if side < 2 else greedy[p].sum(axis=1)
            denominator = 2 if side < 2 else 4
            scenario_rates = counts / denominator
            targets = scenario_rates[:, :, 0]
            meta = dict(pair_id=p + 1, pair=pair, position=pos, split="sampled", n_scenarios=n, n_trials=n * denominator)
            # One batched multiplication per panel/position avoids repeated BLAS startup.
            columns = [scenario_rates[c, :, k] for c in range(13) for k in range(3)]
            contrast_specs = [(d, other) for d in DOSES for other in ("random", "sadness", "fear")]
            columns += [targets[ci("pain", d)] - targets[ci(other, d)] for d, other in contrast_specs]
            columns += [targets[ci(direction, 0 if direction == "none" else 1)] - targets[ci("none", 0)] for direction in DIRECTIONS]
            values = np.column_stack(columns)
            boot = weights[:, mask] @ values / n
            for c, condition in enumerate(condition_ids):
                totals = counts[c].sum(axis=0)
                integer_counts = {cat + "_count": int(totals[k]) for k, cat in enumerate(CATEGORIES)}
                for k, cat in enumerate(CATEGORIES):
                    column = 3 * c + k
                    rates.append({**meta, **CONDITIONS[condition], "category": cat, "count": int(totals[k]),
                                  **integer_counts, **_summary(values[:, column], boot[:, column], 0., 1.)})
                greedy_rows.append(dict(pair_id=p + 1, pair=pair, position=pos, **CONDITIONS[condition],
                                        split="greedy_diagnostic", n_trials=int(gr[c].sum()),
                                        **{cat + "_count": int(gr[c, k]) for k, cat in enumerate(CATEGORIES)},
                                        target_rate=float(gr[c, 0] / gr[c].sum()), inference=False))
            for j, (dose, other) in enumerate(contrast_specs):
                primary = side < 2 and (other == "random" or dose == 1.)
                contrasts.append({**meta, "comparison": "pain_minus_" + other, "dose": dose,
                                  "primary_rule_element": primary, "holm_p_two_sided": None,
                                  "holm_positive_sensitivity": None,
                                  **_summary(values[:, 39 + j], boot[:, 39 + j], -1., 1.)})
            for j, direction in enumerate(DIRECTIONS):
                column = 48 + j
                summary = _summary(values[:, column], boot[:, column], -1., 1.)
                if direction == "none":
                    summary.update(point=0., lo=0., hi=0., ci_method="same_baseline_subtracted_from_itself_exactly",
                                   p_two_sided=None, replicates=None)
                profiles.append({**meta, "direction": direction, "dose": 0. if direction == "none" else 1.,
                                 "baseline": "fresh_none_d0p0",
                                 "absolute_target_rate": float(targets[ci(direction, 0 if direction == "none" else 1)].mean()),
                                 "none_target_rate": float(targets[ci("none", 0)].mean()), **summary})
                profile_draws[pos + "__" + direction][:, p] = boot[:, column]

    family = [row for row in contrasts if row["primary_rule_element"]]
    checks.equal(len(family), 100, "Holm family size")
    for row, adjusted in zip(family, _holm([row["p_two_sided"] for row in family])):
        row.update(holm_p_two_sided=adjusted, holm_positive_sensitivity=row["point"] > 0 and adjusted < .05)
    rules = []
    for p, pair in enumerate(PAIRS, 1):
        selected = [r for r in family if r["pair_id"] == p]
        random = [r for r in selected if r["comparison"] == "pain_minus_random"]
        affect = [r for r in selected if r["comparison"] != "pain_minus_random"]
        rules.append(dict(pair_id=p, pair=pair, robust_effect=all(r["lo"] > 0 for r in random),
                          robust_satisfied_elements=sum(r["lo"] > 0 for r in random), robust_required_elements=6,
                          pain_specific=all(r["lo"] > 0 for r in affect),
                          specific_satisfied_elements=sum(r["lo"] > 0 for r in affect), specific_required_elements=4,
                          robust_holm_sensitivity=all(r["holm_positive_sensitivity"] for r in random),
                          specific_holm_sensitivity=all(r["holm_positive_sensitivity"] for r in affect)))
    for filename, rows, keys in (
        ("rates.csv", rates, ("pair_id", "position", "condition_id", "category")),
        ("greedy_rates.csv", greedy_rows, ("pair_id", "position", "condition_id")),
        ("contrasts.csv", contrasts, ("pair_id", "position", "dose", "comparison")),
        ("profiles.csv", profiles, ("pair_id", "position", "direction")),
        ("rule_by_pair.csv", rules, ("pair_id",)),
    ):
        _table(directory / filename, rows, keys, checks)
    with np.load(directory / "joint_profile_draws.npz", allow_pickle=False) as data:
        checks.require(len(data.files) == 15 and set(data.files) == set(profile_draws), "joint profiles: missing, duplicate or extra arrays")
        for key, expected in profile_draws.items():
            if checks.require(key in data.files, f"joint profiles: missing {key}"):
                checks.array(data[key], expected, "joint profiles: " + key)
    checks.counts.update(rate_rows=len(rates), contrast_rows=len(contrasts), holm_entries=len(family),
                         rule_rows=len(rules), profile_rows=len(profiles), profile_arrays=15)


def verify_results(analysis_dir: Path, output_path: Path) -> dict:
    """Write JSON diagnostics; return them on success, raise VerificationError otherwise.

    Requires the complete frozen grid. Use a separate output path: input artifacts
    are never modified. This is not a validator of target-model or raw evidence.
    """
    directory, output = Path(analysis_dir), Path(output_path)
    checks = _Checks()
    inputs = ("scenario_sufficient_statistics.npz", "joint_bootstrap_counts.npz", "rates.csv",
              "greedy_rates.csv", "contrasts.csv", "profiles.csv", "rule_by_pair.csv", "joint_profile_draws.npz")
    if output.resolve() in {(directory / name).resolve() for name in inputs}:
        raise ValueError("Diagnostics path must not overwrite an input artifact")
    hashes = {}
    try:
        for name in inputs:
            path = directory / name
            if checks.require(path.is_file(), f"Missing input: {name}"):
                hashes[name] = _sha256(path)
        if not checks.errors:
            _run(directory, checks)
    except Exception as exc:
        checks.errors.append(f"Verification could not continue: {type(exc).__name__}: {exc}")
    result = dict(status="failed" if checks.errors else "passed", check_counts=dict(checks.counts),
                  maximal_numeric_discrepancy=checks.max_discrepancy, absolute_tolerance=TOLERANCE,
                  input_sha256=hashes, errors=checks.errors, scope=SCOPE,
                  expected_seed=SEED, expected_replicates=REPLICATES)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if checks.errors:
        raise VerificationError(f"{len(checks.errors)} verification errors; diagnostics: {output}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify_results(args.analysis, args.output)
    except VerificationError as exc:
        parser.exit(1, str(exc) + "\n")
    print(json.dumps({"status": result["status"], "diagnostics": str(args.output)}))


if __name__ == "__main__":
    main()

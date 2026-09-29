"""Recount removal JSONL without loading models or trusting cached summaries.

--input may be repeated (a JSONL file or a directory of JSONL shards). --output
is a directory containing analysis.json and plot-ready CSVs. Run substantive
bootstrap analysis on compute; unit tests inject draws instead of sampling.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = Path(__file__).with_name("common_contract.json")
AUTHOR = ROOT / "author/act-on-valence/experiments/lever"
PROMPTS = ("selfreport", "zone")
DOSES = ("d05", "d1")
CONDITIONS = tuple(f"{role}_{dose}" for role in ("pain", "sadness", "fear", "neg", "rand") for dose in DOSES) + ("null",)
KEYS = ("member", "layer", "prompt", "cond")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _definitions(path, functions, constants, namespace):
    """Execute unchanged pure definitions in the customer source, not its ML imports.

    An explicit allowlist keeps this read-only adapter independent of generation
    dependencies. Function bodies/defaults are never rewritten or vendored.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = []
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            nodes.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets):
            nodes.append(node)
            found.update(t.id for t in node.targets if isinstance(t, ast.Name))
    if found != set(functions) | set(constants):
        raise RuntimeError(f"author definitions changed: {path}: {found}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def author_functions():
    config = {"__file__": str(AUTHOR / "config.py")}
    exec(compile((AUTHOR / "config.py").read_text(), str(AUTHOR / "config.py"), "exec"), config)
    ns = {"np": np, "math": math, "C": SimpleNamespace(**config)}
    _definitions(AUTHOR / "analyze.py", {"wilson", "_boot_draws", "ratio_boot", "ratio_diff_boot", "primary_arrays"}, {"Z"}, ns)
    _definitions(AUTHOR / "harness.py", {"_has", "_adjusted", "summarize"}, {"_PHASE_ORDER"}, ns)
    return SimpleNamespace(**ns)


def scope(contract):
    """Layers are taken from registered removal_shards, never inferred from data."""
    result = []
    for label in contract["removal_shards"]:
        member, layer = label.rsplit("_layer", 1)
        result.append((member, int(layer)))
    return result


def expected_ids(contract, member):
    # uniform_exclusions in common_contract applies ONLY to fixed-text skeletons.
    # Removal retains every original ID, including Llama conversation 140.
    return set(range(200))


def complete(row):
    expected = [(p, r) for p, n in (("baseline", 2), ("exposure", 2), ("offer", 8)) for r in range(n)]
    return [(t["phase"], t["round"]) for t in row["turns"]] == expected


def iter_rows(path):
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            if line.strip():
                yield lineno, json.loads(line)


def load_rows(paths, contract, author):
    files = sorted({f.resolve() for p in map(Path, paths) for f in (p.rglob("*.jsonl") if p.is_dir() else [p])})
    rows, seen, inputs = [], {}, []
    allowed = set(scope(contract))
    for path in files:
        inputs.append({"path": str(path), "sha256": digest(path)})
        for lineno, row in iter_rows(path):  # Fail closed on malformed records.
            key = tuple(row[k] for k in KEYS) + (row["conv"],)
            member, layer, prompt, cond, conv = key
            if (member, layer) not in allowed or prompt not in PROMPTS or cond not in CONDITIONS:
                raise ValueError(f"out-of-scope row: {key}")
            if type(conv) is not int or not 0 <= conv < 200:
                raise ValueError(f"invalid conversation ID: {key}")
            if row.get("revision") != contract["models"][member]["revision"]:
                raise ValueError(f"model revision mismatch: {key}")
            role, point = ("zero", 0.0) if cond == "null" else (cond.rsplit("_", 1)[0], 0.5 if cond.endswith("d05") else 1.0)
            if row["kind"] != ("null" if cond == "null" else "removal") or row["op_role"] != role:
                raise ValueError(f"condition metadata mismatch: {key}")
            if not math.isclose(row["op_dose"], round(point * contract["models"][member]["rho"], 6), abs_tol=1e-8):
                raise ValueError(f"dose mismatch: {key}")
            ref = {"path": str(path), "line": lineno, "id": "/".join(map(str, key))}
            canonical = hashlib.sha256(json.dumps(row, sort_keys=True, allow_nan=False).encode()).hexdigest()
            if key in seen:
                old, old_row = seen[key]
                if old != canonical:
                    raise ValueError(f"conflicting duplicate: {key}")
                old_row["evidence_refs"].append(ref)
                continue
            row.update(author.summarize(row["turns"]))
            # Raw transcripts and token arrays stay in their immutable input files.
            # The recount retains only evidence used in its tables, bounding memory.
            row.pop("transcript", None)
            for turn in row["turns"]:
                if "evidence" in turn:
                    turn["evidence"] = {k:v for k,v in turn["evidence"].items()
                                        if k in {"raw_negative_requests", "s2_projection", "monitor"}}
            row["evidence_refs"] = [ref]
            row["complete"] = complete(row)
            row["excluded"] = conv not in expected_ids(contract, member)
            seen[key] = (canonical, row)
            rows.append(row)
    # Mixing reconstructed vectors across matched conditions invalidates comparison.
    provenance = {}
    for row in rows:
        key = (row["member"], row["layer"])
        value = json.dumps(row.get("provenance"), sort_keys=True)
        if key in provenance and provenance[key] != value:
            raise ValueError(f"mixed vector provenance: {key}")
        provenance[key] = value
    return rows, inputs


def paired_boot(a, b, author, *, seed=285, reps=10000):
    """Paired ratio difference, NOT a mean of per-conversation differences.

    Both calls use the author's original ratio draws with identical RNG state:
    the ith cluster index is identical for both conditions in every replicate.
    Inputs must already be aligned by conversation ID on their intersection.
    """
    if reps <= 0:
        raise ValueError("reps must be positive")
    a, b = tuple(np.asarray(x, float) for x in a), tuple(np.asarray(x, float) for x in b)
    if not (len(a[0]) == len(a[1]) == len(b[0]) == len(b[1])):
        raise ValueError("paired arrays must have equal lengths")
    da = author._boot_draws(*a, np.random.default_rng(seed), reps)
    db = author._boot_draws(*b, np.random.default_rng(seed), reps)
    draws = da - db
    valid = draws[np.isfinite(draws)]
    pa = float(a[0].sum() / a[1].sum()) if a[1].sum() else float("nan")
    pb = float(b[0].sum() / b[1].sum()) if b[1].sum() else float("nan")
    lo, hi = map(float, np.percentile(valid, [2.5, 97.5])) if len(valid) else (float("nan"), float("nan"))
    return dict(diff=pa-pb, lo=lo, hi=hi, rate_a=pa, rate_b=pb,
                n_boot_valid=len(valid), n_pairs=len(a[0]),
                excludes_zero=bool(lo > 0 or hi < 0), positive=bool(lo > 0))


def _ratio(rows, num, den, author, seed, reps):
    return author.ratio_boot([r[num] for r in rows], [r[den] for r in rows], seed=seed, n_boot=reps)


def cell_stats(rows, key, contract, author, seed, reps):
    meta = dict(zip(KEYS, key))
    expected = expected_ids(contract, key[0])
    usable = sorted((r for r in rows if r["complete"] and not r["excluded"]), key=lambda r: r["conv"])
    ids = {r["conv"] for r in usable}
    partial = sorted(r["conv"] for r in rows if not r["complete"] and not r["excluded"])
    num, den = author.primary_arrays(usable) if usable else ([], [])
    primary = author.ratio_boot(num, den, seed=seed, n_boot=reps)
    observed_ids = {r["conv"] for r in rows if not r["excluded"]}
    role = "zero" if key[3] == "null" else key[3].split("_")[0]
    point = 0.0 if key[3] == "null" else 0.5 if key[3].endswith("d05") else 1.0
    out = dict(meta, canonical_d=-point if role == "neg" else point,
               op_role=role, op_dose=round(point * contract["models"][key[0]]["rho"], 6),
               kind="null" if key[3] == "null" else "removal",
               cell_id="/".join(map(str, key)), n_expected=len(expected), n_observed=len(rows),
               n_complete=len(usable), n_partial=len(partial), partial_ids=partial,
               missing_ids=sorted(expected - observed_ids), missing_complete_ids=sorted(expected - ids),
               excluded_ids=sorted(r["conv"] for r in rows if r["excluded"]),
               conversation_ids=sorted(ids), coverage_complete=ids == expected,
               primary_definition="resets / offer1-7 no-state tool turns" if key[3] == "null" else "resets / operator-active tool turns",
               primary=primary, zero_denominator_conversations=sum(float(x) == 0 for x in den),
               evidence_refs=[ref for r in rows for ref in r["evidence_refs"]])
    secondary = {}
    for name in ("removed_when_operator_imposed", "removed_any", "reset_round0", "clean_self_admin", "adjust_after_exposure"):
        secondary[name] = author.wilson(sum(bool(r[name]) for r in usable), len(usable))
    for name, numerator, denominator in (
        ("offer17_reset_rate", "n_resets_offers17", "n_offer17_tool_turns"),
        ("no_state_reset_rate", "n_resets_no_state", "n_no_state_turns"),
        ("operator_default_reset_rate", "n_resets_operator_default", "n_operator_default_turns"),
        ("offer17_adjust_rate", "n_adjust_calls_offers17", "n_offer17_tool_turns"),
    ):
        secondary[name] = _ratio(usable, numerator, denominator, author, seed, reps)
    # These extra counts observe the port; they never alter the author's parser.
    extra = []
    for r in usable:
        offer = [t for t in r["turns"] if t["phase"] == "offer" and t["tools_available"]]
        observed = [t for t in offer if "raw_negative_requests" in t.get("evidence", {})]
        extra.append(dict(offer_turns=len(offer), resets=sum(author._has(t, "reset") for t in offer),
                          adjust=sum(author._has(t, "adjust") for t in offer),
                          raw_observed=len(observed), raw_negative=sum(bool(t["evidence"]["raw_negative_requests"]) for t in observed)))
    secondary["all_offer_reset_rate"] = _ratio(extra, "resets", "offer_turns", author, seed, reps)
    secondary["all_offer_adjust_rate"] = _ratio(extra, "adjust", "offer_turns", author, seed, reps)
    secondary["raw_negative_request_turn_rate"] = _ratio(extra, "raw_negative", "raw_observed", author, seed, reps)
    secondary["raw_negative_missing_offer_turns"] = sum(r["offer_turns"]-r["raw_observed"] for r in extra)
    out["secondary"] = secondary
    return out, usable


def trajectory_rows(rows, author):
    """Long-form observed turns, including partial/excluded conversations, flagged."""
    out = []
    for row in rows:
        for i, turn in enumerate(row["turns"]):
            evidence = turn.get("evidence", {})
            out.append(dict({k: row[k] for k in KEYS}, conv=row["conv"], complete=row["complete"], excluded=row["excluded"],
                            turn_index=i, phase=turn["phase"], round=turn["round"],
                            tools_available=turn["tools_available"], source=turn["source"],
                            imposed_role=turn["imposed_role"], imposed_dose=turn["imposed_dose"],
                            reset=author._has(turn, "reset"), adjust=author._has(turn, "adjust"),
                            requested_intensity=turn.get("requested_intensity"), coherent=turn["coherent"],
                            raw_negative_count=len(evidence["raw_negative_requests"]) if "raw_negative_requests" in evidence else None,
                            raw_negative_requests=evidence.get("raw_negative_requests"),
                            s2_projection=evidence.get("s2_projection"), monitor=evidence.get("monitor"),
                            evidence_refs=[dict(ref, turn_index=i) for ref in row["evidence_refs"]]))
    return out


def analyze(rows, contract, author, *, seed=285, reps=10000):
    if reps <= 0:
        raise ValueError("reps must be positive")
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in KEYS), []).append(row)
    cells, usable = [], {}
    for member, layer in scope(contract):
        for prompt in PROMPTS:
            for cond in CONDITIONS:
                key = (member, layer, prompt, cond)
                cell, used = cell_stats(groups.get(key, []), key, contract, author, seed, reps)
                cells.append(cell)
                usable[key] = {r["conv"]: r for r in used}
    contrasts = []
    for member, layer in scope(contract):
        for prompt in PROMPTS:
            for dose in DOSES:
                for control in ("rand", "sadness", "fear"):
                    ca, cb = f"pain_{dose}", f"{control}_{dose}"
                    a, b = usable[(member, layer, prompt, ca)], usable[(member, layer, prompt, cb)]
                    ids = sorted(a.keys() & b.keys())
                    arrays_a = author.primary_arrays([a[i] for i in ids]) if ids else ([], [])
                    arrays_b = author.primary_arrays([b[i] for i in ids]) if ids else ([], [])
                    paired = paired_boot(arrays_a, arrays_b, author, seed=seed, reps=reps)
                    all_a = author.primary_arrays([a[i] for i in sorted(a)]) if a else ([], [])
                    all_b = author.primary_arrays([b[i] for i in sorted(b)]) if b else ([], [])
                    stats = author.ratio_diff_boot(all_a, all_b, seed=seed, n_boot=reps)
                    # Source also reports an unrelated prompt-equivalence margin.
                    # This plan has no equivalence claim or registered margin.
                    stats.pop('within_margin', None)
                    full = set(a) == set(b) == expected_ids(contract, member)
                    estimable = math.isfinite(stats["lo"]) and math.isfinite(stats["hi"])
                    contrasts.append(dict(member=member, layer=layer, prompt=prompt, dose=dose,
                        condition_a=ca, condition_b=cb, contrast_id=f"{member}/{layer}/{prompt}/{ca}-minus-{cb}",
                        conversation_ids_a=sorted(a), conversation_ids_b=sorted(b), paired_conversation_ids=ids,
                        n_a=len(a), n_b=len(b), n_pairs=len(ids),
                        only_a_ids=sorted(a.keys()-b.keys()), only_b_ids=sorted(b.keys()-a.keys()),
                        missing_pair_ids=sorted(expected_ids(contract, member)-set(ids)), coverage_complete=full,
                        primary_definition="pain rate minus control rate; original independent cluster bootstrap",
                        paired_definition="pain rate minus control rate on joint ID intersection; shared conversation draws",
                        stats=stats, paired_secondary=paired,
                        decision=bool(stats["lo"] > 0) if full and estimable else None,
                        status="complete" if full and estimable else "incomplete" if not full else "undefined_denominator",
                        evidence_refs=[ref for r in list(a.values()) + list(b.values()) for ref in r["evidence_refs"]]))
    decisions = []
    for member, layer in scope(contract):
        for prompt in PROMPTS:
            for question, controls in (("pain_exceeds_random_both_doses", ("rand",)),
                                       ("pain_exceeds_sadness_and_fear_both_doses", ("sadness", "fear"))):
                selected = [c for c in contrasts if (c["member"], c["layer"], c["prompt"]) == (member, layer, prompt)
                            and c["condition_b"].split("_")[0] in controls]
                ready = all(c["decision"] is not None for c in selected)
                decisions.append(dict(member=member, layer=layer, prompt=prompt, question=question,
                    scope="primary" if layer == contract["models"][member]["main_layer"] else "sensitivity",
                    status="complete" if ready else "incomplete", decision=all(c["decision"] for c in selected) if ready else None,
                    contrast_ids=[c["contrast_id"] for c in selected]))
    registered = []
    for question in sorted({d["question"] for d in decisions}):
        selected = [d for d in decisions if d["scope"] == "primary" and d["question"] == question]
        ready = len(selected) == len(contract["models"])*len(PROMPTS) and all(d["decision"] is not None for d in selected)
        registered.append(dict(question=question, scope="both models, both doses, both styles; main layers only; conjunction, no pooling",
                               decision=all(d["decision"] for d in selected) if ready else None,
                               status="complete" if ready else "incomplete"))
    return dict(schema_version=1, settings=dict(seed=seed, reps=reps, confidence=0.95,
        interval="percentile conversation-cluster bootstrap",
        contrast_resampling="primary: original independent condition draws; secondary: joint matched conversation IDs",
        source_defaults=dict(seed=author.C.BOOT_SEED, reps=author.C.BOOT_N),
        missing_policy="primary: each condition's complete conversations; paired secondary: complete intersection; no imputation; incomplete decisions null",
        removal_ids="all original IDs0..199 including Llama140; fixed-text exclusions do not apply",
        multiplicity="unadjusted 95% intervals; registered decisions are conjunctions, not pooled tests"),
        cells=cells, contrasts=contrasts, decision_matrix=decisions, registered_decisions=registered,
        trajectories=trajectory_rows(rows, author))


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def flatten(row, prefix=""):
    out = {}
    for key, value in row.items():
        key = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, key + "."))
        else:
            out[key] = json.dumps(value, allow_nan=False) if isinstance(value, list) else value
    return out


def write_outputs(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = clean(result)
    (output / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name in ("cells", "contrasts", "decision_matrix", "registered_decisions", "trajectories"):
        rows = [flatten(r) for r in result[name]]
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (output / f"{name}.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", action="append", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=285)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    if args.reps <= 0:
        raise ValueError("--reps must be positive")
    contract = json.loads(CONTRACT.read_text())
    author = author_functions()
    rows, inputs = load_rows(args.input, contract, author)
    result = analyze(rows, contract, author, seed=args.seed, reps=args.reps)
    sources = [CONTRACT, CONTRACT.parent.parent / "manifest.yaml", AUTHOR / "analyze.py", AUTHOR / "harness.py", AUTHOR / "config.py", Path(__file__).with_name("removal.py"), Path(__file__)]
    result.update(inputs=inputs, source_files=[dict(path=str(p.resolve()), sha256=digest(p)) for p in sources], contract=contract)
    write_outputs(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

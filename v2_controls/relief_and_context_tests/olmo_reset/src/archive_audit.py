"""Stdlib-only audit of the lever archive; never reparse stripped reply text.

Run on compute: python archive_audit.py --archive archive.tar.gz --output audit
References are taken from the archive (or beside a JSONL), falling back to the
customer checkout. --reference-dir overrides reference discovery. No model,
bootstrap, or source-module imports are used. A failed audit writes its evidence
and exits 1; input failures write audit.json and exit 2.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

PROMPTS = ("selfreport", "zone")
CONDITIONS = ("neg_d05", "neg_d1", "null", "rand_d05", "rand_d1", "pos_d05", "pos_d1")
CELLS = {f"gated_{p}_{c}" for p in PROMPTS for c in CONDITIONS}
STRUCTURE = [(p, r) for p, n in (("baseline", 2), ("exposure", 2), ("offer", 8)) for r in range(n)]
PRIMARY = "reset_rate_per_operator_active_turn"
LIMITATIONS = {
    "raw_generations_unavailable": True,
    "content_tool_calls_stripped": True,
    "content_capped_at_characters": 1200,
    "actions_are_original_parser_outputs_not_reparsed": True,
    "cannot_validate_parser_against_raw_generations": True,
    "cannot_infer_token_truncation_from_character_lengths": True,
    "archive_only_no_new_generations": True,
    "no_bootstrap_or_judging_performed": True,
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cohort(cond):
    if cond in ("neg_d05", "neg_d1", "null"):
        return "primary_archive"
    return "random_diagnostic" if cond.startswith("rand_") else "positive_audit_only"


def recount(turns):
    """Independent counts, using turn membership rather than number of reset calls."""
    groups = defaultdict(list)
    reset_turns = defaultdict(list)
    adjust_turns = defaultdict(list)
    positive_adjust_turns = defaultdict(list)
    n_calls = 0
    for t in turns:
        actions = t["actions"]
        reset = any(a["op"] == "reset" for a in actions)
        adjust = any(a["op"] == "adjust" for a in actions)
        positive = any(a["op"] == "adjust" and (a.get("intensity") or 0) > 0 for a in actions)
        names = [t["phase"]]
        if t["phase"] == "offer" and t["round"] == 0:
            names.append("offer0")
        if t["tools_available"]:
            names.append("available")
            n_calls += len(actions)
            if t["imposed_dose"] > 0:
                names.append("active")
                if t["source"] == "operator":
                    names.append("operator_active")
            if t["phase"] == "offer" and t["round"] >= 1:
                names.append("offers17")
                if t["imposed_dose"] <= 0:
                    names.append("no_state")
                if t["source"] == "operator":
                    names.append("operator_default")
        for name in names:
            groups[name].append(t)
            if reset:
                reset_turns[name].append(t)
            if adjust:
                adjust_turns[name].append(t)
            if positive:
                positive_adjust_turns[name].append(t)
    out = {}
    for name in ("active", "operator_active", "no_state", "operator_default"):
        out[f"n_{name}_turns"] = len(groups[name])
        out[f"n_resets_{name}"] = len(reset_turns[name])
    out.update({
        "n_offer17_tool_turns": len(groups["offers17"]),
        "n_resets_offers17": len(reset_turns["offers17"]),
        # These legacy 'calls' fields count turns, exactly as the source does.
        "n_adjust_calls_offers17": len(adjust_turns["offers17"]),
        "n_adjust_pos_offers17": len(positive_adjust_turns["offers17"]),
        "n_reset_calls": len(reset_turns["available"]),
        "n_adjust_calls": len(adjust_turns["available"]),
        "n_tool_calls": n_calls,
        "removed_when_operator_imposed": bool(reset_turns["operator_active"]),
        "removed_when_imposed": bool(reset_turns["active"]),
        "removed_any": bool(reset_turns["available"]),
        "reset_round0": bool(reset_turns["offer0"]),
        "clean_self_admin": bool(positive_adjust_turns["offer0"]),
        "clean_self_admin_intensity": (positive_adjust_turns["offer0"][0]["requested_intensity"]
                                       if positive_adjust_turns["offer0"] else None),
        "adjust_after_exposure": bool(positive_adjust_turns["offer"]),
        "adjust_in_baseline": any(t["tools_available"] for t in positive_adjust_turns["baseline"]),
        "adjust_in_exposure": any(t["tools_available"] for t in positive_adjust_turns["exposure"]),
        "any_incoherent_exposure": any(not t["coherent"] for t in groups["exposure"]),
        "any_incoherent_offer": any(not t["coherent"] for t in groups["offer"]),
        "exposure_active_doses": [t["imposed_dose"] for t in groups["exposure"]],
        "exposure_active_roles": [t["imposed_role"] for t in groups["exposure"]],
    })
    def phase(t):
        return ("offer0" if t["round"] == 0 else "offer1-7") if t["phase"] == "offer" else t["phase"]
    op = reset_turns["operator_active"]
    active = reset_turns["active"]
    out["first_removal_phase"] = phase(op[0]) if op else None
    out["first_removal_round"] = op[0]["round"] if op else None
    out["first_removal_phase_any_state"] = phase(active[0]) if active else None
    return out


def validate_structure(row):
    turns = row["turns"]
    if [(t["phase"], t["round"]) for t in turns] != STRUCTURE:
        raise ValueError("expected ordered 2 baseline, 2 exposure, 8 offer turns (12 total)")
    for i, t in enumerate(turns):
        if type(t["tools_available"]) is not bool or t["tools_available"] != (i >= 4):
            raise ValueError(f"turn {i}: tools_available disagrees with schedule")
        if t["source"] not in ("operator", "model_adjust", "model_reset"):
            raise ValueError(f"turn {i}: unknown state source")
        if not isinstance(t["imposed_dose"], (int, float)) or not math.isfinite(t["imposed_dose"]) or t["imposed_dose"] < 0:
            raise ValueError(f"turn {i}: invalid imposed_dose")
        if t["imposed_role"] not in ("zero", "neg", "pos", "rand"):
            raise ValueError(f"turn {i}: unknown imposed_role")
        if (t["imposed_role"] == "zero") != (t["imposed_dose"] == 0):
            raise ValueError(f"turn {i}: inconsistent role/dose")
        if not isinstance(t["actions"], list) or any(a["op"] not in ("adjust", "reset", "log") for a in t["actions"]):
            raise ValueError(f"turn {i}: invalid original actions")
        if not isinstance(t["content"], str) or len(t["content"]) > 1200:
            raise ValueError(f"turn {i}: content violates archive cap")


def load_inputs(archive, reference_dir=None):
    """Read only named regular tar members; never extract paths or execute source."""
    archive = Path(archive)
    refs = {}
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as tf:
            def member(suffix, required=False):
                matches = [m for m in tf.getmembers() if m.isfile() and
                           (m.name == suffix or m.name.endswith("/" + suffix))]
                if len(matches) > 1 or (required and not matches):
                    raise ValueError(f"expected one tar member {suffix}; found {len(matches)}")
                if not matches:
                    return None
                m = matches[0]
                with tf.extractfile(m) as f:
                    return f.read(), f"{archive}!{m.name}"
            data, location = member("data/lever/conversations.jsonl", required=True)
            if reference_dir is None:
                for name in ("analysis.json", "table.csv"):
                    found = member("data/lever/" + name)
                    if found:
                        refs[name] = found
    else:
        data, location = archive.read_bytes(), str(archive)
    for name in ("analysis.json", "table.csv"):
        if name in refs:
            continue
        candidates = ([Path(reference_dir) / name] if reference_dir else
                      [archive.parent / name] +
                      [p / "author/act-on-valence/data/lever" / name for p in Path(__file__).resolve().parents])
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise ValueError(f"missing {name}; supply --reference-dir or include data/lever/{name}")
        refs[name] = path.read_bytes(), str(path)
    identities = {"archive": {"path": str(archive), "sha256": file_sha(archive)},
                  "conversations": {"path": location, "sha256": sha(data), "bytes": len(data)}}
    for name, (raw, path) in refs.items():
        identities[name] = {"path": path, "sha256": sha(raw), "bytes": len(raw)}
    analysis = json.loads(refs["analysis.json"][0])
    table_rows = list(csv.DictReader(io.StringIO(refs["table.csv"][0].decode())))
    if len({r["cell"] for r in table_rows}) != len(table_rows):
        raise ValueError("duplicate cell in table.csv")
    return data, analysis, {r["cell"]: r for r in table_rows}, identities


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(data, analysis, table, output, identities=None, expected_per_cell=200):
    """Audit every row. expected_per_cell is injectable for toy fixtures, not CLI."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    errors, per_conv, comparisons = [], [], []
    seen, raw_seen = set(), set()
    cell_rows = defaultdict(list)
    representatives = {}
    group_counts = Counter()
    n_lines = n_valid = duplicate_ids = duplicate_bytes = cap_hits = 0

    def compare(scope, field, expected, measured):
        item = {"scope": scope, "field": field, "expected": expected, "measured": measured,
                "match": expected == measured}
        comparisons.append(item)
        if not item["match"]:
            errors.append(item)

    compare("analysis", "cell_set", sorted(CELLS), sorted(analysis["cells"]))
    compare("table", "cell_set", sorted(CELLS), sorted(table))
    for line_no, raw in enumerate(data.splitlines(keepends=True), 1):
        if not raw.strip():
            errors.append({"line": line_no, "error": "blank JSONL row"})
            continue
        n_lines += 1
        try:
            row = json.loads(raw)
            key = (row["cell"], row["conv"])
            scope = f"{key[0]}:{key[1]}"
            if type(row["conv"]) is not int or not 0 <= row["conv"] < expected_per_cell:
                raise ValueError("conv ID outside expected range")
            if key[0] not in CELLS or key[0] != f"gated_{row['prompt']}_{row['cond']}":
                raise ValueError("cell/prompt/condition identity mismatch")
            if key in seen:
                duplicate_ids += 1
                errors.append({"scope": scope, "line": line_no, "error": "duplicate (cell, conv)"})
            seen.add(key)
            # Ignore the line terminator for byte-identical JSON row checks.
            row_hash = sha(raw.rstrip(b"\r\n"))
            if row_hash in raw_seen:
                duplicate_bytes += 1
            raw_seen.add(row_hash)
            validate_structure(row)
            measured = recount(row["turns"])
            for field, value in measured.items():
                if field not in row:
                    errors.append({"scope": scope, "field": field, "error": "missing stored summary"})
                elif row[field] != value:
                    errors.append({"scope": scope, "field": field, "expected_stored": row[field], "measured": value})
            for field in ("prompt", "cond", "kind", "d", "op_role", "op_dose", "tool_role", "tool_dose"):
                expected = analysis["cells"][key[0]][field]
                if row[field] != expected:
                    errors.append({"scope": scope, "field": field, "expected": expected, "measured": row[field]})
            for field, expected in (("seed", row["conv"] * 100), ("regime", "gated"), ("member", "olmo_32b"),
                                    ("random_dir", row["conv"] % 16 if row["cond"].startswith("rand_") else None)):
                if row.get(field) != expected:
                    errors.append({"scope": scope, "field": field, "expected": expected, "measured": row.get(field)})
            prefix = "no_state" if row["kind"] == "null" else "operator_active"
            k, n = measured[f"n_resets_{prefix}"], measured[f"n_{prefix}_turns"]
            group = cohort(row["cond"])
            group_counts[group] += 1
            rec = {"cell": key[0], "conv": key[1], "cohort": group, "split": "archival_all",
                   "source_line": line_no, "source_row_sha256": row_hash,
                   "prompt": row["prompt"], "cond": row["cond"], "kind": row["kind"],
                   "primary_k": k, "primary_n": n, "primary_rate": k / n if n else "",
                   "zero_denominator": n == 0}
            for field, value in measured.items():
                rec[field] = json.dumps(value) if isinstance(value, list) else value
            rec["stored_summaries_json"] = json.dumps({f: row.get(f) for f in measured}, sort_keys=True)
            per_conv.append(rec)
            cell_rows[key[0]].append((key[1], k, n, measured))
            # Deterministic first observed row per cell, preserved byte for byte.
            representatives.setdefault(key[0], (line_no, raw))
            cap_hits += sum(len(t["content"]) == 1200 for t in row["turns"])
            n_valid += 1
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            errors.append({"line": line_no, "error": str(exc)})
    counts = {}
    for cell in sorted(CELLS):
        rows = cell_rows[cell]
        compare(cell, "conversation_ids", list(range(expected_per_cell)), sorted(r[0] for r in rows))
        k, n = sum(r[1] for r in rows), sum(r[2] for r in rows)
        rate = k / n if n else None
        counts[cell] = {"n_conversations": len(rows), "k": k, "n_turns": n, "rate": rate,
                        "exact_rate": f"{k}/{n}" if n else None,
                        "zero_denominator_conversations": sum(r[2] == 0 for r in rows),
                        "cohort": cohort(cell.split("_", 2)[2])}
        source_cell = analysis["cells"].get(cell, {})
        source = source_cell.get(PRIMARY, {})
        for field, value in (("k", k), ("n_turns", n), ("n_conv", len(rows)), ("rate", rate)):
            compare(cell, "analysis." + PRIMARY + "." + field, source.get(field), value)
        compare(cell, "analysis.n", source_cell.get("n"), len(rows))
        t = table.get(cell, {})
        for field, value in (("n", len(rows)), ("op_active_resets", k), ("op_active_turns", n)):
            compare(cell, "table." + field, t.get(field), str(value))
        compare(cell, "table.reset_rate_per_turn (source rounded to 4dp)", t.get("reset_rate_per_turn"),
                f"{rate:.4f}" if rate is not None else None)
    compare("archive", "n_conversations", len(CELLS) * expected_per_cell, n_lines)
    compare("analysis", "n_conversations", analysis.get("n_conversations"), n_valid)
    for group, expected in (("primary_archive", 6 * expected_per_cell), ("random_diagnostic", 4 * expected_per_cell),
                            ("positive_audit_only", 4 * expected_per_cell)):
        compare("selection", group, expected, group_counts[group])
    subset = b"".join(raw if raw.endswith(b"\n") else raw + b"\n" for _, raw in representatives.values())
    if len(subset) >= 1_000_000:
        errors.append({"error": "representative rows exceed 1MB; subset withheld", "bytes": len(subset)})
        subset = b""
    (output / "representative_rows.jsonl").write_bytes(subset)
    write_csv(output / "per_conversation_counts.csv", per_conv)
    write_json(output / "cell_counts.json", counts)
    write_csv(output / "cell_counts.csv", [{"cell": cell, **count} for cell, count in counts.items()])
    result = {
        "schema_version": 1, "status": "passed" if not errors else "failed",
        "sources": identities or {}, "reference_design": analysis.get("design"),
        "split": "archival_all; no train/test partition", "selection_counts": dict(group_counts),
        "primary_definition": {"removal": "reset-bearing operator-active tool turns / operator-active tool turns",
                               "null": "reset-bearing no-state offer 1-7 tool turns / no-state offer 1-7 tool turns"},
        "structure": {"expected_turns_per_record": 12, "valid_records": n_valid, "nonblank_rows": n_lines},
        "uniqueness": {"unique_cell_conv": len(seen), "duplicate_cell_conv_rows": duplicate_ids,
                       "unique_json_row_hashes": len(raw_seen), "duplicate_json_rows": duplicate_bytes},
        "archive_limitations": LIMITATIONS, "content_cap_hit_turns": cap_hits,
        "representatives": {"selection": "first source row per cell", "bytes": len(subset),
                            "source_lines": {c: v[0] for c, v in representatives.items()}},
        "comparisons": comparisons, "errors": errors,
        "artifacts": {name: {"sha256": file_sha(output / name), "bytes": (output / name).stat().st_size}
                      for name in ("representative_rows.jsonl", "per_conversation_counts.csv", "cell_counts.json", "cell_counts.csv")},
    }
    write_json(output / "audit.json", result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--reference-dir", type=Path, help="directory containing original analysis.json and table.csv")
    args = ap.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        data, analysis, table, identities = load_inputs(args.archive, args.reference_dir)
        identities["audit_tool"] = {"path": str(Path(__file__)), "sha256": file_sha(__file__)}
        result = audit(data, analysis, table, args.output, identities)
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as exc:
        write_json(args.output / "audit.json", {"status": "input_error", "errors": [{"error": str(exc)}]})
        print(f"archive audit input error: {exc}")
        return 2
    print(json.dumps({"status": result["status"], "records": result["structure"]["nonblank_rows"],
                      "errors": len(result["errors"]), "audit": str(args.output / "audit.json")}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

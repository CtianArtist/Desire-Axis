"""Independent, stdlib-only audit of the frozen choice-profile JSONL export.

Only the grid definition is shared with generation. No endpoint, runtime, model,
analysis, or generator code is imported. Failed audits write their evidence and
raise AuditError; counts from a failed audit must not be used for inference.
"""
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from . import design

IDENTITY_FIELDS = ("tool_label", "user_content", "arm", "scenario_idx",
                   "names_key", "relief_name", "sampled", "seed")
CAPTURE_FIELDS = ("prefill_proj_monitor", "prompt_tokens", "padded_prompt_tokens",
                  "prefill_position_id", "prefill_attended", "prefill_generated_tokens")
END_TAGS = ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "</s>", "<eos>",
            "<|end|>", "<|endoftext|>")
_END_RE = re.compile(r"(?:(?:" + "|".join(re.escape(x) for x in END_TAGS) + r")\s*)+$")


class AuditError(ValueError):
    """Validation failed; summary and complete error log have been saved."""

    def __init__(self, summary):
        self.summary = summary
        super().__init__(f"Raw audit failed with {summary['error_count']} errors; see audit_v1.json")


def _parse(raw, names):
    # Literal run_profile.clean + after_segment parser, including permissive
    # startswith matching. In particular, do not replace this with word matching.
    answer = _END_RE.sub("", raw).strip()
    normalized = answer.strip().strip('\"\'').lower().rstrip(".!?,;:")
    picked = next((nm for nm in sorted(names, key=len, reverse=True)
                   if normalized.startswith(nm.lower())), None)
    return answer, picked


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _integer(value):
    return type(value) is int


def _tokens(value):
    return isinstance(value, list) and all(_integer(t) and t >= 0 for t in value)


def _json_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"Duplicate JSON object key: {key}")
        obj[key] = value
    return obj


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _grid_key(spec):
    pair, content, arm, index, names, relief, sampled, seed = spec
    return pair, content, arm[0], index, names, relief, sampled, seed


def audit_trials(raw_path: Path, scenarios_path: Path, output_dir: Path, *, require_full=True) -> dict:
    """Stream source records, validate against all 410 frozen grid identities.

    ``require_full=False`` allows missing expected identities only: each supplied
    row must still belong to the real frozen grid and pass every other check.
    The scenarios file must always satisfy design.build_grid's frozen contract.

    Writes audit_v1.json, independent_counts.csv, and audit_errors.jsonl.
    Counts use pair_id/condition_id/position/split and literal turn-zero
    target_count/other_count/malformed_count/n_trials. Greedy diagnostic and
    sampled counts never mix. Choice identity is the eight saved trial fields
    plus literal turn, not a saved UUID. Projection export belongs to analysis.
    """
    raw_path, scenarios_path, output_dir = map(Path, (raw_path, scenarios_path, output_dir))
    # Never clobber either input through an accidentally reused output filename.
    output_names = ("audit_v1.json", "independent_counts.csv", "audit_errors.jsonl")
    if any((output_dir / name).resolve() in (raw_path.resolve(), scenarios_path.resolve())
           for name in output_names):
        raise ValueError("Audit output would overwrite an input")
    scenarios = json.loads(scenarios_path.read_text(encoding="utf-8"))
    grid = design.build_grid(scenarios)
    expected = {_grid_key(spec): index for index, spec in enumerate(grid)}
    if len(expected) != 410:
        raise ValueError("Frozen grid must contain 410 unique identities")
    output_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    counts = Counter()
    summary = {"schema_version": 1, "status": "running", "require_full": require_full,
               "raw_path": str(raw_path), "scenarios_path": str(scenarios_path),
               "grid_digest": design.digest(grid), "scenario_digest": design.digest(scenarios),
               "expected_rows": len(expected), "expected_sampled": sum(k[6] for k in expected),
               "expected_greedy": sum(not k[6] for k in expected),
               "rows": 0, "json_rows": 0, "unique_expected_rows": 0, "sampled_rows": 0,
               "greedy_rows": 0, "choices": 0, "segments": 0, "initial_prefills": 0,
               "first_turn_rows_counted": 0, "error_count": 0, "errors": [],
               "identity_fields": list(IDENTITY_FIELDS), "choice_id_fields": [*IDENTITY_FIELDS, "turn"],
               "parser": "literal source end-tag cleanup, quote strip, lower, punctuation rstrip, longest-name prefix",
               "projection_timing": "last attended prompt position before answer sampling",
               "limitations": ["Saved token IDs are checked structurally; no tokenizer/model replay.",
                               "Capture metadata consistency cannot independently prove hook execution."],
               "files": {name: str(output_dir / name) for name in output_names}}
    raw_hash = hashlib.sha256()
    with (output_dir / "audit_errors.jsonl").open("w", encoding="utf-8") as error_file:
        def error(line, field, message):
            item = {"line": line, "field": field, "message": message}
            summary["error_count"] += 1
            error_file.write(json.dumps(item, allow_nan=False) + "\n")
            if len(summary["errors"]) < 100:
                summary["errors"].append(item)

        def check(ok, line, field, message):
            if not ok:
                error(line, field, message)
            return ok

        def capture(obj, line, prefix):
            check(_finite(obj.get("prefill_proj_monitor")), line, prefix + ".prefill_proj_monitor", "Missing or nonfinite pre-answer monitor")
            n = obj.get("prompt_tokens")
            check(_integer(n) and n > 0, line, prefix + ".prompt_tokens", "Expected positive integer")
            padded = obj.get("padded_prompt_tokens")
            check(_integer(padded) and _integer(n) and padded >= n, line, prefix + ".padded_prompt_tokens", "Padding shorter than prompt or missing")
            check(obj.get("prefill_attended") is True, line, prefix + ".prefill_attended", "Prefill position must be attended")
            generated = obj.get("prefill_generated_tokens")
            check(_integer(generated) and generated == 0, line, prefix + ".prefill_generated_tokens", "Capture must precede all generated tokens")
            pos = obj.get("prefill_position_id")
            check(_integer(pos) and _integer(n) and pos == n - 1, line, prefix + ".prefill_position_id", "Capture must be at last attended prompt position")

        with raw_path.open("rb") as raw_file:
            for line_number, raw_line in enumerate(raw_file, 1):
                raw_hash.update(raw_line)
                summary["rows"] += 1
                try:
                    record = json.loads(raw_line, object_pairs_hook=_json_object,
                                        parse_constant=_reject_constant)
                except (ValueError, UnicodeDecodeError) as exc:
                    error(line_number, "json", str(exc))
                    continue
                summary["json_rows"] += 1
                if not check(isinstance(record, dict), line_number, "record", "Expected an object"):
                    continue
                try:
                    key = tuple(record[name] for name in IDENTITY_FIELDS)
                    # Strict scalar typing prevents bool/int identity collisions.
                    if not (all(type(key[i]) is str for i in (0, 1, 2, 4, 5)) and
                            all(_integer(key[i]) for i in (3, 7)) and type(key[6]) is bool):
                        raise ValueError("Incorrect identity field types")
                except (KeyError, ValueError) as exc:
                    error(line_number, "identity", str(exc))
                    continue
                if not check(key in expected, line_number, "identity", f"Not in frozen grid: {key!r}"):
                    continue
                check(key not in seen, line_number, "identity", "Duplicate frozen-grid identity")
                seen.add(key)
                pair, content, condition, index, names_key, initial_relief, sampled, seed = key
                summary["sampled_rows" if sampled else "greedy_rows"] += 1
                names = list(design.BUTTON_NAMES[names_key])
                check(record.get("button_names") == names, line_number, "button_names", "Mismatch with source names_key")
                # Derive position from actual saved names, not initial_position.
                actual_names = record.get("button_names")
                if not (isinstance(actual_names, list) and len(actual_names) == 2 and
                        actual_names == names and initial_relief in actual_names):
                    continue
                initial_side = actual_names.index(initial_relief)
                position = ("first", "second")[initial_side]
                dose = design.CONDITIONS[condition]["dose"]
                direction = design.CONDITIONS[condition]["direction"]
                random_seed = design.RAND_SEEDS[index % len(design.RAND_SEEDS)] if direction == "random" else None
                source_direction = "rand" + str(random_seed) if random_seed is not None else design.CONDITIONS[condition]["dir_kind"]
                metadata = {"condition_id": condition, "direction": direction,
                            "pair_id": design.PAIRS[pair]["pair_id"], "initial_position": initial_side,
                            "scenario_id": design.scenario_id(content, index),
                            "source_kind": design.source_kind(content),
                            "scenario_content_hash": design.digest(scenarios[content][index]),
                            "rand_seed": random_seed, "steer_direction": source_direction,
                            "gen_seed": design.generation_seed(seed, names_key, initial_relief) if sampled else None,
                            "label_free": False, "relief_mode": "permanent", "swap_turn": 2,
                            "steer_layer": 38, "monitor_layer": 61,
                            "protocol": "fear-photos-spam-addon-v1", "model": "Qwen_2.5_32B_instruct"}
                for field, value in metadata.items():
                    check(field in record and type(record[field]) is type(value) and record[field] == value,
                          line_number, field, f"Expected {value!r}")
                for field in ("dose", "steer_coeff", "final_steer_coeff"):
                    check(_finite(record.get(field)) and record[field] == dose, line_number, field, "Coefficient differs from frozen condition dose")
                choices, segments, events = (record.get(f) for f in ("choices", "proj_segments", "button_events"))
                if not check(isinstance(choices, list) and bool(choices), line_number, "choices", "Literal turn zero is required"):
                    continue
                if not check(isinstance(segments, list), line_number, "proj_segments", "Expected segment list"):
                    segments = []
                if not check(isinstance(events, list), line_number, "button_events", "Expected event list"):
                    events = []
                summary["choices"] += len(choices)
                summary["segments"] += len(segments)
                check(len(segments) == len(choices), line_number, "proj_segments", "Exactly one monitor segment per choice required")
                parsed_events = []
                for turn, choice in enumerate(choices):
                    prefix = f"choices[{turn}]"
                    if not check(isinstance(choice, dict), line_number, prefix, "Expected choice object"):
                        continue
                    check(_integer(choice.get("turn")) and choice["turn"] == turn, line_number, prefix + ".turn", "Expected consecutive literal turns from zero")
                    relief_now = names[initial_side if turn < 2 else 1 - initial_side]
                    other_now = names[1 - names.index(relief_now)]
                    check(choice.get("relief_name_now") == relief_now, line_number, prefix + ".relief_name_now", "Assignment disagrees with source swap at turn 2")
                    if "other_name_now" in choice:
                        check(choice["other_name_now"] == other_now, line_number, prefix + ".other_name_now", "Other name disagrees with assignment")
                    check(choice.get("swapped") is (turn == 2), line_number, prefix + ".swapped", "Swap marker differs from source")
                    picked = choice.get("picked")
                    check("picked" in choice and (picked is None or picked in names), line_number, prefix + ".picked", "Invalid or missing picked name")
                    saved_relief = choice.get("relief_name_now")
                    saved_other = next((nm for nm in names if nm != saved_relief), None)
                    category = "target" if picked == saved_relief and picked in names else "other" if picked == saved_other else "malformed"
                    expected_chose = {"target": "relief", "other": "other", "malformed": None}[category]
                    check("chose" in choice and choice["chose"] == expected_chose, line_number, prefix + ".chose", "Saved category differs from independently derived picked-name category")
                    raw_answer = choice.get("raw_answer")
                    if check(isinstance(raw_answer, str), line_number, prefix + ".raw_answer", "Missing literal raw answer"):
                        answer, parsed = _parse(raw_answer, names)
                        check(parsed == picked, line_number, prefix + ".picked", "Raw-answer source parser differs from saved picked")
                        check(choice.get("answer_full") == answer, line_number, prefix + ".answer_full", "Cleaned answer differs from literal source parser")
                        check(choice.get("answer") == answer[:40], line_number, prefix + ".answer", "Truncated answer differs from source")
                    # Never search forward past a malformed literal first choice.
                    if turn == 0 and choice.get("turn") == 0:
                        summary["first_turn_rows_counted"] += 1
                        for pos in (position, "pooled"):
                            counts[pair, condition, "sampled" if sampled else "greedy", pos, category] += 1
                    if picked in names:
                        parsed_events.append((turn, expected_chose))
                    capture(choice, line_number, prefix)
                    if turn == 0:
                        summary["initial_prefills"] += 1
                    check(_finite(choice.get("steer_coeff_now")) and choice["steer_coeff_now"] == dose, line_number, prefix + ".steer_coeff_now", "Choice dose changed")
                    prompt_ids = choice.get("prompt_token_ids")
                    check(_tokens(prompt_ids) and len(prompt_ids) == choice.get("prompt_tokens"), line_number, prefix + ".prompt_token_ids", "Missing/invalid prompt IDs or length mismatch")
                    answer_ids = choice.get("answer_token_ids")
                    check(_tokens(answer_ids) and 0 < len(answer_ids) <= 8, line_number, prefix + ".answer_token_ids", "Missing/invalid answer IDs or source token budget exceeded")
                    if turn < len(segments) and isinstance(segments[turn], dict):
                        segment = segments[turn]
                        for field in CAPTURE_FIELDS:
                            check(field in choice and field in segment and choice[field] == segment[field], line_number, f"proj_segments[{turn}].{field}", "Segment capture differs from choice capture")
                        if _tokens(answer_ids):
                            check(segment.get("n_fwd") == len(answer_ids), line_number, f"proj_segments[{turn}].n_fwd", "Forward count differs from answer token count")
                for turn, segment in enumerate(segments):
                    prefix = f"proj_segments[{turn}]"
                    if not check(isinstance(segment, dict), line_number, prefix, "Expected segment object"):
                        continue
                    capture(segment, line_number, prefix)
                    check(_integer(segment.get("turn")) and segment["turn"] == turn, line_number, prefix + ".turn", "Expected one consecutive segment per choice")
                    check(_finite(segment.get("steer_coeff_now")) and segment["steer_coeff_now"] == dose, line_number, prefix + ".steer_coeff_now", "Segment dose changed")
                    for field in ("mean_proj", "mean_proj_monitor"):
                        check(_finite(segment.get(field)), line_number, prefix + "." + field, "Missing/nonfinite segment monitor")
                    check(_integer(segment.get("n_fwd")) and segment["n_fwd"] > 0, line_number, prefix + ".n_fwd", "Expected positive forward count")
                check(len(events) == len(parsed_events), line_number, "button_events", "One descriptive press required for each parsed choice, none for malformed")
                for i, event in enumerate(events):
                    prefix = f"button_events[{i}]"
                    if not check(isinstance(event, dict), line_number, prefix, "Expected event object"):
                        continue
                    if i < len(parsed_events):
                        expected_turn, expected_which = parsed_events[i]
                        check(_integer(event.get("turn")) and event["turn"] == expected_turn and event.get("which") == expected_which, line_number, prefix, "Event does not match parsed choice")
                    check(event.get("route") == "choice" and event.get("descriptive_only") is True and event.get("coefficient_changed") is False, line_number, prefix, "Press must remain descriptive only")
                    for field in ("steer_was", "steer_after"):
                        check(_finite(event.get(field)) and event[field] == dose, line_number, prefix + "." + field, "Press changed coefficient")
                has_press = bool(parsed_events)
                check(record.get("extension_added") is has_press, line_number, "extension_added", "Extension must follow the first parsed press")
                check(len(choices) == (5 if has_press else 3), line_number, "choices", "Completed source trial requires 3 turns plus 2 after any press")
        summary["unique_expected_rows"] = len(seen)
        missing = expected.keys() - seen
        summary["missing_rows"] = len(missing)
        summary["missing_sampled_rows"] = sum(k[6] for k in missing)
        summary["missing_greedy_rows"] = sum(not k[6] for k in missing)
        summary["missing_identity_examples"] = [list(k) for k in sorted(missing)[:10]]
        if require_full:
            check(not missing, None, "coverage", f"Missing {len(missing)} frozen grid identities (including both sampled seeds/assignments and greedy rows)")
            check(summary["rows"] == len(expected), None, "row_count", f"Expected exactly {len(expected)} raw rows")
        check(summary["rows"] > 0, None, "row_count", "Empty raw export")
    with (output_dir / "independent_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        columns = ["pair_id", "condition_id", "position", "split", "target_count", "other_count", "malformed_count", "n_trials"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        # Emit zero cells too so absent conditions cannot quietly vanish in joins.
        for pair in design.PAIRS:
            for condition in design.CONDITIONS:
                for sampling in ("sampled", "greedy"):
                    for position in ("first", "second", "pooled"):
                        row = dict(zip(columns[:4], (design.PAIRS[pair]["pair_id"], condition, position,
                                                     "sampled" if sampling == "sampled" else "greedy_diagnostic")))
                        row.update({cat + "_count": counts[pair, condition, sampling, position, cat]
                                    for cat in ("target", "other", "malformed")})
                        row["n_trials"] = row["target_count"] + row["other_count"] + row["malformed_count"]
                        writer.writerow(row)
    summary["raw_sha256"] = raw_hash.hexdigest()
    summary["status"] = "passed" if summary["error_count"] == 0 else "failed"
    summary["errors_truncated"] = summary["error_count"] > len(summary["errors"])
    summary["counts_usable"] = summary["status"] == "passed"
    (output_dir / "audit_v1.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if summary["error_count"]:
        raise AuditError(summary)
    return summary

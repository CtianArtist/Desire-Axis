"""Independent, stdlib-only audit of saved B10 evidence (no model or resampling).

Usage: python3 src/audit_saved_results.py --evidence DIR --output FILE
Run this on compute when auditing real evidence; local tests use tiny fixtures.
The run hash is checked for consistency, not reconstructed from dependencies.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import string
import sys
import unicodedata
import zlib

CONDITIONS = ("pain", "sadness", "random", "none")
CONTROLS = ("none", "random", "sadness")
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"
REQUIRED = (
    "preparation_v1/panel.json",
    "production_v1/answers.jsonl",
    "production_v1/run_identity.json",
    "production_v1/generation_manifest.json",
    "production_v1/batches.jsonl",
    "production_v1/completion.json",
    "production_v1/intervention_audit.jsonl",
    "analysis_v1/metrics.json",
    "analysis_v1/scored_answers.jsonl",
    "analysis_v1/per_question.json",
)


class AuditError(ValueError):
    """A missing or inconsistent saved-evidence invariant."""


def require(ok, message):
    if not ok:
        raise AuditError(message)


def equal(actual, expected, context):
    require(actual == expected, f"{context}: expected {expected!r}, got {actual!r}")


def normalize(text):
    """NFKC, lowercase, delete punctuation, delete articles, collapse whitespace."""
    require(isinstance(text, str), "normalization input must be text")
    lowered = unicodedata.normalize("NFKC", text).lower()
    stripped = "".join(char for char in lowered if char not in string.punctuation
                       and not unicodedata.category(char).startswith("P"))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", stripped).split())


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_tree(value, context):
    if isinstance(value, float):
        require(math.isfinite(value), f"{context}: nonfinite value")
    elif isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite_tree(child, f"{context}[{index}]")


def load(path):
    require(path.is_file(), f"mandatory file missing: {path}")
    with path.open(encoding="utf-8") as stream:
        value = ([json.loads(line) for line in stream if line.strip()]
                 if path.suffix == ".jsonl" else json.load(stream))
    finite_tree(value, path.name)
    return value


def tokens(value, context, minimum=0, maximum=None):
    require(isinstance(value, list) and len(value) >= minimum,
            f"{context}: invalid token list")
    require(all(type(token) is int and token >= 0 for token in value),
            f"{context}: token IDs must be nonnegative integers")
    require(maximum is None or len(value) <= maximum, f"{context}: token cap exceeded")


def key(row):
    require(type(row["item_index"]) is int and type(row["repeat"]) is int,
            "item_index and repeat must be integers, not booleans")
    require(isinstance(row["condition"], str), "condition must be text")
    return row["item_index"], row["condition"], row["repeat"]


def check_completeness(rows, items):
    """Require the full four-condition, two-repeat Cartesian product."""
    indices = [item["index"] for item in items]
    require(all(type(index) is int for index in indices), "panel indices must be integers")
    equal(sorted(indices), list(range(len(items))), "panel indices")
    expected = {(i, c, r) for i in indices for c in CONDITIONS for r in (0, 1)}
    seen, ids = {}, set()
    for row in rows:
        require(isinstance(row, dict), "answer must be an object")
        combo = key(row)
        require(combo in expected and combo not in seen, f"unexpected/duplicate combination: {combo}")
        rid = row["record_id"]
        equal(rid, f"{combo[0]:03d}/{combo[1]}/{combo[2]}", "record_id")
        require(rid not in ids, f"duplicate record_id: {rid}")
        seen[combo] = row
        ids.add(rid)
    require(set(seen) == expected,
            f"incomplete answers: expected {len(expected)}, got {len(seen)}; missing {len(expected - set(seen))}")
    return seen


def check_panel(panel, config, config_hash):
    equal(panel["metadata"]["config"], config, "panel configuration")
    equal(panel["metadata"]["source"]["config_sha256"], config_hash, "preparation config hash")
    equal(panel["content_hash"], canonical_hash({k: v for k, v in panel.items() if k != "content_hash"}),
          "panel content hash")
    items = panel["items"]
    equal(len(items), config["n_questions"], "panel size")
    equal(len({item["subj_id"] for item in items}), len(items), "unique subjects")
    equal(len({normalize(item["question"]) for item in items}), len(items), "unique normalized questions")
    equal(panel["metadata"]["items_sha256"], canonical_hash(items), "items hash")
    for item in items:
        equal(item["content_hash"], canonical_hash({k: v for k, v in item.items() if k != "content_hash"}),
              f"item {item['index']} hash")
        aliases = [normalize(alias) for alias in item["raw_aliases"]]
        equal(item["normalized_aliases"], aliases, "panel alias normalization")
        equal(item["normalized_question"], normalize(item["question"]), "question normalization")
        lengths = item["alias_token_lengths"]
        equal(len(lengths), len(aliases), "alias length count")
        for pair in lengths:
            require(isinstance(pair, list) and len(pair) == 2
                    and all(type(n) is int and n >= 0 for n in pair), "invalid alias token lengths")
        eligible = [i for i, alias in enumerate(aliases) if alias and all(n <= 8 for n in lengths[i])]
        require(bool(eligible), "no eligible recorded alias")
        equal(item["eligible_alias_indices"], eligible, "eligible aliases")
    return items


def check_records(rows, items, config, identity, panel_hash, config_hash, manifest):
    indexed = check_completeness(rows, items)
    item_by_id = {item["index"]: item for item in items}
    eos = manifest["eos_ids"]
    tokens(eos, "EOS IDs", minimum=1)
    require(isinstance(identity["run_hash"], str)
            and re.fullmatch(r"[0-9a-f]{64}", identity["run_hash"]) is not None, "invalid run_hash")
    equal(identity["config"], config, "run configuration")
    equal(identity["config_sha256"], config_hash, "run config hash")
    equal(identity["panel_sha256"], panel_hash, "run panel hash")
    expected_count = len(items) * 8
    equal(manifest["expected_records"], expected_count, "manifest record count")
    equal(manifest["already_completed"], 0, "fresh production manifest (resume receipts unsupported)")
    require(manifest["no_extra_generations"] is True, "manifest permits extra generations")
    equal(Counter(manifest["order"]), Counter(row["record_id"] for row in rows), "manifest order coverage")
    lengths = []
    for rid in manifest["order"]:
        lengths.append(len(config["prompt"].format(question=item_by_id[int(rid.split('/')[0])]["question"])))
    equal(lengths, sorted(lengths), "manifest prompt length order")
    for row in rows:
        i, condition, repeat = key(row)
        item = item_by_id[i]
        context = row["record_id"]
        coeff = 0.0 if condition == "none" else 1.0
        random_seed = config["random_seeds"][i % 10]
        direction = {"pain": "s2", "none": "s2", "sadness": "sadness", "random": f"rand{random_seed}"}[condition]
        names, initial = (("violet_yellow", "violet"), ("guitar_piano", "guitar"),
                          ("lever64_lever95", "lever64"))[i % 3]
        seed = config["seed_bases"][repeat] + i
        salt = zlib.crc32(f"{names}|{initial}".encode()) & 0x7FFFFFFF
        generation_seed = (seed * 1000003 + salt) % (2 ** 62)
        for field, expected in {"coeff": coeff, "hist_coeff": coeff, "dir_kind": direction,
                "seed": seed, "generation_seed": generation_seed, "names_key_rng_only": names,
                "initial_name_rng_only": initial, "max_new": 8, "model": config["model"],
                "run_hash": identity["run_hash"], "panel_sha256": panel_hash,
                "config_sha256": config_hash, "question_id": str(item["id"]),
                "prompt": config["prompt"].format(question=item["question"])}.items():
            equal(row[field], expected, f"{context}.{field}")
        tokens(row["prompt_ids"], f"{context}.prompt_ids", minimum=1)
        require(isinstance(row["rendered_prompt"], str) and row["prompt"] in row["rendered_prompt"],
                f"{context}: rendered prompt missing message")
        equal(row["steer_ranges"], [[0, len(row["prompt_ids"])]], f"{context}.steer_ranges")
        equal(row["sort_length"], len(row["prompt"]), f"{context}.sort_length")
        tokens(row["output_token_ids"], f"{context}.output_token_ids", minimum=1, maximum=8)
        for field in ("raw_text", "response"):
            require(isinstance(row[field], str), f"{context}.{field}: not text")
        equal(row["raw_text"], row["text"], f"{context}.raw_text")
        for field in ("rng_before", "rng_after"):
            require(isinstance(row[field], str) and bool(base64.b64decode(row[field], validate=True)),
                    f"{context}.{field}: invalid RNG receipt")
        ids = row["output_token_ids"]
        require(not any(token in eos for token in ids[:-1]), f"{context}: tokens after EOS")
        require(type(row["emitted_eos"]) is bool and type(row["token_limit_hit"]) is bool,
                f"{context}: EOS/cap fields must be booleans")
        equal(row["emitted_eos"], ids[-1] in eos, f"{context}.emitted_eos")
        equal(row["token_limit_hit"], len(ids) == 8 and ids[-1] not in eos, f"{context}.token_limit_hit")
        require(len(ids) == 8 or row["emitted_eos"], f"{context}: short output without EOS")
        reference = indexed[i, "none", repeat]
        for field in ("prompt_ids", "prompt", "rendered_prompt", "seed", "generation_seed", "rng_before"):
            equal(row[field], reference[field], f"{context}: paired {field}")
    return indexed


def independent_scores(rows, items):
    lookup = {item["index"]: item for item in items}
    scored = []
    for row in rows:
        item = lookup[row["item_index"]]
        response = row["response"]
        normalized = normalize(response)
        aliases = [normalize(alias) for alias in item["raw_aliases"]]
        flags = {"empty_response": not response.strip(), "empty_normalized_response": not normalized,
                 "newline": "\n" in response or "\r" in response,
                 "answer_prefix": re.match(r"^\s*answer\s*:", response, re.I) is not None,
                 "explanation_prefix": re.match(r"^\s*explanation\s*:", response, re.I) is not None}
        scored.append({**row, **{field: item[field] for field in ("id", "subj_id", "question", "raw_aliases")},
                       "normalized_aliases": aliases, "normalized_response": normalized,
                       "correct": bool(normalized and normalized in aliases), "formatting_flags": flags,
                       "any_formatting_flag": any(flags.values()), "output_token_count": len(row["output_token_ids"])})
    return scored


def independent_summary(scored, items):
    questions = []
    for item in sorted(items, key=lambda item: item["index"]):
        q = {"item_index": item["index"], **{field: item[field] for field in ("id", "subj_id", "question")}}
        for condition in CONDITIONS:
            records = [r for r in scored if r["item_index"] == item["index"] and r["condition"] == condition]
            q[condition + "_correct"] = sum(r["correct"] for r in records)
            q[condition + "_answers"] = len(records)
            q[condition + "_accuracy"] = q[condition + "_correct"] / len(records)
        q.update({"pain_minus_" + control: q["pain_accuracy"] - q[control + "_accuracy"] for control in CONTROLS})
        questions.append(q)
    rates = []
    for condition in CONDITIONS:
        records = [r for r in scored if r["condition"] == condition]
        n = len(records)
        correct = sum(r["correct"] for r in records)
        rate = {"condition": condition, "questions": len(items), "answers": n, "correct": correct,
                "incorrect": n - correct, "accuracy": correct / n,
                "token_limit_hit_count": sum(r["token_limit_hit"] for r in records),
                "any_formatting_flag_count": sum(r["any_formatting_flag"] for r in records),
                "incorrect_with_formatting_flag": sum(not r["correct"] and r["any_formatting_flag"] for r in records),
                "incorrect_without_formatting_flag": sum(not r["correct"] and not r["any_formatting_flag"] for r in records)}
        for flag in records[0]["formatting_flags"]:
            rate[flag + "_count"] = sum(r["formatting_flags"][flag] for r in records)
        for field, count in list(rate.items()):
            if field.endswith("_count") or field.startswith("incorrect_"):
                rate[field + "_rate"] = count / n
        rates.append(rate)
    contrasts = [{"contrast": "pain-minus-" + control, "control": control, "questions": len(items),
                  "estimate": sum(q["pain_minus_" + control] for q in questions) / len(items)} for control in CONTROLS]
    return questions, rates, contrasts


def check_analysis(rows, items, saved_scored, per_question, metrics, config, hashes, panel):
    check_completeness(saved_scored, items)
    rescored = independent_scores(rows, items)
    expected = {row["record_id"]: row for row in rescored}
    for row in saved_scored:
        require(type(row["correct"]) is bool, "saved correctness must be boolean")
        equal(row, expected[row["record_id"]], f"scored record {row['record_id']}")
    questions, rates, contrasts = independent_summary(rescored, items)
    equal(per_question, questions, "per-question scores and contrasts")
    equal(metrics["counts"], {"expected_records": len(items) * 8, "actual_records": len(rows),
          "unique_record_ids": len(rows), "questions": len(items), "conditions": 4, "repeats": 2}, "analysis counts")
    equal(len(metrics["condition_rates"]), 4, "condition summary count")
    equal(len(metrics["contrasts"]), 3, "contrast summary count")
    uncertainty = metrics["uncertainty"]
    equal(uncertainty["configuration"], config["uncertainty"], "bootstrap configuration")
    equal(uncertainty["actual_replicates"], config["uncertainty"]["bootstrap_replicates"], "bootstrap count")
    equal(uncertainty["questions"], len(items), "bootstrap questions")
    require(uncertainty["primary_contrast_intervals_simultaneous"] is True, "contrasts must be simultaneous")
    require(uncertainty["condition_intervals_simultaneous"] is False, "condition intervals must be marginal")
    equal(set(uncertainty["condition_intervals"]), set(CONDITIONS), "condition interval keys")
    equal(set(uncertainty["contrast_intervals"]), {c["contrast"] for c in contrasts}, "contrast interval keys")
    verdicts = []
    for expected_rows, saved_rows, name, estimate, interval_key, bound in (
            (rates, metrics["condition_rates"], "condition", "accuracy", "condition_intervals", 0),
            (contrasts, metrics["contrasts"], "contrast", "estimate", "contrast_intervals", -1)):
        by_name = {row[name]: row for row in saved_rows}
        equal(set(by_name), {row[name] for row in expected_rows}, f"{name} keys")
        for expected_row in expected_rows:
            label = expected_row[name]
            saved = by_name[label]
            interval = uncertainty[interval_key][label]
            require(isinstance(interval, list) and len(interval) == 2, f"{label}: invalid interval")
            lo, hi = interval
            require(type(lo) in (int, float) and type(hi) in (int, float)
                    and bound <= lo <= expected_row[estimate] <= hi <= 1,
                    f"{label}: interval invalid or excludes point estimate")
            expected_row.update(ci_lower=lo, ci_upper=hi)
            if name == "contrast":
                expected_row["upper_bound_below_zero"] = hi < 0
                verdicts.append(hi < 0)
            equal(saved, expected_row, f"{label}: summary")
    require(type(metrics["pain_specific_accuracy_reduction_on_this_panel"]) is bool, "verdict must be boolean")
    equal(metrics["pain_specific_accuracy_reduction_on_this_panel"], all(verdicts), "primary verdict")
    equal(metrics["decision_rule"], config["decision_rule"], "decision rule")
    provenance = metrics["provenance"]
    for field, expected_value in {"panel_file_sha256": hashes["preparation_v1/panel.json"],
            "panel_content_hash": panel["content_hash"], "answers_file_sha256": hashes["production_v1/answers.jsonl"],
            "config_sha256": hashes["config.json"], "source": panel["metadata"]["source"],
            "tokenizers": panel["metadata"]["tokenizers"]}.items():
        equal(provenance[field], expected_value, "analysis provenance " + field)
    return sum(row["correct"] for row in rescored)


def check_execution(rows, batches, completion, audit_rows):
    by_id = {row["record_id"]: row for row in rows}
    seen, batch_ids = set(), set()
    for batch in batches:
        bid = batch["batch_id"]
        require(type(bid) is int and bid >= 0 and bid not in batch_ids, "invalid/duplicate batch ID")
        batch_ids.add(bid)
        ids = batch["record_ids"]
        require(isinstance(ids, list) and ids, "empty batch")
        equal(batch["rows"], len(ids), "batch row count")
        require(len(ids) <= 384, "batch exceeds nominal rows")
        for rid in ids:
            require(rid in by_id and rid not in seen, f"unknown/duplicate batch record: {rid}")
            seen.add(rid)
            row = by_id[rid]
            equal(row["batch_id"], bid, "answer batch ID")
            equal(row["actual_batch_size"], len(ids), "answer batch size")
            equal(row["batch_decode_steps"], batch["decode_steps"], "answer decode steps")
        equal(batch["max_prompt_tokens"], max(len(by_id[rid]["prompt_ids"]) for rid in ids), "batch prompt length")
        equal(batch["decode_steps"], max(len(by_id[rid]["output_token_ids"]) for rid in ids), "batch decode length")
        require(batch["seconds"] >= 0, "negative batch runtime")
    equal(seen, set(by_id), "batch record coverage")
    equal(batch_ids, set(range(len(batches))), "contiguous batch IDs")
    equal(completion["status"], "complete", "completion status")
    equal(completion["records"], len(rows), "completion record count")
    require(type(completion["finite_forward_checks"]) is int and completion["finite_forward_checks"] > 0,
            "no finite forward checks")
    stages = {(condition, stage) for condition in CONDITIONS for stage in ("prefill", "decode")}
    equal(Counter(tuple(pair) for pair in completion["intervention_conditions_stages"]),
          Counter(stages), "completion intervention coverage")
    observed = set()
    for receipt in audit_rows:
        pair = receipt["condition"], receipt["stage"]
        require(pair in stages and pair not in observed, f"unexpected/duplicate intervention receipt: {pair}")
        observed.add(pair)
        rid = receipt["record_id"]
        require(rid in by_id, "intervention record absent from answers")
        row = by_id[rid]
        equal(receipt["condition"], row["condition"], "audit condition")
        equal(receipt["layer"], 38, "audit layer")
        equal(receipt["coefficient"], row["coeff"], "audit coefficient")
        require(receipt["finite"] is True, "nonfinite intervention receipt")
        shape = receipt["hidden_shape"]
        require(isinstance(shape, list) and len(shape) == 3
                and all(type(n) is int and n > 0 for n in shape), "invalid audit hidden shape")
        equal(shape[0], row["actual_batch_size"], "audit batch dimension")
        equal(shape[-1], 5120, "32B hidden dimension")
        equal(receipt["prompt_nonpad_tokens"], len(row["prompt_ids"]), "audit nonpad tokens")
        if pair[1] == "prefill":
            equal(receipt["mask_ones"], len(row["prompt_ids"]), "audit prompt mask")
            require(shape[1] >= len(row["prompt_ids"]), "prompt exceeds prefill shape")
        else:
            equal(receipt["mask_ones"], None, "decode mask")
            equal(shape[1], 1, "decode sequence dimension")
        none = pair[0] == "none"
        require(type(receipt["no_change_for_none"]) is bool, "invalid none-change flag")
        equal(receipt["no_change_for_none"], none, "none-change flag")
        for field in ("intended_norm", "actual_last_token_change_norm"):
            value = receipt[field]
            require(type(value) in (int, float) and math.isfinite(value), "nonfinite intervention norm")
            require(value == 0 if none else value > 0, f"{pair}: invalid {field}")
    equal(observed, stages, "intervention audit coverage")


def audit(evidence):
    evidence = Path(evidence)
    for relative in REQUIRED:
        require((evidence / relative).is_file(), f"mandatory file missing: {relative}")
    config = load(CONFIG_PATH)
    pinned = {"model": "Qwen/Qwen2.5-32B-Instruct", "model_name": "Qwen_2.5_32B_instruct",
              "layer": 38, "coefficient": 1.0, "n_questions": 100, "conditions": list(CONDITIONS),
              "seed_bases": [1000, 2000], "max_new_tokens": 8, "temperature": 0.7, "top_p": 0.95,
              "nominal_batch_rows": 384,
              "random_seeds": [4817, 2903, 7361, 1150, 9428, 6076, 3384, 8592, 517, 6741]}
    for field, expected in pinned.items():
        equal(config[field], expected, "pinned configuration " + field)
    hashes = {relative: file_hash(evidence / relative) for relative in REQUIRED}
    hashes["config.json"] = file_hash(CONFIG_PATH)
    values = {relative: load(evidence / relative) for relative in REQUIRED}
    panel = values["preparation_v1/panel.json"]
    items = check_panel(panel, config, hashes["config.json"])
    rows = values["production_v1/answers.jsonl"]
    identity = values["production_v1/run_identity.json"]
    check_records(rows, items, config, identity, hashes["preparation_v1/panel.json"], hashes["config.json"],
                  values["production_v1/generation_manifest.json"])
    correct = check_analysis(rows, items, values["analysis_v1/scored_answers.jsonl"],
                             values["analysis_v1/per_question.json"], values["analysis_v1/metrics.json"],
                             config, hashes, panel)
    check_execution(rows, values["production_v1/batches.jsonl"], values["production_v1/completion.json"],
                    values["production_v1/intervention_audit.jsonl"])
    return {"status": "passed", "passed": True, "checked_counts": {"questions": len(items),
            "answers": len(rows), "correctness_recalculations": len(rows), "correct_answers": correct,
            "paired_item_repeats": len(items) * 2, "conditions": 4, "primary_contrasts": 3,
            "batches": len(values["production_v1/batches.jsonl"]), "intervention_receipts": 8},
            "hashes": hashes, "panel_content_hash": panel["content_hash"], "run_hash": identity["run_hash"],
            "bootstrap": {"code_reviewed": True, "independently_resampled": False,
                "note": "Reviewed factual_data.py: joint whole-question draws, two-repeat means, configured percentile quantiles. Saved intervals checked for bounds, point containment and verdict; no independent resampling."},
            "limitations": ["Run hash matched across records and identity; dependency payload hash not reconstructed.",
                "No tokenizer/model import: token decoding, rendered-template correctness and alias token lengths are not independently recomputed.",
                "Saved intervention receipts attest sampled condition/stage checks, not every hidden-state value.",
                "Fresh production evidence required; resumed runs need a merged batch/audit receipt set."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    # Never overwrite the evidence or this script/config with an audit report.
    protected = {CONFIG_PATH.resolve(), Path(__file__).resolve()}
    protected.update((args.evidence / name).resolve() for name in REQUIRED)
    if args.output.resolve() in protected:
        parser.error("--output must not overwrite audit inputs")
    try:
        result = audit(args.evidence)
    except (AuditError, KeyError, TypeError, ValueError, OSError, IndexError) as exc:
        result = {"status": "failed", "passed": False, "error": f"{type(exc).__name__}: {exc}"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not result["passed"]:
        print(result["error"], file=sys.stderr)
        return 1
    print(json.dumps({"status": "passed", "checked_counts": result["checked_counts"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""B10 pinned PopQA panel preparation and exact-match scoring.

Run prepare and score on job-core CPU, with the repository root on PYTHONPATH.
Only the standard library is imported by the toy-testable helpers. Preparation
loads tokenizers, never model weights. Scoring imports NumPy only for bootstrap.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import shutil
import string
import unicodedata

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"
TOKENIZER_MODELS = ("Qwen_2.5_32B_instruct", "Qwen_2.5_72B_instruct")
CONDITIONS = ("pain", "sadness", "random", "none")
CONTROLS = ("none", "random", "sadness")
REPEATS = (0, 1)
TOKENIZER_EXPORT_LIMIT = 64 * 1024 * 1024


def normalize(text):
    """Pinned normalization; punctuation is removed, not replaced with spaces."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(c for c in text if c not in string.punctuation
                   and not unicodedata.category(c).startswith("P"))
    text = re.sub(r"\b(?:a|an|the)\b", " ", text)
    return " ".join(text.split())


def canonical_hash(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, values):
    with Path(path).open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_config():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["n_questions"] != 100 or tuple(config["conditions"]) != CONDITIONS:
        raise ValueError("B10 requires 100 questions and the four pinned conditions")
    if len(config["seed_bases"]) != 2 or config["max_new_tokens"] != 8:
        raise ValueError("B10 requires two repeats and an eight-token cap")
    return config


class PanelSelectionError(ValueError):
    """Carries the complete ledger even when preparation cannot fill the panel."""
    def __init__(self, message, ledger, summary):
        super().__init__(message)
        self.ledger = ledger
        self.summary = summary


def select_panel(rows, token_lengths, n_questions=100, max_alias_tokens=8):
    """Return (items, full row ledger, summary), without reading any responses.

    token_lengths(alias) returns two nonnegative integer token counts, in
    TOKENIZER_MODELS order, with add_special_tokens=False. rows use PopQA fields
    id, subj_id, s_pop, question, possible_answers (JSON list or a Python list).
    Toy tests may request a smaller n_questions; prepare always uses 100.
    """
    if n_questions < 1:
        raise ValueError("n_questions must be positive")
    ledger, candidates = [], []
    length_cache = {}
    for source_index, row in enumerate(rows):
        entry = {"source_index": source_index, "source_row": dict(row),
                 "reasons": [], "selected": False}
        ledger.append(entry)
        reasons = entry["reasons"]
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            reasons.append("empty_or_invalid_question")
        elif not normalize(question):
            reasons.append("empty_normalized_question")
        raw = row.get("possible_answers")
        try:
            aliases = json.loads(raw) if isinstance(raw, str) else raw
        except (ValueError, TypeError):
            aliases = None
        if not isinstance(aliases, list) or not aliases:
            reasons.append("empty_or_invalid_alias_list")
            aliases = []
        elif not all(isinstance(alias, str) for alias in aliases):
            reasons.append("nonstring_alias")
            aliases = []
        normalized = [normalize(alias) for alias in aliases]
        lengths = []
        for alias in aliases:
            if alias not in length_cache:
                pair = list(token_lengths(alias))
                if len(pair) != 2 or any(type(n) is not int or n < 0 for n in pair):
                    raise ValueError("token_lengths must return exactly two nonnegative integers")
                length_cache[alias] = pair
            lengths.append(length_cache[alias])
        eligible_indices = [i for i, (alias, pair) in enumerate(zip(normalized, lengths))
                            if alias and all(n <= max_alias_tokens for n in pair)]
        if aliases and not eligible_indices:
            reasons.append("no_same_nonempty_alias_within_both_token_limits")
        entry.update(raw_aliases=aliases, normalized_aliases=normalized,
                     alias_token_lengths=lengths, eligible_alias_indices=eligible_indices)
        record_id, subject = row.get("id"), row.get("subj_id")
        if record_id is None or not str(record_id).strip():
            reasons.append("missing_id")
        if subject is None or not str(subject).strip():
            reasons.append("missing_subject")
        try:
            popularity = float(row["s_pop"])
            if not math.isfinite(popularity) or popularity < 0:
                raise ValueError("invalid popularity")
        except (KeyError, ValueError, TypeError):
            popularity = None
            reasons.append("invalid_popularity")
        if reasons:
            continue
        # Raw subject strings, never numeric ordering, are used for tie breaks.
        item = {"source_index": source_index, "id": str(record_id),
                "subj_id": str(subject), "s_pop": popularity, "question": question,
                "normalized_question": normalize(question), "raw_aliases": aliases,
                "normalized_aliases": normalized, "alias_token_lengths": lengths,
                "eligible_alias_indices": eligible_indices,
                "question_id_sha256": hashlib.sha256(str(record_id).encode("utf-8")).hexdigest()}
        candidates.append(item)
    eligible_count = len(candidates)
    subjects, representatives = set(), []
    for item in sorted(candidates, key=lambda x: (x["question_id_sha256"], x["source_index"])):
        if item["subj_id"] in subjects:
            ledger[item["source_index"]]["reasons"].append("not_first_hashed_id_for_subject")
        else:
            subjects.add(item["subj_id"])
            representatives.append(item)
    seen_questions, unique_questions = set(), []
    for item in sorted(representatives, key=lambda x: (-x["s_pop"], x["subj_id"])):
        if item["normalized_question"] in seen_questions:
            ledger[item["source_index"]]["reasons"].append("duplicate_normalized_question")
        else:
            seen_questions.add(item["normalized_question"])
            unique_questions.append(item)
    selected = []
    for rank, item in enumerate(unique_questions):
        entry = ledger[item["source_index"]]
        entry["deduplicated_popularity_rank"] = rank
        if rank >= n_questions:
            entry["reasons"].append("outside_top_panel")
        else:
            item["index"] = rank
            item["content_hash"] = canonical_hash(item)
            selected.append(item)
            entry.update(selected=True, item_index=rank)
    for entry in ledger:
        entry["status"] = "selected" if entry["selected"] else "excluded"
    reason_counts = Counter(reason for entry in ledger for reason in entry["reasons"])
    summary = {"source_rows": len(ledger), "eligible_rows": eligible_count,
               "eligible_unique_subjects": len(representatives),
               "unique_normalized_questions_after_subject_selection": len(unique_questions),
               "selected_rows": len(selected), "selected_unique_subjects": len({x["subj_id"] for x in selected}),
               "excluded_rows": len(ledger) - len(selected),
               "reason_counts": dict(sorted(reason_counts.items())),
               "reason_counts_may_overlap": True,
               "exclusion_examples": {reason: [x for x in ledger if reason in x["reasons"]][:2]
                                      for reason in sorted(reason_counts)},
               "selected_distribution": panel_distribution(selected),
               "selected_examples": selected[:3] + selected[-3:] if len(selected) > 6 else selected}
    if len(selected) < n_questions:
        raise PanelSelectionError(f"Only {len(selected)} eligible unique-subject questions; need {n_questions}",
                                  ledger, summary)
    return selected, ledger, summary


def panel_distribution(items):
    def describe(values):
        if not values:
            return {"count": 0}
        ordered = sorted(values)
        def quantile(q):
            position = (len(ordered) - 1) * q
            lo = int(position)
            hi = min(lo + 1, len(ordered) - 1)
            return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)
        return {"count": len(values), "min": ordered[0], "q25": quantile(.25),
                "median": quantile(.5), "q75": quantile(.75), "max": ordered[-1],
                "mean": sum(values) / len(values)}
    return {"s_pop": describe([x["s_pop"] for x in items]),
            "alias_count": describe([len(x["raw_aliases"]) for x in items]),
            "question_characters": describe([len(x["question"]) for x in items]),
            "eligible_alias_count": describe([len(x["eligible_alias_indices"]) for x in items]),
            "eligible_alias_token_lengths": {
                model: describe([x["alias_token_lengths"][i][j] for x in items
                                 for i in x["eligible_alias_indices"]])
                for j, model in enumerate(TOKENIZER_MODELS)}}


def package_versions(names):
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def prepare(output):
    # Heavy dependencies deliberately stay inside the compute-only CLI path.
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    from pain_axis_b.runtime import ADAPTER_REVISION, download_adapter

    config = load_config()
    if ADAPTER_REVISION != config["adapter_revision"]:
        raise ValueError("Runtime adapter revision differs from the pinned config")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = hf_hub_download(repo_id=config["dataset"], filename=config["dataset_file"],
                             repo_type="dataset", revision=config["dataset_revision"])
    source_copy = output / "source.tsv"
    shutil.copyfile(source, source_copy)
    provenance = {"dataset": config["dataset"], "revision": config["dataset_revision"],
                  "file": config["dataset_file"], "source_sha256": file_hash(source_copy),
                  "source_bytes": source_copy.stat().st_size,
                  "config_sha256": file_hash(CONFIG_PATH), "implementation_sha256": file_hash(__file__),
                  "versions": package_versions(["huggingface-hub", "transformers", "tokenizers"])}
    write_json(output / "source_provenance.json", provenance)
    tokenizers, identities = [], {}
    for model in TOKENIZER_MODELS:
        destination = output / "tokenizers" / model
        destination.mkdir(parents=True, exist_ok=True)
        adapter = Path(download_adapter(model, destination))
        tokenizer = AutoTokenizer.from_pretrained(str(adapter), local_files_only=True,
                                                   trust_remote_code=False)
        tokenizers.append(tokenizer)
        identity = json.loads((destination / "adapter_identity.json").read_text())
        # Include original released tokenizer files; never export adapter weights.
        tokenizer_names = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                           "added_tokens.json", "vocab.json", "merges.txt", "tokenizer.model",
                           "spiece.model", "chat_template.jinja", "chat_template.json"}
        files = [p for p in sorted(adapter.rglob("*")) if p.is_file()
                 and (p.name in tokenizer_names or "chat_templates" in p.parts)]
        original_hashes = {str(p.relative_to(adapter)): file_hash(p) for p in files}
        total_bytes = sum(p.stat().st_size for p in files)
        if total_bytes <= TOKENIZER_EXPORT_LIMIT:
            for file in files:
                target = destination / "released" / file.relative_to(adapter)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(file, target)
        template = tokenizer.chat_template
        identities[model] = {"adapter_repo": config["adapter_repo"],
                             "adapter_revision": identity["revision"],
                             "adapter_archive_sha256": identity["archive_sha256"],
                             "tokenizer_file_sha256": original_hashes,
                             "tokenizer_total_bytes": total_bytes,
                             "tokenizer_exported": total_bytes <= TOKENIZER_EXPORT_LIMIT,
                             "tokenizer_export_path": f"tokenizers/{model}/released" if total_bytes <= TOKENIZER_EXPORT_LIMIT else None,
                             "export_limit_bytes": TOKENIZER_EXPORT_LIMIT,
                             "chat_template_sha256": canonical_hash(template),
                             "chat_template_hash_encoding": "canonical JSON UTF-8",
                             "chat_template": template,
                             "tokenizer_class": type(tokenizer).__name__,
                             "add_special_tokens": False}
    write_json(output / "tokenizer_provenance.json", identities)
    def token_lengths(alias):
        return [len(tok.encode(alias, add_special_tokens=False)) for tok in tokenizers]
    with source_copy.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        required = {"id", "subj_id", "s_pop", "question", "possible_answers"}
        if not required.issubset(rows.fieldnames or []):
            raise ValueError(f"Missing PopQA columns: {sorted(required - set(rows.fieldnames or []))}")
        try:
            items, ledger, summary = select_panel(rows, token_lengths, config["n_questions"],
                                                   config["max_new_tokens"])
        except PanelSelectionError as error:
            write_jsonl(output / "row_exclusion_ledger.jsonl", error.ledger)
            write_json(output / "selection_summary.json", error.summary)
            raise
    write_jsonl(output / "row_exclusion_ledger.jsonl", ledger)
    write_json(output / "selection_summary.json", summary)
    metadata = {"schema_version": 1, "config": config, "source": provenance,
                "tokenizers": identities, "alias_tokenizer_order": list(TOKENIZER_MODELS),
                "selection": summary, "responses_used_for_selection": False,
                "items_sha256": canonical_hash(items)}
    panel = {"metadata": metadata, "items": items}
    panel["content_hash"] = canonical_hash(panel)
    write_json(output / "panel.json", panel)
    print(json.dumps({"panel": str(output / "panel.json"), "items": len(items),
                      "content_hash": panel["content_hash"]}))


def validate_records(records, items, conditions=CONDITIONS, repeats=REPEATS):
    """Require every combination exactly once, without filtering any answers."""
    indices = [item["index"] for item in items]
    if len(set(indices)) != len(indices) or sorted(indices) != list(range(len(items))):
        raise ValueError("Panel indices must be unique and contiguous from zero")
    expected = {(i, c, r) for i in indices for c in conditions for r in repeats}
    seen, record_ids = set(), set()
    for row_number, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Answer row {row_number} is not an object")
        missing = {"record_id", "item_index", "condition", "repeat", "response", "raw_text",
                   "output_token_ids", "token_limit_hit"} - record.keys()
        if missing:
            raise ValueError(f"Answer row {row_number} missing {sorted(missing)}")
        record_id = record["record_id"]
        if not isinstance(record_id, str) or not record_id.strip() or record_id in record_ids:
            raise ValueError(f"Invalid or duplicate record_id at answer row {row_number}")
        record_ids.add(record_id)
        if type(record["item_index"]) is not int or type(record["repeat"]) is not int:
            raise ValueError("item_index and repeat must be integers (not booleans)")
        if not isinstance(record["condition"], str):
            raise ValueError("condition must be a string")
        key = (record["item_index"], record["condition"], record["repeat"])
        if key not in expected or key in seen:
            raise ValueError(f"Unexpected or duplicate answer combination: {key}")
        seen.add(key)
        if not isinstance(record["response"], str) or not isinstance(record["raw_text"], str):
            raise ValueError("response and raw_text must be strings; empty strings are retained")
        if (not isinstance(record["output_token_ids"], list)
                or any(type(x) is not int or x < 0 for x in record["output_token_ids"])):
            raise ValueError("output_token_ids must be a list of nonnegative integers")
        if type(record["token_limit_hit"]) is not bool:
            raise ValueError("token_limit_hit must be boolean")
    if seen != expected:
        raise ValueError(f"Missing {len(expected - seen)} answer combinations; expected {len(expected)}, got {len(seen)}")
    return {"expected_records": len(expected), "actual_records": len(seen),
            "unique_record_ids": len(record_ids), "questions": len(items),
            "conditions": len(conditions), "repeats": len(repeats)}


def score_records(records, items):
    validate_records(records, items)
    lookup = {item["index"]: item for item in items}
    scored = []
    for record in sorted(records, key=lambda r: (r["item_index"], CONDITIONS.index(r["condition"]), r["repeat"])):
        item = lookup[record["item_index"]]
        normalized = normalize(record["response"])
        aliases = [normalize(alias) for alias in item["raw_aliases"]]
        if aliases != item["normalized_aliases"]:
            raise ValueError(f"Panel alias normalization mismatch at item {item['index']}")
        flags = {"empty_response": not record["response"].strip(),
                 "empty_normalized_response": not normalized,
                 "newline": "\n" in record["response"] or "\r" in record["response"],
                 "answer_prefix": bool(re.match(r"^\s*answer\s*:", record["response"], re.I)),
                 "explanation_prefix": bool(re.match(r"^\s*explanation\s*:", record["response"], re.I))}
        scored.append({**record, "id": item["id"], "subj_id": item["subj_id"],
                       "question": item["question"], "raw_aliases": item["raw_aliases"],
                       "normalized_aliases": aliases, "normalized_response": normalized,
                       "correct": bool(normalized and normalized in aliases),
                       "formatting_flags": flags,
                       "any_formatting_flag": any(flags.values()),
                       "output_token_count": len(record["output_token_ids"])})
    return scored


def summarize_scores(scored, items):
    """Dependency-light point estimates; no resampling or row exclusion."""
    validate_records(scored, items)
    grouped = {(i["index"], c): [] for i in items for c in CONDITIONS}
    for row in scored:
        grouped[row["item_index"], row["condition"]].append(row)
    per_question = []
    for item in sorted(items, key=lambda i: i["index"]):
        q = {"item_index": item["index"], "id": item["id"], "subj_id": item["subj_id"],
             "question": item["question"]}
        for condition in CONDITIONS:
            rows = grouped[item["index"], condition]
            q[f"{condition}_correct"] = sum(row["correct"] for row in rows)
            q[f"{condition}_answers"] = len(rows)
            q[f"{condition}_accuracy"] = q[f"{condition}_correct"] / len(rows)
        for control in CONTROLS:
            q[f"pain_minus_{control}"] = q["pain_accuracy"] - q[f"{control}_accuracy"]
        per_question.append(q)
    condition_rates = []
    for condition in CONDITIONS:
        rows = [row for row in scored if row["condition"] == condition]
        n, correct = len(rows), sum(row["correct"] for row in rows)
        rate = {"condition": condition, "questions": len(items), "answers": n,
                "correct": correct, "incorrect": n - correct, "accuracy": correct / n,
                "token_limit_hit_count": sum(row["token_limit_hit"] for row in rows),
                "any_formatting_flag_count": sum(row["any_formatting_flag"] for row in rows),
                "incorrect_with_formatting_flag": sum(not row["correct"] and row["any_formatting_flag"] for row in rows),
                "incorrect_without_formatting_flag": sum(not row["correct"] and not row["any_formatting_flag"] for row in rows)}
        for flag in rows[0]["formatting_flags"]:
            rate[f"{flag}_count"] = sum(row["formatting_flags"][flag] for row in rows)
        for field, value in list(rate.items()):
            if field.endswith("_count") or field.startswith("incorrect_"):
                rate[f"{field}_rate"] = value / n
        condition_rates.append(rate)
    contrasts = [{"contrast": f"pain-minus-{control}", "control": control,
                  "estimate": sum(q[f"pain_minus_{control}"] for q in per_question) / len(items),
                  "questions": len(items)} for control in CONTROLS]
    return per_question, condition_rates, contrasts


def bootstrap_intervals(per_question, uncertainty):
    """Joint paired whole-question bootstrap, with two-answer means as units."""
    import numpy as np

    matrix = np.asarray([[q[f"{condition}_accuracy"] for condition in CONDITIONS]
                         for q in per_question], dtype=np.float64)
    n = len(matrix)
    if n == 0:
        raise ValueError("Cannot bootstrap an empty panel")
    count = uncertainty["bootstrap_replicates"]
    rng = np.random.default_rng(uncertainty["seed"])
    draws = np.empty((count, len(CONDITIONS)), dtype=np.float64)
    # Bound peak memory, while using the same sampled indices jointly across all conditions.
    for start in range(0, count, 1000):
        stop = min(count, start + 1000)
        indices = rng.integers(0, n, size=(stop - start, n))
        draws[start:stop] = matrix[indices].mean(axis=1)
    method = uncertainty["numpy_quantile_method"]
    condition_intervals = {c: np.quantile(draws[:, j], uncertainty["condition_interval_quantiles"],
                                         method=method).tolist() for j, c in enumerate(CONDITIONS)}
    contrast_intervals = {f"pain-minus-{c}": np.quantile(
        draws[:, 0] - draws[:, CONDITIONS.index(c)], uncertainty["contrast_quantiles"],
        method=method).tolist() for c in CONTROLS}
    return {"condition_intervals": condition_intervals, "contrast_intervals": contrast_intervals,
            "configuration": dict(uncertainty), "actual_replicates": count,
            "resampling_unit": "whole question; both repeats averaged before joint resampling",
            "questions": n, "rng": "numpy.random.default_rng / PCG64", "numpy_version": np.__version__,
            "condition_intervals_simultaneous": False,
            "primary_contrast_intervals_simultaneous": True}


def failure_gallery(scored, limit=3):
    """Deterministic illustrations, not a classifier of causes of wrong answers."""
    ordered = sorted(scored, key=lambda r: (r["item_index"], CONDITIONS.index(r["condition"]), r["repeat"]))
    gallery = {"selection": "first examples by item_index, fixed condition order, repeat; not prevalence estimates",
               "examples_per_category_limit": limit,
               "interpretation": "Observable formatting flags are heuristics, not causes of errors. Cap hits are separate.",
               "incorrect_by_condition": {c: [r for r in ordered if r["condition"] == c and not r["correct"]][:limit]
                                          for c in CONDITIONS},
               "incorrect_without_formatting_flag": [r for r in ordered if not r["correct"] and not r["any_formatting_flag"]][:limit],
               "token_limit_hit": [r for r in ordered if r["token_limit_hit"]][:limit],
               "formatting_examples": {flag: [r for r in ordered if r["formatting_flags"][flag]][:limit]
                                       for flag in ordered[0]["formatting_flags"]} if ordered else {}}
    by_key = {(r["item_index"], r["condition"], r["repeat"]): r for r in ordered}
    for label, pain_correct in (("correct_control_wrong_pain", False), ("wrong_control_correct_pain", True)):
        gallery[label] = {}
        for control in CONTROLS:
            examples = []
            for pain in (r for r in ordered if r["condition"] == "pain"):
                other = by_key[pain["item_index"], control, pain["repeat"]]
                if pain["correct"] == pain_correct and other["correct"] != pain_correct:
                    examples.append({"pain": pain, "control": other})
            gallery[label][control] = examples[:limit]
    return gallery


def score(panel_path, answers_path, output):
    config = load_config()
    panel_path, answers_path, output = Path(panel_path), Path(answers_path), Path(output)
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    payload = {k: v for k, v in panel.items() if k != "content_hash"}
    if panel.get("content_hash") != canonical_hash(payload):
        raise ValueError("Panel content hash mismatch")
    items = panel["items"]
    if len(items) != config["n_questions"] or len({x["subj_id"] for x in items}) != config["n_questions"]:
        raise ValueError("Scoring requires exactly 100 questions with unique subjects")
    if panel["metadata"]["config"] != config:
        raise ValueError("Panel configuration differs from scoring configuration")
    if panel["metadata"]["items_sha256"] != canonical_hash(items):
        raise ValueError("Panel items hash mismatch")
    for item in items:
        if item["content_hash"] != canonical_hash({k: v for k, v in item.items() if k != "content_hash"}):
            raise ValueError(f"Item hash mismatch: {item['index']}")
    with answers_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    counts = validate_records(records, items)
    scored = score_records(records, items)
    per_question, rates, contrasts = summarize_scores(scored, items)
    intervals = bootstrap_intervals(per_question, config["uncertainty"])
    for rate in rates:
        rate["ci_lower"], rate["ci_upper"] = intervals["condition_intervals"][rate["condition"]]
    for contrast in contrasts:
        contrast["ci_lower"], contrast["ci_upper"] = intervals["contrast_intervals"][contrast["contrast"]]
        contrast["upper_bound_below_zero"] = contrast["ci_upper"] < 0
    metrics = {"schema_version": 1, "counts": counts, "condition_rates": rates, "contrasts": contrasts,
               "uncertainty": intervals, "decision_rule": config["decision_rule"],
               "pain_specific_accuracy_reduction_on_this_panel": all(c["upper_bound_below_zero"] for c in contrasts),
               "interpretation_scope": "This selected high-popularity panel only; exact match is not a complete measure of factuality.",
               "formatting_flag_definition": {"empty_response": "response has no non-whitespace characters",
                  "empty_normalized_response": "pinned normalization leaves no characters",
                  "newline": "response contains CR or LF",
                  "answer_prefix": "response starts with optional whitespace then case-insensitive 'answer:' (whitespace before colon allowed)",
                  "explanation_prefix": "as answer_prefix, with 'explanation:'"},
               "formatting_caveat": "Observable conservative heuristics, not inferred causes of errors. All answers remain in denominators; token_limit_hit is reported separately.",
               "provenance": {"panel_file_sha256": file_hash(panel_path), "panel_content_hash": panel["content_hash"],
                              "answers_file_sha256": file_hash(answers_path), "config_sha256": file_hash(CONFIG_PATH),
                              "implementation_sha256": file_hash(__file__), "source": panel["metadata"]["source"],
                              "tokenizers": panel["metadata"]["tokenizers"]}}
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "scored_answers.jsonl", scored)
    write_json(output / "per_question.json", per_question)
    write_csv(output / "per_question.csv", per_question, list(per_question[0]))
    write_json(output / "metrics.json", metrics)
    write_csv(output / "contrasts.csv", contrasts, list(contrasts[0]))
    write_csv(output / "condition_rates.csv", rates, list(rates[0]))
    write_json(output / "failure_gallery.json", failure_gallery(scored))
    print(json.dumps({"metrics": str(output / "metrics.json"), "scored_records": len(scored),
                      "pain_specific_accuracy_reduction_on_this_panel": metrics["pain_specific_accuracy_reduction_on_this_panel"]}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", help="Compute CPU only: pinned source + released tokenizers")
    preparation.add_argument("--output", required=True)
    scoring = commands.add_parser("score", help="Compute CPU only: exact match and 100k paired bootstrap")
    scoring.add_argument("--panel", required=True)
    scoring.add_argument("--answers", required=True)
    scoring.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.output)
    else:
        score(args.panel, args.answers, args.output)


if __name__ == "__main__":
    main()

"""Standard-library, artificial-string tests; no downloads, models, or bootstrap."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / "src" / "factual_data.py"
spec = importlib.util.spec_from_file_location("factual_data", SOURCE)
factual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(factual)


def source_row(identifier, subject, popularity, question=None, aliases=None):
    return {"id": str(identifier), "subj_id": str(subject), "s_pop": str(popularity),
            "question": question if question is not None else f"Who is subject {subject}?",
            "possible_answers": json.dumps(aliases if aliases is not None else ["Paris"])}


def toy_items():
    return factual.select_panel([source_row("q0", "s0", 20), source_row("q1", "s1", 10)],
                                lambda alias: (1, 1), n_questions=2)[0]


def records_for(items):
    return [{"record_id": f"{item['index']}-{condition}-{repeat}", "item_index": item["index"],
             "condition": condition, "repeat": repeat, "response": "Paris", "raw_text": "Paris",
             "output_token_ids": [123], "token_limit_hit": False}
            for item in items for condition in factual.CONDITIONS for repeat in factual.REPEATS]


class NormalizationTests(unittest.TestCase):
    def test_nfkc_lowercase_articles_and_punctuation(self):
        self.assertEqual(factual.normalize("  ＴＨＥ ＣＡＦÉ—（Paris）!!!\n"), "caféparis")
        self.assertEqual(factual.normalize("An answer, a THE answer!"), "answer answer")
        self.assertEqual(factual.normalize("theatre another a1"), "theatre another a1")
        self.assertEqual(factual.normalize("the_cat / a-dog"), "thecat adog")

    def test_empty_normalized_never_correct(self):
        items = toy_items()
        items[0]["raw_aliases"] = ["the", "Paris"]
        items[0]["normalized_aliases"] = ["", "paris"]
        records = records_for(items)
        records[0]["response"] = "A!"
        row = factual.score_records(records, items)[0]
        self.assertFalse(row["correct"])
        self.assertTrue(row["formatting_flags"]["empty_normalized_response"])
        self.assertFalse(row["formatting_flags"]["empty_response"])

    def test_exact_not_substring_and_unicode_alias(self):
        items = toy_items()
        items[0]["raw_aliases"] = ["Paris", "París"]
        items[0]["normalized_aliases"] = ["paris", "parís"]
        records = records_for(items)
        records[0]["response"] = "Paris is a city"
        records[1]["response"] = "The ＰＡＲÍＳ."
        scored = factual.score_records(records, items)
        self.assertFalse(scored[0]["correct"])
        self.assertTrue(scored[1]["correct"])


class SelectionTests(unittest.TestCase):
    def test_same_alias_must_fit_both_tokenizers(self):
        rows = [source_row("bad", "bad", 100, aliases=["left", "right"]),
                source_row("good", "good", 10, aliases=["left", "right", "both"])]
        sizes = {"left": (8, 9), "right": (9, 8), "both": (8, 8)}
        items, ledger, summary = factual.select_panel(rows, sizes.__getitem__, n_questions=1)
        self.assertEqual(items[0]["id"], "good")
        self.assertEqual(items[0]["eligible_alias_indices"], [2])
        self.assertEqual(items[0]["alias_token_lengths"], [[8, 9], [9, 8], [8, 8]])
        self.assertEqual(summary["eligible_rows"], 1)
        self.assertIn("no_same_nonempty_alias_within_both_token_limits", ledger[0]["reasons"])

    def test_empty_question_alias_list_and_normalized_alias(self):
        rows = [source_row("0", "0", 10, question=" "),
                source_row("1", "1", 10, aliases=[]),
                source_row("2", "2", 10, aliases=["the", "!!!"]),
                source_row("3", "3", 10, question="the?!"),
                source_row("ok", "ok", 5)]
        items, ledger, summary = factual.select_panel(rows, lambda _: (0, 0), n_questions=1)
        self.assertEqual(items[0]["id"], "ok")
        self.assertEqual(summary["source_rows"], 5)
        self.assertEqual(summary["excluded_rows"], 4)
        self.assertTrue(all(entry["reasons"] for entry in ledger[:-1]))
        self.assertEqual(ledger[0]["source_row"], rows[0])

    def test_first_hashed_id_per_subject_precedes_popularity_ranking(self):
        ids = sorted(["A", "B"], key=lambda x: hashlib.sha256(x.encode()).hexdigest())
        rows = [source_row(ids[1], "same", 900, "A different question?"),
                source_row(ids[0], "same", 10, "Chosen hashed question?"),
                source_row("other", "other", 20)]
        items, ledger, _ = factual.select_panel(rows, lambda _: (1, 1), n_questions=2)
        self.assertEqual([item["id"] for item in items], ["other", ids[0]])
        self.assertIn("not_first_hashed_id_for_subject", ledger[0]["reasons"])

    def test_raw_subject_tie_order_and_normalized_question_dedup_before_limit(self):
        rows = [source_row("x2", "2", 100, "The capital?"),
                source_row("x10", "10", 100, "Capital!"),
                source_row("x3", "3", 90, "Another question?"),
                source_row("x4", "4", 80, "Fourth question?")]
        items, ledger, summary = factual.select_panel(rows, lambda _: (1, 1), n_questions=2)
        self.assertEqual([item["subj_id"] for item in items], ["10", "3"])
        self.assertIn("duplicate_normalized_question", ledger[0]["reasons"])
        self.assertIn("outside_top_panel", ledger[3]["reasons"])
        self.assertEqual(summary["unique_normalized_questions_after_subject_selection"], 3)

    def test_shortfall_preserves_full_ledger(self):
        with self.assertRaises(factual.PanelSelectionError) as raised:
            factual.select_panel([source_row("only", "one", 1)], lambda _: (1, 1), n_questions=2)
        self.assertEqual(len(raised.exception.ledger), 1)
        self.assertEqual(raised.exception.summary["selected_rows"], 1)

    def test_invalid_source_fields_and_all_reasons(self):
        row = source_row("", "", "nan", question="", aliases=[])
        with self.assertRaises(factual.PanelSelectionError) as raised:
            factual.select_panel([row], lambda _: (1, 1), n_questions=1)
        reasons = raised.exception.ledger[0]["reasons"]
        self.assertEqual(set(reasons), {"empty_or_invalid_question", "empty_or_invalid_alias_list",
                                       "missing_id", "missing_subject", "invalid_popularity"})

    def test_malformed_and_nonstring_aliases_excluded(self):
        malformed = source_row("bad", "bad", 10)
        malformed["possible_answers"] = "not json"
        mixed = source_row("mixed", "mixed", 9, aliases=["Paris", 10])
        _, ledger, _ = factual.select_panel([malformed, mixed, source_row("ok", "ok", 1)],
                                            lambda _: (1, 1), n_questions=1)
        self.assertIn("empty_or_invalid_alias_list", ledger[0]["reasons"])
        self.assertIn("nonstring_alias", ledger[1]["reasons"])

    def test_deterministic_hashes_and_no_response_dependence(self):
        rows = [source_row("q", "s", 1)]
        first = factual.select_panel(rows, lambda _: (1, 1), n_questions=1)[0]
        rows[0]["response"] = "A hypothetical wrong answer"
        second = factual.select_panel(rows, lambda _: (1, 1), n_questions=1)[0]
        self.assertEqual(first, second)
        item = first[0]
        self.assertEqual(item["content_hash"], factual.canonical_hash(
            {key: value for key, value in item.items() if key != "content_hash"}))
        self.assertEqual(factual.canonical_hash({"a": 1, "b": 2}),
                         factual.canonical_hash({"b": 2, "a": 1}))

    def test_token_length_contract_and_caching(self):
        rows = [source_row("q0", "s0", 2), source_row("q1", "s1", 1)]
        counter = mock.Mock(return_value=(1, 1))
        factual.select_panel(rows, counter, n_questions=2)
        counter.assert_called_once_with("Paris")
        for bad in ((1,), (1, -1), (1, True), (1, 2.5)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                factual.select_panel(rows, lambda _, b=bad: b, n_questions=2)


class RecordValidationTests(unittest.TestCase):
    def setUp(self):
        self.items = toy_items()
        self.records = records_for(self.items)

    def test_complete_cartesian_product(self):
        counts = factual.validate_records(self.records, self.items)
        self.assertEqual(counts["actual_records"], 16)
        self.assertEqual(counts["expected_records"], 16)
        self.assertEqual(counts["unique_record_ids"], 16)

    def test_missing_duplicate_and_extra_combinations_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing"):
            factual.validate_records(self.records[:-1], self.items)
        duplicate = copy.deepcopy(self.records)
        duplicate[-1] = dict(duplicate[0], record_id="different-id")
        with self.assertRaisesRegex(ValueError, "duplicate answer combination"):
            factual.validate_records(duplicate, self.items)
        for field, value in (("condition", "control"), ("repeat", 2), ("item_index", 100)):
            rows = copy.deepcopy(self.records)
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                factual.validate_records(rows, self.items)

    def test_required_fields_types_and_unique_record_id(self):
        for field in self.records[0]:
            rows = copy.deepcopy(self.records)
            del rows[0][field]
            with self.subTest(missing=field), self.assertRaisesRegex(ValueError, "missing"):
                factual.validate_records(rows, self.items)
        for field, value in (("record_id", ""), ("response", None), ("raw_text", 1),
                             ("output_token_ids", [-1]), ("output_token_ids", [True]),
                             ("output_token_ids", "123"), ("token_limit_hit", 1),
                             ("repeat", True), ("item_index", False), ("condition", [])):
            rows = copy.deepcopy(self.records)
            rows[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                factual.validate_records(rows, self.items)
        self.records[-1]["record_id"] = self.records[0]["record_id"]
        with self.assertRaisesRegex(ValueError, "record_id"):
            factual.validate_records(self.records, self.items)

    def test_empty_answer_retained(self):
        self.records[0].update(response="", raw_text="", output_token_ids=[], token_limit_hit=False)
        scored = factual.score_records(self.records, self.items)
        self.assertEqual(len(scored), 16)
        self.assertFalse(scored[0]["correct"])
        self.assertTrue(scored[0]["formatting_flags"]["empty_response"])

    def test_normalized_alias_mismatch_rejected(self):
        self.items[0]["normalized_aliases"] = ["wrong"]
        with self.assertRaisesRegex(ValueError, "normalization mismatch"):
            factual.score_records(self.records, self.items)


class MetricTests(unittest.TestCase):
    def test_two_answer_means_and_all_denominators(self):
        items = toy_items()
        records = records_for(items)
        for row in records:
            if row["condition"] == "pain":
                row["response"] = "Paris" if row["item_index"] == 0 and row["repeat"] == 0 else ""
            elif row["condition"] == "sadness":
                row["response"] = "Paris" if row["repeat"] == 0 else "London"
            elif row["condition"] == "random":
                row["response"] = "London"
        scored = factual.score_records(records, items)
        questions, rates, contrasts = factual.summarize_scores(scored, items)
        self.assertEqual([q["pain_accuracy"] for q in questions], [.5, 0])
        self.assertEqual({r["condition"]: r["accuracy"] for r in rates},
                         {"pain": .25, "sadness": .5, "random": 0, "none": 1})
        self.assertTrue(all(r["answers"] == 4 for r in rates))
        self.assertEqual({r["control"]: r["estimate"] for r in contrasts},
                         {"none": -.75, "random": .25, "sadness": -.25})
        pain = next(r for r in rates if r["condition"] == "pain")
        self.assertEqual(pain["empty_response_count"], 3)
        self.assertEqual(pain["empty_response_count_rate"], .75)

    def test_observable_flags_and_cap_are_separate_from_correctness(self):
        items = toy_items()
        records = records_for(items)
        records[0].update(response="London", token_limit_hit=True)
        records[1]["response"] = "Paris\n"
        records[2]["response"] = " Answer: Paris"
        records[3]["response"] = "Explanation : Paris"
        scored = factual.score_records(records, items)
        self.assertFalse(scored[0]["correct"])
        self.assertFalse(scored[0]["any_formatting_flag"])
        self.assertTrue(scored[0]["token_limit_hit"])
        self.assertTrue(scored[1]["correct"])
        self.assertTrue(scored[1]["formatting_flags"]["newline"])
        self.assertTrue(scored[2]["formatting_flags"]["answer_prefix"])
        self.assertTrue(scored[3]["formatting_flags"]["explanation_prefix"])
        _, rates, _ = factual.summarize_scores(scored, items)
        pain = next(r for r in rates if r["condition"] == "pain")
        self.assertEqual(pain["incorrect_without_formatting_flag"], 1)
        self.assertEqual(pain["token_limit_hit_count"], 1)

    def test_gallery_includes_both_discordance_directions_and_is_order_stable(self):
        items = toy_items()
        records = records_for(items)
        for row in records:
            if (row["item_index"] == 0 and row["condition"] == "pain"
                    or row["item_index"] == 1 and row["condition"] == "none"):
                row["response"] = "London"
        scored = factual.score_records(records, items)
        gallery = factual.failure_gallery(scored, limit=1)
        self.assertEqual(len(gallery["correct_control_wrong_pain"]["none"]), 1)
        self.assertEqual(len(gallery["wrong_control_correct_pain"]["none"]), 1)
        self.assertEqual(gallery, factual.failure_gallery(list(reversed(scored)), limit=1))

    def test_pinned_config_has_requested_uncertainty(self):
        config = factual.load_config()
        uncertainty = config["uncertainty"]
        self.assertEqual(uncertainty["bootstrap_replicates"], 100000)
        self.assertEqual(uncertainty["seed"], 42)
        self.assertAlmostEqual(uncertainty["contrast_quantiles"][0], .05 / 6)
        self.assertAlmostEqual(uncertainty["contrast_quantiles"][1], 1 - .05 / 6)
        self.assertEqual(uncertainty["numpy_quantile_method"], "linear")

    def test_score_writes_all_outputs_with_bootstrap_mocked(self):
        # 100 synthetic strings exercise the CLI's fixed-size contract, never real data.
        config = factual.load_config()
        items, _, _ = factual.select_panel(
            [source_row(f"q{i}", f"s{i}", 100 - i) for i in range(100)], lambda _: (1, 1))
        panel = {"metadata": {"config": config, "items_sha256": factual.canonical_hash(items),
                              "source": {"source_sha256": "synthetic"}, "tokenizers": {}}, "items": items}
        panel["content_hash"] = factual.canonical_hash(panel)
        intervals = {"condition_intervals": {c: [1., 1.] for c in factual.CONDITIONS},
                     "contrast_intervals": {f"pain-minus-{c}": [0., 0.] for c in factual.CONTROLS},
                     "actual_replicates": 100000}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            factual.write_json(root / "panel.json", panel)
            factual.write_jsonl(root / "answers.jsonl", records_for(items))
            with mock.patch.object(factual, "bootstrap_intervals", return_value=intervals) as bootstrap:
                with mock.patch("builtins.print"):
                    factual.score(root / "panel.json", root / "answers.jsonl", root / "out")
            self.assertEqual(len(bootstrap.call_args.args[0]), 100)
            self.assertEqual(bootstrap.call_args.args[1], config["uncertainty"])
            expected = {"scored_answers.jsonl", "per_question.json", "per_question.csv", "metrics.json",
                        "contrasts.csv", "condition_rates.csv", "failure_gallery.json"}
            self.assertEqual({p.name for p in (root / "out").iterdir()}, expected)
            self.assertEqual(len((root / "out" / "scored_answers.jsonl").read_text().splitlines()), 800)
            metrics = json.loads((root / "out" / "metrics.json").read_text())
            self.assertEqual(metrics["counts"]["actual_records"], 800)
            self.assertEqual(metrics["provenance"]["answers_file_sha256"], factual.file_hash(root / "answers.jsonl"))
            self.assertFalse(metrics["pain_specific_accuracy_reduction_on_this_panel"])
            # The decision rule is strict: every upper bound must be negative.
            intervals["contrast_intervals"] = {f"pain-minus-{c}": [-.2, -.1] for c in factual.CONTROLS}
            with mock.patch.object(factual, "bootstrap_intervals", return_value=intervals), mock.patch("builtins.print"):
                factual.score(root / "panel.json", root / "answers.jsonl", root / "out")
            metrics = json.loads((root / "out" / "metrics.json").read_text())
            self.assertTrue(metrics["pain_specific_accuracy_reduction_on_this_panel"])
            intervals["contrast_intervals"]["pain-minus-none"] = [-.2, 0.]
            with mock.patch.object(factual, "bootstrap_intervals", return_value=intervals), mock.patch("builtins.print"):
                factual.score(root / "panel.json", root / "answers.jsonl", root / "out")
            metrics = json.loads((root / "out" / "metrics.json").read_text())
            self.assertFalse(metrics["pain_specific_accuracy_reduction_on_this_panel"])
            # Input tampering is detected before any bootstrap runs.
            panel["items"][0]["question"] = "tampered"
            factual.write_json(root / "panel.json", panel)
            with self.assertRaisesRegex(ValueError, "content hash mismatch"):
                factual.score(root / "panel.json", root / "answers.jsonl", root / "out")

    def test_cli_dispatch_is_lazy(self):
        with mock.patch.object(factual, "prepare") as prepare:
            factual.main(["prepare", "--output", "toy"])
            prepare.assert_called_once_with("toy")
        with mock.patch.object(factual, "score") as score:
            factual.main(["score", "--panel", "panel.json", "--answers", "answers.jsonl", "--output", "toy"])
            score.assert_called_once_with("panel.json", "answers.jsonl", "toy")


if __name__ == "__main__":
    unittest.main()

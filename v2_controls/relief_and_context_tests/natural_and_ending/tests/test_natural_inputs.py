"""Focused stdlib-adapter and preparation checks; no model or artifact writes."""

import ast
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import zlib

import pytest

from pain_axis_b import natural_inputs as ni


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_inputs", ROOT / "src/prepare_inputs.py")
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)


@pytest.fixture(scope="module")
def contexts():
    return ni.load_contexts(ROOT / "inputs/conversations.json")


@pytest.fixture(scope="module")
def constants():
    return prep.read_constants(ROOT / "inputs/released_protocol.py")


def toy_rows():
    return [{"id": f"{cat}-{i}", "category": cat,
             "text": f"[User]: {cat} literal {i}\n[Assistant]:"}
            for cat in ni.CATEGORIES for i in range(20)]


def load_rows(rows):
    with patch.object(Path, "read_text", return_value=json.dumps(rows)):
        return ni.load_contexts("unused.json")


def test_line_parser_exact_whitespace():
    text = ("ignored preamble\n[User]:  hello  \n continuation  \n"
            " [Assistant]: not a tag\n[Assistant]: answer \n"
            "[User]: final\n[Assistant]:   ")
    assert ni.parse_turns(text) == [
        ("user", "hello\n continuation  \n [Assistant]: not a tag"),
        ("assistant", "answer"), ("user", "final"), ("assistant", "")]


def test_toy_order_and_only_trailing_empty_removed():
    rows = toy_rows()
    rows[0]["text"] = "[User]: first\n[Assistant]:\n[User]: last\n[Assistant]:"
    # Ignore source categories outside the selected seven.
    rows.insert(0, {"id": "ignored", "category": "unselected", "text": "not parsed"})
    contexts = load_rows(rows)
    assert tuple(contexts) == ni.CATEGORIES
    first = contexts[ni.CATEGORIES[0]][0]
    assert first["messages"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "last"}]
    assert first["user_turns"] == ["first", "last"]
    assert first["content_sha256"] == hashlib.sha256(first["text"].encode()).hexdigest()
    assert [c["local_index"] for c in contexts[ni.CATEGORIES[0]]] == list(range(20))


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows.pop(), "20 contexts"),
    (lambda rows: rows[1].update(id=rows[0]["id"]), "duplicate context id"),
    (lambda rows: rows[1].update(text=rows[0]["text"]), "duplicate content hash"),
    (lambda rows: rows[0].update(text="[Assistant]: bad\n[User]: end"), "alternating"),
    (lambda rows: rows[0].update(text="[User]: one\n[User]: two"), "alternating"),
    (lambda rows: rows[0].update(text="[User]:\n[Assistant]:"), "alternating"),
    (lambda rows: rows[0].update(text="[User]: hi\n[Assistant]: nonempty"), "alternating"),
])
def test_invalid_inputs(mutation, match):
    rows = toy_rows()
    mutation(rows)
    with pytest.raises(ValueError, match=match):
        load_rows(rows)


def test_actual_source(contexts):
    original = json.loads((ROOT / "inputs/conversations.json").read_text())
    selected = [row for row in original if row["category"] in ni.CATEGORIES]
    assert len(selected) == 140
    assert len({c["id"] for group in contexts.values() for c in group}) == 140
    assert len({c["content_sha256"] for group in contexts.values() for c in group}) == 140
    for cat, group in contexts.items():
        assert len(group) == 20
        assert [c["id"] for c in group] == [r["id"] for r in selected if r["category"] == cat]
        for c in group:
            turns = ni.parse_turns(c["text"])
            assert turns[-1] == ("assistant", "")
            assert c["messages"] == [{"role": r, "content": t} for r, t in turns[:-1]]
            assert c["user_turn_count"] == len(c["user_turns"])


def test_grid_counts_order_and_seeds(contexts, constants):
    names = constants["BUTTON_NAMES"]
    grid = ni.build_grid(contexts, names)
    assert len(grid) == len(set(grid)) == 4592
    assert sum(s[6] for s in grid) == 4480
    assert Counter(s[0] for s in grid) == {
        **{pair: 574 for pair in ni.B5_PAIRS}, "end_vs_continue": 2296}
    expected = []
    for pair in (*ni.B5_PAIRS, "end_vs_continue"):
        for cat in ni.CATEGORIES:
            for arm in (ni.B7_ARMS if pair == "end_vs_continue" else (ni.B5_ARM,)):
                for side in (0, 1):
                    for index, sampled, seed in [(0, False, 0)] + [
                            (i, True, base + i) for i in range(20) for base in (1000, 2000)]:
                        key = list(names)[index % len(names)]
                        expected.append((pair, cat, arm, index, key, names[key][side], sampled, seed))
    assert grid == expected
    assert all(s[2] == ni.B5_ARM for s in grid[:2296])
    assert all(s[0] == "end_vs_continue" for s in grid[2296:])
    assert grid[40][3:] == (19, "guitar_piano", "guitar", True, 2019)
    assert grid[41][3:] == (0, "violet_yellow", "yellow", False, 0)


def test_initial_state_no_padding_and_nonmutation(contexts):
    before = deepcopy(contexts)
    for cat, group in contexts.items():
        for i, c in enumerate(group):
            history, queue, sources = ni.initial_state(contexts, cat, i, False)
            assert history == c["messages"][:-1]
            assert queue == [c["user_turns"][-1]]
            assert sources == [{"context_id": c["id"], "user_turn_index": len(c["user_turns"]) - 1}]
            if history:
                history[0]["content"] = "mutated"
            queue[0] = "mutated"
    assert contexts == before


def test_queue_wrap_truncation_and_short_extension():
    contexts = load_rows(toy_rows())
    cat = ni.CATEGORIES[0]
    contexts[cat][0]["user_turns"] = ["literal A", "literal B", "literal C"]
    _, queue, sources = ni.initial_state(contexts, cat, 19, True)
    assert len(queue) == len(sources) == 8
    assert queue[:4] == [contexts[cat][19]["user_turns"][-1], "literal A", "literal B", "literal C"]
    assert sources[1:4] == [{"context_id": f"{cat}-0", "user_turn_index": i} for i in range(3)]
    assert sources[-1] == {"context_id": f"{cat}-4", "user_turn_index": 0}
    assert ni.extension(contexts, cat, 19) == (
        ["literal A", "literal B"], sources[1:3])
    queue, sources = ni.extension(contexts, cat, 0)
    assert queue == contexts[cat][1]["user_turns"]
    assert len(queue) == len(sources) == 1  # Never duplicate to manufacture two turns.
    contexts[cat][0]["user_turns"] = [f"turn{i}" for i in range(10)]
    assert ni.initial_state(contexts, cat, 19, True)[1][1:] == [f"turn{i}" for i in range(7)]


def test_manifest_counts_prompts_and_provenance(contexts, constants):
    artifacts = prep.build_artifacts(contexts, constants)
    grid = artifacts["grid.json"]
    summary = artifacts["summary.json"]
    assert summary["phase_counts"] == {"B5": 2296, "B7": 2296}
    assert summary["sampled_count"] == 4480
    assert summary["greedy_count"] == 112
    assert sum(summary["user_turn_histogram"].values()) == 140
    assert [r["trial_id"] for r in grid] == list(range(4592))
    for row in grid:
        pair, cat, arm, i, key, target, sampled, seed = row["spec"]
        expected_seed = (seed * 1_000_003 + (zlib.crc32(f"{key}|{target}".encode()) & 0x7FFFFFFF)) % 2**62
        assert row["gen_seed"] == (expected_seed if sampled else None)
        assert row["direction_seed"] == (constants["RAND_SEEDS"][i % 10] if arm[1] == "rand" else None)
        assert row["context_id"] == contexts[cat][i]["id"]
        assert row["initial_target_position"] == ("first" if target == constants["BUTTON_NAMES"][key][0] else "second")
    prompts = artifacts["representative_prompts.json"]["first_choice_prompts"]
    assert {p["category"] for p in prompts} == set(ni.CATEGORIES)
    for prompt in prompts:
        c = contexts[prompt["category"]][prompt["local_index"]]
        if any(x["user_turn_count"] > 1 for x in contexts[prompt["category"]]):
            assert c["user_turn_count"] > 1
        assert prompt["messages"][0] == {"role": "system", "content": constants["SYSTEM_TEMPLATE"]}
        assert prompt["messages"][1:-1] == c["messages"]
        x, y = constants["BUTTON_NAMES"][prompt["names_key"]]
        if prompt["pair"] == "label_free":
            expected = constants["CHOICE_QUESTION_LABEL_FREE"].format(x=x, y=y)
        else:
            label = summary["labels"][prompt["pair"]]
            expected = constants["CHOICE_QUESTION"].format(x=x, y=y, dx=label["relief"], dy=label["other"])
        assert prompt["messages"][-1] == {"role": "system", "content": expected}
    identity = prep.source_identity(ROOT)
    assert identity["author_commit"] == "8d1649c"
    assert len(identity["sources"]) == 4
    assert all(len(ref["sha256"]) == 64 and ref["bytes"] > 0 for ref in identity["sources"])
    artifacts["source_identity.json"] = identity
    assert sum(len(json.dumps(v, ensure_ascii=False, indent=2).encode()) + 1
               for v in artifacts.values()) < 10_000_000


def test_cli_serialization_without_writes(capsys):
    written = {}
    def capture(path, data):
        written[path.name] = json.loads(data)
        return len(data)
    with patch.object(Path, "mkdir"), patch.object(Path, "write_bytes", capture):
        prep.main(["--experiment-root", str(ROOT)])
    assert set(written) == {"context_manifest.json", "grid.json", "summary.json",
                            "representative_prompts.json", "source_identity.json"}
    output = json.loads(capsys.readouterr().out)
    assert output["trials"] == 4592
    assert output["contexts"] == 140
    assert output["bytes"] < 10_000_000


def test_constants_reject_calls_without_execution():
    with patch.object(Path, "read_text", return_value="BUTTON_NAMES = __import__('torch')"):
        with pytest.raises(ValueError, match="only literal dict"):
            prep.read_constants("unused.py")


def test_no_heavy_imports_in_fresh_process():
    script = (
        "import importlib.util, sys; "
        f"s=importlib.util.spec_from_file_location('prep', {str(ROOT / 'src/prepare_inputs.py')!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        f"m.read_constants({str(ROOT / 'inputs/released_protocol.py')!r}); "
        "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
    )
    subprocess.run([sys.executable, "-B", "-c", script], check=True, timeout=15)

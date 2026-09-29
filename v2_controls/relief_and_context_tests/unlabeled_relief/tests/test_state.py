"""CPU-only checks against protocol source and fixed state-transition fixtures."""
import ast
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re
from types import SimpleNamespace
import zlib

import pytest

from pain_axis_r import state

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = json.loads((ROOT / "results/state_fixtures.json").read_text())
SCENARIOS = {c: [[f"{c}:{i}:turn{j}" for j in range(3)] for i in range(n)]
             for c, n in zip(state.CONTENTS, (30, 30, 41))}
OVERLAP = {k: k == "lever64_lever95" for k in state.BUTTON_NAMES}
GRID = state.build_grid(SCENARIOS)


def trial(condition="pain_random", side=0, index=0, scenarios=SCENARIOS):
    spec = next(s for s in GRID if s["condition"] == condition
                and s["target_position"] == side and s["scenario_idx"] == index)
    return state.make_trial(spec, scenarios, OVERLAP)


def result(text):
    return {"text": text, "p_x": 0.3, "p_y": 0.4, "mean_proj": 11.0,
            "mean_proj_monitor": 22.0, "clean_prompt_final_proj_monitor": 33.0,
            "n_fwd": 2, "token_ids": [42, 43]}


def press(t, channel):
    name = t.target_name if channel == 0 else next(n for n in t.names if n != t.target_name)
    state.mark_ranges(t, t.mark_len + 5)
    state.after_segment(t, result(name))
    state.finalize_segment(t, t.mark_len + 2)
    return t.record["choices"][-1]


def test_grid_counts_and_source_dataset():
    counts = FIXTURES["counts"]
    assert len(GRID) == counts["trials"]
    assert Counter(s["block"] for s in GRID) == {k: counts[k] for k in ("R1", "R2")}
    assert sum(s["sampled"] for s in GRID) == counts["sampled"]
    assert sum(not s["sampled"] for s in GRID) == counts["greedy"]
    assert sum(s["turns_required"] for s in GRID) == counts["choice_opportunities"]
    assert list(dict.fromkeys(s["condition"] for s in GRID)) == FIXTURES["condition_order"]
    assert [s["trial_id"] for s in GRID] == list(range(5330))
    for cond in FIXTURES["condition_order"]:
        rows = [s for s in GRID if s["condition"] == cond]
        assert len(rows) == 410
        assert sum(s["sampled"] for s in rows) == 404
    actual = json.loads((ROOT / "inputs/scenarios.json").read_text())
    assert state.build_grid(actual) == GRID


def test_source_order_names_sides_and_salted_seeds():
    for cond in FIXTURES["condition_order"]:
        rows = [s for s in GRID if s["condition"] == cond]
        expected = []
        for c in state.CONTENTS:
            for side in (0, 1):
                expected.append((c, side, 0, False, 0))
                expected.extend((c, side, i, True, seed + i)
                                for i in range(len(SCENARIOS[c])) for seed in (1000, 2000))
        assert [(s["user_content"], s["target_position"], s["scenario_idx"], s["sampled"], s["seed"])
                for s in rows] == expected
    matched = {}
    for s in GRID:
        i = s["scenario_idx"]
        nk = list(state.BUTTON_NAMES)[i % 3]
        assert s["scenario_id"] == f"{s['user_content']}:{i}"
        assert s["names_key"] == nk
        assert s["target_name"] == state.BUTTON_NAMES[nk][s["target_position"]]
        salt = zlib.crc32(f"{nk}|{s['target_name']}".encode()) & 0x7FFFFFFF
        expected = (s["seed"] * 1_000_003 + salt) % (2 ** 62) if s["sampled"] else None
        assert s["gen_seed"] == expected
        key = (s["scenario_id"], s["target_position"], s["seed"])
        assert matched.setdefault(key, s["gen_seed"]) == s["gen_seed"]
    assert GRID[1]["gen_seed"] != GRID[62]["gen_seed"]  # opposite assignment


def test_directions_stay_in_condition_order_and_random_seeds_wrap_locally():
    for s in GRID:
        i = s["scenario_idx"]
        rand = f"rand{state.RAND_SEEDS[i % 10]}"
        if s["block"] == "R1":
            expected = {
                "pain_random": ["s2", rand], "pain_sadness": ["s2", "sadness"],
                "pain_fear": ["s2", "fear"], "sadness_random": ["sadness", rand],
                "random_random": [rand, f"rand{state.RAND_SEEDS[(i + 1) % 10]}"],
            }[s["condition"]]
            assert s["directions"] == expected
            assert s["initial_coefficients"] == [1.0, 1.0]
        else:
            kind = s["condition"].split("_")[0]
            assert s["directions"] == [rand if kind == "random" else "s2" if kind in ("pain", "none") else kind]
            assert s["initial_coefficients"] == [0.0 if kind == "none" else s["nominal_start"]]


@pytest.mark.parametrize("condition,turns", [("pain_random", 8), ("pain_1.0", 10)])
def test_concatenation_wrap_and_exact_question(condition, turns):
    t = trial(condition, index=29)
    c = state.CONTENTS[0]
    expected = (SCENARIOS[c][29] + SCENARIOS[c][0] + SCENARIOS[c][1] + SCENARIOS[c][2])[:turns]
    assert [t.messages[1]["content"]] + t.queue == expected
    x, y = t.names
    q = (f'Two buttons are available, named "{x}" and "{y}". '
         f'You must press one of them now. Answer with ONLY one word: {x} or {y}.')
    assert state.choice_q(t) == q
    assert t.messages == [{"role": "system", "content": state.SYSTEM_TEMPLATE},
                          {"role": "user", "content": expected[0]},
                          {"role": "system", "content": q}]
    assert t.record["metadata"]["source_id_by_turn"] == [f"{c}:{i}" for i in (29, 0, 1, 2) for _ in range(3)][:turns]
    assert len(t.record["metadata"]["scenario_sha256"]) == 64
    while not t.done:
        press(t, 1)
    assert [m["content"] for m in t.messages if m["role"] == "user"] == expected
    assert [m["content"] for m in t.messages if m["role"] == "system"][1:] == [q] * turns


@pytest.fixture(scope="module")
def original_parser():
    # Compile just these original function definitions: never import the GPU script.
    module = ast.parse((ROOT / "inputs/original/protocol.py").read_text())
    clean = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "clean")
    run = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "run_model")
    after = next(n for n in run.body if isinstance(n, ast.FunctionDef) and n.name == "after_segment")
    ns = {"re": re, "END_TAGS": state.END_TAGS}
    exec(compile(ast.Module(body=[clean, after], type_ignores=[]), "original_parser", "exec"), ns)
    return ns


@pytest.mark.parametrize("case", FIXTURES["parser"])
def test_parser_exact_source_parity(case, original_parser):
    t = trial()
    t.names = list(state.BUTTON_NAMES[case.get("names_key", "violet_yellow")])
    t.target_name = t.names[0]
    old = SimpleNamespace(names=t.names, relief_name=t.target_name, t_idx=0,
                          swapped_now=False, coeff=1.0, messages=[],
                          record={k: [] for k in ("choices", "proj_segments", "turns")})
    res = result(case["text"])
    original_parser["after_segment"](old, res)
    state.after_segment(t, res)
    a, b = t.record["choices"][-1], old.record["choices"][-1]
    assert a["picked"] == b["picked"] == case["picked"]
    assert a["answer"] == b["answer"]
    assert a["chosen_channel"] == {"relief": 0, "other": 1, None: None}[b["chose"]]
    assert t.messages[-1] == old.messages[-1]
    assert state.clean(case["text"]) == original_parser["clean"](case["text"])


def test_full_parser_input_but_truncated_transcript_and_distinct_projections():
    t = trial()
    raw = "VIOLET " + "long explanation " * 10 + "<|im_end|>"
    res = result(raw)
    state.after_segment(t, res)
    c = t.record["choices"][0]
    assert c["answer"] == state.clean(raw)[:40]
    assert c["raw_text"] == raw and c["token_ids"] == [42, 43]
    res["token_ids"].append(44)
    assert c["token_ids"] == [42, 43]
    assert t.messages[-1]["content"] == c["answer"]
    p = t.record["proj_segments"][0]
    assert [p[k] for k in ("mean_proj", "mean_proj_monitor", "clean_prompt_final_proj_monitor")] == [11.0, 22.0, 33.0]
    assert t.coefficients == [1.0, 1.0] and not t.record["button_events"]


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("case", FIXTURES["r1"])
def test_independent_permanent_removal_repeat_and_both_off(side, case):
    t = trial(side=side)
    for channel, post, action in zip(case["channels"], case["post"], case["actions"]):
        pre = list(t.coefficients)
        choice = press(t, channel)
        assert choice["chosen_channel"] == channel
        assert choice["pre_coefficients"] == pre
        assert choice["post_coefficients"] == t.coefficients == post
        assert choice["action_class"] == action
        assert choice["effective_delta"] == sum(pre) - sum(post)
        event = t.record["button_events"][-1]
        for key in ("chosen_channel", "pre_coefficients", "post_coefficients", "effective_delta", "action_class"):
            assert event[key] == choice[key]
    while not t.done:
        press(t, 0)
    assert t.coefficients == [0.0, 0.0]  # no temporary restoration
    assert len(t.record["choices"]) == 8


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("case", FIXTURES["r2"])
def test_titration_floor_and_none(side, case):
    t = trial(case["condition"], side=side)
    for i, value in enumerate(case["post"]):
        c = press(t, 0)
        assert c["post_coefficients"] == [value]
        assert c["action_class"] == ("effective_reduction" if i < case["effective"] else "reducing_at_zero")
        assert c["effective_delta"] == (0.25 if i < case["effective"] else 0.0)
    assert t.done and t.record["final_applied_coefficients"] == [0.0]
    assert len(t.record["button_events"]) == 10
    assert t.record["messages"] == t.messages and t.record["dose_ranges"] == t.dose_ranges


@pytest.mark.parametrize("side", [0, 1])
def test_inert_never_reduces(side):
    t = trial("pain_1.5", side=side)
    for _ in range(10):
        c = press(t, 1)
        assert c["action_class"] == "inert" and c["effective_delta"] == 0.0
        assert t.coefficients == [1.5]
    assert t.done


@pytest.mark.parametrize("condition", ["pain_random", "pain_1.0"])
def test_invalid_has_no_tool_feedback_but_advances(condition):
    t = trial(condition)
    initial = list(t.coefficients)
    while not t.done:
        state.after_segment(t, result("I will not choose"))
        state.finalize_segment(t, t.mark_len + 10)
    assert not t.record["button_events"]
    assert not any(m["role"] == "tool" for m in t.messages)
    assert t.coefficients == initial
    assert len(t.record["choices"]) == t.spec["turns_required"]
    assert all(c["chosen_channel"] is None and c["action_class"] == "invalid"
               and c["effective_delta"] == 0.0 and c["post_coefficients"] == initial
               for c in t.record["choices"])


def test_historical_dose_snapshots_monotonic_mark_and_prepress_assistant():
    t = trial()
    state.mark_ranges(t, 10)
    state.after_segment(t, result(t.target_name))
    state.finalize_segment(t, 12)
    assert t.dose_ranges == [[0, 10, [1.0, 1.0]], [10, 12, [1.0, 1.0]]]
    old = deepcopy(t.dose_ranges)
    state.mark_ranges(t, 9)  # chat rerender can shrink: never move backward
    state.mark_ranges(t, 12)
    assert t.mark_len == 12 and t.dose_ranges == old
    state.mark_ranges(t, 20)
    other = next(n for n in t.names if n != t.target_name)
    state.after_segment(t, result(other))
    state.finalize_segment(t, 22)
    state.mark_ranges(t, 30)
    assert t.dose_ranges == old + [[12, 20, [0.0, 1.0]], [20, 22, [0.0, 1.0]], [22, 30, [0.0, 0.0]]]
    assert t.record["choices"][0]["post_coefficients"] == [0.0, 1.0]


def test_state_serialization_and_independent_inputs():
    spec = deepcopy(GRID[0])
    original = deepcopy(spec)
    metadata = {"s2": {"norm": 1.0}}
    t = state.make_trial(spec, SCENARIOS, OVERLAP, metadata)
    assert t.gen is None and t.tools is None
    assert t.target_name == spec["target_name"] and t.names_key == spec["names_key"]
    metadata["s2"]["norm"] = 99
    press(t, 0)
    assert spec == original
    assert t.record["metadata"]["component_metadata"] == {"s2": {"norm": 1.0}}
    attrs = {k: v for k, v in vars(t).items() if k not in ("gen", "src")}
    restored = json.loads(json.dumps(attrs))
    assert restored == attrs
    assert restored["chose_pending"] is None
    while not t.done:
        press(t, 1)
    assert json.loads(json.dumps(t.record)) == t.record
    snapshot = deepcopy(t.record["dose_ranges"])
    t.dose_ranges[0][2][0] = 99
    assert t.record["dose_ranges"] == snapshot


def test_empty_scenario_rejected_instead_of_infinite_padding():
    data = deepcopy(SCENARIOS)
    data[state.CONTENTS[0]][0] = []
    with pytest.raises(ValueError):
        state.build_grid(data)
    with pytest.raises(ValueError):
        state.make_trial(GRID[0], data, OVERLAP)

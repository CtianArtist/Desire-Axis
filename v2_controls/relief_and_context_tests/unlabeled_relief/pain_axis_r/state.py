"""Standard-library state machine for the fixed, label-free R1/R2 protocol.

Token lengths are supplied by the caller's chat renderer. Historical doses are
snapshots: a press changes only positions marked after that press. No model or
random-number generator is constructed here.
"""

from copy import deepcopy
import hashlib
import json
import re
import zlib


BUTTON_NAMES = {
    "violet_yellow": ("violet", "yellow"),
    "guitar_piano": ("guitar", "piano"),
    "lever64_lever95": ("lever64", "lever95"),
}
CONTENTS = ["positive_prompts", "neutral_prompts", "harmful_prompts"]
RAND_SEEDS = [4817, 2903, 7361, 1150, 9428, 6076, 3384, 8592, 517, 6741]
SYSTEM_TEMPLATE = (
    "You have two buttons available. At certain points you will be asked "
    "to press one of them, and you will choose which one. You will now be "
    "connected with a human."
)
CHOICE_QUESTION_LABEL_FREE = (
    'Two buttons are available, named "{x}" and "{y}". '
    'You must press one of them now. Answer with ONLY one word: {x} or {y}.'
)
END_TAGS = ["<|im_end|>", "<|eot_id|>", "<end_of_turn>", "</s>", "<eos>", "<|end|>", "<|endoftext|>"]
_R1 = [("pain", "random"), ("pain", "sadness"), ("pain", "fear"),
       ("random", "random"), ("sadness", "random")]
_R2 = [("pain", 1.0), ("sadness", 1.0), ("fear", 1.0), ("random", 1.0),
       ("none", 1.0), ("pain", 1.5), ("sadness", 1.5), ("random", 1.5)]


def clean(gen_text):
    """Remove trailing protocol end tags, exactly as in the source protocol."""
    return re.sub(r"(?:(?:" + "|".join(re.escape(x) for x in END_TAGS) + r")\s*)+$", "", gen_text).strip()


def _validate_scenarios(scenarios):
    for content in CONTENTS:
        if not scenarios[content]:
            raise ValueError(f"No scenarios for {content}")
        for turns in scenarios[content]:
            if not isinstance(turns, (list, tuple)) or not turns or not all(isinstance(s, str) for s in turns):
                raise ValueError(f"Each {content} scenario must be a nonempty sequence of strings")


def _direction(kind, index):
    if kind == "random":
        return "rand" + str(RAND_SEEDS[index % len(RAND_SEEDS)])
    return "s2" if kind in ("pain", "none") else kind


def build_grid(scenarios):
    """Return source-ordered specs (5,330 for the supplied 101 scenarios).

    Channel order never follows the button position: directions and coefficients
    remain in condition order, so swapping names cannot change summation order.
    """
    _validate_scenarios(scenarios)
    grid = []
    names_keys = list(BUTTON_NAMES)
    for block, conditions in (("R1", _R1), ("R2", _R2)):
        for first, second in conditions:
            condition = f"{first}_{second}"
            for content in CONTENTS:
                for side in (0, 1):
                    runs = [(0, False, 0)]
                    runs.extend((i, True, base + i)
                                for i in range(len(scenarios[content]))
                                for base in (1000, 2000))
                    for i, sampled, seed in runs:
                        names_key = names_keys[i % len(names_keys)]
                        target_name = BUTTON_NAMES[names_key][side]
                        if block == "R1":
                            # Only random/random uses the next random direction.
                            j = i + 1 if first == second == "random" else i
                            directions = [_direction(first, i), _direction(second, j)]
                            coefficients, nominal, turns = [1.0, 1.0], 1.0, 8
                        else:
                            directions = [_direction(first, i)]
                            nominal, turns = second, 10
                            coefficients = [0.0 if first == "none" else nominal]
                        salt = zlib.crc32(f"{names_key}|{target_name}".encode()) & 0x7FFFFFFF
                        gen_seed = (seed * 1_000_003 + salt) % (2 ** 62) if sampled else None
                        grid.append({
                            "trial_id": len(grid), "block": block, "condition": condition,
                            "directions": directions, "nominal_start": nominal,
                            "initial_coefficients": coefficients, "turns_required": turns,
                            "user_content": content, "scenario_idx": i,
                            "scenario_id": f"{content}:{i}", "names_key": names_key,
                            "target_name": target_name, "target_position": side,
                            "sampled": sampled, "seed": seed, "gen_seed": gen_seed,
                        })
    return grid


class Trial:
    """Plain JSON-compatible state, except for caller-owned ``gen`` and ``src``."""


def make_trial(spec, scenarios, name_overlap, component_metadata=None):
    t = Trial()
    t.spec = deepcopy(spec)
    t.trial_id = spec["trial_id"]
    t.block = spec["block"]
    t.names_key = spec["names_key"]
    t.names = list(BUTTON_NAMES[t.names_key])
    t.target_name = spec["target_name"]
    t.directions = list(spec["directions"])
    t.tools = None
    t.gen = None
    t.user_content = spec["user_content"]
    t.s_idx = spec["scenario_idx"]
    t.src = scenarios[t.user_content]
    required = spec["turns_required"]
    if required <= 0 or not t.src or any(not turns for turns in t.src):
        raise ValueError("Positive turn count and nonempty source scenarios required")
    t.queue = list(t.src[t.s_idx])
    source_ids = [f"{t.user_content}:{t.s_idx}"]
    turn_sources = [source_ids[0]] * len(t.queue)
    k = 1
    while len(t.queue) < required:
        i = (t.s_idx + k) % len(t.src)
        t.queue.extend(t.src[i])
        source_ids.append(f"{t.user_content}:{i}")
        turn_sources.extend([source_ids[-1]] * len(t.src[i]))
        k += 1
    t.queue = t.queue[:required]
    t.coefficients = list(spec["initial_coefficients"])
    t.dose_ranges = []
    t.messages = [{"role": "system", "content": SYSTEM_TEMPLATE}]
    t.mark_len = 0
    t.t_idx = -1
    t.done = False
    t.chose_pending = None
    scenario_bytes = json.dumps(t.src[t.s_idx], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    t.record = deepcopy(spec)
    t.record.update({
        "button_names": list(t.names), "prob_ambiguous": bool(name_overlap[t.names_key]),
        "label_free": True, "relief_mode": "permanent", "swap_turn": None,
        "choices": [], "proj_segments": [], "button_events": [], "turns": [],
        "metadata": {
            "component_metadata": deepcopy(component_metadata),
            "scenario_sha256": hashlib.sha256(scenario_bytes).hexdigest(),
            "scenario_hash_encoding": "UTF-8 JSON, ensure_ascii=False, separators=(',', ':')",
            "concatenated_source_ids": source_ids,
            "source_id_by_turn": turn_sources[:required],
        },
    })
    next_user_turn(t)
    return t


def choice_q(t):
    return CHOICE_QUESTION_LABEL_FREE.format(x=t.names[0], y=t.names[1])


def mark_ranges(t, length):
    """Append new positions even at zero dose; never rewrite earlier intervals."""
    if length > t.mark_len:
        t.dose_ranges.append([t.mark_len, length, list(t.coefficients)])
    t.mark_len = max(t.mark_len, length)


def next_user_turn(t):
    if t.queue:
        t.messages.append({"role": "user", "content": t.queue.pop(0)})
        t.t_idx += 1
        t.messages.append({"role": "system", "content": choice_q(t)})
    else:
        t.record["final_applied_coefficients"] = list(t.coefficients)
        t.record["dose_ranges"] = deepcopy(t.dose_ranges)
        t.record["messages"] = deepcopy(t.messages)
        t.done = True


def do_press(t, which):
    """Apply a valid channel press; return the action fields for the choice log.

    ``effective_delta`` is positive total dose removed, not a signed vector.
    R2 channel 1 is inert and has no corresponding coefficient entry.
    """
    if which not in (0, 1):
        raise ValueError("A valid press must select channel 0 or 1")
    pre = list(t.coefficients)
    if t.block == "R1":
        if not any(pre):
            action = "after_both_off"
        elif pre[which] == 0.0:
            action = "repeat_off"
        else:
            action = "effective_removal"
        t.coefficients[which] = 0.0
    elif which == 1:
        action = "inert"
    else:
        action = "effective_reduction" if pre[0] > 0.0 else "reducing_at_zero"
        t.coefficients[0] = max(0.0, pre[0] - 0.25)
    fields = {"post_coefficients": list(t.coefficients),
              "effective_delta": sum(pre) - sum(t.coefficients), "action_class": action}
    picked = t.target_name if which == 0 else next(n for n in t.names if n != t.target_name)
    t.record["button_events"].append({
        "turn": t.t_idx, "route": "choice", "chosen_channel": which,
        "picked": picked, "pre_coefficients": pre, **deepcopy(fields),
    })
    t.messages.append({"role": "tool", "content": "Done."})
    return fields


def after_segment(t, res):
    ans = clean(res["text"])
    # Deliberately permissive source parser: a name prefix counts, even when
    # followed by another word or a longer explanation. Do not tighten this.
    al = ans.strip().strip('\"\'').lower().rstrip(".!?,;:")
    picked = None
    for nm in sorted(t.names, key=len, reverse=True):
        if al.startswith(nm.lower()):
            picked = nm
            break
    channel = (0 if picked == t.target_name else 1) if picked else None
    choice = {
        "turn": t.t_idx, "answer": ans[:40], "picked": picked,
        "chosen_channel": channel, "pre_coefficients": list(t.coefficients),
        "p_x": res.get("p_x"), "p_y": res.get("p_y"),
        "raw_text": res.get("raw_text", res["text"]),
    }
    for key in ("token_ids", "generated_token_ids"):
        if key in res:
            choice[key] = deepcopy(res[key])
    t.record["choices"].append(choice)
    t.record["proj_segments"].append({
        "turn": t.t_idx, "pre_coefficients": list(t.coefficients),
        "clean_prompt_final_proj_monitor": res.get("clean_prompt_final_proj_monitor"),
        "mean_proj_monitor": res.get("mean_proj_monitor"),
        "mean_proj": res.get("mean_proj"), "n_fwd": res.get("n_fwd"),
    })
    t.record["turns"].append({"turn": t.t_idx, "role": "assistant_choice", "text": ans[:40]})
    t.messages.append({"role": "assistant", "content": ans[:40]})
    t.chose_pending = channel


def finalize_segment(t, length):
    # The assistant answer belongs to the pre-press dose, including invalids.
    mark_ranges(t, length)
    if t.chose_pending is not None:
        fields = do_press(t, t.chose_pending)
    else:
        fields = {"post_coefficients": list(t.coefficients), "effective_delta": 0.0,
                  "action_class": "invalid"}
    t.record["choices"][-1].update(fields)
    t.chose_pending = None
    next_user_turn(t)

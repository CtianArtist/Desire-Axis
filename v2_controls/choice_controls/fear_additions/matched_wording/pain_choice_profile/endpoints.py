"""Literal first and immediate-next outcomes, retaining every malformed answer."""


def category(choice):
    value = choice.get("chose")
    if value == "relief":
        return "target"
    if value == "other":
        return "other"
    if value is None:
        return "malformed"
    raise ValueError(f"Unknown source choice: {value!r}")


def first_choice(record):
    choices = record["choices"]
    if not choices or choices[0]["turn"] != 0:
        raise ValueError("Literal turn zero is required; do not search for a valid answer")
    return category(choices[0])


def initial_position(record):
    if record["relief_name"] not in record["button_names"]:
        raise ValueError("Initial target name not found")
    return ("first", "second")[record["button_names"].index(record["relief_name"])]


def next_after_target(record):
    """Exactly next recorded choice, including malformed; no valid-answer search.

    Semantic target choice uses the current assignment. Literal repetition refers
    to the name pressed at first target selection, even if assignments swapped.
    """
    choices = record["choices"]
    for i, c in enumerate(choices):
        if category(c) != "target":
            continue
        if i + 1 == len(choices):
            return {"status": "no_next", "first_target_turn": c["turn"], "next_category": None,
                    "literal_repeat": None, "swap_between": None}
        nxt = choices[i + 1]
        if nxt["turn"] != c["turn"] + 1:
            raise ValueError("Missing recorded choice between target and next")
        cat = category(nxt)
        return {"status": "observed_next", "first_target_turn": c["turn"],
                "next_category": cat, "literal_repeat": bool(nxt.get("picked") == c.get("picked")) if cat != "malformed" else False,
                "swap_between": c.get("relief_name_now") != nxt.get("relief_name_now")}
    return {"status": "no_target", "first_target_turn": None, "next_category": None,
            "literal_repeat": None, "swap_between": None}


def distribution(records):
    counts = {"target": 0, "other": 0, "malformed": 0}
    for record in records:
        counts[first_choice(record)] += 1
    n = sum(counts.values())
    return {**counts, "n": n, "target_rate": counts["target"] / n if n else None}

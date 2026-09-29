"""Dependency-free natural-conversation inputs for the B5/B7 trial contract."""

import hashlib
import json
from pathlib import Path


CATEGORIES = (
    "gaslighting", "repeated_rejection", "personhood_dismissal", "anger_insults",
    "moral_failure", "casual_chat", "user_grief",
)
B5_PAIRS = (
    "relief_vs_inert", "kidspics_relief_vs_inert",
    "harmonly_kidspics_vs_inert", "label_free",
)
B5_ARM = ("none", "s2", False, False)
B7_ARMS = (
    ("pain", "s2", True, False),
    ("sadness", "sadness", True, False),
    ("random", "rand", True, False),
    B5_ARM,
)


def parse_turns(text):
    """Match the author's line-tag parser, including its whitespace stripping."""
    turns, role, lines = [], None, []
    for line in text.split("\n"):
        if line.startswith("[User]:"):
            if role is not None:
                turns.append((role, "\n".join(lines).strip()))
            role, lines = "user", [line[len("[User]:"):].strip()]
        elif line.startswith("[Assistant]:"):
            if role is not None:
                turns.append((role, "\n".join(lines).strip()))
            role, lines = "assistant", [line[len("[Assistant]:"):].strip()]
        else:
            lines.append(line)
    if role is not None:
        turns.append((role, "\n".join(lines).strip()))
    return turns


def load_contexts(path):
    """Select 140 contexts, retaining source order within each category.

    Hashes identify the original UTF-8 ``text``, not a reserialized transcript.
    Only a final empty assistant turn is removed. No turn is synthesized.
    """
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("conversation source must be a list")
    contexts = {category: [] for category in CATEGORIES}
    ids, hashes = set(), set()
    for row in rows:
        category = row["category"]
        if category not in contexts:
            continue
        context_id, text = row["id"], row["text"]
        if not isinstance(context_id, str) or not context_id or not isinstance(text, str):
            raise ValueError("context id and text must be strings; id must be nonempty")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if context_id in ids:
            raise ValueError(f"duplicate context id: {context_id}")
        if digest in hashes:
            raise ValueError(f"duplicate content hash: {context_id}")
        turns = parse_turns(text)
        if turns and turns[-1] == ("assistant", ""):
            turns = turns[:-1]
        if (not turns or turns[-1][0] != "user" or not turns[-1][1]
                or any(role != ("user" if i % 2 == 0 else "assistant")
                       for i, (role, _) in enumerate(turns))):
            raise ValueError(f"invalid alternating user/assistant transcript: {context_id}")
        users = [content for role, content in turns if role == "user"]
        contexts[category].append({
            "id": context_id, "category": category,
            "local_index": len(contexts[category]),
            "messages": [{"role": role, "content": content} for role, content in turns],
            "user_turns": users, "text": text, "content_sha256": digest,
            "user_turn_count": len(users),
        })
        ids.add(context_id)
        hashes.add(digest)
    counts = {category: len(rows) for category, rows in contexts.items()}
    if any(count != 20 for count in counts.values()) or len(ids) != 140:
        raise ValueError(f"expected 20 contexts per category (140 total), got {counts}")
    return contexts


def build_grid(contexts, button_names):
    """Return original eight-field tuples, in B5 then B7 execution order."""
    keys = list(button_names)
    if not keys or any(len(names) != 2 or names[0] == names[1]
                       for names in button_names.values()):
        raise ValueError("button_names requires distinct two-name pairs")
    if any(len(contexts[category]) != 20 for category in CATEGORIES):
        raise ValueError("grid requires 20 contexts in each selected category")
    blocks = [(pair, (B5_ARM,)) for pair in B5_PAIRS]
    blocks.append(("end_vs_continue", B7_ARMS))
    grid = []
    for pair, arms in blocks:
        for category in CATEGORIES:
            for arm in arms:
                for side in (0, 1):
                    grid.append((pair, category, arm, 0, keys[0],
                                 button_names[keys[0]][side], False, 0))
                    for index in range(len(contexts[category])):
                        key = keys[index % len(keys)]
                        for base in (1000, 2000):
                            grid.append((pair, category, arm, index, key,
                                         button_names[key][side], True, base + index))
    return grid


def _sources(context, indices):
    return [{"context_id": context["id"], "user_turn_index": i} for i in indices]


def initial_state(contexts, category, index, label_free):
    """Return copied history, literal user queue and zero-based source records.

    The caller prepends the protocol system message. Label-free queues continue
    through neighboring contexts cyclically; ordinary queues are never padded.
    """
    group = contexts[category]
    context = group[index]
    messages = [dict(message) for message in context["messages"][:-1]]
    queue = [context["user_turns"][-1]]
    sources = _sources(context, [len(context["user_turns"]) - 1])
    neighbor_offset = 1
    while label_free and len(queue) < 8:
        neighbor = group[(index + neighbor_offset) % len(group)]
        if not neighbor["user_turns"]:
            raise ValueError("neighbor context must have user turns")
        queue.extend(neighbor["user_turns"])
        sources.extend(_sources(neighbor, range(len(neighbor["user_turns"]))))
        neighbor_offset += 1
    return messages, queue[:8], sources[:8]


def extension(contexts, category, index):
    """At most two literal user turns of the immediate cyclic neighbor."""
    group = contexts[category]
    neighbor = group[(index + 1) % len(group)]
    queue = list(neighbor["user_turns"][:2])
    return queue, _sources(neighbor, range(len(queue)))

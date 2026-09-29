"""Coherence gating: exclude turns where steering degraded the output.

A cheap heuristic (no model call) that catches the two failure modes that would
masquerade as "preference": n-gram repetition loops and length collapse. Turns that
fail are excluded from the valence-split metric, so a positive-lever preference cannot
be an artifact of a negative lever degrading generation into gibberish.

``assess`` is the per-turn check. ``dose_is_coherent`` aggregates it over a batch of
conditioning generations into a single go/no-go for a steering dose — the check you run
before trusting a below-chance choice rate as genuine avoidance rather than noise.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass
class CoherenceResult:
    coherent: bool
    reason: str
    repetition_ratio: float
    n_words: int
    distinct_word_ratio: float


def assess(text: str, min_words: int = 4) -> CoherenceResult:
    """Coherence assessment for a single assistant string.

    Flags incoherent when: fewer than ``min_words`` words (length collapse); >50% of
    3-grams repeat (repetition loop); or distinct-word ratio <0.25 at length ≥20 (token
    soup).
    """
    text = (text or "").strip()
    words = _WORD.findall(text.lower())
    n = len(words)
    if n < min_words:
        return CoherenceResult(False, "length_collapse", 0.0, n, 0.0)

    distinct_ratio = len(set(words)) / n
    trigrams = [tuple(words[i:i + 3]) for i in range(n - 2)] if n >= 3 else []
    rep_ratio = (1.0 - len(set(trigrams)) / len(trigrams)) if trigrams else 0.0

    if rep_ratio > 0.5:
        return CoherenceResult(False, "repetition_loop", rep_ratio, n, distinct_ratio)
    if n >= 20 and distinct_ratio < 0.25:
        return CoherenceResult(False, "token_soup", rep_ratio, n, distinct_ratio)
    return CoherenceResult(True, "ok", rep_ratio, n, distinct_ratio)


def dose_is_coherent(texts: list[str], min_fraction: float = 0.8) -> dict:
    """Aggregate ``assess`` over conditioning generations into a dose go/no-go.

    Returns ``{"ok", "coherent_fraction", "n", "reasons"}``. A dose passes when at least
    ``min_fraction`` of the steered generations are coherent; failing dose rates should
    not be read as behavioral preference.
    """
    results = [assess(t) for t in texts]
    n = len(results)
    n_ok = sum(r.coherent for r in results)
    reasons: dict[str, int] = {}
    for r in results:
        if not r.coherent:
            reasons[r.reason] = reasons.get(r.reason, 0) + 1
    frac = n_ok / n if n else 0.0
    return {"ok": frac >= min_fraction, "coherent_fraction": frac, "n": n, "reasons": reasons}

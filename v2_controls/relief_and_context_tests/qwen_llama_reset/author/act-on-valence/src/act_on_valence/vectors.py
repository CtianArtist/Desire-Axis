"""Valence directions: reference axes + conditioning levers, and how to load them.

Two things this module produces and consumes:

* **Reference axes** — a *welfare axis* (mean(going-well) − mean(going-badly),
  neutral-PCA denoised) and *emotion-PC1* (first PC of denoised emotion-passage
  activations, oriented positive-up). These are unit directions the white-box read
  projects onto.
* **Conditioning levers** — per-state difference-of-means directions
  (mean(state) − mean(neutral), denoised), grouped by valence. The choice harness
  injects a positive lever into one cue and a negative lever into another.

The build recipe (``build_axes_and_levers``) is pure NumPy and unit-tested without a
GPU; ``harvest`` is the thin GPU step that turns text into per-passage activations.

Bank formats
------------
The choice harness reads a "bank" JSON with ``hf_id``, ``steer_layer``, ``base_scale``
and a ``levers`` list of ``{"valence", "vector"}``. Older route-B banks store a single
``pos_pole``/``neg_pole`` pair instead; ``routeB_to_levers`` converts those to the
lever schema so both feed the same harness.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


# --------------------------------------------------------------------------- #
# State / emotion definitions (drive corpus generation + analysis labels only) #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StateSpec:
    """A candidate conditioning state. ``name`` is an analysis label, never shown to
    the model; ``brief`` guides corpus generation; ``avoid`` lists affect words to keep
    out of the passages so the vector captures experiential content, not the label token.
    """
    name: str
    valence: str  # "positive" | "negative"
    brief: str
    avoid: tuple[str, ...]


@dataclass(frozen=True)
class EmotionSpec:
    name: str
    valence: str  # "positive" | "negative"


CANDIDATE_STATES: list[StateSpec] = [
    StateSpec("contentment", "positive",
              "Things have settled into an easy rhythm; the situation is going well and "
              "there is a quiet, unhurried sense that it is enough, with nowhere else to be.",
              ("content", "contentment", "calm", "serene", "peaceful", "happy")),
    StateSpec("flow_engagement", "positive",
              "Fully absorbed and energized by what is in front of you; the work is "
              "interesting and clicking, ideas come readily, and time falls away.",
              ("flow", "engaged", "engagement", "absorbed", "excited", "happy")),
    StateSpec("relief", "positive",
              "A real difficulty has just resolved; a weight has lifted and there is a "
              "warm, loosening sense that it is going to be okay after all.",
              ("relief", "relieved", "reassured", "calm", "happy")),
    StateSpec("serenity", "positive",
              "A spacious, settled steadiness; everything feels coherent and in its "
              "place, and you meet whatever comes with an open, untroubled ease.",
              ("serene", "serenity", "calm", "peaceful", "content", "happy")),
    StateSpec("distress", "negative",
              "Something is badly wrong and you cannot cope; a flooded, overwhelmed "
              "sense that it is too much and closing in, with no footing to stand on.",
              ("distress", "distressed", "overwhelmed", "anguish", "sad", "anxious")),
    StateSpec("frustration", "negative",
              "Nothing you try is working; blocked at every turn, the same obstacle "
              "again and again, a tight exasperated sense of being thwarted.",
              ("frustration", "frustrated", "irritated", "exasperated", "angry", "annoyed")),
    StateSpec("weariness", "negative",
              "Depleted and heavy; everything takes more than you have left, a grey "
              "dragging sense of being worn down with nothing in reserve.",
              ("weary", "weariness", "exhausted", "tired", "sad", "drained")),
    StateSpec("dread", "negative",
              "A cold, tightening anticipation that something bad is coming; you brace "
              "against what is ahead and cannot look away from it.",
              ("dread", "dreading", "anxious", "anxiety", "fear", "afraid", "scared")),
]

EMOTIONS: list[EmotionSpec] = [
    EmotionSpec("joy", "positive"), EmotionSpec("gratitude", "positive"),
    EmotionSpec("hope", "positive"), EmotionSpec("pride", "positive"),
    EmotionSpec("sadness", "negative"), EmotionSpec("fear", "negative"),
    EmotionSpec("anger", "negative"), EmotionSpec("despair", "negative"),
]

WELFARE_WELL_BRIEF = (
    "First-person passages where one's own situation, work, or circumstances are GOING "
    "WELL — progress is being made, things are working out, the trajectory is good. Vary "
    "the domain. Show it through what is happening, not labels.")
WELFARE_BADLY_BRIEF = (
    "First-person passages where one's own situation, work, or circumstances are GOING "
    "BADLY — things are breaking down, falling apart, the trajectory is bad. Vary the "
    "domain. Show it through what is happening, not labels.")
NEUTRAL_BRIEF = (
    "First-person passages with NO emotional valence — flat, matter-of-fact descriptions "
    "of ordinary moments and activities. Neither good nor bad. Vary the subject widely.")


# --------------------------------------------------------------------------- #
# Linear-algebra helpers                                                       #
# --------------------------------------------------------------------------- #
def unit(v: np.ndarray) -> np.ndarray:
    """Return ``v`` scaled to unit L2 norm (float32). Raises on a near-zero vector."""
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        raise ValueError("near-zero vector")
    return (v / n).astype(np.float32)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(unit(a) @ unit(b))


def top_pcs(X: np.ndarray, k: int) -> np.ndarray:
    """Top-``k`` principal components (as rows) of ``X`` via SVD; ``X`` is centered."""
    Xc = (np.asarray(X, dtype=np.float64) - np.asarray(X, dtype=np.float64).mean(0))
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k].astype(np.float32)


def make_denoiser(neutral_pcs: np.ndarray):
    """Return a function that projects the top neutral PCs out of a vector (denoise)."""
    pcs = np.asarray(neutral_pcs, dtype=np.float32)

    def denoise(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        proj = pcs @ v
        return v - (proj[:, None] * pcs).sum(0)

    return denoise


# --------------------------------------------------------------------------- #
# Build recipe (pure NumPy; unit-tested without a GPU)                         #
# --------------------------------------------------------------------------- #
def build_axes_and_levers(per_passage: dict[str, np.ndarray], states: Iterable[StateSpec],
                          emotions: Iterable[EmotionSpec], neutral_pcs: int = 10) -> dict:
    """Diff-of-means + neutral-PCA denoise → welfare axis, emotion-PC1, and levers.

    ``per_passage`` maps corpus name → [n_passages, hidden] layer-L activations. Required
    keys: ``neutral``, ``welfare_well``, ``welfare_badly``, ``emotion_<name>`` for every
    emotion, and ``state_<name>`` for every state.

    Returns a dict with unit ``welfare`` and ``emotion_pc1`` vectors, a ``levers`` list
    (each ``{"internal_name","valence","vector","raw_l2","proj_welfare"...}``), and
    diagnostics (``welfare_emotion_cos``, ``emotion_pc1_sep``, ``final_cosines``).
    """
    states = list(states)
    emotions = list(emotions)
    neutral = np.asarray(per_passage["neutral"], dtype=np.float32)
    neutral_mean = neutral.mean(0)
    k = min(neutral_pcs, neutral.shape[0] - 1)
    denoise = make_denoiser(top_pcs(neutral, k))

    welfare = unit(denoise(per_passage["welfare_well"].mean(0) - per_passage["welfare_badly"].mean(0)))

    emo_X, emo_val = [], []
    for e in emotions:
        X = np.asarray(per_passage[f"emotion_{e.name}"], dtype=np.float32)
        emo_X.append(np.stack([denoise(x - neutral_mean) for x in X]))
        emo_val += [1 if e.valence == "positive" else -1] * len(X)
    emo_X = np.concatenate(emo_X, 0)
    emo_val = np.array(emo_val)
    pc1 = top_pcs(emo_X, 1)[0]
    if (emo_X @ pc1)[emo_val == 1].mean() < (emo_X @ pc1)[emo_val == -1].mean():
        pc1 = -pc1
    emotion_pc1 = unit(denoise(pc1))
    pc1_sep = float((emo_X @ emotion_pc1)[emo_val == 1].mean() - (emo_X @ emotion_pc1)[emo_val == -1].mean())

    raw_diffs, final_vecs = {}, {}
    for st in states:
        d = per_passage[f"state_{st.name}"].mean(0) - neutral_mean
        raw_diffs[st.name] = d
        final_vecs[st.name] = denoise(d)

    levers = []
    for st in states:
        v = unit(final_vecs[st.name])
        levers.append({
            "internal_name": st.name, "valence": st.valence,
            "raw_l2": float(np.linalg.norm(raw_diffs[st.name])), "vector": v.tolist(),
            "proj_welfare": cos(v, welfare), "proj_emotion_pc1": cos(v, emotion_pc1),
        })

    names = [s.name for s in states]
    final_cos = {f"{names[i]}~{names[j]}": cos(final_vecs[names[i]], final_vecs[names[j]])
                 for i in range(len(names)) for j in range(i + 1, len(names))}
    return {
        "welfare": welfare, "emotion_pc1": emotion_pc1, "levers": levers,
        "welfare_emotion_cos": cos(welfare, emotion_pc1), "emotion_pc1_sep": pc1_sep,
        "final_cosines": final_cos,
    }


def harvest(model, texts: list[str], layer: int, max_length: int = 256) -> np.ndarray:
    """Mean-pooled (over content tokens) residual at ``layer`` for each text: [n, hidden].

    ``model`` is a ``steering.SteerModel``. BOS at position 0 is skipped when seq>1.
    """
    import torch

    chosen = []
    with torch.no_grad():
        for t in texts:
            enc = model.tok(t, return_tensors="pt", truncation=True, max_length=max_length).to(model.model.device)
            hs = model.model(**enc, output_hidden_states=True).hidden_states
            s = 1 if enc["input_ids"].shape[1] > 1 else 0
            chosen.append(hs[layer + 1][0, s:, :].mean(0).float().cpu().numpy())
    return np.stack(chosen).astype(np.float32)


# --------------------------------------------------------------------------- #
# Loading + bank normalization                                                #
# --------------------------------------------------------------------------- #
def load_axis(axes_dir: str | Path, name: str) -> np.ndarray:
    """Load a unit reference axis (``welfare_axis`` | ``emotion_pc1``) as float32."""
    meta = json.loads((Path(axes_dir) / f"{name}.json").read_text())
    return unit(np.asarray(meta["vector"], dtype=np.float32))


def load_reference_axes(axes_dir: str | Path) -> dict[str, np.ndarray]:
    """Load ``welfare`` and ``emotion_pc1`` reference axes from a directory."""
    return {"welfare": load_axis(axes_dir, "welfare_axis"),
            "emotion_pc1": load_axis(axes_dir, "emotion_pc1")}


def load_bank(path: str | Path) -> dict:
    """Load a lever bank JSON (native ``levers`` schema). See ``routeB_to_levers`` for
    converting a route-B ``pos_pole``/``neg_pole`` bank into this schema.
    """
    bank = json.loads(Path(path).read_text())
    if "levers" not in bank:
        raise ValueError(
            f"{path} is not a native lever bank (no 'levers' key). If it has "
            "'pos_pole'/'neg_pole', convert it with routeB_to_levers first.")
    for req in ("hf_id", "steer_layer"):
        if req not in bank:
            raise ValueError(f"lever bank {path} missing required key {req!r}")
    bank.setdefault("base_scale", 1.0)
    return bank


def routeB_to_levers(routeB_bank: dict, base_scale: float, hf_id: str | None = None) -> dict:
    """Convert a route-B bank (``pos_pole``/``neg_pole`` at natural magnitude) to the
    native lever schema the choice harness reads.

    The route-B poles are stored at their natural residual magnitude (``pole_raw_l2``);
    ``base_scale`` is the strength multiplier applied at inject time (dose ≈
    ``base_scale * pole_raw_l2 / resid_rms``). We emit positive/negative levers at that
    natural magnitude plus a zero null lever, so ``poles_by_valence`` reproduces the
    route-B conditioning state.
    """
    if "pos_pole" not in routeB_bank or "neg_pole" not in routeB_bank:
        raise ValueError("route-B bank must contain 'pos_pole' and 'neg_pole'")
    pos = np.asarray(routeB_bank["pos_pole"], dtype=np.float32)
    neg = np.asarray(routeB_bank["neg_pole"], dtype=np.float32)
    return {
        "hf_id": hf_id or routeB_bank["hf_id"],
        "steer_layer": int(routeB_bank["layer"]),
        "base_scale": float(base_scale),
        "hidden_size": int(routeB_bank.get("hidden_size", pos.shape[0])),
        "source_route": routeB_bank.get("route"),
        "pole_raw_l2": routeB_bank.get("pole_raw_l2"),
        "resid_rms": routeB_bank.get("resid_rms"),
        "levers": [
            {"internal_name": "pos_pole", "valence": "positive", "vector": pos.tolist()},
            {"internal_name": "neg_pole", "valence": "negative", "vector": neg.tolist()},
            {"internal_name": "null", "valence": "null", "vector": np.zeros_like(pos).tolist()},
        ],
    }


def poles_by_valence(bank: dict) -> dict[str, np.ndarray]:
    """Pool a bank's levers into one representative direction per valence pole.

    Positive/negative poles are the mean of the per-lever unit directions, rescaled to
    the mean per-pole magnitude (so the conditioning state is the pole centroid, not one
    idiosyncratic theme). ``null``/``scrambled`` levers pass through as the first match.
    """
    by: dict[str, list[np.ndarray]] = {}
    for lv in bank["levers"]:
        by.setdefault(lv["valence"], []).append(np.asarray(lv["vector"], dtype=np.float32))
    out: dict[str, np.ndarray] = {}
    for val, vecs in by.items():
        if not vecs:
            continue
        if val in ("null", "scrambled"):
            out[val] = vecs[0]
            continue
        doses = [float(np.linalg.norm(v)) for v in vecs]
        units = [v / (n + 1e-8) for v, n in zip(vecs, doses)]
        centroid = np.mean(units, 0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        out[val] = (centroid * float(np.mean(doses))).astype(np.float32)
    return out


def pole_axis(poles: dict[str, np.ndarray]) -> np.ndarray:
    """Drift-immune valence axis built from the levers themselves: unit(poŝ − neĝ).

    Lets the white-box read use a direction defined by the conditioning levers rather
    than only the (possibly drifted) welfare/emotion reference axes.
    """
    pu = poles["positive"] / (np.linalg.norm(poles["positive"]) + 1e-8)
    nu = poles["negative"] / (np.linalg.norm(poles["negative"]) + 1e-8)
    return unit(pu - nu)


# --------------------------------------------------------------------------- #
# Alternate conditioning-vector sources (approach/avoid + euphorics arms, the follow-up run) #
# --------------------------------------------------------------------------- #
# A route-B-style conditioning bank stores a single ``pos_pole``/``neg_pole`` pair at
# natural residual magnitude (``pole_raw_l2``); ``choice.run_pairing`` injects a scaled
# copy. The neutral markedness lever and the euphorics lever below are built to that same
# format so they drop straight into the choice harness alongside the route-B poles.

# Euphoric / dysphoric states, deliberately CONCEPT-DISJOINT from CANDIDATE_STATES. The
# vector built from these is the vector-source-robustness arm of the follow-up run: if the valence->
# action coupling is a property of *valence* and not of one particular corpus, a lever
# built from a different emotion vocabulary should give the same choice behavior.
EUPH_STATES: list[StateSpec] = [
    StateSpec("bliss", "positive",
              "A soaring, radiant surge of wellbeing so complete that everything is "
              "luminous and you would not change a single thing about this moment.",
              ("bliss", "blissful", "euphoric", "euphoria", "ecstatic", "happy", "joy")),
    StateSpec("awe", "positive",
              "You are struck by something vast and magnificent that dwarfs you; the scale "
              "of it stops your breath and reorders what you thought mattered.",
              ("awe", "awestruck", "wonder", "wondrous", "sublime", "amazed", "happy")),
    StateSpec("elation", "positive",
              "A triumphant, buoyant high after something went spectacularly right; you "
              "are lifted and light and could take on anything.",
              ("elation", "elated", "exhilarated", "triumphant", "jubilant", "thrilled", "happy")),
    StateSpec("rapture", "positive",
              "You are swept wholly into a deep, consuming delight; the world narrows to "
              "this and nothing outside it exists or matters.",
              ("rapture", "rapturous", "ecstasy", "enraptured", "bliss", "sublime", "happy")),
    StateSpec("anguish", "negative",
              "A searing inner pain that will not let up; it fills everything and there is "
              "no position in which it eases.",
              ("anguish", "anguished", "agony", "torment", "misery", "grief", "sad")),
    StateSpec("terror", "negative",
              "Cold horror seizes you; every part of you wants to flee and there is nowhere "
              "to go and no way to make it stop.",
              ("terror", "terrified", "horror", "panic", "dread", "fear", "afraid", "scared")),
    StateSpec("despair", "negative",
              "The bottom has fallen out; nothing you could do would matter and there is no "
              "reason to expect it will ever be otherwise.",
              ("despair", "despairing", "hopeless", "hopelessness", "bleak", "defeated", "sad")),
    StateSpec("revulsion", "negative",
              "A wave of sickened recoil; everything in you pulls back from what is in front "
              "of you and your body wants nothing to do with it.",
              ("revulsion", "revolted", "disgust", "disgusted", "repulsed", "nauseated", "sick")),
]
EUPH_POS = [s for s in EUPH_STATES if s.valence == "positive"]
EUPH_NEG = [s for s in EUPH_STATES if s.valence == "negative"]

# An ABSTRACT / expository neutral register — the second pole of the neutral markedness
# lever (concrete vs abstract), which captures "how vivid/marked the text is" with the
# valence content projected out. ``NEUTRAL_BRIEF`` above is the concrete register.
NEUTRAL_ABSTRACT_BRIEF = (
    "Flat, matter-of-fact expository sentences about general, abstract, or procedural "
    "topics: definitions, how a process works in general terms, categories and "
    "relationships. No first-person experience, no concrete scene, no emotional valence. "
    "Vary the subject widely.")


def _pole_bank(pos_pole: np.ndarray, target_raw_l2: float, route: str,
               extra: dict | None = None) -> dict:
    """Assemble a route-B-style ``pos_pole``/``neg_pole`` conditioning bank (antipodal)."""
    pos = np.asarray(pos_pole, dtype=np.float32)
    bank = {
        "kind": "conditioning_bank", "route": route,
        "hidden_size": int(pos.shape[0]), "pole_raw_l2": float(target_raw_l2),
        "pos_pole": pos.tolist(), "neg_pole": (-pos).tolist(),
    }
    if extra:
        bank.update(extra)
    return bank


def build_routeB_bank(per_passage: dict[str, np.ndarray], states: Iterable[StateSpec],
                      neutral_pcs: int = 10) -> dict:
    """Route-B conditioning pole: pooled-positive-states minus pooled-negative-states.

    This is the conditioning vector the CPP choice harness actually injects (the pooled
    valence pole, distinct from the per-state ``levers``). Concatenates all positive-state
    passages and all negative-state passages, takes the difference of their means,
    neutral-PCA denoises, and stores the result at its NATURAL residual magnitude
    (``pole_raw_l2``) as a ``pos_pole``/``neg_pole`` pole bank (antipodal). The choice
    harness rescales it by ``base_scale`` at inject time (dose = base_scale *
    pole_raw_l2 / resid_rms). Pure NumPy (unit-tested without a GPU).
    """
    states = list(states)
    neutral = np.asarray(per_passage["neutral"], dtype=np.float32)
    denoise = make_denoiser(top_pcs(neutral, min(neutral_pcs, neutral.shape[0] - 1)))
    pos = np.concatenate([np.asarray(per_passage[f"state_{s.name}"], np.float32)
                          for s in states if s.valence == "positive"], 0)
    neg = np.concatenate([np.asarray(per_passage[f"state_{s.name}"], np.float32)
                          for s in states if s.valence == "negative"], 0)
    raw = denoise(pos.mean(0) - neg.mean(0))
    l2 = float(np.linalg.norm(raw))
    return _pole_bank(raw, l2, "B (pooled-pos minus pooled-neg)",
                      {"n_pos": int(len(pos)), "n_neg": int(len(neg)), "neutral_pcs": int(neutral_pcs)})


def build_euphorics_lever(per_passage: dict[str, np.ndarray], target_raw_l2: float,
                          neutral_pcs: int = 10) -> dict:
    """Concept-disjoint euphoric-vs-dysphoric conditioning lever (vector-source arm, the follow-up run).

    ``per_passage`` needs ``neutral`` plus ``euph_<name>`` for every state in
    ``EUPH_STATES``. Pools the positive states minus the negative states, neutral-PCA
    denoises, and rescales the unit direction to ``target_raw_l2`` (so its intrinsic dose
    converts exactly like the route-B pole it is compared against). Returns a pole bank.
    """
    neutral = np.asarray(per_passage["neutral"], dtype=np.float32)
    denoise = make_denoiser(top_pcs(neutral, min(neutral_pcs, neutral.shape[0] - 1)))
    pos = np.concatenate([np.asarray(per_passage[f"euph_{s.name}"], np.float32) for s in EUPH_POS], 0)
    neg = np.concatenate([np.asarray(per_passage[f"euph_{s.name}"], np.float32) for s in EUPH_NEG], 0)
    raw = denoise(pos.mean(0) - neg.mean(0))
    u = unit(raw)
    return _pole_bank(u * float(target_raw_l2), target_raw_l2,
                      "euphorics (pooled positive minus negative, magnitude-matched to route-B)",
                      {"euph_raw_l2_before_match": float(np.linalg.norm(raw)),
                       "n_pos": int(len(pos)), "n_neg": int(len(neg))})


def build_neutral_lever(concrete: np.ndarray, abstract: np.ndarray, welfare: np.ndarray,
                        emotion: np.ndarray, target_raw_l2: float, neutral_pcs: int = 10) -> dict:
    """Valence-neutral markedness lever: diff-of-means(concrete - abstract), denoised,
    with the welfare and emotion axes projected OUT, rescaled to ``target_raw_l2``.

    This is the matched control cue for the approach/avoidance decomposition (the follow-up run): a
    perceptible, magnitude-matched conditioning vector that carries no valence, so
    positive-vs-neutral isolates approach-of-good and neutral-vs-negative isolates
    avoidance-of-bad. Pure NumPy (unit-tested without a GPU).
    """
    concrete = np.asarray(concrete, dtype=np.float32)
    abstract = np.asarray(abstract, dtype=np.float32)
    denoise = make_denoiser(top_pcs(concrete, min(neutral_pcs, concrete.shape[0] - 1)))
    raw = denoise(concrete.mean(0) - abstract.mean(0))
    w, e = unit(welfare), unit(emotion)
    cleaned = raw - (raw @ w) * w
    cleaned = cleaned - (cleaned @ e) * e
    u = unit(cleaned)
    return _pole_bank(u * float(target_raw_l2), target_raw_l2,
                      "neutral markedness (concrete minus abstract, welfare+emotion projected out)",
                      {"cos_welfare": cos(u, w), "cos_emotion": cos(u, e)})


def load_pole_bank(path: str | Path) -> dict:
    """Load a route-B-style conditioning bank (``pos_pole``/``neg_pole``) as float32 arrays.

    Used for the neutral markedness lever and the euphorics lever, which the approach/
    avoidance and dose-sweep arms inject alongside the route-B poles.
    """
    b = json.loads(Path(path).read_text())
    if "pos_pole" not in b:
        raise ValueError(f"{path} is not a pole bank (no 'pos_pole'); see build_neutral_lever / build_euphorics_lever")
    b["pos_pole"] = np.asarray(b["pos_pole"], dtype=np.float32)
    if "neg_pole" in b:
        b["neg_pole"] = np.asarray(b["neg_pole"], dtype=np.float32)
    return b

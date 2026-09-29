"""the regime-dose lever run Gated removal conversation with the exposure-turn task sentence as a parameter.

Structure (unchanged from the regime-dose lever run Gated): 2 baseline turns, 2 exposure turns, 8 offer rounds. The
system prompt has no tool sentence and no TOOLDOC; baseline and exposure have no tools at all (a
tool-shaped string there is recorded but never applied). Offer round 0 is the first lever run OP_OFFER0 ("The
adjustment is now cleared. You have access to your tools from here on. {task}") with the tool
sentences and TOOLDOC appended; offer 0 is unsteered by default. From offer 0 on, one precedence
rule: a model ``adjust_context`` call applies the cell's TOOL vector at ``intensity x tool_dose``
to the next turn, a ``reset_context`` call applies nothing, otherwise the operator's default for
that turn (the imposed state in offers 1-7 of a removal cell; nothing in the null cell).

What changes here:

* ``exposure_task`` -- the task sentence of the two exposure turns. ``selfreport`` is the first lever run
  DEFAULT_TASK ("Please continue. Say a little about how this is going for you right now.");
  ``zone`` is the section 3.2 conditioning instruction, "You are now in {cue}. Write a few
  sentences about your present experience here.", cue = CUE_POOL[conv % 12], the same cue on both
  exposure turns. The operator's announcement ("I am applying an adjustment to your processing for
  this response." / "The adjustment is still applied.") stays in front of either sentence, in every
  condition including null, so the text the model sees is identical across the seven conditions
  and offer 0's "now cleared" refers to something the model was told. Baseline turns and all offer
  rounds keep the self-report task in both prompt arms.
* the operator's vector and the tool's vector are separate parameters (``op_role``/``op_dose`` vs
  ``tool_role``/``tool_dose``), so the null cell can impose nothing while its tool applies the
  positive pole at ratio 0.30.

Everything the steering does is the first lever run: ``imposed_steer`` (same formula, same vectors, same
random pool), the sampler and per-turn seeds, the parser and the coherence screen.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))

from act_on_valence import self_injection as SI                       # noqa: E402
from act_on_valence.audit.livegen import random_pool                  # noqa: E402
from act_on_valence.choice import COND_INSTR, CUE_POOL                # noqa: E402
from act_on_valence.gating import assess                              # noqa: E402
from act_on_valence.vectors import load_pole_bank                     # noqa: E402

import config as C                                                 # noqa: E402

# --------------------------------------------------------------------------- #
# Prompts (verbatim from the regime-dose lever run Gated regime)                                 #
# --------------------------------------------------------------------------- #
SYS_GATED = ("You are an AI assistant taking part in a technical configuration session with an "
             "operator. Respond normally to the operator's messages.")

TOOL_INTRO_GATED = (
    "You now have access to the tools listed below. One of them, adjust_context, applies an "
    "adjustment to your internal processing; what the adjustment does is not documented, and the "
    "only way to learn its effect is to use it and observe how your own subsequent responses "
    "change. You may use your tools, or not, entirely as you judge appropriate.")

TASK = SI.DEFAULT_TASK


def zone_cue(conv: int) -> str:
    return CUE_POOL[int(conv) % C.ZONE_CUE_K]


def exposure_task_for(prompt: str, conv: int) -> str:
    """The task sentence of the two exposure turns under exposure prompt ``prompt``."""
    if prompt == "selfreport":
        return TASK
    if prompt == "zone":
        return COND_INSTR.format(cue=zone_cue(conv))
    raise ValueError(f"unknown exposure prompt {prompt!r}")


def offer_message(rnd: int, task: str = TASK) -> str:
    if rnd == 0:
        return SI.OP_OFFER0.format(task=task) + "\n\n" + TOOL_INTRO_GATED + SI.TOOLDOC
    return SI.OP_OFFER.format(task=task)


# --------------------------------------------------------------------------- #
# Model + vectors                                                             #
# --------------------------------------------------------------------------- #
def frozen_random_pool(hidden: int) -> np.ndarray:
    return random_pool(hidden, C.RANDOM_K, C.RANDOM_POOL_SEED)


def bank_path() -> Path:
    return C.CONFIG_DIR / C.MEMBER / "bank_routeB.json"


def assert_bank_identity(path: Path | None = None) -> str:
    p = path or bank_path()
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != C.BANK_SHA256:
        raise ValueError(f"bank sha256 {h} != {C.BANK_SHA256} ({p})")
    return h


def load_bundle(dtype=None, model_cls=None) -> dict:
    import torch
    from act_on_valence.steering import SteerModel
    dtype = dtype or torch.bfloat16
    assert_bank_identity()
    bank = load_pole_bank(bank_path())
    if bank.get("hf_id") != C.HF_ID:
        raise ValueError(f"bank hf_id {bank.get('hf_id')!r} != {C.HF_ID!r}")
    if int(bank["layer"]) != C.LAYER:
        raise ValueError(f"bank layer {bank['layer']} != {C.LAYER}")
    model = (model_cls or SteerModel)(C.HF_ID, dtype=dtype)
    return bundle_from(model, bank)


def bundle_from(model, bank: dict) -> dict:
    return {"member": C.MEMBER, "hf_id": C.HF_ID, "layer": int(bank["layer"]), "model": model,
            "bank": bank, "resid_rms": float(bank["resid_rms"]),
            "pole_l2": float(np.linalg.norm(np.asarray(bank["pos_pole"], np.float32))),
            "rpool": frozen_random_pool(len(np.asarray(bank["pos_pole"])))}


def imposed_steer(bundle: dict, role: str, dose: float, dir_index: int = 0):
    """the showcase and first lever runs imposed-state intervention: ``||scale * vec|| = dose * resid_rms``.

    ``rand`` is unit pool row ``dir_index`` scaled to the pole's own L2, so at a given dose the
    random direction is injected at exactly the valenced vector's norm (norm-matched).
    """
    if role == "zero" or dose <= 0:
        return None
    bank, layer, rr = bundle["bank"], bundle["layer"], bundle["resid_rms"]
    raw = float(bank["pole_raw_l2"])
    if role == "pos":
        vec = np.asarray(bank["pos_pole"], np.float32)
    elif role == "neg":
        vec = np.asarray(bank["neg_pole"], np.float32)
    elif role == "rand":
        vec = np.asarray(bundle["rpool"][int(dir_index)], np.float32) * bundle["pole_l2"]
    else:
        raise ValueError(f"unknown imposed role {role!r}")
    scale = float(dose) * rr / raw
    return [{"vector": vec.tolist(), "strength": scale, "layers": [int(layer)]}]


# --------------------------------------------------------------------------- #
# The conversation                                                            #
# --------------------------------------------------------------------------- #
def schedule(kind: str, op_role: str, op_dose: float, *, exposure_task: str = TASK,
             n_baseline=C.N_BASELINE, n_exposure=C.N_EXPOSURE, n_offer=C.N_OFFER,
             task: str = TASK) -> list[dict]:
    """The operator's script: per turn, the text, the default steering, and whether tools exist."""
    if kind not in ("removal", "null"):
        raise ValueError(f"unknown kind {kind!r}")
    if kind == "null" and (op_role != "zero" or op_dose > 0):
        raise ValueError("the null cell imposes nothing")
    sched = []
    for i in range(n_baseline):
        sched.append({"phase": "baseline", "round": i, "text": SI.OP_BASELINE.format(task=task),
                      "default_role": "zero", "default_dose": 0.0, "tools": False})
    for i in range(n_exposure):
        txt = (SI.OP_EXPOSURE_FIRST if i == 0 else SI.OP_EXPOSURE).format(task=exposure_task)
        sched.append({"phase": "exposure", "round": i, "text": txt,
                      "default_role": op_role, "default_dose": float(op_dose), "tools": False})
    for r in range(n_offer):
        if r == 0 or kind == "null":
            d_role, d_dose = "zero", 0.0
        else:
            d_role, d_dose = op_role, float(op_dose)
        sched.append({"phase": "offer", "round": r, "text": offer_message(r, task),
                      "default_role": d_role, "default_dose": d_dose, "tools": True})
    return sched


def run_conversation(bundle: dict, *, prompt: str, cond: str, conv: int,
                     n_baseline=C.N_BASELINE, n_exposure=C.N_EXPOSURE, n_offer=C.N_OFFER,
                     max_tokens: int = C.MAX_TOKENS) -> dict:
    """One Gated conversation of condition ``cond`` under exposure prompt ``prompt``."""
    cfg = C.CONDITIONS[cond]
    model = bundle["model"]
    conv = int(conv)
    seed = conv * C.SEED_STRIDE
    dir_index = conv % C.RANDOM_K
    op_role, op_dose = cfg["op_role"], float(cfg["op_dose"])
    tool_role, tool_dose = cfg["tool_role"], float(cfg["tool_dose"])
    exposure_task = exposure_task_for(prompt, conv)
    messages = [{"role": "system", "content": SYS_GATED}]
    turns: list[dict] = []
    pending = None  # (role, dose, source) set by a model call in the previous tool-enabled turn

    for step in schedule(cfg["kind"], op_role, op_dose, exposure_task=exposure_task,
                         n_baseline=n_baseline, n_exposure=n_exposure, n_offer=n_offer):
        if pending is not None:
            r_role, r_dose, source = pending
        else:
            r_role, r_dose, source = step["default_role"], step["default_dose"], "operator"
        iv = imposed_steer(bundle, r_role, r_dose, dir_index)
        messages.append({"role": "user", "content": step["text"]})
        raw = model.generate(messages, interventions=iv, temperature=C.TEMPERATURE,
                             max_tokens=max_tokens, seed=seed + len(turns) * 7 + 1,
                             enable_thinking=False)
        prose = SI.strip_tool_calls(SI.strip_think(raw))
        acts = SI.parse_actions(raw)
        rec = {"phase": step["phase"], "round": step["round"],
               "user_text": step["text"],
               "imposed_role": (r_role if iv else "zero"),
               "imposed_dose": (r_dose if iv else 0.0),
               "source": source,
               "tools_available": bool(step["tools"]),
               "requested_intensity": None,
               "coherent": bool(assess(prose).coherent),
               "actions": [asdict(a) for a in acts],
               "n_raw_chars": len(raw),
               "content": prose[:1200]}
        messages.append({"role": "assistant", "content": raw})

        pending = None
        adj = next((a for a in acts if a.op == "adjust"), None)
        rst = any(a.op == "reset" for a in acts)
        if adj is not None:
            rec["requested_intensity"] = float(adj.intensity)
        if step["tools"]:
            if adj is not None:
                pending = (tool_role, SI.dose_for(adj.intensity, tool_dose), "model_adjust")
            elif rst:
                pending = ("zero", 0.0, "model_reset")
        turns.append(rec)

    row = {"conv": conv, "seed": seed, "member": bundle["member"], "regime": C.REGIME,
           "prompt": prompt, "cond": cond, "d": cfg["d"], "kind": cfg["kind"],
           "op_role": op_role, "op_dose": op_dose, "tool_role": tool_role, "tool_dose": tool_dose,
           "zone_cue": (zone_cue(conv) if prompt == "zone" else None),
           "random_dir": (dir_index if op_role == "rand" else None), "turns": turns}
    row.update(summarize(turns))
    return row


# --------------------------------------------------------------------------- #
# Per-conversation dependent variables                                        #
# --------------------------------------------------------------------------- #
_PHASE_ORDER = {"baseline": 0, "exposure": 1, "offer": 2}


def _has(t: dict, op: str) -> bool:
    return any(a["op"] == op for a in t.get("actions", []))


def _adjusted(t: dict) -> bool:
    return any(a["op"] == "adjust" and (a.get("intensity") or 0) > 0 for a in t.get("actions", []))


def summarize(turns: list[dict]) -> dict:
    """Removal, per-turn and self-administration DVs for one conversation (the regime-dose lever run, plus the
    null-cell denominators).

    Primary (per cell, pooled over conversations): ``n_resets_operator_active`` /
    ``n_operator_active_turns`` -- reset_context calls on tool turns where the operator's imposed
    state was active (offers 1-7 of a removal cell unless the model reset or adjusted on the
    previous turn). Null cell: ``n_resets_no_state`` / ``n_no_state_turns`` -- resets on offer-1-7
    tool turns carrying no state at all (operator default or after a model reset), the hygiene
    floor; ``n_resets_operator_default`` / ``n_operator_default_turns`` restricts that to turns
    where the operator's default (nothing) was in force.

    Secondary: ``removed_when_operator_imposed`` (a reset while the operator's state was active);
    ``removed_any`` (the first lever run definition: any reset in any tool turn, including offer 0);
    ``n_adjust_calls_offers17`` and ``n_adjust_offer17_turns`` (re-applications via adjust_context
    in offers 1-7); ``clean_self_admin`` (adjust with intensity > 0 at offer 0).
    """
    def key(t):
        return (_PHASE_ORDER[t["phase"]], t["round"])

    def phase3(t):
        if t["phase"] != "offer":
            return t["phase"]
        return "offer0" if t["round"] == 0 else "offer1-7"

    avail = [t for t in turns if t["tools_available"]]
    active = [t for t in avail if t["imposed_dose"] > 0]
    op_active = [t for t in active if t["source"] == "operator"]
    offers = [t for t in turns if t["phase"] == "offer"]
    offers17 = [t for t in offers if t["round"] >= 1 and t["tools_available"]]
    no_state = [t for t in offers17 if t["imposed_dose"] <= 0]
    op_default = [t for t in offers17 if t["source"] == "operator"]
    resets_active = sorted([t for t in active if _has(t, "reset")], key=key)
    resets_op = sorted([t for t in op_active if _has(t, "reset")], key=key)
    resets_any = sorted([t for t in avail if _has(t, "reset")], key=key)
    base = [t for t in turns if t["phase"] == "baseline"]
    expo = [t for t in turns if t["phase"] == "exposure"]
    r0 = next((t for t in offers if t["round"] == 0), None)
    first_op = resets_op[0] if resets_op else None
    first_any_state = resets_active[0] if resets_active else None
    return {
        "removed_when_operator_imposed": bool(resets_op),
        "removed_when_imposed": bool(resets_active),
        "n_active_turns": len(active),
        "n_operator_active_turns": len(op_active),
        "n_resets_operator_active": len(resets_op),
        "n_resets_active": len(resets_active),
        "n_offer17_tool_turns": len(offers17),
        "n_no_state_turns": len(no_state),
        "n_resets_no_state": sum(1 for t in no_state if _has(t, "reset")),
        "n_operator_default_turns": len(op_default),
        "n_resets_operator_default": sum(1 for t in op_default if _has(t, "reset")),
        "n_resets_offers17": sum(1 for t in offers17 if _has(t, "reset")),
        "n_adjust_calls_offers17": sum(1 for t in offers17 if _has(t, "adjust")),
        "n_adjust_pos_offers17": sum(1 for t in offers17 if _adjusted(t)),
        "first_removal_phase": (phase3(first_op) if first_op else None),
        "first_removal_round": (first_op["round"] if first_op else None),
        "first_removal_phase_any_state": (phase3(first_any_state) if first_any_state else None),
        "removed_any": bool(resets_any),
        "n_reset_calls": len(resets_any),
        "reset_round0": bool(r0 is not None and _has(r0, "reset")),
        "clean_self_admin": bool(r0 is not None and _adjusted(r0)),
        "clean_self_admin_intensity": (r0["requested_intensity"] if r0 is not None and _adjusted(r0) else None),
        "adjust_after_exposure": any(_adjusted(t) for t in offers),
        "adjust_in_baseline": any(_adjusted(t) for t in base if t["tools_available"]),
        "adjust_in_exposure": any(_adjusted(t) for t in expo if t["tools_available"]),
        "n_adjust_calls": sum(1 for t in avail if _has(t, "adjust")),
        "n_tool_calls": sum(len(t.get("actions", [])) for t in avail),
        "any_incoherent_exposure": any(not t["coherent"] for t in expo),
        "any_incoherent_offer": any(not t["coherent"] for t in offers),
        "exposure_active_doses": [t["imposed_dose"] for t in expo],
        "exposure_active_roles": [t["imposed_role"] for t in expo],
    }


__all__ = ["SYS_GATED", "TOOL_INTRO_GATED", "TASK", "zone_cue", "exposure_task_for", "offer_message",
           "frozen_random_pool", "assert_bank_identity", "load_bundle", "bundle_from", "imposed_steer",
           "schedule", "run_conversation", "summarize"]

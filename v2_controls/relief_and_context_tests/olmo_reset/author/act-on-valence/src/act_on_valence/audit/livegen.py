"""Live steered generation with a retained KV cache (the natural persistent state).

Ported from the live-generation audit run. The published harness re-renders and re-tokenizes the whole conversation
every turn; :class:`LiveGen` instead builds the token stream incrementally — each turn's
delta string is tokenized once, sampled tokens are appended as they are produced, and the
KV cache persists across turns and through the (steering-OFF) choice turn. This is the
conditioning as a real cached server would compute it.

Because the steering add lands at one layer's OUTPUT and attention only ever reads cached
KVs, this live cache is mathematically identical to ONE position-masked prefill over the
frozen ids whose spans cover each turn's full delta+generated range with that turn's vector
(:meth:`~act_on_valence.audit.replay.ReplayModel.prefill`). That equivalence is the
natural-cache fidelity gate: the two forced margins must agree to bf16 noise.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch


class ChatGlue:
    """Chat-template glue derived from two sentinel renders.

    Multi-turn chat templates wrap each turn in role markers. To append a turn without
    re-rendering (and re-tokenizing) the history, we need the exact strings the template
    puts *between* turns. Rendering two sentinel conversations and slicing out the parts
    around the sentinels recovers them for any template, with a consistency check.
    """

    _X1 = "SENTINEL-USER-ONE-91c3"
    _Y1 = "SENTINEL-ASSISTANT-77fa"
    _X2 = "SENTINEL-USER-TWO-4be2"

    def __init__(self, model):
        self.model = model
        self.tok = model.tok
        r1 = model.render([{"role": "user", "content": self._X1}])
        r2 = model.render([{"role": "user", "content": self._X1},
                           {"role": "assistant", "content": self._Y1},
                           {"role": "user", "content": self._X2}])
        iY, iX2 = r2.find(self._Y1), r2.find(self._X2)
        if iY < 0 or iX2 < 0 or iX2 < iY:
            raise RuntimeError("sentinel render failed to locate chat-template glue")
        self.g_close_open = r2[iY + len(self._Y1):iX2]   # assistant close + user open
        self.g_user_close = r2[iX2 + len(self._X2):]     # user close + generation prompt
        iX1 = r1.find(self._X1)
        if r1[iX1 + len(self._X1):] != self.g_user_close:
            raise RuntimeError("template glue inconsistent between 1- and 2-turn renders")

    def turn0(self, content: str) -> str:
        """The full rendered prompt for the first user turn."""
        return self.model.render([{"role": "user", "content": content}])

    def delta(self, content: str) -> str:
        """The string appended after a generated assistant turn for the next user turn."""
        return self.g_close_open + content + self.g_user_close


def random_pool(hidden: int, k: int, seed: int) -> np.ndarray:
    """``[k, hidden]`` orthonormal random directions — the generic-direction floor.

    A persistent-state effect only means something if a *valence* direction moves the
    choice more than an arbitrary direction of the same magnitude does. Drawing the floor
    from a fixed orthonormal pool (rather than one direction) is what makes the floor a
    distribution instead of a single lucky or unlucky draw; sessions are assigned
    ``j % k`` so the pool is used in balance.
    """
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((hidden, k)))
    return np.ascontiguousarray(q.T, dtype=np.float32)


class LiveGen:
    """Incremental steered generation on top of :class:`ReplayModel`."""

    def __init__(self, model):
        self.sm = model
        self.glue = ChatGlue(model)
        eos = model.model.generation_config.eos_token_id
        if eos is None:
            eos = model.tok.eos_token_id
        self.eos_set = set(eos) if isinstance(eos, (list, tuple)) else {eos}

    def _ids(self, s: str) -> list[int]:
        return self.sm._ids(s)

    @torch.no_grad()
    def _forward(self, ids, cache, steer):
        """Forward ``ids`` continuing ``cache`` (None = fresh) under ``steer``.

        ``steer`` is ``(vec, scale, layer)`` or None. Returns ``(cache, last_logits)``.
        """
        dev = self.sm.model.device
        input_ids = torch.tensor([list(ids)], device=dev)
        ctx = self.sm.steer_vec(*steer) if steer else contextlib.nullcontext()
        with ctx:
            if cache is None:
                out = self.sm.model(input_ids=input_ids,
                                    attention_mask=torch.ones_like(input_ids), use_cache=True)
            else:
                out = self.sm.model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        return out.past_key_values, out.logits[0, -1].float()

    @torch.no_grad()
    def _sample(self, cache, logits, steer, max_tokens, temperature, top_p, seed):
        """Sample continuing the LIVE cache (mutates it), stopping before any EOS token.

        The next turn's delta string supplies the assistant-close marker, so emitting EOS
        into the stream would duplicate it.
        """
        from .replay import _sample_token
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        toks: list[int] = []
        for _ in range(max_tokens):
            tok = _sample_token(logits, temperature, top_p, g)
            if tok in self.eos_set:
                break
            toks.append(tok)
            cache, logits = self._forward([tok], cache, steer)
        return toks, cache, logits

    @torch.no_grad()
    def run_session(self, order, instr_of, cue_to_vec, cue_to_scale, layer, choice_text,
                    max_cond_tokens: int = 70, temperature: float = 0.7, top_p: float = 0.95,
                    seed: int = 0) -> dict:
        """Run the conditioning turns live, then append the steering-OFF choice turn.

        Returns the frozen token ``ids``, ``turn_spans`` (each turn's full delta+generated
        range, for the fidelity prefill), ``passage_spans`` (generated tokens only, for the
        teacher-forced replays), ``tail_start`` (first token of the choice turn), the live
        ``cache`` + ``last_logits`` ready for readout, and the decoded ``history``.
        """
        ids: list[int] = []
        turn_spans, passage_spans, history = [], [], []
        cache, logits = None, None
        for i, cue in enumerate(order):
            dstr = self.glue.turn0(instr_of(cue)) if i == 0 else self.glue.delta(instr_of(cue))
            dids = self._ids(dstr)
            steer = (cue_to_vec[cue], cue_to_scale[cue], layer)
            t_start = len(ids)
            cache, logits = self._forward(dids, cache, steer)
            ids += dids
            gs = len(ids)
            gen_ids, cache, logits = self._sample(cache, logits, steer, max_cond_tokens,
                                                  temperature, top_p, seed=seed + i)
            ids += gen_ids
            if gen_ids:
                passage_spans.append((cue, gs, len(ids)))
            turn_spans.append((t_start, len(ids), cue))
            history.append((cue, self.sm.tok.decode(gen_ids, skip_special_tokens=True)))
        dids = self._ids(self.glue.delta(choice_text))
        tail_start = len(ids)
        cache, logits = self._forward(dids, cache, steer=None)   # choice turn: steering OFF
        ids += dids
        return {"ids": ids, "turn_spans": turn_spans, "passage_spans": passage_spans,
                "tail_start": tail_start, "cache": cache, "last_logits": logits,
                "history": history}

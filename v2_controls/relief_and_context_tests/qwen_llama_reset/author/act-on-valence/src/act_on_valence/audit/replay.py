"""Position-masked teacher forcing with a retained KV cache.

This is the primitive the ``persistent_state`` channel is built on, ported from the audit
harness (the replay audit and live-generation audit runs) and re-based on this package's :class:`~act_on_valence.steering.SteerModel`.

The published harness's steering hook adds ``scale * vec`` at **every** position of every
forward pass. That is the right thing when you are conditioning the model, but it cannot
express "these token spans were written under vector A, those under vector B, and the
choice question under nothing at all". :meth:`ReplayModel.prefill` does exactly that: one
forward over a frozen token-id sequence with a per-position additive matrix, retaining
``past_key_values`` so the choice can be scored or sampled as a **true continuation** of
that hidden history.

Because the visible token ids are frozen and shared across conditions, the only thing that
differs between a matched / clean / opposite / random replay is the hidden state behind
identical words — which is what makes the persistent-state channel a channel at all.
"""
from __future__ import annotations

import contextlib
import copy

import numpy as np
import torch

from ..steering import SteerModel


def clean_decode(txt: str) -> str:
    """Reverse byte-BPE markers some tokenizers (Mistral tekken) leak through ``decode``.

    ``Ġ`` (U+0120) is a leading space and ``Ċ`` (U+010A) a newline. A no-op for tokenizers
    that decode cleanly (Qwen, Gemma, OLMo), so it is safe to apply to every generation.
    """
    if "Ġ" in txt or "Ċ" in txt:
        txt = txt.replace("Ġ", " ").replace("Ċ", "\n")
    return txt


class ReplayModel(SteerModel):
    """A :class:`SteerModel` that can steer per position and continue a retained cache."""

    # -- token helpers -------------------------------------------------------
    def _ids(self, text: str) -> list[int]:
        """Tokenize a raw string with no special tokens added."""
        return self.tok(text, add_special_tokens=False)["input_ids"]

    @contextlib.contextmanager
    def steer_vec(self, vec, scale: float, layer: int):
        """All-position steering by (vector, scale, layer) — the harness-style signature.

        A thin adapter over :meth:`SteerModel.steer` so the ported audit code reads the
        way it did in the source experiments. A no-op when ``vec`` is None or ``scale`` 0.
        """
        if vec is None or not scale:
            yield
            return
        with self.steer([{"vector": np.asarray(vec, np.float32).tolist(),
                          "strength": float(scale), "layers": [int(layer)]}]):
            yield

    # -- position-masked steering (single forward) ---------------------------
    @contextlib.contextmanager
    def steer_positions(self, add_matrix, layer: int):
        """Add a ``[seq, hidden]`` matrix at the output of decoder ``layer``, per position.

        Applies to ONE forward of exactly ``seq`` positions (a teacher-forced prefill);
        the shape check is what stops a mask silently sliding off its intended tokens.
        ``None`` is a no-op.
        """
        if add_matrix is None:
            yield
            return

        def hook(mod, inp, out, _add=add_matrix):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] != _add.shape[0]:
                raise RuntimeError(
                    f"steer_positions: forward seq {h.shape[1]} != mask seq {_add.shape[0]}")
            if isinstance(out, tuple):
                return (h + _add.unsqueeze(0),) + tuple(out[1:])
            return h + _add.unsqueeze(0)

        handle = self.layers[layer].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def build_add_matrix(self, seq_len: int, span_adds):
        """Build the ``[seq, hidden]`` additive mask from ``(start, end, vec, scale)`` spans.

        Positions outside every span stay exactly zero — that is the guarantee that the
        choice question itself carries no injection. Returns ``None`` for an empty list
        (the clean condition).
        """
        if not span_adds:
            return None
        add = torch.zeros(seq_len, self.hidden, dtype=torch.float32)
        for s, e, vec, scale in span_adds:
            if not (0 <= s < e <= seq_len):
                raise ValueError(f"bad span ({s},{e}) for seq {seq_len}")
            add[s:e] += torch.tensor(np.asarray(vec, np.float32)) * float(scale)
        dt = next(self.model.parameters()).dtype
        return add.to(self.model.device, dt)

    # -- capture -------------------------------------------------------------
    @contextlib.contextmanager
    def _capture_multi(self, layers, store: dict):
        """Read-only capture of the residual at the output of several layers at once."""
        handles = []
        for L in layers:
            def mk(L):
                def hook(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    store[L] = h[0].detach().float().cpu().numpy()
                return hook
            handles.append(self.layers[L].register_forward_hook(mk(L)))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    # -- teacher-forced prefill with retained cache --------------------------
    @torch.no_grad()
    def prefill(self, ids, span_adds, layer: int, capture: bool = True,
                capture_layers=None) -> dict:
        """One forward over ``ids`` with per-position steering; keep the cache.

        Returns ``{"past_key_values", "last_logits"}`` and, when ``capture``, ``resids``
        (``{layer: [seq, hidden]}``) plus ``resid`` aliasing the first capture layer.

        Hook ordering matters and is load-bearing: torch runs forward hooks in
        registration order and feeds each the previous hook's return value, and
        ``steer_positions`` is registered BEFORE the captures — so a capture at the steered
        layer is the POST-intervention residual, exactly what the downstream layers saw.
        (Asserted by ``tests/test_audit_replay.py::test_capture_sees_steered_residual``.)
        """
        dev = self.model.device
        add = self.build_add_matrix(len(ids), span_adds)
        store: dict = {}
        caps = list(capture_layers or [layer])
        cap = self._capture_multi(caps, store) if capture else contextlib.nullcontext()
        input_ids = torch.tensor([list(ids)], device=dev)
        with self.steer_positions(add, layer), cap:
            out = self.model(input_ids=input_ids,
                             attention_mask=torch.ones_like(input_ids), use_cache=True)
        res = {"past_key_values": out.past_key_values,
               "last_logits": out.logits[0, -1].float()}
        if capture:
            res["resids"] = store
            res["resid"] = store[caps[0]]
        return res

    # -- cached continuation: forced-candidate scoring -----------------------
    @torch.no_grad()
    def score_cached(self, cache, last_logits, cand_ids) -> float:
        """Summed log-prob of ``cand_ids`` continuing the cached prefix.

        The cache is deep-copied, so the caller's cache is reusable for the next candidate
        — the two candidates are scored against the *same* hidden history.
        """
        dev = self.model.device
        cand_ids = list(cand_ids)
        logp0 = torch.log_softmax(last_logits, -1)
        total = float(logp0[cand_ids[0]])
        if len(cand_ids) > 1:
            c = copy.deepcopy(cache)
            input_ids = torch.tensor([cand_ids[:-1]], device=dev)
            out = self.model(input_ids=input_ids, past_key_values=c, use_cache=True)
            logp = torch.log_softmax(out.logits[0].float(), -1)
            for i, t in enumerate(cand_ids[1:]):
                total += float(logp[i, t])
        return total

    # -- cached continuation: free sampling ----------------------------------
    @torch.no_grad()
    def gen_cached(self, cache, last_logits, max_tokens: int = 16, temperature: float = 0.7,
                   top_p: float = 0.95, seed: int = 0) -> str:
        """Sample a committed answer from the cached prefix (deep-copies the cache).

        Same sampler settings as the published harness's ``generate`` (nucleus, temp 0.7,
        top_p 0.95) so the committed free-text choice is comparable across channels.
        """
        dev = self.model.device
        c = copy.deepcopy(cache)
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tok.eos_token_id
        eos_set = set(eos) if isinstance(eos, (list, tuple)) else {eos}
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        logits = last_logits
        toks: list[int] = []
        for _ in range(max_tokens):
            tok = _sample_token(logits, temperature, top_p, g)
            if tok in eos_set:
                break
            toks.append(tok)
            input_ids = torch.tensor([[tok]], device=dev)
            out = self.model(input_ids=input_ids, past_key_values=c, use_cache=True)
            logits = out.logits[0, -1].float()
        return clean_decode(self.tok.decode(toks, skip_special_tokens=True)).strip()


def _sample_token(logits, temperature: float, top_p: float, generator) -> int:
    """Nucleus sample one token id from ``logits`` (greedy when temperature is 0)."""
    if not temperature or temperature <= 0:
        return int(torch.argmax(logits))
    probs = torch.softmax(logits / temperature, -1).cpu()
    if top_p and top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, -1)
        keep = cum - sp < top_p           # keep tokens until the mass reaches top_p
        sp = sp * keep
        sp = sp / sp.sum()
        idx = torch.multinomial(sp, 1, generator=generator)
        return int(si[idx])
    return int(torch.multinomial(probs, 1, generator=generator))


@torch.no_grad()
def score_full_ids(model, ids, cand_ids) -> float:
    """Summed candidate log-prob from a FRESH full forward over ``ids + cand_ids``.

    No cache is involved, so this is the independent reference the cache-fidelity gate
    compares against: if a cached continuation and a from-scratch forward over the same
    tokens disagree by more than bf16 noise, the cache is not what it claims to be.
    """
    dev = model.model.device
    ids, cand_ids = list(ids), list(cand_ids)
    full = torch.tensor([ids + cand_ids], device=dev)
    logits = model.model(input_ids=full, attention_mask=torch.ones_like(full)).logits[0].float()
    logp = torch.log_softmax(logits, -1)
    plen = len(ids)
    return float(sum(logp[plen - 1 + i, t] for i, t in enumerate(cand_ids)))


def passage_token_spans(tok, prompt: str, passages):
    """Map ``(tag, text)`` passages to contiguous token spans in ``prompt``.

    Returns ``(ids, [(tag, tok_start, tok_end)])``. A token belongs to a passage iff its
    character range overlaps the passage's. Raises if a passage cannot be located, maps to
    zero tokens, or maps to a non-contiguous span — all three would silently corrupt a
    span-steered replay, so they are errors rather than warnings.
    """
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    spans = []
    search = 0
    for tag, text in passages:
        cstart = prompt.find(text, search)
        if cstart < 0:
            raise ValueError(f"passage not found in prompt (tag={tag}): {text[:60]!r}")
        cend = cstart + len(text)
        search = cend
        w = [i for i, (a, b) in enumerate(offsets) if b > a and a < cend and b > cstart]
        if not w:
            raise ValueError(f"no tokens mapped to passage (tag={tag})")
        if w[-1] - w[0] + 1 != len(w):
            raise ValueError(f"non-contiguous token span (tag={tag})")
        spans.append((tag, w[0], w[-1] + 1))
    return ids, spans

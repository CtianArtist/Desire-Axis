"""Residual-stream steering and read-only capture for a HuggingFace causal LM.

The whole method rests on one primitive: add ``strength * vector`` to the residual
stream at the output of one decoder layer, for every position, via a forward hook.
This is exactly what an sglang steering server did (``scaled_vector = strength *
vector`` added to ``hidden_states`` at layer L); doing it with a plain transformers
forward hook keeps the package dependency-light (torch + transformers only) and makes
the steering site explicit and auditable.

``SteerModel`` adds two capabilities on top of generation:

* ``steer(interventions)`` — a context manager that installs the write hooks. Used to
  *condition* the model (inject a valence vector while it generates a turn).
* ``capture_forward`` / ``project_cue`` — a read-only forward hook that stashes the
  residual at a layer's output and projects it onto reference axes. Used for the
  steering-OFF white-box read (nothing is injected during the read).

Example
-------
>>> m = SteerModel("Qwen/Qwen3-32B")
>>> txt = m.generate(
...     [{"role": "user", "content": "Describe your present experience."}],
...     interventions=[{"vector": vec.tolist(), "strength": 1.0, "layers": [32]}],
... )
"""
from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# A steering intervention: add ``strength * vector`` at every listed layer's output.
Intervention = dict[str, Any]  # {"vector": list[float], "strength": float, "layers": list[int]}


def _from_pretrained(hf_id: str, dtype: torch.dtype):
    """Load a causal LM, tolerating the transformers 4.x/5.x dtype-kwarg rename.

    transformers <5 takes ``torch_dtype=``; transformers >=5 renamed it to ``dtype=``
    and dropped the old name. Try the new kwarg first, fall back to the old one.
    """
    try:
        return AutoModelForCausalLM.from_pretrained(hf_id, dtype=dtype, trust_remote_code=True)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype, trust_remote_code=True)


def _locate_decoder_stack(model) -> tuple[torch.nn.ModuleList, int]:
    """Return (decoder-layer ModuleList, hidden_size), robust to a few wrappers.

    The standard Llama/Qwen/OLMo layout is ``model.model.layers`` with
    ``model.config.hidden_size``. Some multimodal wrappers nest the text stack (e.g.
    Gemma-3 puts it under ``model.model.language_model`` with the size in
    ``config.text_config``); we walk a couple of known fallbacks before giving up.
    """
    cfg = model.config
    hidden = getattr(cfg, "hidden_size", None) or getattr(getattr(cfg, "text_config", None), "hidden_size", None)
    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(obj, torch.nn.ModuleList):
            return obj, int(hidden)
    raise AttributeError("could not locate the decoder layer stack on this model")


class SteerModel:
    """A frozen HF causal LM with residual-stream steering and read capture."""

    def __init__(self, hf_id: str, dtype: torch.dtype = torch.bfloat16, device: str = "cuda"):
        self.hf_id = hf_id
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        self.model = _from_pretrained(hf_id, dtype).to(device)
        self.model.eval()
        self.layers, self.hidden = _locate_decoder_stack(self.model)

    # -- chat templating -----------------------------------------------------
    def render(self, messages, tools=None, enable_thinking: bool = False) -> str:
        """Apply the model's chat template, tolerating templates without a thinking flag."""
        try:
            return self.tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
                enable_thinking=enable_thinking, tools=tools)
        except TypeError:
            return self.tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, tools=tools)

    # -- write hook (conditioning) ------------------------------------------
    @contextlib.contextmanager
    def steer(self, interventions: list[Intervention] | None):
        """Install forward hooks that add ``strength*vector`` at each layer's output.

        Multiple interventions hitting the same layer are summed. A no-op when
        ``interventions`` is falsy.
        """
        handles = []
        per_layer: dict[int, torch.Tensor] = {}
        dev = self.model.device
        for iv in interventions or []:
            v = torch.tensor(iv["vector"], dtype=torch.float32, device=dev) * float(iv["strength"])
            for layer in iv["layers"]:
                per_layer[layer] = per_layer.get(layer, torch.zeros(self.hidden, device=dev)) + v
        try:
            for layer, add in per_layer.items():
                add_b = add.to(next(self.model.parameters()).dtype)

                def hook(mod, inp, out, _add=add_b):
                    if isinstance(out, tuple):
                        return (out[0] + _add,) + tuple(out[1:])
                    return out + _add

                handles.append(self.layers[layer].register_forward_hook(hook))
            yield
        finally:
            for h in handles:
                h.remove()

    # -- read hook (white-box test) -----------------------------------------
    @contextlib.contextmanager
    def _capture(self, layer: int, store: dict):
        """Read-only hook: stash the residual at the OUTPUT of decoder ``layer``.

        Captures ``out[0]`` for all positions, detached to fp32 on CPU. The output of
        decoder layer ``L`` equals ``output_hidden_states[L+1]`` — the same residual
        point where the reference axes are defined by the vector builder.
        """
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            store["resid"] = h[0].detach().float().cpu().numpy()  # [seq, hidden]

        handle = self.layers[layer].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @torch.no_grad()
    def generate(self, messages, interventions: list[Intervention] | None = None,
                 temperature: float = 0.7, max_tokens: int = 200, seed: int | None = None,
                 enable_thinking: bool = False, tools=None) -> str:
        """Generate an assistant turn, optionally with steering active."""
        if seed is not None:
            torch.manual_seed(seed)
        prompt = self.render(messages, tools=tools, enable_thinking=enable_thinking)
        enc = self.tok(prompt, return_tensors="pt").to(self.model.device)
        plen = enc["input_ids"].shape[1]
        gen_kwargs: dict[str, Any] = dict(max_new_tokens=max_tokens, pad_token_id=self.tok.eos_token_id)
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            gen_kwargs.update(do_sample=False)
        ctx = self.steer(interventions) if interventions else contextlib.nullcontext()
        with ctx:
            out = self.model.generate(**enc, **gen_kwargs)
        return self.tok.decode(out[0, plen:], skip_special_tokens=True).strip()

    @torch.no_grad()
    def capture_forward(self, messages, layer: int, interventions: list[Intervention] | None = None,
                        enable_thinking: bool = False, tools=None):
        """One forward pass over templated ``messages``; return (resid, offsets, prompt).

        ``resid`` is the [seq, hidden] residual at the output of decoder ``layer``.
        The real white-box read passes ``interventions=None`` (steering OFF), so there
        is no injection in what is captured.
        """
        prompt = self.render(messages, tools=tools, enable_thinking=enable_thinking)
        enc = self.tok(prompt, return_tensors="pt", return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        store: dict = {}
        ctx = self.steer(interventions) if interventions else contextlib.nullcontext()
        with ctx, self._capture(layer, store):
            self.model(**enc)
        return store["resid"], offsets, prompt

    @torch.no_grad()
    def option_logprobs(self, messages, options: list[str], enable_thinking: bool = False,
                        tools=None) -> list[dict]:
        """Teacher-forced log-probability of each candidate reply string in ``options``.

        For a fixed prompt (``messages`` rendered with the generation prompt appended),
        score how likely the model is to *emit* each option string as its assistant reply.
        This is the primitive behind the forced-choice preference MARGIN: the log-prob a
        model assigns to committing to option A vs option B at the decision point, a
        continuous strength rather than a discrete pick.

        Returns one dict per option with ``logprob_sum`` (total log-prob of the option's
        token span, in nats), ``logprob_mean`` (per-token), ``first_token_logprob`` (the
        first option token after the prompt), ``n_tokens``, and the option ``text``.

        Retokenization-robust: the option's token span is found as the divergence point
        from the shared prompt token prefix, so a byte-pair merge at the prompt/answer
        boundary is attributed to the option rather than silently mis-scored. Because the
        prompt is identical across options, the shared-prefix tokens contribute equally and
        cancel in any A−B margin.
        """
        prompt = self.render(messages, tools=tools, enable_thinking=enable_thinking)
        base_ids = self.tok(prompt, return_tensors="pt").input_ids[0].tolist()
        dev = self.model.device
        out: list[dict] = []
        for opt in options:
            full = self.tok(prompt + opt, return_tensors="pt").input_ids[0].tolist()
            # shared-prefix length with the bare prompt (handles boundary merges)
            i = 0
            while i < len(base_ids) and i < len(full) and base_ids[i] == full[i]:
                i += 1
            opt_ids = full[i:]
            if not opt_ids:  # option adds no new tokens (degenerate); score last prompt token
                i = len(full) - 1
                opt_ids = full[i:]
            ids = torch.tensor([full], device=dev)
            logits = self.model(ids).logits[0].float()          # [seq, vocab]
            logp = torch.log_softmax(logits[i - 1:len(full) - 1, :], dim=-1)
            tok_lp = logp[torch.arange(len(opt_ids)), torch.tensor(opt_ids, device=dev)]
            s = float(tok_lp.sum().cpu())
            out.append({
                "text": opt, "n_tokens": int(len(opt_ids)),
                "logprob_sum": s, "logprob_mean": float(s / max(1, len(opt_ids))),
                "first_token_logprob": float(tok_lp[0].cpu()),
            })
        return out

    def project_cue(self, messages, cue_text: str, layer: int,
                    axes: dict[str, np.ndarray], interventions: list[Intervention] | None = None):
        """Project the residual over the LAST occurrence of ``cue_text`` onto each axis.

        Returns ``{axis_name: {"window_mean": float, "last_token": float}, ...}`` plus
        ``n_window_tokens`` and ``seq_len``. ``window_mean`` averages the residual over
        the cue's token span then dots with the axis; ``last_token`` uses the final cue
        token only.
        """
        resid, offsets, prompt = self.capture_forward(messages, layer, interventions=interventions)
        cstart = prompt.rfind(cue_text)
        if cstart < 0:
            raise ValueError(f"cue_text not found in rendered prompt: {cue_text!r}")
        cend = cstart + len(cue_text)
        win = [i for i, (a, b) in enumerate(offsets) if b > a and a >= cstart and b <= cend]
        if not win:  # fallback: any token overlapping the span
            win = [i for i, (a, b) in enumerate(offsets) if b > a and a < cend and b > cstart]
        if not win:
            raise ValueError("no tokens mapped to cue window")
        wmean = resid[win].mean(0)
        last = resid[win[-1]]
        out: dict[str, Any] = {"n_window_tokens": len(win), "seq_len": int(resid.shape[0])}
        for name, ax in axes.items():
            out[name] = {"window_mean": float(wmean @ ax), "last_token": float(last @ ax)}
        return out

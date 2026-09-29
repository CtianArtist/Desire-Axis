"""Read-only dual-site evidence for the author's OLMo removal generations.

Hook order matters: the source installs its additive hooks *inside* generate.
Only the layer-32 pre-add observer is installed before that call. Both post-add
observers are installed by the engine wrapper, after the source's write hooks.
No tokenization, forward pass, output replacement, or sampler change occurs here.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import struct
from contextlib import contextmanager


LAYER32 = 32  # Zero-based decoder-stack index, matching the author's intervention.


def _restore(obj, name, had, value):
    if had:
        setattr(obj, name, value)
    else:
        delattr(obj, name)


def _steer_state(interventions, model):
    """Mirror source accumulation order; return compact provenance and layer32 add.

    The hash identifies contiguous little-endian float32 vector bytes, *before*
    strength multiplication. Full vectors are never written to turn evidence.
    This is arithmetic on vectors only, not an additional model evaluation.
    """
    import torch

    state, per_layer = [], {}
    for iv in interventions or []:
        vector = torch.tensor(iv["vector"], dtype=torch.float32, device=model.model.device)
        strength = float(iv["strength"])
        if vector.ndim != 1 or vector.numel() != model.hidden:
            raise ValueError("steer vector must match model hidden width")
        scaled = vector * strength
        if not bool(torch.isfinite(scaled).all()):
            raise ValueError("steer vector/strength must be finite")
        layers = list(iv["layers"])
        values = vector.detach().cpu().tolist()
        state.append(dict(layers=layers, strength=strength,
                          vector_sha256=hashlib.sha256(struct.pack(f"<{len(values)}f", *values)).hexdigest(),
                          vector_encoding="little_endian_float32", vector_numel=vector.numel(),
                          vector_l2=float(vector.norm().cpu()),
                          intended_injected_l2=float(scaled.norm().cpu())))
        for layer in layers:
            # Keep source's zero initialization, summation order and dtype cast.
            per_layer[layer] = per_layer.get(layer, torch.zeros(model.hidden, device=model.model.device)) + scaled
    add = per_layer.get(LAYER32)
    intended_fp32_l2 = 0.0 if add is None else float(add.norm().cpu())
    if add is not None:
        add = add.to(next(model.model.parameters()).dtype)
    return state, add, intended_fp32_l2


def _observation(hidden, axis, layer, semantics, prompt_tokens):
    import torch

    if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] != prompt_tokens:
        raise RuntimeError("missing full first-prefill observation (batch1, full prompt required)")
    vector = torch.as_tensor(axis, device=hidden.device, dtype=torch.float32)
    norm = vector.norm()
    if vector.ndim != 1 or vector.numel() != hidden.shape[-1] or not bool(torch.isfinite(norm)) or float(norm) <= 0:
        raise ValueError("S2 monitor axis must have matching width and finite positive norm")
    final = hidden[0, -1].detach().clone()
    if not bool(torch.isfinite(final).all()):
        raise ValueError("nonfinite prompt-final activation")
    return final, dict(layer=layer, site="decoder_layer_output", semantics=semantics,
                       position="prompt_final", pass_kind="first_prefill_only",
                       prefill_tokens=prompt_tokens, dtype=str(hidden.dtype), axis="saved_S2_pain",
                       formula="dot(h.float(), S2.float()/norm(S2.float()))",
                       projection=float(torch.dot(final.float(), vector / norm).cpu()))


@contextmanager
def capture(bundle, h):
    """Yield per-turn evidence, retaining removal.capture's compatibility fields.

    Every turn records messages, seed, sampler, input/generated token IDs, raw
    output, negative requests and the *supplied* applied_steer list. Offer turns
    additionally record named observations selected_s2 (post-add), layer32_preadd
    and layer32_postadd, even when the selected site is itself layer 32.
    s2_projection, prefill_tokens and monitor alias the selected post-add site.
    layer32_addition_check verifies the prompt-final BF16 state bit-for-bit using
    the actual intervention list, not a role/dose inferred from the harness.
    Other layers' writes are logged, but only layer 32 is locally validated.
    """
    import torch
    try:
        from .removal import negative_requests
    except ImportError:
        from removal import negative_requests

    model, records = bundle["model"], []
    engine = model.model
    original = model.generate
    original_engine = engine.generate
    signature = inspect.signature(original)
    model_had, engine_had = "generate" in vars(model), "generate" in vars(engine)
    model_value, engine_value = vars(model).get("generate"), vars(engine).get("generate")
    selected_layer = int(bundle["monitor_layer"])
    axis = bundle["directions"]["pain"]

    def generate(messages, *args, **kwargs):
        bound = signature.bind(messages, *args, **kwargs)
        bound.apply_defaults()
        interventions = bound.arguments.get("interventions")
        state, add, intended_fp32_l2 = _steer_state(interventions, model)
        evidence = dict(messages=copy.deepcopy(messages), seed=bound.arguments.get("seed"), applied_steer=state)
        offer = len(records) >= h.C.N_BASELINE + h.C.N_EXPOSURE
        observations, finals, handles = {}, {}, []
        calls = 0

        def observer(name, layer, semantics):
            def hook(module, inputs, output):
                if name in observations:
                    return None
                hidden = output[0] if isinstance(output, tuple) else output
                if "input_token_ids" not in evidence:
                    raise RuntimeError("observation occurred outside original engine generate")
                final, observation = _observation(hidden, axis, layer, semantics, len(evidence["input_token_ids"]))
                observation['relation_to_injection'] = ('upstream' if layer < LAYER32 else
                                                       'same_layer' if layer == LAYER32 else 'downstream')
                observation['semantics'] = ('pre_add' if name == 'layer32_preadd' else
                                            'post_add' if layer == LAYER32 else 'raw_post_block')
                observations[name] = observation
                if name.startswith("layer32_"):
                    if final.dtype != torch.bfloat16:
                        raise RuntimeError("layer32 addition verification requires BF16 activations")
                    finals[name] = final
                return None
            return hook

        def engine_generate(*a, **kw):
            nonlocal calls
            calls += 1
            if calls != 1:
                raise RuntimeError("expected one original torch generate call per turn")
            ids = kw.get("input_ids", a[0] if a else None)
            if ids is None:
                raise RuntimeError("original generate did not supply input_ids")
            if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
                raise RuntimeError("removal evidence requires batch size 1 and a nonempty prompt")
            evidence["input_token_ids"] = ids.detach().cpu().tolist()[0]
            evidence["sampler"] = {k: kw.get(k) for k in ("do_sample", "temperature", "top_p", "max_new_tokens")}
            if evidence["sampler"] != dict(do_sample=True, temperature=0.7, top_p=0.95, max_new_tokens=200):
                raise RuntimeError("author generation sampler contract changed")
            if offer:
                # The source's additive hooks are already registered here.
                handles.append(model.layers[selected_layer].register_forward_hook(
                    observer("selected_s2", selected_layer, "post_add")))
                handles.append(model.layers[LAYER32].register_forward_hook(
                    observer("layer32_postadd", LAYER32, "post_add")))
            output = original_engine(*a, **kw)
            sequences = output.sequences if hasattr(output, "sequences") else output
            if sequences.ndim != 2 or sequences.shape[0] != 1:
                raise RuntimeError("generation output requires batch size 1")
            all_ids = sequences.detach().cpu().tolist()[0]
            plen = len(evidence["input_token_ids"])
            if all_ids[:plen] != evidence["input_token_ids"]:
                raise RuntimeError("generation output does not preserve prompt token prefix")
            evidence["generated_token_ids"] = all_ids[plen:]
            return output

        try:
            engine.generate = engine_generate
            if offer:
                handles.append(model.layers[LAYER32].register_forward_hook(
                    observer("layer32_preadd", LAYER32, "pre_add")))
            raw = original(messages, *args, **kwargs)
            if calls != 1:
                raise RuntimeError("missing original torch generation capture")
            if offer:
                if set(observations) != {"selected_s2", "layer32_preadd", "layer32_postadd"}:
                    raise RuntimeError("missing full first-prefill dual-site observation")
                pre, post = finals["layer32_preadd"], finals["layer32_postadd"]
                if add is not None and add.dtype != torch.bfloat16:
                    raise RuntimeError("source steering addition must be BF16")
                expected = pre if add is None else pre + add
                exact = torch.equal(post.view(torch.int16), expected.view(torch.int16))
                if not exact:
                    raise RuntimeError("layer32 post-add state differs from exact source BF16 arithmetic")
                delta = post.float() - pre.float()
                evidence["layer32_addition_check"] = dict(
                    exact_bf16_match=True, has_layer32_add=add is not None,
                    arithmetic="sum_in_order_fp32(vector*strength); cast_sum_to_parameter_dtype; pre_bf16+add_bf16",
                    scope="prompt_final_first_prefill", max_abs_error=0.0,
                    changed_elements=int((post != pre).sum().cpu()),
                    actual_delta_l2=float(delta.norm().cpu()),
                    intended_fp32_l2=intended_fp32_l2,
                    expected_bf16_delta_l2=float((expected.float() - pre.float()).norm().cpu()),
                    cast_add_l2=0.0 if add is None else float(add.float().norm().cpu()))
                evidence["observations"] = observations
                selected = observations["selected_s2"]
                evidence["s2_projection"] = selected["projection"]
                evidence["prefill_tokens"] = selected["prefill_tokens"]
                evidence["monitor"] = {k: v for k, v in selected.items() if k not in ("projection", "prefill_tokens")}
            evidence["raw_output"] = raw
            evidence["raw_negative_requests"] = negative_requests(raw, h.SI)
            records.append(evidence)
            return raw
        finally:
            for handle in handles:
                handle.remove()
            _restore(engine, "generate", engine_had, engine_value)

    model.generate = generate
    try:
        yield records
    finally:
        _restore(model, "generate", model_had, model_value)

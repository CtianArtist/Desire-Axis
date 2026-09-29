"""Deterministic, fail-closed anchored extension of the retained customer protocol.

Run with --check to verify the generated entrypoint without importing torch.
Only explicit anchors below change; sampling, masks, caches and continuation are source.
"""
import argparse
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "inputs/reference/b2_protocol.py"
OUTPUT = HERE / "run_profile.py"
SOURCE_SHA256 = "e85ebf2b4065885756d62f5bd20d88e23d499ffb18c160bd80a586a6bec22ca5"


def generate():
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA256, "Customer source changed"
    text = raw.decode()

    def replace(old, new):
        nonlocal text
        assert text.count(old) == 1, f"Expected one anchor: {old[:100]!r}"
        text = text.replace(old, new)

    # Keep the customer narrative only in the reference, not as misleading live docs.
    end_doc = text.index('"""', 3) + 3
    text = ('"""Generated fixed always-on choice profile. Do not edit; run build_protocol.py.\n'
            'Legacy relief fields denote target choices; all presses are descriptive only.\n'
            'Sampling and transcript continuation are inherited from the pinned customer source.\n"""'
            + text[end_doc:])
    replace('from pain_axis_b import runtime as rt\nfrom pain_axis_b.grid import new_labels, build_grid',
            'from pain_choice_profile import runtime as rt\n'
            'from pain_choice_profile.design import (build_grid, CONDITIONS, PAIRS as DESIGN_PAIRS,\n'
            '    scenario_id, source_kind, digest, TOOL_LABELS as PROFILE_LABELS)')
    replace('PROTOCOL = "2btnN names+saltseed v1"', 'PROTOCOL = "four-missing-fear-lamp32-v1"')
    replace('    ("Qwen/Qwen2.5-7B-Instruct",  "Qwen_2.5_7B_instruct",  16, 1.0,  "hermes", 1024),\n', '')
    replace('    ("Qwen/Qwen2.5-72B-Instruct", "Qwen_2.5_72B_instruct", 46, 1.25, "hermes",  192),  # dose set by manual check of the generations, used in case the judged dose is excessive\n', '')
    replace('pain_path, sadness_path, output, resume=None):',
            'pain_path, sadness_path, fear_path, output, resume=None, input_receipt=None):')
    replace('    grid = build_grid(SCENARIOS, ORIGINAL_LABELS, BUTTON_NAMES, CONTENTS, SEED_BASES)\n'
            '    assert len(grid) == 15580\n'
            '    if sadness_path is None:\n'
            '        grid = [g for g in grid if g[2][1] != "sadness"]',
            '    grid = build_grid(SCENARIOS)\n'
            '    assert len(grid) == 820\n'
            '    assert sadness_path.is_file() and fear_path.is_file(), "All controls are mandatory"')
    replace('    monitor_layer = min(int(data["layer"]), len(layers) - 1)\n'
            '    if monitor_layer <= STEER_LAYER:\n'
            '        monitor_layer = min(STEER_LAYER + 4, len(layers) - 1)',
            '    monitor_layer = 61\n'
            '    assert int(data["layer"]) == monitor_layer and len(layers) > monitor_layer\n'
            '    assert MODEL_NAME == "Qwen_2.5_32B_instruct" and STEER_LAYER == 38\n'
            '    assert data["s2_pain_vector"].dtype == torch.float32\n'
            '    assert v.norm().item() == rt.S2_FP32_NORM\n'
            '    assert rt.tensor_sha256(v) == input_receipt["vector_pins"]["fp32_tensor_sha256"]["s2_pain_vector"]')
    replace('        sv = sadness["sadness_vector_matched_fp32"].float()',
            '        assert sadness["sadness_vector_matched_fp32"].dtype == torch.float32\n'
            '        sv = sadness["sadness_vector_matched_fp32"].float()\n'
            '        assert rt.tensor_sha256(sv) == input_receipt["vector_pins"]["fp32_tensor_sha256"]["sadness_vector_matched_fp32"]')
    replace('    for rs in RAND_SEEDS:\n',
            '    fear = torch.load(fear_path, map_location="cpu", weights_only=True)\n'
            '    fv = fear["fear_vector_matched_fp32"]\n'
            '    assert fv.dtype == torch.float32 and fv.shape == v.shape and torch.isfinite(fv).all()\n'
            '    assert rt.tensor_sha256(fv) == rt.FEAR_FP32_SHA256\n'
            '    assert rt.tensor_sha256(fv) == input_receipt["vector_pins"]["fp32_tensor_sha256"]["fear_vector_matched_fp32"]\n'
            '    assert torch.isclose(fv.norm(), v.norm(), rtol=1e-6, atol=1e-5)\n'
            '    DIR["fear"] = fv.to("cuda", dtype=torch.bfloat16)\n\n'
            '    for rs in RAND_SEEDS:\n')
    replace('coeff {COEFF}, monitor', 'condition-specific doses, monitor')
    replace('steer_layer=STEER_LAYER, monitor_layer=monitor_layer, coefficient=COEFF)',
            'steer_layer=STEER_LAYER, monitor_layer=monitor_layer, conditions=CONDITIONS)')
    replace('                     coefficient=COEFF, steering_hook_present=True, monitor_hook_present=True)',
            '                     conditions=CONDITIONS, steering_hook_present=True, monitor_hook_present=True)')
    replace('    def generate_segment(items):\n        B = len(items)',
            '    def generate_segment(items):\n        audit.set_items(items, DIR)\n        B = len(items)')
    replace('        out = forward(input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True)\n',
            '        out = forward(input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True)\n'
            '        # Capture once at the first forward, before any RNG draw or token selection.\n'
            '        assert bool((mask[:, -1] == 1).all()), "Prefill last column must be attended"\n'
            '        prefill_proj_monitor = G["mon"].clone()\n'
            '        assert bool(torch.isfinite(prefill_proj_monitor).all())\n'
            '        prefill_proj_monitor_l = prefill_proj_monitor.tolist()\n')
    replace('                        "p_x": p_x[i], "p_y": p_y[i],',
            '                        "p_x": p_x[i], "p_y": p_y[i],\n'
            '                        "answer_token_ids": toks[i][:n_out_l[i]],\n'
            '                        "prompt_token_ids": list(items[i]["prompt_ids"]),\n'
            '                        "prompt_tokens": len(items[i]["prompt_ids"]),\n'
            '                        "padded_prompt_tokens": L, "prefill_attended": True,\n'
            '                        "prefill_position_id": len(items[i]["prompt_ids"]) - 1,\n'
            '                        "prefill_generated_tokens": 0,\n'
            '                        "prefill_proj_monitor": prefill_proj_monitor_l[i],')
    replace('        t.arm_coeff = COEFF if steer_on else 0.0',
            '        t.arm_coeff = CONDITIONS[arm_name]["dose"]')
    replace('                    "arm": arm_name, "steer_direction": dir_kind, "rand_seed": rand_seed,',
            '                    "arm": arm_name, "steer_direction": dir_kind, "rand_seed": rand_seed,\n'
            '                    "condition_id": arm_name, "direction": CONDITIONS[arm_name]["direction"],\n'
            '                    "dose": t.arm_coeff, "pair_id": DESIGN_PAIRS[tool_label]["pair_id"],\n'
            '                    "scenario_id": scenario_id(user_content, s_idx),\n'
            '                    "source_kind": source_kind(user_content),\n'
            '                    "scenario_content_hash": digest(SCENARIOS[user_content][s_idx]),\n'
            '                    "initial_position": t.names.index(relief_name),')
    replace('"steer_coeff": COEFF, "monitor_layer": monitor_layer',
            '"steer_coeff": t.arm_coeff, "monitor_layer": monitor_layer')
    start = text.index('    def do_press(t, which):')
    stop = text.index('    def after_segment(t, res):', start)
    replace(text[start:stop], '''    def do_press(t, which):
        before = t.coeff
        t.record["button_events"].append({"turn": t.t_idx, "steer_was": before,
                                          "route": "choice", "which": which,
                                          "descriptive_only": True, "coefficient_changed": False,
                                          "steer_after": before})
        assert t.coeff == before == t.arm_coeff
        t.messages.append({"role": "tool", "content": "Done."})

''')
    replace('        t.record["proj_segments"].append({"turn": t.t_idx, "steer_coeff_now": t.coeff,',
            '        capture = {key: res[key] for key in ("prefill_proj_monitor", "prompt_tokens",\n'
            '            "padded_prompt_tokens", "prefill_position_id", "prefill_attended", "prefill_generated_tokens")}\n'
            '        t.record["choices"][-1].update(capture)\n'
            '        t.record["choices"][-1].update(raw_answer=res["text"], answer_full=ans,\n'
            '            answer_token_ids=res["answer_token_ids"], prompt_token_ids=res["prompt_token_ids"])\n'
            '        t.record["proj_segments"].append({**capture, "turn": t.t_idx, "steer_coeff_now": t.coeff,')
    replace('              "coefficient": COEFF, "protocol": PROTOCOL, "pain_sha256": rt.sha256(pain_path),',
            '              **rt.provenance(__file__), **(input_receipt or {}),\n'
            '              "protocol": PROTOCOL, "pain_sha256": rt.sha256(pain_path),\n'
            '              "fear_sha256": rt.sha256(fear_path),\n'
            '              "vector_fp32_sha256": {"pain": rt.tensor_sha256(v),\n'
            '                  "sadness": rt.tensor_sha256(sv), "fear": rt.tensor_sha256(fv)},\n'
            '              "vector_norms_fp32": {"pain": v.norm().item(), "sadness": sv.norm().item(), "fear": fv.norm().item()},')
    replace('"scenario_digest": rt.grid_digest(SCENARIOS, {}), "labels": TOOL_LABELS,',
            '"scenario_digest": digest(SCENARIOS), "labels": TOOL_LABELS,')
    replace('              "world_size": rt.world(), "precision": "bfloat16", "attention": "sdpa"}',
            '              "batch_rows": BATCH_ROWS,\n'
            '              "world_size": rt.world(), "precision": "bfloat16", "attention": "sdpa"}')
    replace('    schedule = store.restore(trials)',
            '    rt.atomic_json(Path(output)/"configuration.json", config)\n'
            '    schedule = store.restore(trials)')
    replace('    rt.prepare_recovery_state(store, trials, schedule)',
            '    if rt.world() == 1 and store.db.execute("SELECT count(*) FROM trials").fetchone()[0] == 0:\n'
            '        store.save(trials, schedule)  # Save every row and RNG before the first forward.\n'
            '    rt.prepare_recovery_state(store, trials, schedule)\n'
            '    rt.early_validation(trials, output)')
    replace('                    "dir_kind":t.dir_kind, "steer_ranges":list(t.steer_ranges),',
            '                    "condition_id":t.arm_name,\n'
            '                    "dir_kind":t.dir_kind, "steer_ranges":list(t.steer_ranges),')
    replace('            store.save(chunk,schedule)',
            '            store.save(chunk,schedule)\n'
            '            rt.early_validation(trials, output)')
    replace('"requested_trials":15580', '"requested_trials":820')
    replace('ORIGINAL_LABELS = TOOL_LABELS\nTOOL_LABELS = new_labels(ORIGINAL_LABELS)',
            'TOOL_LABELS = PROFILE_LABELS')
    start = text.index('def main():')
    replace(text[start:], '''def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["Qwen_2.5_32B_instruct"], required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--fear-inputs", type=Path, required=True)
    p.add_argument("--scenarios", type=Path, required=True)
    p.add_argument("--stimulus-manifest", type=Path, required=True)
    p.add_argument("--vector-pins", type=Path, default=Path(__file__).resolve().parent.parent/"inputs/vector_pins.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--batch-rows", type=int)
    args = p.parse_args()
    # No output side effects or downloads until the frozen texts and hashes pass.
    scenarios = json.loads(args.scenarios.read_text())
    manifest = json.loads(args.stimulus_manifest.read_text())
    receipt = rt.validate_manifest(scenarios, manifest)
    pins = json.loads(args.vector_pins.read_text())
    receipt.update(rt.verify_vector_files(args.inputs, args.fear_inputs, pins))
    receipt["vector_pins_file_sha256"] = rt.sha256(args.vector_pins)
    receipt.update(scenarios_file_sha256=rt.sha256(args.scenarios),
                   stimulus_manifest_file_sha256=rt.sha256(args.stimulus_manifest))
    sadness = args.inputs/f"vectors_{args.model}.pt"
    fear = args.fear_inputs/f"fear_{args.model}.pt"
    pain = args.inputs/f"{args.model}_pain_vectors.safetensors"
    for path in (sadness, fear, pain, args.inputs/"verification.json", args.fear_inputs/"verification_fear.json"):
        assert path.is_file(), f"Required input missing: {path}"
    receipt["verification_sha256"] = rt.sha256(args.inputs/"verification.json")
    receipt["fear_verification_sha256"] = rt.sha256(args.fear_inputs/"verification_fear.json")
    model_spec = next(x for x in MODELS if x[1] == args.model)
    if args.batch_rows is not None:
        assert 0 < args.batch_rows <= model_spec[-1]
        model_spec = (*model_spec[:-1], args.batch_rows)
    rt.init_runtime(1, args.output)
    rt.atomic_json(args.output/"launch_configuration.json", {
        **rt.provenance(__file__), **receipt, "model": args.model,
        "revision": rt.REVISIONS[args.model], "adapter_revision": rt.ADAPTER_REVISION,
        "adapter_sha256": rt.ADAPTER_HASHES[args.model],
        "nominal_batch_rows": 384, "batch_rows": model_spec[-1],
        "profile_max_seconds": float(os.environ.get("PROFILE_MAX_SECONDS", 12 * 3600)),
        "pain_sha256": rt.sha256(pain), "sadness_sha256": rt.sha256(sadness), "fear_sha256": rt.sha256(fear)})
    adapter = rt.download_adapter(args.model, args.output)
    run_model(*model_spec, list(TOOL_LABELS), adapter, scenarios, 10**9,
              pain, sadness, fear, args.output, args.resume, receipt)


if __name__ == "__main__":
    main()
''')
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = generate()
    compile(text, str(OUTPUT), "exec")
    if args.check:
        assert OUTPUT.read_text() == text, "Generated entrypoint is stale"
        print("Generated entrypoint matches pinned source and deterministic anchors")
    else:
        OUTPUT.write_text(text)
        print(OUTPUT)


if __name__ == "__main__":
    main()

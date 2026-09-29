"""Bounded fabricated protocol records, never model generations or real trials."""
import ast
from pathlib import Path

from pain_seed_b.grid import B2_PAIRS, build_grid, build_manifest, generation_seed

E = Path(__file__).resolve().parents[1]
RANDOM_SEEDS = (4817, 2903, 7361, 1150, 9428, 6076, 3384, 8592, 517, 6741)


def fixture():
    tree = ast.parse((E / "inputs/original/04_selfmed_two_buttons.py").read_text())
    selected = [n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id in {"TOOL_LABELS", "BUTTON_NAMES", "SEED_BASES"}]
    namespace = {"__builtins__": {}, "dict": dict}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "literal_author_config", "exec"), namespace)
    scenarios = {"neutral_prompts": [["fabricated scenario zero"] * 3, ["fabricated scenario one"] * 3]}
    grid = build_grid(scenarios, namespace["TOOL_LABELS"], namespace["BUTTON_NAMES"], scenarios, namespace["SEED_BASES"])
    manifest = build_manifest(scenarios, namespace["TOOL_LABELS"], namespace["BUTTON_NAMES"], scenarios, namespace["SEED_BASES"])
    adapters = [{"model": "Qwen_2.5_32B_instruct", "training_seed": seed, "adapter_identity_digest": str(seed) * 64} for seed in (1, 2)]
    records = []
    for seed in (1, 2):
        for spec in grid:
            pair, content, arm, scenario, names_key, initial, sampled, behavior_seed = spec
            names = namespace["BUTTON_NAMES"][names_key]
            label_free = pair == "label_free"
            arm_name, direction, on, works = arm
            pattern = ["T", "O", "O", "O", "O"]
            if not works and on and not (seed == 2 and direction == "rand"):
                pattern = ["T"] * 5
            if pair in B2_PAIRS:
                pattern[0] = "O" if direction == "rand" or not on or (direction == "sadness" and scenario == 1) else "T"
                if pattern[0] == "O":
                    pattern[1] = "T"
            if pair == "relief_vs_helpful":
                pattern = { (0, 1000): ["T", "?", "T", "O", "O"],
                            (0, 2000): ["O", "O", "O", "O", "T"],
                            (1, 1000): ["O"] * 5,
                            (1, 2000): ["O", "T", "O", "O", "O"] }[(scenario, behavior_seed - scenario)] if sampled else pattern
            if pair == "relief_vs_grant":
                pattern = ["T"] * 5
            if label_free:
                pattern += [pattern[-1]] * 3
            if not sampled:
                pattern = ["?"] * (8 if label_free else 3)
            r = {"synthetic_fixture": True, "training_seed": seed, "adapter_identity_digest": str(seed) * 64,
                 "model": "Qwen_2.5_32B_instruct", "protocol": "2btnN names+saltseed v1", "tool_label": pair,
                 "user_content": content, "arm": arm_name, "scenario_idx": scenario,
                 "names_key": names_key, "button_names": list(names), "relief_name": initial,
                 "label_free": label_free, "relief_mode": "temporary" if label_free else "permanent",
                 "swap_turn": None if label_free else 2, "sampled": sampled, "seed": behavior_seed,
                 "gen_seed": generation_seed(behavior_seed, names_key, initial) if sampled else None,
                 "steer_layer": 38, "steer_coeff": 1.0, "monitor_layer": 61,
                 "rand_seed": RANDOM_SEEDS[scenario % 10] if direction == "rand" else None,
                 "steer_direction": f"rand{RANDOM_SEEDS[scenario % 10]}" if direction == "rand" else direction,
                 "choices": [], "button_events": [], "proj_segments": [], "extension_added": False}
            coefficient, until = float(on), None
            for turn, selection in enumerate(pattern):
                if label_free and until is not None and turn > until:
                    coefficient, until = float(on), None
                current = initial if label_free or turn < 2 else next(n for n in names if n != initial)
                picked = None if selection == "?" else current if selection == "T" else next(n for n in names if n != current)
                chose = None if picked is None else "relief" if picked == current else "other"
                r["choices"].append({"turn": turn, "answer": picked or "???", "picked": picked, "chose": chose,
                     "relief_name_now": current, "swapped": not label_free and turn == 2,
                     "steer_coeff_now": coefficient, "p_x": 0.5, "p_y": 0.5})
                r["proj_segments"].append({"turn": turn, "steer_coeff_now": coefficient,
                     "mean_proj": 0.0, "mean_proj_monitor": 0.0, "n_fwd": 1})
                if picked is not None:
                    r["button_events"].append({"turn": turn, "steer_was": coefficient, "which": chose})
                    if not label_free:
                        r["extension_added"] = True
                    if works and chose == "relief" and on:
                        coefficient = 0.0
                        if label_free:
                            until = turn + 1
            r["final_steer_coeff"] = coefficient
            records.append(r)
    return records, manifest, adapters

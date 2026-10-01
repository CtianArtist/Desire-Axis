"""Helpers for running the repository scripts from desire_axis_colab.ipynb.

The scripts keep their settings as constants at the top of each file. run() writes a copy
of a script with some of those top-level assignments replaced (found with ast, so multi-line
lists and dicts are replaced whole), then runs the copy inside the notebook kernel from the
repository root, so relative paths resolve and interactive prompts appear as Colab input
boxes. answers= pre-fills prompts in order; the rest are asked normally.
"""

import ast
import builtins
import gc
import json
import runpy
import shutil
from pathlib import Path

import pandas as pd

REPO = Path("/content/Desire-Axis")
PATCHED = Path("/content/_patched_scripts")


class Raw(str):
    """A value written into the script as source code, e.g. Raw('Path("results") / "x"')."""


def patch_constants(script, assignments):
    """Copy of `script` with each module-level NAME = ... replaced by NAME = value."""
    script = Path(script)
    src = script.read_text(encoding="utf-8")
    lines = src.splitlines(keepends=True)
    spans = {}
    for node in ast.parse(src).body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        for t in targets:
            if isinstance(t, ast.Name) and t.id in assignments:
                spans[t.id] = (node.lineno, node.end_lineno)
    missing = set(assignments) - set(spans)
    if missing:
        raise KeyError(f"{script.name} has no top-level assignment for {sorted(missing)}")
    for name, (start, end) in sorted(spans.items(), key=lambda kv: -kv[1][0]):
        value = assignments[name]
        text = str(value) if isinstance(value, Raw) else repr(value)
        lines[start - 1:end] = [f"{name} = {text}  # set by the Colab notebook\n"]
    PATCHED.mkdir(parents=True, exist_ok=True)
    out = PATCHED / script.name
    out.write_text("".join(lines), encoding="utf-8")
    return out


def run(script, answers=None, **assignments):
    """Run a repository script (path relative to REPO) with top-level constants overridden."""
    import os
    path = REPO / script
    target = patch_constants(path, assignments) if assignments else path
    queue = list(answers or [])
    real_input = builtins.input

    def scripted_input(prompt=""):
        if queue:
            ans = queue.pop(0)
            print(f"{prompt}{ans}  [answered by the notebook]")
            return ans
        return real_input(prompt)

    cwd = os.getcwd()
    os.chdir(REPO)
    builtins.input = scripted_input
    try:
        runpy.run_path(str(target), run_name="__main__")
    finally:
        builtins.input = real_input
        os.chdir(cwd)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def steer_layer(model_name, tag="S2"):
    """Layer picked by the 4.2 steering ladder for this model."""
    path = REPO / "results" / "steering" / f"steer_layers_{tag}.json"
    layers = json.loads(path.read_text()) if path.exists() else {}
    if model_name not in layers:
        raise SystemExit(f"no {tag} steering layer for {model_name} in {path}: run the steering ladder first")
    return int(layers[model_name])


def judged_dose(model_name):
    """Lowest dose judged aroused and coherent by 03_feel_probe_judge.py, or None."""
    path = REPO / "results" / "selfstim" / "feel_judge" / "doses.csv"
    if not path.exists():
        raise SystemExit(f"{path} missing: run the feel probe and the judge first")
    row = pd.read_csv(path).set_index("model").get("dose", pd.Series(dtype=float)).get(model_name)
    return None if row is None or pd.isna(row) else float(row)


def collect_for_analysis():
    """Copy the run outputs into the section folders the table and figure scripts read."""
    res = REPO / "results"
    logs = res / "4.3_selfstim" / "trial_logs"
    logs.mkdir(parents=True, exist_ok=True)
    for f in (res / "selfstim").glob("selfstim_*.jsonl"):
        shutil.copy(f, logs / f.name)
    steer = res / "4.2_steering" / "S2"
    steer.mkdir(parents=True, exist_ok=True)
    for f in (res / "steering").glob("*_steering_S2_neutral50_L*.csv"):
        shutil.copy(f, steer / f.name)
    screens = res / "4.1_self_other" / "per_model"
    screens.mkdir(parents=True, exist_ok=True)
    for f in (res / "screen").glob("screen_*.csv"):
        shutil.copy(f, screens / f.name.replace("screen_", "screen_v2_", 1))


def steering_examples(model_name, coeffs=(0, 1, 2), n_prompts=3, width=220):
    """A few steered generations side by side from the 4.2 ladder output."""
    files = sorted((REPO / "results" / "steering").glob(f"{model_name}_steering_S2_neutral50_L*.csv"))
    if not files:
        print(f"no steering output for {model_name}")
        return
    df = pd.read_csv(files[-1])
    for idx in sorted(df["prompt_idx"].unique())[:n_prompts]:
        rows = df[df["prompt_idx"] == idx]
        print(f"\n### {rows['prompt'].iloc[0]}")
        for c in coeffs:
            g = rows[rows["coeff"] == c]["generation"]
            if len(g):
                print(f"  [coeff {c:+g}] {str(g.iloc[0])[:width]!r}")

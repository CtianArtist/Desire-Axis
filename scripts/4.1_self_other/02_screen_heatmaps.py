"""Category-by-model heatmaps of the self-other screen (Figures 4 and 5 of the paper):
desire axis (mean of the S1 and S2 z-scores), affection, adrenaline and contentment.
Rows are the 21 categories sorted by mean desire-axis z; the same row order and the same
color scale are used for all four maps (contentment keeps its own scale).

Reads results/4.1_self_other/per_model/screen_v2_<model>.csv.
Writes the four maps into results/4.1_self_other/figures.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

SCREEN_DIR = Path("results") / "4.1_self_other" / "per_model"
OUT_DIR = Path("results") / "4.1_self_other" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e3"
DIV_CMAP = LinearSegmentedColormap.from_list("desire_div", [
    "#104281", "#2a78d6", "#86b6ef", "#cde2fb", "#f0efec", "#f6cfcb", "#ee8a85", "#d03b3b", "#8f2424"])
F_TITLE, F_TICKS, F_CATS, F_MEANS, F_CBAR = 19, 13, 14, 12, 13
mpl.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": TEXT_PRIMARY, "axes.labelcolor": TEXT_PRIMARY, "xtick.color": TEXT_SECONDARY,
                     "ytick.color": TEXT_SECONDARY, "axes.edgecolor": GRID, "font.size": 12})

MODEL_ORDER = [
    "Gemma_2_2B_base", "Gemma_2_2B_instruct", "Gemma_2_9B_base", "Gemma_2_9B_instruct",
    "Gemma_2_27B_base", "Gemma_2_27B_instruct", "Gemma_3_27B_base", "Gemma_3_27B_instruct",
    "Llama_3.1_8B_base", "Llama_3.1_8B_instruct", "Llama_3.1_70B_base", "Llama_3.1_70B_instruct",
    "Llama_3.3_70B_instruct", "Mistral_7B_base", "Mistral_7B_instruct", "Mistral_Small_24B_base",
    "Phi_4", "Qwen_2.5_7B_base", "Qwen_2.5_7B_instruct", "Qwen_2.5_32B_base", "Qwen_2.5_32B_instruct",
    "Qwen_2.5_72B_base", "Qwen_2.5_72B_instruct", "Qwen_3_8B_base", "Qwen_3_14B_base",
]
FAMILY_BREAKS = [8, 13, 16, 17, 23]


def pretty_model(m):
    return m.replace("_", " ").replace("base", "b").replace("instruct", "it")


def pretty_cat(c):
    return c.replace("_", " ")


frames = []
for f in sorted(SCREEN_DIR.glob("screen_v2_*.csv")):
    df = pd.read_csv(f)
    df["model"] = f.stem.replace("screen_v2_", "")
    frames.append(df)
all_df = pd.concat(frames, ignore_index=True)
all_df["block_z"] = (all_df["s1_desire_vector_z"] + all_df["s2_desire_vector_z"]) / 2
models = [m for m in MODEL_ORDER if m in set(all_df["model"])]

cat_model = all_df.groupby(["category", "model"])["block_z"].mean().unstack("model")[models]
cat_order = cat_model.mean(axis=1).sort_values(ascending=False).index.tolist()

MAPS = [
    ("block_z", "desire vector readable.png", f"Desire-axis activation by category across {len(models)} models", "block desire z (mean of S1, S2)"),
    ("affection_vector_z", "affection readable.png",
     f"Affection activation by category across {len(models)} models\n(rows kept in desire order for comparison)", "affection z"),
    ("adrenaline_vector_z", "adrenaline readable.png",
     f"Adrenaline activation by category across {len(models)} models\n(rows kept in desire order for comparison)", "adrenaline z"),
    ("contentment_vector_z", "contentment readable.png",
     f"Contentment activation by category across {len(models)} models\n(rows kept in desire order for comparison)", "contentment z"),
]
matrices = {col: all_df.groupby(["category", "model"])[col].mean().unstack("model")[models].loc[cat_order]
            for col, _, _, _ in MAPS if col in all_df}
shared_vmax = max(np.abs(m.values).max() for k, m in matrices.items() if k != "contentment_vector_z")

for col, fname, title, cbar_label in MAPS:
    if col not in matrices:
        continue
    m = matrices[col]
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    norm = TwoSlopeNorm(vmin=-shared_vmax, vcenter=0.0, vmax=shared_vmax)
    im = ax.imshow(m.values, aspect="auto", cmap=DIV_CMAP, norm=norm)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([pretty_model(mn) for mn in models], rotation=45, ha="right", fontsize=F_TICKS)
    ax.set_yticks(range(len(cat_order)))
    ax.set_yticklabels([pretty_cat(c) for c in cat_order], fontsize=F_CATS)
    for b in FAMILY_BREAKS:
        ax.axvline(b - 0.5, color=SURFACE, linewidth=2)
    for i, cat in enumerate(cat_order):
        ax.text(len(models) - 0.25, i, f"{m.loc[cat].mean():+.2f}", va="center", ha="left",
                fontsize=F_MEANS, color=TEXT_SECONDARY, clip_on=False)
    ax.text(len(models) - 0.25, -0.9, "mean", fontsize=F_MEANS, color=TEXT_SECONDARY, ha="left", clip_on=False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.06)
    cbar.set_label(cbar_label, fontsize=F_CBAR, color=TEXT_SECONDARY)
    cbar.ax.tick_params(labelsize=F_CBAR - 1)
    cbar.outline.set_visible(False)
    ax.set_title(title, fontsize=F_TITLE, loc="left", pad=14)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=200)
    plt.close(fig)
    print("wrote", fname)

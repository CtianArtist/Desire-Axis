"""Heatmaps of the mean cosine similarity matrices (Figure 3 of the paper and the
pooled-control robustness variant). Reads the MEAN files written by
04_run_all_similarity.py in results/3.3_validation/cosine_similarity and writes
fig_similarity_raw and fig_similarity_alldenoise there as PDF and 300 dpi PNG.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib import font_manager

RESULTS = Path("results") / "3.3_validation" / "cosine_similarity"
ORDER = ["S1_desire", "S2_desire", "Adrenaline", "Affection", "Joy",
         "BodySens", "Excitement", "Random", "Numb", "Contentment"]
DISPLAY = {"S1_desire": "S1 desire", "S2_desire": "S2 desire", "Adrenaline": "Adrenaline", "Affection": "Affection",
           "Joy": "Joy", "BodySens": "BodySens", "Excitement": "Excitement", "Random": "Random",
           "Numb": "Numb", "Contentment": "Contentment"}
N = len(ORDER)

_HAVE = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ("Arial", "Helvetica", "DejaVu Sans") if f in _HAVE), "DejaVu Sans")
INK, INK2, MUTED = "#000000", "#333333", "#8a8a8a"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#f5f4f2", "#c23a39"])
DIVERGING.set_bad(alpha=0.0)
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [FONT, "DejaVu Sans"], "font.size": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
})


def fmt(v):
    return f"{v:.2f}".replace("0.", ".").replace("-.", "−.")


def read_csv(path):
    mat = np.full((N, N), np.nan)
    for r, line in enumerate(path.read_text(encoding="utf-8").strip().splitlines()[1:]):
        for c, cell in enumerate(line.split(",")[1:]):
            if cell:
                mat[r, c] = float(cell)
    return mat


def draw_square(grand, stem):
    norm = TwoSlopeNorm(vmin=-0.8, vcenter=0.0, vmax=0.8)
    show = np.ma.masked_where(np.eye(N, dtype=bool), grand)

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(show, cmap=DIVERGING, norm=norm)
    for i in range(N):
        for j in range(N):
            if i == j:
                ax.text(j, i, "1", ha="center", va="center", fontsize=6.8, color=MUTED)
            else:
                v = grand[i, j]
                ax.text(j, i, fmt(v), ha="center", va="center", fontsize=6.8,
                        color="white" if abs(v) > 0.55 else INK)
    ax.set_xticks(range(N), [DISPLAY[l] for l in ORDER], rotation=45, ha="right", color=INK2)
    ax.set_yticks(range(N), [DISPLAY[l] for l in ORDER], color=INK2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, N), minor=True)
    ax.set_yticks(np.arange(-0.5, N), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, shrink=0.62, pad=0.02, aspect=18, ticks=[-0.8, -0.4, 0, 0.4, 0.8])
    cb.set_label("Cosine similarity", color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2, length=2)
    cb.outline.set_visible(False)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{stem}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


for mean_name, stem in [("similarity_MEAN_all_models.csv", "fig_similarity_raw"),
                        ("similarity_alldenoise_MEAN_all_models.csv", "fig_similarity_alldenoise")]:
    path = RESULTS / mean_name
    if not path.exists():
        raise SystemExit(f"missing {path}; run 04_run_all_similarity.py first")
    draw_square(read_csv(path), str(RESULTS / stem))
    print(f"{stem}: written as .pdf and .png")

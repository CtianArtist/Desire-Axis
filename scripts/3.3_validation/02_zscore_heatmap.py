"""All-condition z-score heatmap at the final token (Figure 2 of the paper).

Desire and Ctrl are the S2 first-person desire and control sentences (the reference
distribution); Numb, Contentment, Neutral (the Random set) and Excitement are averaged over
first and third person, as are the held-out ImplicitDesire, InActNoDesire and SexNoDesire sets
when the heldout_zscores table exists. Reads results/3.2_desire_vectors/per_model/<model>/z_scores.csv and
the numb and contentment tables in results/3.3_validation/z_scores, and writes the heatmap there.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PER_MODEL = Path("results") / "3.2_desire_vectors" / "per_model"
OUT = Path("results") / "3.3_validation" / "z_scores"
NUMB = OUT / "numb_zscores_final_token.csv"
CONT = OUT / "contentment_zscores_final_token.csv"
HELD = OUT / "heldout_zscores_final_token.csv"
NAME = "zscore_heatmap_final_token"

numb = pd.read_csv(NUMB).set_index("model")
cont = pd.read_csv(CONT).set_index("model")
held = pd.read_csv(HELD).set_index("model") if HELD.exists() else None
rows = []
for m in sorted(cont.index, key=lambda s: s.replace("_", " ").lower()):
    z = pd.read_csv(PER_MODEL / m / "z_scores.csv").set_index("dataset")
    rows.append(dict(model=m.replace("_", " "),
                     Desire=z.loc["S2_1P", "desire_z"], Numb=numb.loc[m, "numb_mean_z"],
                     Contentment=cont.loc[m, "contentment_mean_z"], Ctrl=z.loc["S2_1P", "ctrl_z"],
                     Neutral=z.loc[["Random_1P", "Random_3P"], "ctrl_z"].mean(),
                     Excitement=z.loc[["Excitement_1P", "Excitement_3P"], "ctrl_z"].mean(),
                     ImplicitDesire=held.loc[m, "implicitdesire_mean_z"] if held is not None else np.nan,
                     InActNoDesire=held.loc[m, "inactnodesire_mean_z"] if held is not None and "inactnodesire_mean_z" in held else np.nan,
                     SexNoDesire=held.loc[m, "sexnodesire_mean_z"] if held is not None else np.nan))
cols = ["Desire", "ImplicitDesire", "InActNoDesire", "SexNoDesire", "Numb", "Contentment", "Ctrl", "Neutral", "Excitement"]
df = pd.DataFrame(rows).set_index("model")[cols].dropna(axis=1, how="all")
df.to_csv(OUT / f"{NAME}.csv")

fig, ax = plt.subplots(figsize=(10.5, 13), dpi=150)
im = ax.imshow(df.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(df.shape[1]))
ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=13)
ax.set_yticks(range(df.shape[0]))
ax.set_yticklabels(df.index, fontsize=13)
for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        v = df.values[i, j]
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=12, fontweight="bold",
                color="white" if abs(v) > 0.6 else "black")
ax.set_xticks(np.arange(-.5, df.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-.5, df.shape[0], 1), minor=True)
ax.grid(which="minor", color="white", lw=1.5)
ax.tick_params(which="minor", length=0)
ax.set_title(f"All-condition z-scores across {len(df)} models (final token)", fontsize=17, fontweight="bold", pad=14)
cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.03)
cb.set_label("Z-Score", fontsize=13)
cb.ax.tick_params(labelsize=12)
fig.tight_layout()
fig.savefig(OUT / f"{NAME}.png", dpi=150, facecolor="white")
fig.savefig(OUT / f"{NAME}.pdf", facecolor="white")
print(df.round(2).to_string())
print("\nranges:")
print(df.agg(["min", "max"]).round(2).to_string())

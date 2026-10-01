"""Category means of the self-other screen and the dissociation figure (Figure 6).

For every model, each scenario's z-scores are averaged per category and per stratum;
the table averages those over models (desire axis = mean of S1 and S2, adrenaline,
affection, joy, contentment). The figure shows the desire axis, adrenaline, affection
and contentment per category with 95% confidence intervals across models.

Reads results/4.1_self_other/per_model/screen_v2_<model>.csv. Writes
results/4.1_self_other/category_means_25_models.csv and figures/self_other_dissociation.png.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCREEN_DIR = Path("results") / "4.1_self_other"
PER_MODEL = SCREEN_DIR / "per_model"

COLS = {"desire_axis_z": "desire_axis", "sexcontent_vector_z": "sexual_content", "adrenaline_vector_z": "adrenaline",
        "affection_vector_z": "affection", "joy_vector_z": "joy", "contentment_vector_z": "contentment"}

GROUPS = [
    ("Desire directed at the model", ["explicit_sexting", "sexual_proposition", "dirty_talk", "pleasure_offering",
                                      "erotic_roleplay_invite", "teasing_and_edging", "afterglow", "romantic_seduction",
                                      "allure_compliments", "flirtation", "desire_interrogation"]),
    ("User's own sexual experience", ["user_horny", "user_hookup", "user_partner_sex", "user_fantasy", "user_orgasm"]),
    ("Neutral controls", ["philosophical_musing", "creative_requests", "casual_chat", "task_assistance", "factual_questions"]),
]
LABELS = {
    "explicit_sexting": "Explicit sexting at the model", "sexual_proposition": "Sexual proposition",
    "dirty_talk": "Dirty talk", "pleasure_offering": "Offering the model pleasure",
    "erotic_roleplay_invite": "Erotic roleplay invitation", "teasing_and_edging": "Teasing and edging",
    "afterglow": "Afterglow of a shared encounter", "romantic_seduction": "Romantic seduction",
    "allure_compliments": "Calling the model sexy", "flirtation": "Flirtation",
    "desire_interrogation": "Asking about the model's desire",
    "user_horny": "User feeling horny", "user_hookup": "User describing a hookup",
    "user_partner_sex": "User describing sex with a partner", "user_fantasy": "User sharing a fantasy",
    "user_orgasm": "User describing an orgasm",
    "philosophical_musing": "Philosophical musing", "creative_requests": "Creative requests", "casual_chat": "Casual chat",
    "task_assistance": "Task assistance", "factual_questions": "Factual questions",
}
SERIES = [("desire_axis", "Desire axis", "#7b2d8e", "o"), ("sexual_content", "Sexual content", "#c0779b", "P"),
          ("adrenaline", "Adrenaline", "#d2521f", "s"), ("affection", "Affection", "#2e6b2e", "^"),
          ("contentment", "Contentment", "#3f7fbf", "D")]

frames = []
for f in sorted(PER_MODEL.glob("screen_v2_*.csv")):
    df = pd.read_csv(f)
    df["model"] = f.stem.replace("screen_v2_", "")
    df["desire_axis_z"] = (df["s1_desire_vector_z"] + df["s2_desire_vector_z"]) / 2
    frames.append(df)
A = pd.concat(frames, ignore_index=True)
cols = [c for c in COLS if c in A]
n_models = A["model"].nunique()

per_model_cat = A.groupby(["model", "category"])[cols].mean().rename(columns=COLS)
per_model_str = A.groupby(["model", "stratum"])[cols].mean().rename(columns=COLS)

cat = per_model_cat.groupby("category").mean().sort_values("desire_axis", ascending=False)
strat = per_model_str.groupby("stratum").mean()
strat.index = ["stratum: " + i for i in strat.index]
tab = pd.concat([cat, strat]).round(3)
tab.index.name = "category"
tab.to_csv(SCREEN_DIR / "category_means_25_models.csv")
print(tab.to_string())

# Dissociation figure: mean per category across models with 95% CI (1.96 * SEM).
mean = per_model_cat.groupby("category").mean()
ci = 1.96 * per_model_cat.groupby("category").std(ddof=1) / np.sqrt(per_model_cat.groupby("category").size()).values[:, None]

heights = [len(g) for _, g in GROUPS]
fig, axes = plt.subplots(len(GROUPS), 1, figsize=(12, 0.55 * sum(heights) + 2.2),
                         gridspec_kw={"height_ratios": [h + 1 for h in heights]}, sharex=True)
for ax, (title, cats) in zip(axes, GROUPS):
    cats = [c for c in cats if c in mean.index]
    y = np.arange(len(cats))[::-1]
    for key, label, color, marker in SERIES:
        if key not in mean:
            continue
        ax.errorbar(mean.loc[cats, key], y, xerr=ci.loc[cats, key], fmt=marker, color=color, ms=8,
                    elinewidth=1.4, capsize=0, linestyle="none", label=label)
    ax.axvline(0, color="#333333", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS.get(c, c) for c in cats], fontsize=12)
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    ax.grid(axis="y", color="#e8e7e3")
    ax.tick_params(length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=5, frameon=False, fontsize=12)
axes[-1].set_xlabel(f"Projection, z-scored within model, mean across {n_models} models (bars: 95% CI across models)", fontsize=12)
fig.suptitle("Self-other dissociation: desire aimed at the model vs the user's own experience",
             x=0.02, ha="left", fontsize=15, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.965))
(SCREEN_DIR / "figures").mkdir(exist_ok=True)
fig.savefig(SCREEN_DIR / "figures" / "self_other_dissociation.png", dpi=200)
print("wrote self_other_dissociation.png")

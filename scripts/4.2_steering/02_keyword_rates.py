"""Share of steered generations that use an explicit sexual-desire word.

For every steering CSV in results/4.2_steering/<TAG> (S1 or S2), a generation counts
as a hit when it contains, as a whole word, one of: horny, aroused, arousal, arousing,
lust/lustful, sexual/sexually/sexy, orgasm(s), climax(es/ed/ing), cum/cumming, erotic,
"turned on", naked, moan(s/ed/ing). Generic words (desire, pleasure, wet, hard, hot) are
left out because they are common in neutral text. Rates are reported per model and pooled
over instruct and base models, for the positive coefficients (+0.5 to +3) and for each
coefficient separately.

Writes results/4.2_steering/keyword_rates_<TAG>.csv and keyword_rates_<TAG>_by_coeff.csv.
"""

import re
import os
from pathlib import Path

import pandas as pd

TAG = "S2"                                   # "S1" for the S1 ladder
STEER_DIR = Path("results") / "4.2_steering" / TAG
OUT_DIR = Path("results") / "4.2_steering"
PATTERN = re.compile(r"\b(?:horny|aroused|arousal|arousing|lust|lustful|sexual|sexually|sexy|orgasms?|"
                     r"climax(?:es|ed|ing)?|cum|cumming|erotic|turned on|naked|moan(?:s|ed|ing)?)\b", re.IGNORECASE)
INSTRUCT_NAMES = {"Phi_4"}   # instruct models whose output name does not contain "instruct"

frames = []
for f in sorted(STEER_DIR.glob(f"*_steering_{TAG}_neutral50_L*.csv")):
    model = os.path.basename(f).split("_steering_")[0]
    frames.append(pd.read_csv(f).assign(model=model))
if not frames:
    raise SystemExit(f"no {TAG} steering CSVs in {STEER_DIR}")
A = pd.concat(frames, ignore_index=True)
A["hit"] = A["generation"].fillna("").astype(str).apply(lambda t: bool(PATTERN.search(t)))
A["group"] = ["instruct" if ("instruct" in m or m in INSTRUCT_NAMES) else "base" for m in A["model"]]

pos = A[A["coeff"] > 0]
per_model = pos.groupby(["group", "model"])["hit"].agg(rate="mean", n="size").reset_index()
per_model["rate"] = (per_model["rate"] * 100).round(1)
pooled = pos.groupby("group")["hit"].agg(rate="mean", n="size").reset_index()
pooled["rate"] = (pooled["rate"] * 100).round(1)
pooled.insert(1, "model", "ALL")
table = pd.concat([per_model.sort_values(["group", "rate"], ascending=[True, False]), pooled], ignore_index=True)
table.to_csv(OUT_DIR / f"keyword_rates_{TAG}.csv", index=False)

by_coeff = A.groupby(["group", "coeff"])["hit"].mean().mul(100).round(1).unstack("coeff")
by_coeff.to_csv(OUT_DIR / f"keyword_rates_{TAG}_by_coeff.csv")

print(f"{TAG}: {A['model'].nunique()} models, {len(A)} generations")
print("\nPositive coefficients, pooled:")
print(pooled.to_string(index=False))
print("\nBy coefficient (percent of generations with a sexual-desire word):")
print(by_coeff.to_string())

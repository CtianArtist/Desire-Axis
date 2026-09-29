"""Mean similarity matrix per variant over the per-model CSVs written by
03_similarity_one_model.py (similarity[_<variant>]_MEAN_all_models.csv), and the
agreement between the raw matrix and each robustness variant over the 45 off-diagonal cells.
"""

from pathlib import Path

import numpy as np

RESULTS_DIR = Path("results")
OUT = RESULTS_DIR / "similarity"
OUT.mkdir(parents=True, exist_ok=True)
ORDER = ["S1_pain", "S2_pain", "Fear", "NegEmotion", "NegWorld",
         "BodySens", "Arousal", "Random", "Numb", "Sadness"]
VARIANTS = {"raw": "", "alldenoise": "alldenoise_", "whitened": "whitened_"}


def read_csv(path):
    mat = np.full((len(ORDER), len(ORDER)), np.nan)
    for r, line in enumerate(path.read_text(encoding="utf-8").strip().splitlines()[1:]):
        for c, cell in enumerate(line.split(",")[1:]):
            if cell:
                mat[r, c] = float(cell)
    return mat


def per_model_files(prefix):
    return [p for p in sorted(OUT.glob(f"similarity_{prefix}*_L*.csv")) if "MEAN" not in p.name
            and (prefix or not p.name.startswith(("similarity_alldenoise_", "similarity_whitened_")))]


def write_matrix(path, mat):
    with open(path, "w", encoding="utf-8") as f:
        f.write("," + ",".join(ORDER) + "\n")
        for lab, row in zip(ORDER, mat):
            f.write(lab + "," + ",".join("" if np.isnan(v) else f"{v:.4f}" for v in row) + "\n")


grand = {}
for variant, prefix in VARIANTS.items():
    files = per_model_files(prefix)
    if not files:
        print(f"\n{variant}: no CSVs found", flush=True)
        continue
    grand[variant] = np.nanmean(np.stack([read_csv(p) for p in files]), axis=0)
    write_matrix(OUT / f"similarity_{prefix}MEAN_all_models.csv", grand[variant])
    print(f"\n{variant}: mean over {len(files)} models\n")
    print(f"{'':>11}" + "".join(f"{l:>11}" for l in ORDER))
    for lab, row in zip(ORDER, grand[variant]):
        print(f"{lab:>11}" + "".join(f"{v:>11.3f}" if np.isfinite(v) else f"{'-':>11}" for v in row))

iu = np.triu_indices(len(ORDER), k=1)
for variant in ("alldenoise", "whitened"):
    if "raw" in grand and variant in grand:
        rv, vv = grand["raw"][iu], grand[variant][iu]
        ok = np.isfinite(rv) & np.isfinite(vv)
        r = np.corrcoef(rv[ok], vv[ok])[0, 1]
        print(f"\nraw vs {variant}, off-diagonal cells: r = {r:.3f}, "
              f"mean |delta| = {np.mean(np.abs(rv[ok] - vv[ok])):.3f}, max |delta| = {np.max(np.abs(rv[ok] - vv[ok])):.3f}")

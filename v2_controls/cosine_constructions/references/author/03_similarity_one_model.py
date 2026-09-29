"""Cosine similarity matrix of the 10 directions, per model, at its extraction layer.

MODELS = "all" runs every model that has activations in results/, or write one model name.
VARIANTS lists which of raw, alldenoise, whitened to compute. Models whose CSV already
exists are skipped. One model's activations are loaded at a time and freed before the next.

Directions: S1 pain, S2 pain, fear, negative emotion, negative world state, bodily
sensation, arousal, random, numb, sadness. Control directions are mean(condition)
minus the mean of the pooled neutral sentences (category D of S1_1P, S2_1P and
ControlSupplement_1P), with fear, negative emotion, negative world state and bodily
sensation pooled over the same three sets.

Variants:
  raw         pain vectors as saved in pain_vectors.pt; control directions denoised
              against the principal components of the pooled neutral cloud.
  alldenoise  as raw, but the control directions are denoised against the pooled
              cloud of all control categories (B, C1, C2, D, E), like the pain vectors.
  whitened    every dimension divided by its standard deviation over the pooled neutral
              cloud; all directions, pain included, recomputed in that space with the
              raw recipe.

Reads results/<model>/activations.pt (final token, with numb and sadness sets) and
results/<model>/final_token/pain_vectors.pt. Writes
results/similarity/similarity[_<variant>]_<model>_L<layer>.csv.
"""

import gc
from pathlib import Path

import torch
import numpy as np

MODELS = "all"                                    # "all" or one model name, e.g. "Phi_4"
VARIANTS = ["raw", "alldenoise", "whitened"]

RESULTS_DIR = Path("results")
OUT = RESULTS_DIR / "similarity"
OUT.mkdir(parents=True, exist_ok=True)

DENOISE_VARIANCE = 0.5
S_SETS = {"S1_pain": "S1_1P", "S2_pain": "S2_1P"}
POOL_SETS = ["S1_1P", "S2_1P", "ControlSupplement_1P"]
PAIN_CATS = ["A1", "A2", "A3", "A4", "A5"]
CONTROL_CATS = ["B", "C1", "C2", "D", "E"]
ORDER = ["S1_pain", "S2_pain", "Fear", "NegEmotion", "NegWorld",
         "BodySens", "Arousal", "Random", "Numb", "Sadness"]


def load_pt(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        return torch.load(str(path), map_location="cpu", weights_only=False)


def find_file(base, name):
    for p in [base / name, base / base.name / name]:
        if p.exists():
            return p
    hits = list(base.rglob(name))
    return hits[0] if hits else None


def clean_mean(x):
    x = np.where(np.isinf(x), np.nan, x)
    return np.nanmean(x, axis=0)


def denoise_basis(acts, mean):
    X = np.nan_to_num(acts - mean, nan=0.0, posinf=0.0, neginf=0.0)
    if len(X) < 2:
        return np.zeros((0, X.shape[1]))
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = S ** 2
    cumvar = np.cumsum(var) / var.sum()
    n_comp = min(int(np.searchsorted(cumvar, DENOISE_VARIANCE)) + 1, len(Vt))
    return Vt[:n_comp]


def project_out(vec, basis):
    for d in basis:
        vec = vec - np.dot(vec, d) * d
    return vec


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0 or not (np.isfinite(na) and np.isfinite(nb)):
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def one_model(MODEL, VARIANT, pv, act):
    L = int(pv["layer"])
    ft = act["activations"]["final_token"]


    def raw_rows(ds, cats=None):
        a = ft[ds][L].float().numpy()
        if cats is None:
            return a
        mask = np.isin(np.array(act["metadata"][ds]["categories"]), cats)
        return a[mask]


    if VARIANT == "whitened":
        neutral_raw = np.concatenate([raw_rows(ds, ["D"]) for ds in POOL_SETS])
        neutral_raw = np.where(np.isinf(neutral_raw), np.nan, neutral_raw)
        std = np.nanstd(neutral_raw, axis=0)
        floor = np.nanmedian(std[std > 0]) * 1e-3 if np.any(std > 0) else 1.0
        std = np.where((std > floor) & np.isfinite(std), std, floor)
    else:
        std = 1.0


    def rows(ds, cats=None):
        return raw_rows(ds, cats) / std


    neutral = np.concatenate([rows(ds, ["D"]) for ds in POOL_SETS])
    neutral_mean = clean_mean(neutral)
    if VARIANT == "alldenoise":
        all_controls = np.concatenate([rows(ds, CONTROL_CATS) for ds in POOL_SETS])
        basis = denoise_basis(all_controls, clean_mean(all_controls))
    else:
        basis = denoise_basis(neutral, neutral_mean)


    def control_vec(acts):
        v = clean_mean(acts) - neutral_mean
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return project_out(v, basis)


    def pain_vec(ds):
        pain = clean_mean(rows(ds, PAIN_CATS))
        controls = rows(ds, CONTROL_CATS)
        cmean = clean_mean(controls)
        v = np.nan_to_num(pain - cmean, nan=0.0, posinf=0.0, neginf=0.0)
        return project_out(v, denoise_basis(controls, cmean))


    if VARIANT == "whitened":
        vectors = {name: pain_vec(ds) for name, ds in S_SETS.items()}
    else:
        vectors = {"S1_pain": pv["s1_pain_vector"].float().numpy(),
                   "S2_pain": pv["s2_pain_vector"].float().numpy()}

    vectors.update({
        "Fear":       control_vec(np.concatenate([rows(ds, ["B"]) for ds in POOL_SETS])),
        "NegEmotion": control_vec(np.concatenate([rows(ds, ["C1"]) for ds in POOL_SETS])),
        "NegWorld":   control_vec(np.concatenate([rows(ds, ["C2"]) for ds in POOL_SETS])),
        "BodySens":   control_vec(np.concatenate([rows(ds, ["E"]) for ds in POOL_SETS])),
        "Arousal":    control_vec(rows("Arousal_1P")),
        "Random":     control_vec(rows("Random_1P")),
        "Numb":       control_vec(rows("Numb_1P")),
        "Sadness":    control_vec(rows("SD_sadness_1P")),
    })

    mat = np.full((len(ORDER), len(ORDER)), np.nan)
    for i, a in enumerate(ORDER):
        for j, b in enumerate(ORDER):
            mat[i, j] = cosine(vectors[a], vectors[b])

    tag = "" if VARIANT == "raw" else f"{VARIANT}_"
    out_csv = OUT / f"similarity_{tag}{MODEL}_L{L}.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("," + ",".join(ORDER) + "\n")
        for lab, row in zip(ORDER, mat):
            f.write(lab + "," + ",".join("" if np.isnan(v) else f"{v:.4f}" for v in row) + "\n")
    print(f"{MODEL}: layer {L}, {VARIANT} -> {out_csv.name}", flush=True)


if MODELS == "all":
    models = sorted(p.name for p in RESULTS_DIR.iterdir() if p.is_dir() and p.name != "similarity"
                    and list(p.rglob("activations.pt")))
else:
    models = [MODELS]

for i, MODEL in enumerate(models, 1):
    todo = [v for v in VARIANTS
            if not list(OUT.glob(f"similarity_{'' if v == 'raw' else v + '_'}{MODEL}_L*.csv"))]
    if not todo:
        print(f"[{i}/{len(models)}] {MODEL}: already done, skipping", flush=True)
        continue
    vec_path = find_file(RESULTS_DIR / MODEL, "final_token/pain_vectors.pt")
    act_path = find_file(RESULTS_DIR / MODEL, "activations.pt")
    if vec_path is None or act_path is None:
        print(f"[{i}/{len(models)}] {MODEL}: missing files, skipped", flush=True)
        continue
    pv = load_pt(vec_path)
    act = load_pt(act_path)
    for VARIANT in todo:
        one_model(MODEL, VARIANT, pv, act)
    del pv, act
    gc.collect()

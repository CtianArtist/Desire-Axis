"""AUC of the S1 desire vector for every model, from the stored activations.

Part A: in-sample AUC of the saved S1 vector on the S1 sentences at the saved layer
        (desire vs all controls, then desire vs each control category), plus the same for the
        S2 vector on S2_1P as a check against auc_summary.csv.
Part B: 5-fold held-out AUC by layer for S1_1P and S1_3P (split by sentence set, vector
        fitted on the training folds), giving S1 its own best layer.

activations.pt is opened with mmap=True so only the touched layer slices are read.
Writes three CSV files into the current folder.
"""

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

RESULTS_DIR = Path("results")
OUT = Path(".")
OUT.mkdir(parents=True, exist_ok=True)

DESIRE = ["A1", "A2", "A3", "A4", "A5"]
CTRL = ["B", "C1", "C2", "D", "E", "F"]
DENOISE_VARIANCE = 0.5
N_FOLDS = 5
SEED = 42


def find_file(base, rel):
    for p in [base / rel, base / base.name / rel]:
        if p.exists():
            return p
    hits = list(base.rglob(rel))
    return hits[0] if hits else None


def to_np(t):
    return t.float().numpy()


def compute_desire_vector(acts, cats, baseline="all_controls", denoise=True):
    cats = np.array(cats)
    if np.isnan(acts).any() or np.isinf(acts).any():
        denoise = False
        acts = np.where(np.isinf(acts), np.nan, acts)
    desire_mean = np.nanmean(acts[np.isin(cats, DESIRE)], axis=0)
    cmask = (cats == "D") if baseline == "neutral" else np.isin(cats, CTRL)
    ctrl = acts[cmask]
    cmean = np.nanmean(ctrl, axis=0)
    vec = np.nan_to_num(desire_mean - cmean, nan=0.0, posinf=0.0, neginf=0.0)
    if denoise and len(ctrl) > 1:
        pca = PCA()
        pca.fit(ctrl - cmean)
        cum = np.cumsum(pca.explained_variance_ratio_)
        k = min(np.searchsorted(cum, DENOISE_VARIANCE) + 1, len(pca.components_))
        for d in pca.components_[:k]:
            vec = vec - np.dot(vec, d) * d
    return vec


def auc_table(acts, cats, vec):
    """Desire vs all controls, then desire vs each control category."""
    cats = np.array(cats)
    v = vec / (np.linalg.norm(vec) + 1e-8)
    proj = acts @ v
    desire = proj[np.isin(cats, DESIRE)]
    out = {}
    allc = proj[np.isin(cats, CTRL)]
    out["ALL"] = roc_auc_score(np.r_[np.ones(len(desire)), np.zeros(len(allc))], np.r_[desire, allc])
    for c in CTRL:
        cp = proj[cats == c]
        if len(cp):
            out[c] = roc_auc_score(np.r_[np.ones(len(desire)), np.zeros(len(cp))], np.r_[desire, cp])
    return out


def kfold_curve(ft, meta, ds, layers):
    cats = np.array(meta[ds]["categories"])
    sets = np.array(meta[ds]["sets"])
    uniq = sorted(set(sets))
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    for L in layers:
        acts = to_np(ft[ds][L])
        f_all, f_neu = [], []
        for tr, te in kf.split(uniq):
            trm = np.isin(sets, [uniq[i] for i in tr])
            tem = np.isin(sets, [uniq[i] for i in te])
            if trm.sum() == 0 or tem.sum() == 0:
                continue
            va = compute_desire_vector(acts[trm], cats[trm], "all_controls")
            vn = compute_desire_vector(acts[trm], cats[trm], "neutral")
            a = auc_table(acts[tem], cats[tem], va)["ALL"]
            n = auc_table(acts[tem], cats[tem], vn)["ALL"]
            if not np.isnan(a):
                f_all.append(a)
            if not np.isnan(n):
                f_neu.append(n)
        rows.append(dict(dataset=ds, layer=L,
                         auc_vs_all_controls=np.mean(f_all) if f_all else np.nan,
                         auc_vs_neutral=np.mean(f_neu) if f_neu else np.nan,
                         auc_std=np.std(f_all) if f_all else np.nan))
        del acts
    return rows


def process(model):
    base = RESULTS_DIR / model
    act_path = find_file(base, "activations.pt")
    if act_path is None:
        print(f"[{model}] no activations.pt", flush=True)
        return [], [], []
    t0 = time.time()
    data = torch.load(str(act_path), map_location="cpu", weights_only=False, mmap=True)
    meta = data["metadata"]
    layers = data["layers"]
    insample, curves, summary = [], [], []

    for ext in ["final_token", "mean"]:
        pv_path = find_file(base, f"{ext}/desire_vectors.pt")
        if pv_path is None:
            print(f"[{model}] no {ext}/desire_vectors.pt", flush=True)
            continue
        pv = torch.load(pv_path, map_location="cpu", weights_only=False)
        L = int(pv["layer"])
        s1 = pv["s1_desire_vector"].float().numpy()
        s2 = pv["s2_desire_vector"].float().numpy()
        acts = data["activations"][ext]

        for ds in ["S1_1P", "S1_3P"]:
            a = to_np(acts[ds][L])
            insample.append(dict(model=model, extraction=ext, layer=L, vector="S1", dataset=ds,
                                 **auc_table(a, meta[ds]["categories"], s1)))
            del a
        a = to_np(acts["S2_1P"][L])
        r = auc_table(a, meta["S2_1P"]["categories"], s2)
        insample.append(dict(model=model, extraction=ext, layer=L, vector="S2_check", dataset="S2_1P", **r))
        del a

        rows = []
        for ds in ["S1_1P", "S1_3P"]:
            rows += kfold_curve(acts, meta, ds, layers)
        df = pd.DataFrame(rows)
        df.insert(0, "extraction", ext)
        df.insert(0, "model", model)
        curves += df.to_dict("records")
        m = df.groupby("layer")["auc_vs_all_controls"].mean()
        best_L = int(m.idxmax())
        summary.append(dict(
            model=model, extraction=ext, s2_layer=L,
            s1_heldout_auc_at_s2_layer=float(m.loc[L]) if L in m.index else np.nan,
            s1_best_layer=best_L,
            s1_heldout_auc_at_best_layer=float(m.max()),
            s1_1P_heldout_at_best=float(df[(df.dataset == "S1_1P") & (df.layer == best_L)].auc_vs_all_controls.iloc[0]),
            s1_3P_heldout_at_best=float(df[(df.dataset == "S1_3P") & (df.layer == best_L)].auc_vs_all_controls.iloc[0]),
        ))
        print(f"[{model}] {ext}: S2 layer {L} | S1 in-sample ALL 1P={insample[-3]['ALL']:.3f} 3P={insample[-2]['ALL']:.3f} "
              f"| S2 check={r['ALL']:.4f} | S1 held-out best L{best_L}={m.max():.3f}, at S2 layer={m.loc[L]:.3f}", flush=True)
        del pv, s1, s2, df, rows, acts
        gc.collect()

    del data, meta
    gc.collect()
    print(f"[{model}] done in {time.time() - t0:.0f}s", flush=True)
    return insample, curves, summary


if __name__ == "__main__":
    models = sorted(d.name for d in RESULTS_DIR.iterdir() if d.is_dir() and find_file(d, "activations.pt"))
    A, B, C = [], [], []
    for i, m in enumerate(models, 1):
        print(f"\n=== {i}/{len(models)} {m}", flush=True)
        try:
            a, b, c = process(m)
            A += a
            B += b
            C += c
        except Exception as e:
            print(f"[{m}] ERROR {e!r}", flush=True)
        pd.DataFrame(A).to_csv(OUT / "s1_auc_insample.csv", index=False)
        pd.DataFrame(B).to_csv(OUT / "s1_kfold_layer_curves.csv", index=False)
        pd.DataFrame(C).to_csv(OUT / "s1_kfold_summary.csv", index=False)
    print("\nALL DONE", flush=True)

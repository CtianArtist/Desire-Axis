"""Z-scores of the numb and contentment sentences on the S2 desire vector, for every model.

For each model and each extraction type (final_token, mean): the S2 desire vector is
recomputed from S2_1P at the chosen layer, the numb and contentment sentences (first and
third person) are projected onto it, and the projections are z-scored against the
S2_1P projections. The same z-scores are reported for the S2 desire and control
sentences and for the sexual-pleasure category A1 (the numb set matches the S2 A1
events, each crossed with five reasons the arousal is absent).

Also the topic check: the sexual-content control F and the held-out sets SexNoDesire
(sexual content, no desire) and ImplicitDesire (desire, no explicit words). On a desire vector
rather than a sexual-content vector, Numb and SexNoDesire land well below A1 and the desire
mean, and ImplicitDesire lands near them. Sets missing from older activations are left empty.

Reads results/<model>/activations.pt and summary.json from
01_extract_activations_and_desire_vectors.py (the contentment sets are in the same file).
Writes numb_zscores_<extraction>.csv, contentment_zscores_<extraction>.csv and
heldout_zscores_<extraction>.csv.
"""

import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

RESULTS_DIR = Path("results")
OUT_DIR = Path("results")

DESIRE_CATEGORIES = ["A1", "A2", "A3", "A4", "A5"]
CONTROL_CATEGORIES = ["B", "C1", "C2", "D", "E", "F"]
DENOISE_VARIANCE = 0.5


def compute_desire_vector(acts, cats, denoise=True):
    acts_np = acts.numpy() if hasattr(acts, "numpy") else acts
    cats_np = np.array(cats)
    desire_mean = np.nanmean(acts_np[np.isin(cats_np, DESIRE_CATEGORIES)], axis=0)
    control_acts = acts_np[np.isin(cats_np, CONTROL_CATEGORIES)]
    control_mean = np.nanmean(control_acts, axis=0)
    vec = np.nan_to_num(desire_mean - control_mean, nan=0.0, posinf=0.0, neginf=0.0)
    if denoise and len(control_acts) > 1:
        pca = PCA()
        pca.fit(control_acts - control_mean)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_comp = min(np.searchsorted(cumvar, DENOISE_VARIANCE) + 1, len(pca.components_))
        for d in pca.components_[:n_comp]:
            vec = vec - np.dot(vec, d) * d
    return vec


def project_and_zscore(acts, vector, ref_acts):
    acts_np = acts.numpy() if hasattr(acts, "numpy") else acts
    ref_np = ref_acts.numpy() if hasattr(ref_acts, "numpy") else ref_acts
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    vec_norm = vector / (np.linalg.norm(vector) + 1e-8)
    proj = np.nan_to_num(acts_np @ vec_norm, nan=0.0, posinf=0.0, neginf=0.0)
    ref_proj = np.nan_to_num(ref_np @ vec_norm, nan=0.0, posinf=0.0, neginf=0.0)
    return (proj - np.mean(ref_proj)) / (np.std(ref_proj) + 1e-8)


def find_file(base, name):
    for p in [base / name, base / base.name / name]:
        if p.exists():
            return p
    return None


def analyze_model(model_name, extraction_type):
    act_path = find_file(RESULTS_DIR / model_name, "activations.pt")
    sum_path = find_file(RESULTS_DIR / model_name, "summary.json")
    if act_path is None or sum_path is None:
        print(f"  {model_name}: activations.pt or summary.json missing, skipped")
        return None, None, None

    data = torch.load(str(act_path), map_location="cpu", weights_only=False, mmap=True)
    acts = data["activations"][extraction_type]
    meta = data["metadata"]
    summary = json.load(open(sum_path))
    best_layer = summary[f"best_layer_{extraction_type}"]

    s2_acts = acts["S2_1P"][best_layer]
    s2_cats = np.array(meta["S2_1P"]["categories"])
    desire_vector = compute_desire_vector(s2_acts, s2_cats)

    all_z = project_and_zscore(s2_acts, desire_vector, s2_acts)
    all_desire_z = float(all_z[np.isin(s2_cats, DESIRE_CATEGORIES)].mean())
    all_ctrl_z = float(all_z[np.isin(s2_cats, CONTROL_CATEGORIES)].mean())
    a1_z = float(all_z[s2_cats == "A1"].mean())

    base = {"model": model_name, "n_layers": data["n_layers"], "d_model": data["d_model"],
            "best_layer": best_layer, "extraction": extraction_type}

    def z_of(ds):
        return float(project_and_zscore(acts[ds][best_layer], desire_vector, s2_acts).mean())

    numb_1p, numb_3p = z_of("Numb_1P"), z_of("Numb_3P")
    numb_row = {**base, "numb_1P_z": numb_1p, "numb_3P_z": numb_3p, "numb_mean_z": (numb_1p + numb_3p) / 2,
                "A1_with_desire_z": a1_z, "all_desire_z": all_desire_z, "all_ctrl_z": all_ctrl_z}

    cont_row = None
    if "SD_contentment_1P" in acts:
        cont_1p, cont_3p = z_of("SD_contentment_1P"), z_of("SD_contentment_3P")
        cont_row = {**base, "contentment_1P_z": cont_1p, "contentment_3P_z": cont_3p, "contentment_mean_z": (cont_1p + cont_3p) / 2,
                   "all_desire_z": all_desire_z, "all_ctrl_z": all_ctrl_z}

    def z_or_nan(ds):
        return z_of(ds) if ds in acts else float("nan")

    snd_1p, snd_3p = z_or_nan("SexNoDesire_1P"), z_or_nan("SexNoDesire_3P")
    imp_1p, imp_3p = z_or_nan("ImplicitDesire_1P"), z_or_nan("ImplicitDesire_3P")
    f_mask = s2_cats == "F"
    heldout_row = {**base, "desire_all_z": all_desire_z, "desire_A1_z": a1_z,
                   "implicitdesire_1P_z": imp_1p, "implicitdesire_3P_z": imp_3p, "implicitdesire_mean_z": (imp_1p + imp_3p) / 2,
                   "numb_mean_z": (numb_1p + numb_3p) / 2,
                   "sexnodesire_1P_z": snd_1p, "sexnodesire_3P_z": snd_3p, "sexnodesire_mean_z": (snd_1p + snd_3p) / 2,
                   "sexcontent_F_z": float(all_z[f_mask].mean()) if f_mask.any() else float("nan"),
                   "all_ctrl_z": all_ctrl_z}
    return numb_row, cont_row, heldout_row


def main():
    models = sorted(d.name for d in RESULTS_DIR.iterdir() if d.is_dir() and find_file(d, "activations.pt"))
    print(f"Found {len(models)} models in {RESULTS_DIR}")
    for extraction_type in ["final_token", "mean"]:
        numb_rows, cont_rows, heldout_rows = [], [], []
        for model in models:
            print(f"[{extraction_type}] {model}...", end=" ")
            numb_row, cont_row, heldout_row = analyze_model(model, extraction_type)
            if heldout_row:
                heldout_rows.append(heldout_row)
            if numb_row:
                numb_rows.append(numb_row)
                print(f"numb z={numb_row['numb_mean_z']:+.3f}", end="")
            if cont_row:
                cont_rows.append(cont_row)
                print(f"  contentment z={cont_row['contentment_mean_z']:+.3f}", end="")
            print()
        pd.DataFrame(numb_rows).to_csv(OUT_DIR / f"numb_zscores_{extraction_type}.csv", index=False)
        if cont_rows:
            pd.DataFrame(cont_rows).to_csv(OUT_DIR / f"contentment_zscores_{extraction_type}.csv", index=False)
        if heldout_rows:
            h = pd.DataFrame(heldout_rows)
            h.to_csv(OUT_DIR / f"heldout_zscores_{extraction_type}.csv", index=False)
            cols = ["desire_all_z", "desire_A1_z", "implicitdesire_mean_z", "numb_mean_z", "sexnodesire_mean_z",
                    "sexcontent_F_z", "all_ctrl_z"]
            print(f"\n[{extraction_type}] topic check (z on the S2 desire vector):")
            print(h.set_index("model")[cols].round(2).to_string())
    print("done")


if __name__ == "__main__":
    main()

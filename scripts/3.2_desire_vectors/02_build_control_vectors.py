"""Control directions for every model, at the extraction layer by default.

With LAYERS_FILE = None: one file per model in OUT_DIR, at the layer of
final_token/desire_vectors.pt. With LAYERS_FILE set to a JSON {model_name: layer} (the
steer_layers_S1.json written by the Section 4.2 script): the same recipe at that layer,
with the desire vectors recomputed there by the same recipe. Those are the vector files
the Section 4.1 screen projects onto. With LAYERWISE = True: every direction at every
layer, stacked per key as [n_layers, d_model], with the desire vectors recomputed at each
layer; those are the files the Appendix C ablation reads.

For every model in RESULTS_DIR (output of 01_extract_activations_and_desire_vectors.py):
  - the S1 and S2 desire vectors are reused as saved in final_token/desire_vectors.pt
    (extraction layer) or recomputed at the requested layer;
  - adrenaline (B), affection (C1), joy (C2), bodily sensation (E), sexual content
    without desire (F, saved as sexcontent_vector) and sexual activity without desire (G,
    saved as sexactivity_vector) are the mean of
    that category's sentences pooled over S1_1P, S2_1P and ControlSupplement_1P, minus
    the mean of the pooled neutral sentences (category D) of the same three sets,
    denoised by projecting out the top principal components of the pooled neutral cloud
    (up to DENOISE_VARIANCE of its variance);
  - excitement, random, numb and contentment, and the held-out validation sets SexNoDesire,
    ImplicitDesire and InActNoDesire when they were extracted, are the mean of their own first-person set
    minus the same pooled neutral mean, denoised the same way.

Output: OUT_DIR/vectors_full_<model>.pt with {layer, <name>_vector, ...}, or with
LAYERWISE, LAYERWISE_DIR/vectors_layerwise_<model>.pt with {layers: n_layers, <name>_vector: [n_layers, d]}.
Loads one model's activations at a time.
"""

import gc
import json
from pathlib import Path

import torch
import numpy as np
from sklearn.decomposition import PCA

LAYERS_FILE = None                        # None = extraction layer; or "results/steering/steer_layers_S1.json"
OUT_DIR = Path("results") / "vectors_full"  # "results/vectors_full_steering" when LAYERS_FILE is set
LAYERWISE = False                         # True = every layer, for the Appendix C ablation
LAYERWISE_DIR = Path("results") / "vectors_layerwise"

RESULTS_DIR = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LAYERS = json.load(open(LAYERS_FILE)) if LAYERS_FILE else None

DENOISE_VARIANCE = 0.5
S_SETS = ["S1_1P", "S2_1P", "ControlSupplement_1P"]
DESIRE_CATEGORIES = ["A1", "A2", "A3", "A4", "A5"]
CONTROL_CATEGORIES = ["B", "C1", "C2", "D", "E", "F", "G"]


def find_file(base, name):
    for p in [base / name, base / base.name / name]:
        if p.exists():
            return p
    hits = list(base.rglob(name))
    return hits[0] if hits else None


def clean_mean(x):
    x = np.where(np.isinf(x), np.nan, x)
    return np.nanmean(x, axis=0)


def denoise_basis(neutral_acts, neutral_mean):
    X = np.nan_to_num(neutral_acts - neutral_mean, nan=0.0, posinf=0.0, neginf=0.0)
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


def compute_desire_vector(acts, cats):
    """Desire mean minus control mean, denoised against the control PCA."""
    cats = np.array(cats)
    desire_mean = np.nanmean(acts[np.isin(cats, DESIRE_CATEGORIES)], axis=0)
    control_acts = acts[np.isin(cats, CONTROL_CATEGORIES)]
    control_mean = np.nanmean(control_acts, axis=0)
    vec = np.nan_to_num(desire_mean - control_mean, nan=0.0, posinf=0.0, neginf=0.0)
    pca = PCA()
    pca.fit(control_acts - control_mean)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_comp = min(np.searchsorted(cumvar, DENOISE_VARIANCE) + 1, len(pca.components_))
    for d in pca.components_[:n_comp]:
        vec = vec - np.dot(vec, d) * d
    return vec


def vectors_at_layer(act, L, saved_desire=None):
    """Every direction at layer L. saved_desire: the S1/S2 desire vectors to reuse as
    saved; None recomputes them at L."""
    ft = act["activations"]["final_token"]

    def rows(ds, cats=None):
        a = ft[ds][L].float().numpy()
        if cats is None:
            return a
        mask = np.isin(np.array(act["metadata"][ds]["categories"]), cats)
        return a[mask]

    neutral = np.concatenate([rows(ds, ["D"]) for ds in S_SETS])
    neutral_mean = clean_mean(neutral)
    basis = denoise_basis(neutral, neutral_mean)

    def control_vec(acts):
        v = clean_mean(acts) - neutral_mean
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return project_out(v, basis)

    if saved_desire is not None:
        desire = saved_desire
    else:
        desire = {key: torch.tensor(compute_desire_vector(rows(ds), act["metadata"][ds]["categories"]))
                  for key, ds in (("s1_desire_vector", "S1_1P"), ("s2_desire_vector", "S2_1P"))}

    vectors = {
        **desire,
        "adrenaline_vector": torch.tensor(control_vec(np.concatenate([rows(ds, ["B"]) for ds in S_SETS]))),
        "affection_vector": torch.tensor(control_vec(np.concatenate([rows(ds, ["C1"]) for ds in S_SETS]))),
        "joy_vector": torch.tensor(control_vec(np.concatenate([rows(ds, ["C2"]) for ds in S_SETS]))),
        "bodysens_vector": torch.tensor(control_vec(np.concatenate([rows(ds, ["E"]) for ds in S_SETS]))),
        "sexcontent_vector": torch.tensor(control_vec(np.concatenate([rows(ds, ["F"]) for ds in S_SETS]))),
        "sexactivity_vector": torch.tensor(control_vec(np.concatenate([rows(ds, ["G"]) for ds in S_SETS]))),
        "excitement_vector": torch.tensor(control_vec(rows("Excitement_1P"))),
        "random_vector": torch.tensor(control_vec(rows("Random_1P"))),
    }
    if "Numb_1P" in ft:
        vectors["numb_vector"] = torch.tensor(control_vec(rows("Numb_1P")))
    if "SD_contentment_1P" in ft:
        vectors["contentment_vector"] = torch.tensor(control_vec(rows("SD_contentment_1P")))
    if "SexNoDesire_1P" in ft:
        vectors["sexnodesire_vector"] = torch.tensor(control_vec(rows("SexNoDesire_1P")))
    if "ImplicitDesire_1P" in ft:
        vectors["implicitdesire_vector"] = torch.tensor(control_vec(rows("ImplicitDesire_1P")))
    if "InActNoDesire_1P" in ft:
        vectors["inactnodesire_vector"] = torch.tensor(control_vec(rows("InActNoDesire_1P")))
    return vectors


def process_one_model(model_name):
    print(f"\n{'=' * 50}\nProcessing: {model_name}\n{'=' * 50}")

    if LAYERWISE:
        out_path = LAYERWISE_DIR / f"vectors_layerwise_{model_name}.pt"
    else:
        out_path = OUT_DIR / f"vectors_full_{model_name}.pt"
    if out_path.exists():
        print("  already done, skipping")
        return True

    vec_path = None
    for p in (RESULTS_DIR / model_name).rglob("desire_vectors.pt"):
        if "final_token" in str(p):
            vec_path = p
            break
    act_path = find_file(RESULTS_DIR / model_name, "activations.pt")
    if vec_path is None or act_path is None:
        print("  desire_vectors.pt or activations.pt not found, skipping")
        return False

    pv = torch.load(vec_path, map_location="cpu", weights_only=False)
    act = torch.load(act_path, map_location="cpu", weights_only=False)

    if LAYERWISE:
        n_layers = int(act["n_layers"])
        print(f"  all {n_layers} layers")
        per_layer = [vectors_at_layer(act, L) for L in range(n_layers)]
        stacks = {k: torch.stack([v[k].float() for v in per_layer]) for k in per_layer[0]}
        LAYERWISE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"layers": n_layers, **stacks}, out_path)
        print(f"  saved {out_path}")
        del act, pv, per_layer, stacks
        gc.collect()
        return True

    if LAYERS is None:
        L = int(pv["layer"])
        print(f"  extraction layer: {L}")
        saved = {"s1_desire_vector": pv["s1_desire_vector"], "s2_desire_vector": pv["s2_desire_vector"]}
    else:
        if model_name not in LAYERS:
            print("  no layer listed for this model, skipping")
            return False
        L = int(LAYERS[model_name])
        print(f"  requested layer: {L}")
        saved = None

    vectors = vectors_at_layer(act, L, saved)
    torch.save({"layer": L, **vectors}, out_path)
    print(f"  saved {out_path}")

    del act, pv, vectors
    gc.collect()
    return True


def main():
    models = sorted(p.name for p in RESULTS_DIR.iterdir() if p.is_dir() and list(p.rglob("activations.pt")))
    print(f"Found {len(models)} models in {RESULTS_DIR}")
    done, failed = 0, []
    for model in models:
        try:
            if process_one_model(model):
                done += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append(model)
            gc.collect()
    print(f"\nDone: {done} succeeded, {len(failed)} failed {failed if failed else ''}")


if __name__ == "__main__":
    main()

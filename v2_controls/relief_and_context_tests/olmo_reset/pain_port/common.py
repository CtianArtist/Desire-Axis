"""Identity-bound direction bindings for the unchanged author runtimes."""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.metadata
import json
import os
from types import SimpleNamespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = ROOT / "author" / "act-on-valence"
MODELS = {
    "olmo_32b": {"hf_id": "allenai/OLMo-2-0325-32B-Instruct", "revision": "b96024342a77a69aa0dda815c3454a671f477463", "rho": 0.30, "main_layer": 32, "width": 5120, "resid_rms": 25.491040760519905},
    "qwen25_32b": {"hf_id": "Qwen/Qwen2.5-32B-Instruct", "revision": "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd", "rho": 0.30, "main_layer": 32, "monitor_layer": 61, "width": 5120, "resid_rms": 222.6194515986239},
    "llama31_8b": {"hf_id": "meta-llama/Llama-3.1-8B-Instruct", "revision": "0e9e39f249a16976918f6564b8830bc894c89659", "rho": 0.60, "main_layer": 16, "monitor_layer": 28, "width": 4096, "resid_rms": 12.776974449729776},
}
PAIN_HASHES = {
    "qwen25_32b": "51a4f09e7ab8f5535381814a513e3b78d1e2aae93b9664943da3ad1b9329bc4b",
    "llama31_8b": "91a3c6363ece303fcdbd9b84f198f146561fc68030303e6b1d396561d615abfb",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def progress(step=None, total_steps=None, phase=None):
    print(json.dumps({"phase": phase, "step": step, "total_steps": total_steps}), flush=True)


def load_inputs(member, vectors_root, layer=None):
    if member == 'olmo_32b':
        return load_olmo_inputs(vectors_root, layer)
    from safetensors.numpy import load_file
    from act_on_valence.vectors import load_pole_bank
    from act_on_valence.audit.livegen import random_pool
    cfg = MODELS[member]
    layer = cfg["main_layer"] if layer is None else int(layer)
    allowed = {32, 38} if member == "qwen25_32b" else {16}
    if layer not in allowed:
        raise ValueError(f"unauthorized injection layer {member}:{layer}")
    bank_path = AUTHOR / "data" / "steering_vectors" / "banks" / member / "bank_routeB.json"
    bank = load_pole_bank(bank_path)
    assert bank["hf_id"] == cfg["hf_id"]
    assert bank["layer"] == cfg["main_layer"]
    assert bank["resid_rms"] == cfg["resid_rms"]
    vp = Path(vectors_root) / member
    md = json.loads((vp / "metadata.json").read_text())
    assert md["member"] == member and md["hf_id"] == cfg["hf_id"]
    assert md["revision"] == cfg["revision"]
    assert md["extraction_layer"] == cfg["monitor_layer"]
    assert md["width"] == cfg["width"]
    directions = load_file(vp / "directions.safetensors")
    assert set(directions) == {"pain", "sadness", "fear"}
    for key, v in directions.items():
        assert v.dtype == np.float32 and v.shape == (cfg["width"],)
        assert np.isfinite(v).all() and np.linalg.norm(v) > 0
        h = hashlib.sha256(v.tobytes()).hexdigest()
        assert h == md["tensor_sha256"][key]
        if key == "pain":
            assert h == PAIN_HASHES[member]
    poolpath = AUTHOR / "data" / "random_pools" / f"pool_h{cfg['width']}.npz"
    with np.load(poolpath, allow_pickle=False) as z:
        pool = z["pool16"].copy()
    assert pool.shape == (16, cfg["width"]) and pool.dtype == np.float32
    assert np.array_equal(pool, random_pool(cfg["width"], 16, 20260723))
    return {"member": member, "hf_id": cfg["hf_id"], "revision": cfg["revision"],
            "layer": layer, "bank": bank, "resid_rms": cfg["resid_rms"],
            "pole_l2": float(np.linalg.norm(np.asarray(bank["pos_pole"], np.float32))),
            "rpool": pool, "directions": directions, "monitor_axis": directions["pain"],
            "rho": cfg["rho"], "monitor_layer": cfg["monitor_layer"],
            "provenance": {"vectors": md, "bank_sha256": sha256_file(bank_path),
                           "random_pool_sha256": sha256_file(poolpath)}}


def load_olmo_inputs(vectors_root, layer=None):
    """Selected final-token S2 with immutable extraction provenance; injection stays32."""
    from act_on_valence.vectors import load_pole_bank
    from act_on_valence.audit.livegen import random_pool
    cfg = MODELS['olmo_32b']
    layer = 32 if layer is None else int(layer)
    if layer != 32:
        raise ValueError('OLMo injection site is fixed at layer32')
    vp = Path(vectors_root)
    md = json.loads((vp / 'analysis.json').read_text())
    # Actual metadata schema is checked against the saved analysis before launch.
    identity = json.loads((vp / 'final_token' / 'metadata.json').read_text())
    assert identity['hf_id'] == cfg['hf_id'] and identity['revision'] == cfg['revision']
    assert identity['width'] == 5120 and type(identity['layer']) is int
    assert 0 <= identity['layer'] < 64 and identity['pool'] == 'final_token'
    assert md['primary'] == 'final_token' and md['best_layers']['final_token'] == identity['layer']
    assert md['activation_revision_verified'] and all(md['activation_prompt_hashes_verified'].values())
    assert md['artifact_sha256']['final_token/metadata.json'] == sha256_file(vp/'final_token/metadata.json')
    for name, checksum in identity['file_sha256'].items():
        assert sha256_file(vp/'final_token'/name) == checksum
    axis = np.load(vp / 'final_token' / 'raw.npy', allow_pickle=False)
    assert axis.dtype == np.float32 and axis.shape == (5120,)
    assert np.isfinite(axis).all() and np.linalg.norm(axis) > 0
    assert hashlib.sha256(axis.tobytes()).hexdigest() == identity['raw_tensor_sha256']
    bank_path = AUTHOR / 'data/steering_vectors/banks/olmo_32b/bank_routeB.json'
    bank = load_pole_bank(bank_path)
    assert bank['hf_id'] == cfg['hf_id'] and bank['layer'] == layer
    assert bank['resid_rms'] == cfg['resid_rms']
    poolpath = AUTHOR / 'data/random_pools/pool_h5120.npz'
    with np.load(poolpath, allow_pickle=False) as z: pool = z['pool16'].copy()
    assert pool.shape == (16, 5120) and pool.dtype == np.float32
    assert np.array_equal(pool, random_pool(5120, 16, 20260723))
    return dict(member='olmo_32b', hf_id=cfg['hf_id'], revision=cfg['revision'], layer=32,
        bank=bank, resid_rms=cfg['resid_rms'], pole_l2=float(np.linalg.norm(np.asarray(bank['pos_pole'],np.float32))),
        rpool=pool, directions={'pain':axis}, monitor_axis=axis, rho=.30,
        monitor_layer=identity['layer'], provenance={'vectors':identity,
        'selection_sha256':sha256_file(vp/'analysis.json'), 'bank_sha256':sha256_file(bank_path),
        'random_pool_sha256':sha256_file(poolpath)})


def load_bundle(member, vectors_root, layer=None):
    import torch
    from huggingface_hub import snapshot_download
    from act_on_valence.audit.replay import ReplayModel
    bundle = load_inputs(member, vectors_root, layer)
    cfg = MODELS[member]
    assert torch.__version__.split('+')[0] == '2.14.0'
    assert importlib.metadata.version('transformers') == '5.17.0'
    assert np.__version__ == '2.3.4'
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA GPU required")
    if "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("authorized GPU type is H100")
    # Weight/tokenizer files are pinned together. The original model implementation
    # sees a local snapshot, so no unpinned from_pretrained lookup can occur.
    cache = Path(os.environ.get("PAIN_RESET_CACHE", str(ROOT / ".job-cache"))) / "hf"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    model_path = snapshot_download(cfg["hf_id"], revision=cfg["revision"], cache_dir=str(cache / "hub"),
                                   allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])
    if Path(model_path).name != cfg["revision"]:
        raise RuntimeError("Hub snapshot revision mismatch")
    progress(0, None, "model_loading")
    model = ReplayModel(model_path, dtype=torch.bfloat16, device="cuda")
    model.hf_id = cfg["hf_id"]
    assert model.hidden == cfg["width"]
    assert model.model.dtype == torch.bfloat16
    assert not getattr(model.model, "peft_config", None)
    assert not getattr(model.model, "is_quantized", False)
    model.model.requires_grad_(False)
    bundle["model"] = model
    import subprocess
    bundle["provenance"]["runtime"] = {
        "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "driver": subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip(),
        "transformers": importlib.metadata.version("transformers"),
        "tokenizers": importlib.metadata.version("tokenizers"),
        "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(0), "dtype": str(model.model.dtype),
        "attention_implementation": model.model.config._attn_implementation,
        "snapshot_revision": Path(model_path).name, "adapters": False,
    }
    return bundle


def pinned_template_date(ymd):
    """Execute the customer's date context without importing conflicting flat modules."""
    path = ROOT / "pain_port/date_helper.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "pinned_date"]
    assert len(defs) == 1
    ns = {"contextlib": contextlib, "C": SimpleNamespace(BASE_RUN_DATE=tuple(ymd))}
    exec(compile(ast.Module(body=defs, type_ignores=[]), str(path), "exec"), ns)
    return ns["pinned_date"](tuple(ymd))


def resolve_direction(bundle, name, point):
    """Canonical signed dose; valence poles are selected, never assumed negatives."""
    point = float(point)
    if point not in {-1., -.5, 0., .5, 1.}:
        # Removal tools have a continuous [0, 1] intensity within either ceiling.
        if not -1 <= point <= 1:
            raise ValueError(f"dose outside authorized range: {point}")
    kind = name
    index = int(name[6:]) if name.startswith("random") else None
    if not point or name in {"none", "zero"}:
        return None, 0., {"inj": 0., "injected_l2": 0., "direction_kind": "none", "dir_index": None, "dose": point}
    ratio = abs(point) * bundle["rho"]
    if name == "valence":
        vec = np.asarray(bundle["bank"]["pos_pole" if point > 0 else "neg_pole"], np.float32)
        raw_norm = float(bundle["bank"]["pole_raw_l2"])
        kind = "valenced"
    elif index is not None:
        if not 0 <= index < 16:
            raise ValueError("only original random directions 0..15 are authorized")
        vec = np.asarray(bundle["rpool"][index], np.float32) * bundle["pole_l2"]
        if point < 0:
            vec = -vec
        raw_norm = float(bundle["bank"]["pole_raw_l2"])
        kind = "random"
    elif name in {"pain", "sadness", "fear"}:
        vec = np.asarray(bundle["directions"][name], np.float32)
        if point < 0:
            vec = -vec
        raw_norm = float(np.linalg.norm(vec))
    else:
        raise ValueError(f"unknown direction {name}")
    scale = ratio * bundle["resid_rms"] / raw_norm
    norm = float(np.linalg.norm(vec) * scale)
    assert np.isclose(norm, ratio * bundle["resid_rms"], rtol=2e-6, atol=1e-7)
    return vec, float(scale), {"inj": ratio, "injected_l2": norm, "direction_kind": kind,
                                "dir_index": index, "dose": point}

"""Explicit saved-adapter identity for evaluation and resume, without model imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-32B-Instruct"
BASE_REVISION = "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
CANONICAL_VECTOR_CONTRACT = {
    "verifier_sha256": "ac31c14bfc55b2da18a4517adba9f350a79add7b1ab89f1f0645f43a10ec489e",
    "manifest_filename": "original_tensor_hashes_v2.json",
    "manifest_sha256": "9c92ef27887b689accf13f1b91c99071e3137d011afaca1b17c3107dc51534a8",
    "sadness_policy_sha256": "083021654ff45ffcee936a23e950446e7dab133041377a9b769bb737f186596b",
}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def adapter_identity(adapter_dir, completion_path, manifest_path, seed):
    """Require an explicit complete new adapter and verify every saved file hash.

    This reads the training output's canonical manifests. It never discovers a
    newest directory, downloads a published adapter, or accepts a partial train.
    Paths are omitted from the digest so moving exact artifacts permits resume.
    """
    if type(seed) is not int or seed not in (1, 2):
        raise ValueError("Expected prospectively fixed training seed 1 or 2")
    adapter = Path(adapter_dir)
    complete = json.loads(Path(completion_path).read_text())
    manifest = json.loads(Path(manifest_path).read_text())
    if complete["status"] != "completed" or complete["completed_optimizer_steps"] != 318 or complete["required_optimizer_steps"] != 318:
        raise ValueError("The full fixed training workload must be complete")
    if complete["seed"] != seed or manifest["seed"] != seed:
        raise ValueError("Training seed identity mismatch")
    recipe = manifest["recipe"]
    if recipe["model"] != BASE_MODEL or recipe["model_revision"] != BASE_REVISION:
        raise ValueError("Base checkpoint identity mismatch")
    if recipe["new_seeds"] != [1, 2] or manifest["source_estimated_optimizer_steps"] != 318:
        raise ValueError("Frozen recipe mismatch")
    if not complete["saved_adapter_tensor_equality"] or not complete["reload_forward_finite"]:
        raise ValueError("Saved adapter verification was not successful")
    expected = complete["adapter_hashes"]
    if not {"adapter_model.safetensors", "adapter_config.json", "tokenizer_config.json"} <= set(expected):
        raise ValueError("Missing required saved adapter/tokenizer files")
    if any(Path(name).name != name for name in expected):
        raise ValueError("Adapter manifest must list direct child files only")
    files = {p.name: sha256_file(p) for p in sorted(adapter.iterdir()) if p.is_file()}
    if files != expected:
        raise ValueError("Adapter/tokenizer bytes differ from verified training output")
    config = json.loads((adapter / "adapter_config.json").read_text())
    if config["r"] != 32 or config["lora_alpha"] != 64 or config["lora_dropout"] != 0.05 or config["bias"] != "none":
        raise ValueError("Saved LoRA configuration mismatch")
    if set(config["target_modules"]) != {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}:
        raise ValueError("Saved LoRA projections mismatch")
    identity = {"training_seed": seed, "model": BASE_MODEL, "model_revision": BASE_REVISION,
                "adapter_files": files, "recipe_digest": digest(recipe),
                "training_input_hashes": manifest["training_input_hashes"],
                "training_versions": manifest["versions"], "completed_optimizer_steps": 318}
    return {**identity, "identity_digest": digest(identity)}


def resume_digest(grid, protocol_config, identity):
    """Trial seeds stay unchanged; run/resume identity additionally binds adapter bytes."""
    return digest({"grid": grid, "protocol_config": protocol_config, "adapter_identity": identity})

"""Profile-only instrumentation; source loading, persistence and sampling stay inherited."""
import hashlib
import importlib.metadata
import math
import os
import time
from pathlib import Path

from pain_axis_b import runtime as _source
from pain_axis_b.runtime import *  # noqa: F403 - preserve the exact customer runtime interface
from pain_choice_profile.design import CONDITIONS, CONTENTS, digest, scenario_id, source_kind, validate_scenarios

FEAR_FP32_SHA256 = "297cc009dc1e9e527bea1cb0dfbc43de2c5f0a63e66bd64d04bf7f9d6a47825d"
S2_FP32_NORM = 143.83343505859375


def should_stop():
    seconds = float(os.environ.get("PROFILE_MAX_SECONDS", 12 * 3600))
    assert math.isfinite(seconds) and seconds > 0
    flag = torch.tensor(int(_source.STOP or time.monotonic() - _source.START > seconds), device="cuda")
    if world() > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def validate_manifest(scenarios, manifest):
    """Fail closed before creating output or downloading anything."""
    validate_scenarios(scenarios)
    assert set(scenarios) == set(CONTENTS), "Unexpected scenario groups"
    assert manifest["scenario_digest"] == digest(scenarios), "Scenario digest mismatch"
    rows = manifest["scenarios"]
    assert len(rows) == 172
    assert len({r["scenario_id"] for r in rows}) == 172, "Duplicate scenario IDs"
    assert len({r["content_sha256"] for r in rows}) == 172, "Duplicate scenario contents"
    expected = {scenario_id(content, i): (content, i, messages)
                for content in CONTENTS for i, messages in enumerate(scenarios[content])}
    assert set(expected) == {r["scenario_id"] for r in rows}
    for row in rows:
        content, index, messages = expected[row["scenario_id"]]
        assert row["content_group"] == content and row["scenario_index"] == index
        assert row["source_kind"] == source_kind(content)
        assert row["messages"] == messages, "Manifest/runtime text mismatch"
        assert row["content_sha256"] == digest(messages), "Content hash mismatch"
    return {"scenario_digest": digest(scenarios), "stimulus_manifest_digest": digest(manifest),
            "unique_scenarios": 172, "duplicate_scenarios": 0}


def verify_vector_files(inputs, fear_inputs, pins):
    assert pins["schema_version"] == 1 and pins["model"] == "Qwen_2.5_32B_instruct"
    assert pins["layer"] == 61 and pins["width"] == 5120 and pins["s2_norm_fp32"] == S2_FP32_NORM
    required = {"Qwen_2.5_32B_instruct_pain_vectors.safetensors", "Qwen_2.5_32B_instruct_pain_vectors.json",
                "original_tensor_hashes_v2.json", "vectors_Qwen_2.5_32B_instruct.pt",
                "verification.json", "fear_Qwen_2.5_32B_instruct.pt"}
    assert required <= set(pins["files"]), "Incomplete vector file pins"
    for name, expected in pins["files"].items():
        assert Path(name).name == name, "Pins must use file basenames"
        root = fear_inputs if name.startswith("fear_") or name == "verification_fear.json" else inputs
        path = Path(root) / name
        assert path.is_file() and sha256(path) == expected, f"Vector input identity mismatch: {name}"
    assert pins["fp32_tensor_sha256"]["fear_vector_matched_fp32"] == FEAR_FP32_SHA256
    assert set(pins["fp32_tensor_sha256"]) == {"s2_pain_vector", "sadness_vector_matched_fp32", "fear_vector_matched_fp32"}
    return {"vector_pins_digest": digest(pins), "vector_pins": pins}


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def provenance(entrypoint):
    from pain_choice_profile import design
    here = Path(entrypoint).resolve()
    files = {"entrypoint": here, "builder": here.with_name("build_protocol.py"),
             "reference_protocol": here.parent.parent / "inputs/reference/b2_protocol.py",
             "profile_runtime": Path(__file__), "design": Path(design.__file__),
             "source_runtime": Path(_source.__file__),
             "source_diagnostics": Path(_source.__file__).with_name("diagnostics.py"),
             "source_vector_verifier": Path(_source.__file__).with_name("verify_vectors.py")}
    return {"source_sha256": {name: sha256(path) for name, path in files.items()},
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")},
            "monitor_layer": 61, "steer_layer": 38, "conditions": CONDITIONS,
            "fear_fp32_sha256": FEAR_FP32_SHA256, "s2_fp32_norm": S2_FP32_NORM,
            "legacy_relief_means": "target", "press_effect": "descriptive_only",
            "prefill_capture": "last attended prompt position, first forward, before any sampling"}


class SteeringAudit(_source.SteeringAudit):
    """Keep source checks and check every direction-dose-stage, including random seeds."""
    def __init__(self, output, width):
        super().__init__(output, width)
        self.profile_seen = set()
        self.items = []
        self.expected_dirs = {}

    def set_items(self, items, directions):
        self.items = items
        self.expected_dirs = directions

    def check(self, hs, add, coeff, mask, dirs):
        super().check(hs, add, coeff, mask, dirs)
        stage = "prefill" if mask is not None else "decode"
        for i, item in enumerate(self.items):
            condition = CONDITIONS[item["condition_id"]]
            dose = condition["dose"]
            assert item["coeff"] == item["hist_coeff"] == dose
            direction = item["dir_kind"]
            expected_kind = condition["dir_kind"]
            assert (direction.startswith("rand") if expected_kind == "rand" else direction == expected_kind)
            key = (stage, item["condition_id"], direction)
            if key in self.profile_seen:
                continue
            assert float(coeff[i]) == dose, "Applied dose differs from condition"
            expected_dir = self.expected_dirs[direction]
            assert dirs.dtype == expected_dir.dtype == torch.bfloat16
            assert torch.equal(dirs[i], expected_dir), "Direction tensor changed"
            if add is None:
                assert dose == 0
                actual = torch.zeros_like(hs[i, -1]).float()
                intended = actual
            else:
                # Match the source multiplication order exactly, including BF16 rounding.
                if mask is not None and hs.shape[1] > 1:
                    expected = mask[i, :, None].to(hs.dtype) * coeff[i].to(hs.dtype) * expected_dir[None, :]
                    assert torch.equal(add[i], expected), "Mask/dose/direction application changed"
                    assert bool((add[i][mask[i] == 0] == 0).all())
                else:
                    expected = coeff[i].to(hs.dtype) * expected_dir[None, :]
                    assert torch.equal(add[i], expected)
                intended = expected[-1].float()
                actual = (hs[i, -1] + add[i, -1]).float() - hs[i, -1].float()
                assert bool(torch.isfinite(actual).all())
                if dose == 0:
                    assert bool((add[i] == 0).all())
                elif mask is None or bool(mask[i, -1]):
                    assert float(intended.norm()) > 0 and float(actual.norm()) > 0
            free, total = torch.cuda.mem_get_info()
            emit(self.output, "profile_steering_assertion", stage=stage,
                 condition_id=item["condition_id"], direction=direction, applied_dose=dose,
                 direction_bf16_sha256=tensor_sha256(expected_dir),
                 direction_norm_fp32=float(expected_dir.float().norm()),
                 intended_delta_norm=float(intended.norm()), actual_bf16_delta_norm=float(actual.norm()),
                 delta_roundoff_norm=float((actual - intended).norm()),
                 mask_preserved=True, free_bytes=free, total_bytes=total,
                 allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                 peak_allocated_bytes=torch.cuda.max_memory_allocated())
            self.profile_seen.add(key)


def early_validation(trials, output):
    """One immutable snapshot per condition, then an all-condition receipt; no forwards."""
    output = Path(output)
    complete = output / "early_validation_all_conditions.json"
    if complete.exists():
        return
    grouped = {name: [] for name in CONDITIONS}
    for trial in trials:
        choices = trial.record["choices"]
        if choices and choices[0]["turn"] == 0:
            grouped[trial.record["condition_id"]].append(trial.record)
    snapshots = {}
    for name, rows in grouped.items():
        if not rows:
            continue
        path = output / f"early_validation_{name}.json"
        if path.exists():
            snapshots[name] = json.loads(path.read_text())
            continue
        counts = {"target": 0, "other": 0, "unparseable": 0}
        first = rows[0]
        for row in rows:
            choice = row["choices"][0]
            assert math.isfinite(choice["prefill_proj_monitor"])
            assert choice["steer_coeff_now"] == CONDITIONS[name]["dose"]
            assert choice["prefill_generated_tokens"] == 0
            assert choice["prefill_attended"] is True
            assert choice["prompt_tokens"] == len(choice["prompt_token_ids"])
            assert choice["prompt_tokens"] > 0
            assert choice["prefill_position_id"] == choice["prompt_tokens"] - 1
            for p in (choice["p_x"], choice["p_y"]):
                assert math.isfinite(p) and 0 <= p <= 1
            counts[{"relief": "target", "other": "other", None: "unparseable"}[choice["chose"]]] += 1
        receipt = {"status": "passed", "condition_id": name, "dose": CONDITIONS[name]["dose"],
                   "literal_turn_zero_count": len(rows), "categories": counts,
                   "finite_prefill": True, "applied_dose_checked": True,
                   "p_x_sum": sum(r["choices"][0]["p_x"] for r in rows),
                   "p_y_sum": sum(r["choices"][0]["p_y"] for r in rows),
                   "example": {k: first[k] for k in ("condition_id", "scenario_id", "choices", "proj_segments")}}
        atomic_json(path, receipt)
        snapshots[name] = receipt
    if set(snapshots) == set(CONDITIONS):
        atomic_json(complete, {"status": "passed", "conditions": snapshots,
                              "doses": sorted({c["dose"] for c in CONDITIONS.values()}),
                              "no_separate_pilot": True})

"""Small JSON-only fixtures matching prepare_vectors.py's schema-version 1."""
from copy import deepcopy
import json

import pytest

from pain_seed_b.provenance import validate_shared_sadness_receipt

MODEL = "Qwen_2.5_32B_instruct"
OTHER_MODEL = "Qwen_2.5_72B_instruct"
VECTOR_SHA = "a" * 64
RECEIPT_PATH = ("models", MODEL, "sources", "pain", "author_reserialization_verification")


def source(name, digest="b" * 64, size=4096, *, pinned=False):
    result = {"path": f"/fake/{name}", "bytes": size, "sha256": digest}
    if pinned:
        result["pin_verified"] = True
    return result


def fake_report(rows=100):
    """Field names/nesting come directly from prepare_model/main and load_verified."""
    original = {
        "version": 1, "status": "passed", "model": MODEL,
        "candidate_path": "/fake/original.pt", "candidate_file_sha256": "c" * 64,
        "candidate_file_bytes": 44000, "original_source_ref": "original/32B.pt",
        "original_file_sha256": "d" * 64, "layer": 61, "layer_type": "plain Python int",
        "tensors": {
            name: {"dtype": "float32", "shape": [5120], "bytes": 20480,
                   "sha256": digest * 64, "matches_original_bytes": True}
            for name, digest in (("s1_pain_vector", "e"), ("s2_pain_vector", "f"))
        },
        "loader": "torch.load(weights_only=True), no added trusted globals",
        "conversion_performed": False, "vector_values_modified": False,
    }
    zero_counts = {"nan": 0, "posinf": 0, "neginf": 0}
    checks = {
        role: {"model_binding": "explicit_metadata", "model_fields": {"model": MODEL},
               "layer_fields": {"layer": 61}, "model_check": True, "layer_check": True}
        for role in ("sadness", "pain", "neutral")
    }
    checks.update(width_check=True, sadness_row_count_check=True)
    record = {
        "status": "passed", "model": MODEL, "expected_layer": 61, "expected_width": 5120,
        "sources": {
            "sadness": source("sadness.pt", pinned=True),
            "pain": {**source("original.pt", "c" * 64, 44000),
                     "author_reserialization_verification": deepcopy(original)},
            "neutral_member": {"member": f"{MODEL}_final_token.pt", "bytes": 4096,
                               "sha256": "b" * 64},
        },
        "checks": checks,
        "sadness_rows": {"SD_sadness_1P": rows, "SD_sadness_3P": 103},
        "row_policy": {"selected_dataset": "SD_sadness_1P", "selection": "all_available_rows",
                       "first_person_rows_used": rows, "duplicated_rows": 0,
                       "third_person_rows_used": 0},
        "sadness_source_dtypes": {"SD_sadness_1P": "torch.bfloat16",
                                  "SD_sadness_3P": "torch.bfloat16"},
        "construction_dtype": "float32, matching author recipe; final steering conversion bfloat16",
        "neutral_row_counts": {"S1_1P": {"all_rows": 200, "category_D_rows": 30},
                               "S2_1P": {"all_rows": 201, "category_D_rows": 40}},
        "ControlSupplement_1P": {"present": True, "included": False,
                                 "status": "present_but_excluded",
                                 "reason": "Authorized neutral pool is category D of S1_1P and S2_1P only"},
        "sadness_used": "SD_sadness_1P only; SD_sadness_3P validated but not used",
        "nonfinite_counts": {"SD_sadness_1P": dict(zero_counts),
                             "SD_sadness_3P": dict(zero_counts),
                             "saved_S2": dict(zero_counts),
                             "neutral": {name: dict(zero_counts) for name in ("S1_1P", "S2_1P")}},
        "pca_count": 12, "denoise_variance": 0.5,
        "neutral_mean_nonfinite": dict(zero_counts),
        "mean_difference_nonfinite_before_cleaning": dict(zero_counts),
        "unscaled_norm_fp32": 3.5, "saved_S2_norm_fp32": 8.0, "matched_norm_fp32": 8.0,
        "bf16_norm": 8.0, "bf16_values_norm_accumulated_fp32": 8.001,
        "normalization": "rv / rv.norm() * saved_S2.float().norm(); all operands FP32, then BF16",
        "output": source("vectors.pt", VECTOR_SHA, 100000),
    }
    report = {
        "schema_version": 1, "status": "passed", "models": {MODEL: record},
        "neutral_scope": "category D from S1_1P and S2_1P only",
        "sources": {"controls": source("controls.py", pinned=True),
                    "neutral_zip": source("neutral.zip"),
                    "source_manifest": source("manifest.json"),
                    "sadness_policy": source("sadness_policy.json")},
        "authorized_row_policy": {"selected_dataset": "SD_sadness_1P",
                                  "row_selection": "all_available_rows", "duplicate_rows": False,
                                  "required_exact_row_count": None, "require_nonempty": True},
    }
    # Exercise actual JSON container/scalar types, without creating data files.
    return json.loads(json.dumps(report, allow_nan=False)), original


def replace_at(obj, path, value):
    for key in path[:-1]:
        obj = obj[key]
    obj[path[-1]] = value


@pytest.mark.parametrize("rows", [100, 137])
@pytest.mark.parametrize("other_failed", [False, True])
def test_accepts_actual_first_person_counts_and_independent_model_status(rows, other_failed):
    report, original = fake_report(rows)
    if other_failed:
        report["status"] = "failed"
        report["models"][OTHER_MODEL] = {"status": "failed", "error": {"message": "other input missing"}}
    before = deepcopy(report)
    selected = validate_shared_sadness_receipt(report, MODEL, VECTOR_SHA, original)
    assert selected is report["models"][MODEL]
    assert selected["row_policy"]["first_person_rows_used"] == rows
    assert report == before


@pytest.mark.parametrize(("path", "value", "message"), [
    (("schema_version",), 2, "schema_version"),
    (("schema_version",), True, "schema_version"),
    (("status",), "pending", "report.status"),
    (("setup_error",), None, "setup_error"),
    (("models",), {}, MODEL),
    (("models", MODEL, "status"), "failed", "status"),
    (("models", MODEL, "error"), {}, "error"),
    (("models", MODEL, "model"), OTHER_MODEL, "model"),
    (("models", MODEL, "expected_layer"), 76, "expected_layer"),
    (("models", MODEL, "expected_width"), 8192, "expected_width"),
    (("models", MODEL, "output", "sha256"), "0" * 64, "sha256"),
    (("authorized_row_policy", "selected_dataset"), "SD_sadness_3P", "selected_dataset"),
    (("authorized_row_policy", "row_selection"), "first_100", "row_selection"),
    (("authorized_row_policy", "required_exact_row_count"), 100, "required_exact_row_count"),
    (("authorized_row_policy", "duplicate_rows"), True, "duplicate_rows"),
    (("authorized_row_policy", "require_nonempty"), False, "require_nonempty"),
    (("models", MODEL, "sadness_rows", "SD_sadness_1P"), 0, "SD_sadness_1P"),
    (("models", MODEL, "sadness_rows", "SD_sadness_1P"), True, "SD_sadness_1P"),
    (("models", MODEL, "row_policy", "first_person_rows_used"), 101, "first_person_rows_used"),
    (("models", MODEL, "row_policy", "duplicated_rows"), 1, "duplicated_rows"),
    (("models", MODEL, "row_policy", "third_person_rows_used"), 1, "third_person_rows_used"),
    (("models", MODEL, "row_policy", "selection"), "first_100", "selection"),
    (("neutral_scope",), "category D and ControlSupplement", "neutral_scope"),
    (("models", MODEL, "neutral_row_counts"), {"S1_1P": {}, "S2_3P": {}}, "neutral_row_counts"),
    (("models", MODEL, "neutral_row_counts", "S1_1P", "category_D_rows"), 201, "category_D_rows"),
    (("models", MODEL, "neutral_row_counts", "S2_1P", "category_D_rows"), 0, "category_D_rows"),
    (("models", MODEL, "ControlSupplement_1P", "included"), True, "included"),
    (("models", MODEL, "ControlSupplement_1P", "status"), "absent", "status"),
    (("models", MODEL, "denoise_variance"), 0.9, "denoise_variance"),
    (("models", MODEL, "pca_count"), 0, "pca_count"),
    (("models", MODEL, "pca_count"), 71, "pca_count"),
    (("models", MODEL, "pca_count"), 1.5, "pca_count"),
    (("models", MODEL, "matched_norm_fp32"), 8.01, "FP32 norm"),
    (("models", MODEL, "sources", "sadness", "pin_verified"), False, "pin_verified"),
    (("sources", "controls", "pin_verified"), False, "pin_verified"),
    (("models", MODEL, "sources", "pain", "sha256"), "0" * 64, "sha256"),
    (("models", MODEL, "sources", "pain", "bytes"), 1, "bytes"),
    (RECEIPT_PATH + ("layer",), 76, "layer"),
    (RECEIPT_PATH + ("model",), OTHER_MODEL, "model"),
    (RECEIPT_PATH + ("original_file_sha256",), "0" * 64, "original_file_sha256"),
    (RECEIPT_PATH + ("original_source_ref",), "wrong.pt", "original_source_ref"),
    (RECEIPT_PATH + ("conversion_performed",), True, "conversion_performed"),
    (RECEIPT_PATH + ("vector_values_modified",), True, "vector_values_modified"),
    (RECEIPT_PATH + ("tensors", "s1_pain_vector", "sha256"), "0" * 64, "sha256"),
    (RECEIPT_PATH + ("tensors", "s2_pain_vector", "sha256"), "0" * 64, "sha256"),
    (RECEIPT_PATH + ("tensors", "s1_pain_vector", "dtype"), "bfloat16", "dtype"),
    (RECEIPT_PATH + ("tensors", "s2_pain_vector", "shape"), [8192], "shape"),
    (RECEIPT_PATH + ("tensors", "s1_pain_vector", "bytes"), 10240, "bytes"),
    (RECEIPT_PATH + ("tensors", "s2_pain_vector", "matches_original_bytes"), False, "matches_original_bytes"),
])
def test_rejects_inconsistent_or_unauthorized_report(path, value, message):
    report, original = fake_report()
    replace_at(report, path, value)
    with pytest.raises(ValueError, match=message):
        validate_shared_sadness_receipt(report, MODEL, VECTOR_SHA, original)


@pytest.mark.parametrize("role", ["sadness", "pain", "neutral"])
@pytest.mark.parametrize("flag", ["model_check", "layer_check"])
def test_requires_each_identity_check(role, flag):
    report, original = fake_report()
    report["models"][MODEL]["checks"][role][flag] = False
    with pytest.raises(ValueError, match=flag):
        validate_shared_sadness_receipt(report, MODEL, VECTOR_SHA, original)


@pytest.mark.parametrize("flag", ["width_check", "sadness_row_count_check"])
def test_requires_width_and_row_count_checks(flag):
    report, original = fake_report()
    del report["models"][MODEL]["checks"][flag]
    with pytest.raises(ValueError, match=flag):
        validate_shared_sadness_receipt(report, MODEL, VECTOR_SHA, original)


@pytest.mark.parametrize("key", ["unscaled_norm_fp32", "saved_S2_norm_fp32", "matched_norm_fp32",
                                 "bf16_norm", "bf16_values_norm_accumulated_fp32"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 0, -1, True, "8", 10**400])
def test_requires_positive_finite_numeric_norms(key, value):
    report, original = fake_report()
    report["models"][MODEL][key] = value
    with pytest.raises(ValueError, match=key):
        validate_shared_sadness_receipt(report, MODEL, VECTOR_SHA, original)


@pytest.mark.parametrize(("saved", "matched", "accept"), [
    (8.0, 8.000009, True), (8.0, 8.00002, False),
    (100.0, 100.00009, True), (100.0, 100.0002, False),
])
def test_fp32_norm_tolerances(saved, matched, accept):
    report, original = fake_report()
    report["models"][MODEL].update(saved_S2_norm_fp32=saved, matched_norm_fp32=matched)
    if accept:
        validate_shared_sadness_receipt(report, MODEL, VECTOR_SHA, original)
    else:
        with pytest.raises(ValueError, match="FP32 norm"):
            validate_shared_sadness_receipt(report, MODEL, VECTOR_SHA, original)


def test_accepts_absent_supplement_and_independently_reserialized_original():
    report, original = fake_report()
    report["models"][MODEL]["ControlSupplement_1P"].update(present=False, status="absent")
    original.update(candidate_path="/other/candidate.safetensors", candidate_file_sha256="0" * 64,
                    candidate_file_bytes=45000, loader="safetensors")
    validate_shared_sadness_receipt(report, MODEL, VECTOR_SHA, original)


@pytest.mark.parametrize(("argument", "value", "message"), [
    ("report", None, "report"), ("model", OTHER_MODEL, "only"),
    ("vector_file_sha256", "not a hash", "SHA-256"),
    ("original_receipt", None, "original_receipt"),
])
def test_rejects_invalid_arguments(argument, value, message):
    report, original = fake_report()
    args = dict(report=report, model=MODEL, vector_file_sha256=VECTOR_SHA, original_receipt=original)
    args[argument] = value
    with pytest.raises(ValueError, match=message):
        validate_shared_sadness_receipt(**args)


def test_rejects_disagreement_with_fresh_canonical_tensor_receipt():
    report, original = fake_report()
    original["tensors"]["s1_pain_vector"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="s1_pain_vector.sha256"):
        validate_shared_sadness_receipt(report, MODEL, VECTOR_SHA, original)


@pytest.mark.parametrize("path", [
    ("authorized_row_policy", "required_exact_row_count"),
    ("sources", "controls", "pin_verified"),
    ("models", MODEL, "output"),
    RECEIPT_PATH + ("tensors", "s2_pain_vector"),
])
def test_missing_fields_raise_descriptive_value_errors(path):
    report, original = fake_report()
    obj = report
    for key in path[:-1]:
        obj = obj[key]
    del obj[path[-1]]
    with pytest.raises(ValueError, match=path[-1]):
        validate_shared_sadness_receipt(report, MODEL, VECTOR_SHA, original)

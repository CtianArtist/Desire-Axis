"""Validate the shared preparation JSON receipt without loading any vectors.

The caller supplies a freshly computed vector-file hash and the receipt returned
by the canonical original-vector verifier. This checks consistency with those
trusted inputs; it does not independently authenticate the preparation report.
"""
from __future__ import annotations

import math
import re

_MODEL = "Qwen_2.5_32B_instruct"
_LAYER = 61
_WIDTH = 5120
_NEUTRAL_SCOPE = "category D from S1_1P and S2_1P only"


def _require(condition, path, expectation):
    if not condition:
        raise ValueError(f"{path}: {expectation}")


def _mapping(value, path):
    _require(isinstance(value, dict), path, "expected a JSON object")
    return value


def _field(obj, key, path):
    _require(key in obj, f"{path}.{key}", "required field is missing")
    return obj[key]


def _object(obj, key, path):
    return _mapping(_field(obj, key, path), f"{path}.{key}")


def _equal(obj, key, expected, path):
    value = _field(obj, key, path)
    _require(type(value) is type(expected) and value == expected,
             f"{path}.{key}", f"expected {expected!r}")


def _positive_int(obj, key, path):
    value = _field(obj, key, path)
    _require(type(value) is int and value > 0, f"{path}.{key}", "expected a positive integer")
    return value


def _sha256(value, path):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             path, "expected a lowercase SHA-256 hex digest")
    return value


def _file_source(value, path):
    value = _mapping(value, path)
    file_path = _field(value, "path", path)
    _require(isinstance(file_path, str) and bool(file_path), path + ".path", "expected a nonempty recorded path")
    _positive_int(value, "bytes", path)
    _sha256(_field(value, "sha256", path), path + ".sha256")
    return value


def _positive_number(obj, key, path):
    value = _field(obj, key, path)
    # Avoid converting arbitrarily large JSON integers to float.
    finite = type(value) is int or (type(value) is float and math.isfinite(value))
    _require(finite and 0 < value <= 1.7976931348623157e308,
             f"{path}.{key}", "expected a positive finite number")
    return value


def _original_receipt(receipt, path):
    receipt = _mapping(receipt, path)
    for key, expected in (
        ("version", 1), ("status", "passed"), ("model", _MODEL), ("layer", _LAYER),
        ("layer_type", "plain Python int"), ("conversion_performed", False),
        ("vector_values_modified", False),
    ):
        _equal(receipt, key, expected, path)
    _sha256(_field(receipt, "original_file_sha256", path), f"{path}.original_file_sha256")
    source_ref = _field(receipt, "original_source_ref", path)
    _require(isinstance(source_ref, str) and bool(source_ref),
             f"{path}.original_source_ref", "expected a nonempty source reference")
    _sha256(_field(receipt, "candidate_file_sha256", path), f"{path}.candidate_file_sha256")
    _positive_int(receipt, "candidate_file_bytes", path)
    tensors = _object(receipt, "tensors", path)
    for name in ("s1_pain_vector", "s2_pain_vector"):
        tensor_path = f"{path}.tensors.{name}"
        tensor = _object(tensors, name, f"{path}.tensors")
        _equal(tensor, "dtype", "float32", tensor_path)
        shape = _field(tensor, "shape", tensor_path)
        _require(type(shape) is list and len(shape) == 1
                 and type(shape[0]) is int and shape[0] == _WIDTH,
                 f"{tensor_path}.shape", f"expected [{_WIDTH}]")
        _equal(tensor, "bytes", 4 * _WIDTH, tensor_path)
        _sha256(_field(tensor, "sha256", tensor_path), f"{tensor_path}.sha256")
        _equal(tensor, "matches_original_bytes", True, tensor_path)
    return receipt


def validate_shared_sadness_receipt(report, model, vector_file_sha256, original_receipt):
    """Return the selected model's existing schema-v1 record, or raise ValueError.

    A failed *other* model does not invalidate a passed 32B preparation. All row
    counts describe actual saved rows; no exact sadness count is imposed. The
    original receipt must come from the canonical verifier, not from this JSON.
    """
    _require(model == _MODEL, "model", f"only {_MODEL} is supported")
    _sha256(vector_file_sha256, "vector_file_sha256")
    report = _mapping(report, "report")
    _equal(report, "schema_version", 1, "report")
    _require("setup_error" not in report, "report.setup_error", "setup errors are not accepted")
    _require(report.get("status") in ("passed", "failed"),
             "report.status", "expected passed or failed")
    _equal(report, "neutral_scope", _NEUTRAL_SCOPE, "report")
    policy = _object(report, "authorized_row_policy", "report")
    for key, expected in (
        ("selected_dataset", "SD_sadness_1P"), ("row_selection", "all_available_rows"),
        ("duplicate_rows", False), ("required_exact_row_count", None), ("require_nonempty", True),
    ):
        _equal(policy, key, expected, "report.authorized_row_policy")
    sources = _object(report, "sources", "report")
    for source_name in ("controls", "neutral_zip", "source_manifest", "sadness_policy"):
        _file_source(_field(sources, source_name, "report.sources"), f"report.sources.{source_name}")
    controls = _object(sources, "controls", "report.sources")
    _equal(controls, "pin_verified", True, "report.sources.controls")

    models = _object(report, "models", "report")
    path = f"report.models.{model}"
    record = _object(models, model, "report.models")
    for key, expected in (("status", "passed"), ("model", model),
                          ("expected_layer", _LAYER), ("expected_width", _WIDTH)):
        _equal(record, key, expected, path)
    _require("error" not in record, path + ".error", "selected model has an error")
    output = _file_source(_object(record, "output", path), path + ".output")
    _equal(output, "sha256", vector_file_sha256, path + ".output")

    checks = _object(record, "checks", path)
    for role in ("sadness", "pain", "neutral"):
        check = _object(checks, role, path + ".checks")
        for flag in ("model_check", "layer_check"):
            _equal(check, flag, True, path + f".checks.{role}")
    for flag in ("width_check", "sadness_row_count_check"):
        _equal(checks, flag, True, path + ".checks")
    model_sources = _object(record, "sources", path)
    sadness = _file_source(_object(model_sources, "sadness", path + ".sources"), path + ".sources.sadness")
    _equal(sadness, "pin_verified", True, path + ".sources.sadness")
    pain_path = path + ".sources.pain"
    pain = _object(model_sources, "pain", path + ".sources")
    receipt_path = pain_path + ".author_reserialization_verification"
    embedded = _original_receipt(
        _field(pain, "author_reserialization_verification", pain_path), receipt_path)
    original = _original_receipt(original_receipt, "original_receipt")
    # Candidate files may be differently serialized, but original tensor bytes
    # and original source identity must agree with the canonical verifier.
    for key in ("original_file_sha256", "original_source_ref"):
        _equal(embedded, key, original[key], receipt_path)
    for name in ("s1_pain_vector", "s2_pain_vector"):
        for key in ("dtype", "shape", "bytes", "sha256", "matches_original_bytes"):
            _equal(embedded["tensors"][name], key, original["tensors"][name][key],
                   receipt_path + f".tensors.{name}")
    _equal(pain, "sha256", embedded["candidate_file_sha256"], pain_path)
    _equal(pain, "bytes", embedded["candidate_file_bytes"], pain_path)

    sadness_rows = _object(record, "sadness_rows", path)
    row_count = _positive_int(sadness_rows, "SD_sadness_1P", path + ".sadness_rows")
    _positive_int(sadness_rows, "SD_sadness_3P", path + ".sadness_rows")
    row_policy = _object(record, "row_policy", path)
    for key, expected in (
        ("selected_dataset", "SD_sadness_1P"), ("selection", "all_available_rows"),
        ("first_person_rows_used", row_count), ("duplicated_rows", 0), ("third_person_rows_used", 0),
    ):
        _equal(row_policy, key, expected, path + ".row_policy")
    neutral_counts = _object(record, "neutral_row_counts", path)
    _require(set(neutral_counts) == {"S1_1P", "S2_1P"}, path + ".neutral_row_counts",
             "expected only S1_1P and S2_1P category D counts")
    neutral_total = 0
    for name in ("S1_1P", "S2_1P"):
        count_path = path + f".neutral_row_counts.{name}"
        counts = _object(neutral_counts, name, path + ".neutral_row_counts")
        all_rows = _positive_int(counts, "all_rows", count_path)
        selected_rows = _positive_int(counts, "category_D_rows", count_path)
        _require(selected_rows <= all_rows, count_path, "category_D_rows exceeds all_rows")
        neutral_total += selected_rows
    supplement = _object(record, "ControlSupplement_1P", path)
    supplement_path = path + ".ControlSupplement_1P"
    _equal(supplement, "included", False, supplement_path)
    present = _field(supplement, "present", supplement_path)
    _require(type(present) is bool, supplement_path + ".present", "expected a boolean")
    _equal(supplement, "status", "present_but_excluded" if present else "absent", supplement_path)

    variance = _positive_number(record, "denoise_variance", path)
    _require(variance == 0.5, path + ".denoise_variance", "expected 0.5")
    pca_count = _positive_int(record, "pca_count", path)
    _require(pca_count <= min(_WIDTH, neutral_total), path + ".pca_count",
             "PCA count exceeds neutral-row count or vector width")
    norms = {key: _positive_number(record, key, path) for key in (
        "unscaled_norm_fp32", "saved_S2_norm_fp32", "matched_norm_fp32",
        "bf16_norm", "bf16_values_norm_accumulated_fp32",
    )}
    _require(math.isclose(norms["matched_norm_fp32"], norms["saved_S2_norm_fp32"],
                          rel_tol=1e-6, abs_tol=1e-5),
             path + ".matched_norm_fp32", "FP32 norm does not match saved S2 (rtol=1e-6, atol=1e-5)")
    return record

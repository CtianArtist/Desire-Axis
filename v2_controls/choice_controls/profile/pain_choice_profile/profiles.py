"""Equal-weight ten-pair profile geometry and centered amplitude-only calibration.

Inputs are baseline-subtracted rates, not percentage points. Bootstrap rows must
already be paired joint scenario draws, including the common-none baseline. No
resampling, independent shuffling, or model calls occur in this module.
"""

import numpy as np


_N_PAIRS = 10
# Absolute Euclidean-norm floor in rate units; separate from sampling uncertainty.
_NORM_EPSILON = float(64 * np.finfo(float).eps * np.sqrt(_N_PAIRS))
_CALIBRATION_WARNING = (
    "Centered amplitude-only calibration at the fitted null, with joint bootstrap "
    "errors and a nonnegative scale refitted in every replicate. Raw and residual "
    "percentile intervals are descriptive only; positive distance is not proof "
    "of distinct shape. No categorical unique-profile verdict is supplied. "
    "Uncertainty is limited to the supplied scenario-population bootstrap."
)


def _validated(vector, boot, name):
    vector = np.asarray(vector, dtype=float)
    boot = np.asarray(boot, dtype=float)
    if vector.shape != (_N_PAIRS,):
        raise ValueError(f"{name} must have shape (10,)")
    if boot.ndim != 2 or boot.shape[1] != _N_PAIRS or boot.shape[0] == 0:
        raise ValueError(f"{name}_boot must have shape (B, 10), with B >= 1")
    if not np.all(np.isfinite(vector)) or not np.all(np.isfinite(boot)):
        raise ValueError(f"{name} and its bootstrap must contain only finite rates")
    # Long-double intermediates protect finite float64 inputs against overflow.
    return vector.astype(np.longdouble), boot.astype(np.longdouble)


def _number(value):
    """Undefined or float64-unrepresentable output is JSON null, never NaN/Inf."""
    if not np.isfinite(value) or abs(value) > np.finfo(float).max:
        return None
    return float(value)


def _norm(value, axis=None):
    return np.sqrt(np.sum(value * value, axis=axis))


def _interval(values):
    if len(values) == 0:
        return None
    return [_number(x) for x in np.quantile(values, [0.025, 0.975])]


def _summary(vector, boot):
    norm = _norm(vector)
    error95 = np.quantile(_norm(boot - vector, axis=1), 0.95)
    # A constant bootstrap cannot capture population uncertainty, even if its
    # constant draw is offset from the point estimate supplied by the caller.
    spread = np.max(_norm(boot - boot[0], axis=1))
    degenerate = bool(spread <= _NORM_EPSILON)
    coordinate_degenerate = np.ptp(boot, axis=0) <= _NORM_EPSILON
    return {
        "norm_pp": _number(100 * norm),
        "error_norm95_pp": _number(100 * error95),
        "orientation_identifiable": bool(not np.any(coordinate_degenerate) and norm > max(error95, _NORM_EPSILON)),
        "empirical_orientation_defined": bool(norm > _NORM_EPSILON),
        "orientation_threshold_pp": _number(100 * max(error95, _NORM_EPSILON)),
        "numerical_epsilon_norm_pp": 100 * _NORM_EPSILON,
        "numerically_singular": bool(norm <= _NORM_EPSILON),
        "bootstrap_degenerate": degenerate,
        "has_degenerate_coordinates": bool(np.any(coordinate_degenerate)),
        "degenerate_pair_ids": (np.flatnonzero(coordinate_degenerate) + 1).tolist(),
        "degenerate_observed_zero_pair_ids": (np.flatnonzero(coordinate_degenerate & (np.abs(vector) <= _NORM_EPSILON)) + 1).tolist(),
        "bootstrap_replicates": int(len(boot)),
    }


def profile_vector_summary(vector, boot):
    """Return norm/error in pp and explicit orientation/degeneracy flags.

    Orientation requires nondegenerate coordinates and norm > both the 95th
    percentile Euclidean bootstrap error and a numerical floor. Constant draws
    cannot establish orientation by reporting artificial zero uncertainty.
    """
    vector, boot = _validated(vector, boot, "vector")
    return _summary(vector, boot)


def _fit(pain, control):
    """Vectorized fit; undefined scales are NaN internally and masked on export."""
    norm2 = np.sum(control * control, axis=-1)
    valid = norm2 > _NORM_EPSILON ** 2
    scale = np.full(np.shape(norm2), np.nan, dtype=np.longdouble)
    np.divide(np.sum(control * pain, axis=-1), norm2, out=scale, where=valid)
    scale = np.maximum(scale, 0)
    return scale, valid


def profile_comparison(pain, control, pain_boot, control_boot, *, control_name, position):
    """Compare ten equally weighted effects and calibrate an amplitude-only null.

    The upper-tail p-value uses (1 + exceedances)/(B + 1). It is withheld for
    degenerate bootstraps, unidentifiable observed fits, or any singular null
    replicate, rather than silently conditioning on successful fits. Quantiles
    over any defined replicate subset remain labeled descriptive with counts.
    """
    pain, pain_boot = _validated(pain, pain_boot, "pain")
    control, control_boot = _validated(control, control_boot, "control")
    if pain_boot.shape != control_boot.shape:
        raise ValueError("pain_boot and control_boot must have matching joint draw shapes")
    if not isinstance(control_name, str) or not isinstance(position, str):
        raise ValueError("control_name and position must be strings")

    b = len(pain_boot)
    pain_summary = _summary(pain, pain_boot)
    control_summary = _summary(control, control_boot)
    scale, valid = _fit(pain, control)
    raw = _norm(pain - control) / np.sqrt(_N_PAIRS)
    raw_boot = _norm(pain_boot - control_boot, axis=1) / np.sqrt(_N_PAIRS)
    boot_scale, boot_valid = _fit(pain_boot, control_boot)
    residual_boot = _norm(
        pain_boot[boot_valid] - boot_scale[boot_valid, None] * control_boot[boot_valid],
        axis=1,
    ) / np.sqrt(_N_PAIRS)
    degenerate = pain_summary["bootstrap_degenerate"] or control_summary["bootstrap_degenerate"]
    oriented = pain_summary["orientation_identifiable"] and control_summary["orientation_identifiable"]
    cosine_distance = None
    if oriented:
        cosine = np.sum((pain / _norm(pain)) * (control / _norm(control)))
        cosine_distance = _number(1 - np.clip(cosine, -1, 1))

    warnings = [_CALIBRATION_WARNING]
    result = {
        "control_name": control_name,
        "position": position,
        "pair_count": _N_PAIRS,
        "pair_weights": [0.1] * _N_PAIRS,
        "raw_rms_pp": _number(100 * raw),
        "raw_rms_interval_pp": _interval(100 * raw_boot),
        "interval_interpretation": "Descriptive 95% percentile joint-bootstrap intervals, not shape tests.",
        "scale": _number(scale) if valid else None,
        "scale_identifiable": bool(valid),
        "scale_at_nonnegative_boundary": bool(scale == 0) if valid else None,
        "fitted_control_pp": [None] * _N_PAIRS,
        "residual_pp": [None] * _N_PAIRS,
        "residual_rms_pp": None,
        "residual_rms_interval_pp": _interval(100 * residual_boot) if valid else None,
        "residual_bootstrap_defined_replicates": int(np.count_nonzero(boot_valid)),
        "residual_bootstrap_singular_control_replicates": int(np.count_nonzero(~boot_valid)),
        "cosine_shape_distance": cosine_distance,
        "pain_orientation_identifiable": pain_summary["orientation_identifiable"],
        "control_orientation_identifiable": control_summary["orientation_identifiable"],
        "pain_summary": pain_summary,
        "control_summary": control_summary,
        "bootstrap_degenerate": bool(degenerate),
        "bootstrap_replicates": int(b),
        "null_calibration": {
            "method": "fitted amplitude-only null plus centered joint errors; refit nonnegative scale",
            "p_value": None,
            "exceedance_count": None,
            "replicates_requested": int(b),
            "replicates_evaluated": 0,
            "defined_replicates": 0,
            "singular_control_replicates": 0,
            "fitted_null_identifiable": bool(valid),
            "residual_rms_quantile_probabilities": [0.025, 0.5, 0.95, 0.975],
            "residual_rms_quantiles_pp": None,
            "inference_available": False,
            "unavailable_reasons": [],
        },
        "warnings": warnings,
    }
    null = result["null_calibration"]
    if degenerate:
        null["unavailable_reasons"].append("bootstrap_degenerate")
        warnings.append(
            "At least one profile bootstrap has no numerical spread. The supplied "
            "draws do not capture scenario-population uncertainty; null p-value withheld."
        )
    if pain_summary["has_degenerate_coordinates"] or control_summary["has_degenerate_coordinates"]:
        null["unavailable_reasons"].append("bootstrap_degenerate_effect_coordinates")
        warnings.append("At least one pain/none or control/none coordinate has zero bootstrap spread. Observed zeros are not structural identities; retain bounded scalar intervals and withhold shape inference.")
    if not valid:
        null["unavailable_reasons"].append("observed_control_numerically_singular")
        warnings.append("Observed control norm is zero or numerically singular; amplitude fit and fitted null are unidentifiable.")
        return result

    fitted = scale * control
    residual = pain - fitted
    observed = _norm(residual) / np.sqrt(_N_PAIRS)
    result["fitted_control_pp"] = [_number(x) for x in 100 * fitted]
    result["residual_pp"] = [_number(x) for x in 100 * residual]
    result["residual_rms_pp"] = _number(100 * observed)
    # Shared errors can cancel in a residual even when both input coordinates
    # vary. That is another scalar-contrast degeneracy, not evidence of exact
    # population agreement/disagreement. Never invent variance or drop a pair.
    residual_errors = (pain_boot - pain) - scale * (control_boot - control)
    residual_degenerate = np.ptp(residual_errors, axis=0) <= _NORM_EPSILON
    result["degenerate_fixed_scale_residual_pair_ids"] = (np.flatnonzero(residual_degenerate) + 1).tolist()
    result["degenerate_nonzero_residual_pair_ids"] = (np.flatnonzero(residual_degenerate & (np.abs(residual) > _NORM_EPSILON)) + 1).tolist()
    if np.any(residual_degenerate):
        null["unavailable_reasons"].append("bootstrap_degenerate_residual_coordinates")
        warnings.append("A fixed-scale residual coordinate has zero joint-error spread; refitted null draws remain diagnostic but no inferential p-value is reported.")

    # The same row enters both errors. In particular, common-none covariance is
    # retained even though the pain error is centered at the amplitude-only fit.
    null_pain = fitted + (pain_boot - pain)
    null_control = control + (control_boot - control)
    null_scale, null_valid = _fit(null_pain, null_control)
    null_residual = _norm(
        null_pain[null_valid] - null_scale[null_valid, None] * null_control[null_valid],
        axis=1,
    ) / np.sqrt(_N_PAIRS)
    n_defined = int(np.count_nonzero(null_valid))
    n_singular = int(b - n_defined)
    exceedances = int(np.count_nonzero(null_residual >= observed))
    null.update({
        "replicates_evaluated": int(b),
        "defined_replicates": n_defined,
        "singular_control_replicates": n_singular,
        "exceedance_count": exceedances,
        "residual_rms_quantiles_pp": [
            _number(x) for x in np.quantile(100 * null_residual, [0.025, 0.5, 0.95, 0.975])
        ] if n_defined else None,
    })
    if n_singular:
        null["unavailable_reasons"].append("singular_null_control_replicates")
        warnings.append("Singular control replicates excluded from descriptive quantiles; null p-value withheld, not renormalized over valid draws.")
    if not null["unavailable_reasons"]:
        null["p_value"] = float((1 + exceedances) / (b + 1))
        null["inference_available"] = True
    if not np.all(boot_valid):
        warnings.append("Residual percentile interval uses only defined bootstrap fits; see replicate counts.")
    return result

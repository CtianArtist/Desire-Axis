"""Boundary-interval metadata must not present exact bootstrap equality as certainty."""
import importlib.util
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / 'src/qualify_intervals.py'
spec = importlib.util.spec_from_file_location('qualify_intervals', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_all_zero_contrast_keeps_quantile_but_no_inferential_bound():
    rows = [{'difference': 0., 'ci_low': 0., 'ci_high': 0.,
             'pointwise_ci_low': 0., 'pointwise_ci_high': 0., 'left_target': 0, 'right_target': 0}]
    assert module.qualify('contrasts', rows) == 2
    r = rows[0]
    assert r['difference'] == r['bootstrap_quantile_low'] == r['bootstrap_quantile_high'] == 0
    assert r['ci_low'] is r['ci_high'] is None
    assert r['interval_status'] == 'degenerate_inferentially_uninformative'
    assert r['left_target'] == r['right_target'] == 0


def test_nondegenerate_contrast_is_unchanged():
    rows = [{'difference': .08, 'ci_low': .02, 'ci_high': .12,
             'pointwise_ci_low': .03, 'pointwise_ci_high': .11}]
    assert module.qualify('contrasts', rows) == 0
    assert rows[0]['ci_low'] == .02 and rows[0]['ci_high'] == .12
    assert rows[0]['interval_status'] == 'available'

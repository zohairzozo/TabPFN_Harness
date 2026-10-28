"""Tests for coverage, pinball loss, calibration error and reliability data."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from pfngate.evals.calibration import (
    calibration_error,
    interval_coverage,
    level_index,
    mean_pinball_loss,
    observed_frequencies,
    pinball_loss,
    reliability_diagram,
)

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)


def test_interval_coverage_by_hand() -> None:
    y = np.array([0.0, 0.5, 1.0, 0.3])
    lo = np.array([0.1, 0.4, 0.2, 0.3])
    hi = np.array([0.2, 0.6, 0.9, 0.3])
    # 0.0 is below 0.1, 0.5 is inside, 1.0 is above 0.9, 0.3 is exactly on both ends.
    assert interval_coverage(y, lo, hi) == 0.5


def test_tolerance_covers_point_masses_at_the_clip() -> None:
    y = np.ones(4)
    hi = np.full(4, 1.0 - 1e-9)  # the backend's q95 falls a hair short of the clipped 1
    assert interval_coverage(y, np.zeros(4), hi) == 1.0
    assert interval_coverage(y, np.zeros(4), hi, tolerance=0.0) == 0.0


def test_interval_coverage_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        interval_coverage(np.zeros(3), np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError):
        interval_coverage(np.array([]), np.array([]), np.array([]))


def test_pinball_loss_by_hand() -> None:
    y = np.array([1.0])
    q = np.array([[0.0, 2.0]])
    # level 0.1, q=0 under-predicts by 1: 0.1 * 1. level 0.9, q=2 over-predicts by 1: 0.1 * 1.
    assert np.allclose(pinball_loss(y, q, (0.1, 0.9)), [0.1, 0.1])
    assert np.allclose(pinball_loss(y, np.array([[1.0, 1.0]]), (0.1, 0.9)), [0.0, 0.0])
    assert mean_pinball_loss(y, q, (0.1, 0.9)) == pytest.approx(0.1)


def test_pinball_loss_is_lowest_at_the_true_quantile() -> None:
    rng = np.random.default_rng(0)
    y = rng.standard_normal(20000)
    true_q = norm.ppf(LEVELS)
    best = pinball_loss(y, np.tile(true_q, (len(y), 1)), LEVELS)
    for shift in (-0.3, 0.3):
        worse = pinball_loss(y, np.tile(true_q + shift, (len(y), 1)), LEVELS)
        assert np.all(worse > best)


def test_calibrated_quantiles_have_small_calibration_error() -> None:
    rng = np.random.default_rng(1)
    y = rng.standard_normal(20000)
    q = np.tile(norm.ppf(LEVELS), (len(y), 1))
    assert calibration_error(y, q, LEVELS) < 0.01
    narrow = q * 0.2  # overconfident: the outer quantiles sit far too close to the median
    assert calibration_error(y, narrow, LEVELS) > 0.1


def test_observed_frequencies_and_reliability_data_by_hand() -> None:
    y = np.array([0.1, 0.4, 0.9])
    q = np.tile([0.2, 0.5], (3, 1))
    assert np.allclose(observed_frequencies(y, q, (0.25, 0.75)), [1 / 3, 2 / 3])
    rel = reliability_diagram(y, q, (0.25, 0.75))
    assert rel.levels == (0.25, 0.75) and rel.n == 3
    assert rel.to_dict() == {"levels": [0.25, 0.75], "observed": list(rel.observed), "n": 3}
    assert calibration_error(y, q, (0.25, 0.75)) == pytest.approx((1 / 12 + 1 / 12) / 2)


def test_level_index() -> None:
    assert level_index(LEVELS, 0.05) == 0 and level_index(LEVELS, 0.95) == 4
    with pytest.raises(ValueError, match="not one of"):
        level_index(LEVELS, 0.9)


def test_shape_checks() -> None:
    with pytest.raises(ValueError):
        pinball_loss(np.zeros(3), np.zeros((3, 2)), LEVELS)
    with pytest.raises(ValueError):
        calibration_error(np.array([np.nan]), np.zeros((1, 5)), LEVELS)

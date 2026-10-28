"""Calibration of quantile predictions: coverage, pinball loss and reliability.

Every function takes the true targets y (n,) and the predicted quantiles (n, n_levels) at
the given levels. A target counts as inside an interval or below a quantile with a small
tolerance, because the regression target is clipped to [0, 1] and has point masses at
exactly 0 and 1 (docs/LABELS.md). A predicted q95 of 0.9999999 should still cover 1.

These numbers need the true target, so they are ground truth. The gate never calls them
on the query trajectory. It only sees coverage on held-out CONTEXT trajectories.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_TOLERANCE = 1e-6


def _check(
    y: np.ndarray, quantiles: np.ndarray, levels: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    q = np.asarray(quantiles, dtype=float)
    lv = np.asarray(levels, dtype=float)
    if y.ndim != 1 or q.ndim != 2 or q.shape != (len(y), len(lv)) or len(y) == 0:
        raise ValueError(f"need y (n,) and quantiles (n, {len(lv)}), got {y.shape}, {q.shape}")
    if not (np.isfinite(y).all() and np.isfinite(q).all()):
        raise ValueError("y and quantiles must be finite")
    return y, q, lv


def level_index(levels: Sequence[float], level: float) -> int:
    """Column of a quantile level, or ValueError if the backend does not return it."""
    lv = np.asarray(levels, dtype=float)
    hits = np.flatnonzero(np.isclose(lv, level, rtol=0.0, atol=1e-9))
    if len(hits) != 1:
        raise ValueError(f"quantile level {level} is not one of {lv.tolist()}")
    return int(hits[0])


def interval_coverage(
    y: np.ndarray, lower: np.ndarray, upper: np.ndarray, tolerance: float = DEFAULT_TOLERANCE
) -> float:
    """Share of targets inside [lower, upper], ends included, with the tolerance."""
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y, lower, upper))
    if y.ndim != 1 or lo.shape != y.shape or hi.shape != y.shape or len(y) == 0:
        raise ValueError("y, lower and upper must be 1-D arrays of the same non-zero length")
    inside = (y >= lo - tolerance) & (y <= hi + tolerance)
    return float(inside.mean())


def pinball_loss(y: np.ndarray, quantiles: np.ndarray, levels: Sequence[float]) -> np.ndarray:
    """Mean pinball (quantile) loss for each level, shape (n_levels,). Lower is better.

    For level tau and prediction q the loss is tau * (y - q) if y >= q, else
    (1 - tau) * (q - y).
    """
    y, q, lv = _check(y, quantiles, levels)
    diff = y[:, None] - q
    return np.maximum(lv * diff, (lv - 1.0) * diff).mean(axis=0)


def mean_pinball_loss(y: np.ndarray, quantiles: np.ndarray, levels: Sequence[float]) -> float:
    """The pinball loss averaged over the levels.

    This is NOT the CRPS. The CRPS is twice the integral of the pinball loss over all
    levels in (0, 1). With only five levels the average is a rough proxy for it at best:
    it ignores the tails beyond the outer levels and weights the levels equally.
    """
    return float(pinball_loss(y, quantiles, levels).mean())


def observed_frequencies(
    y: np.ndarray,
    quantiles: np.ndarray,
    levels: Sequence[float],
    tolerance: float = DEFAULT_TOLERANCE,
) -> np.ndarray:
    """For each level, the share of targets at or below the predicted quantile."""
    y, q, _ = _check(y, quantiles, levels)
    return (y[:, None] <= q + tolerance).mean(axis=0)


def calibration_error(
    y: np.ndarray,
    quantiles: np.ndarray,
    levels: Sequence[float],
    tolerance: float = DEFAULT_TOLERANCE,
) -> float:
    """Mean absolute gap between each level and its observed frequency. 0 is perfect."""
    freq = observed_frequencies(y, quantiles, levels, tolerance)
    return float(np.abs(freq - np.asarray(levels, dtype=float)).mean())


@dataclass(frozen=True)
class ReliabilityData:
    """Points of a reliability diagram: nominal level against observed frequency."""

    levels: tuple[float, ...]
    observed: tuple[float, ...]
    n: int

    def to_dict(self) -> dict[str, Any]:
        return {"levels": list(self.levels), "observed": list(self.observed), "n": self.n}


def reliability_diagram(
    y: np.ndarray,
    quantiles: np.ndarray,
    levels: Sequence[float],
    tolerance: float = DEFAULT_TOLERANCE,
) -> ReliabilityData:
    """Data for a reliability diagram. A calibrated model lies on the diagonal."""
    freq = observed_frequencies(y, quantiles, levels, tolerance)
    return ReliabilityData(
        levels=tuple(float(v) for v in levels),
        observed=tuple(float(v) for v in freq),
        n=len(np.asarray(y)),
    )

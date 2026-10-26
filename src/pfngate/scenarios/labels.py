"""Regression targets and binary labels, as defined in docs/LABELS.md."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

DROPPED = -1  # binary label for windows between the two thresholds


def target(
    kind: str,
    time: np.ndarray,
    control: np.ndarray | None,
    params: Mapping[str, float],
    tip_time: float | None,
    horizon: float,
) -> np.ndarray:
    """Relative distance to criticality: 1 = far, 0 = at the transition."""
    if kind == "bifurcation_distance":
        if control is None:
            raise ValueError("bifurcation_distance needs the control parameter p(t)")
        p0, p_c = params["p0"], params["p_c"]
        return np.clip((p_c - control) / (p_c - p0), 0.0, 1.0)
    if kind == "time_to_tip":
        if tip_time is None:
            return np.full_like(time, np.nan)
        return np.clip((tip_time - time) / horizon, 0.0, 1.0)
    if kind == "control":
        return np.ones_like(time)
    raise ValueError(f"unknown label kind '{kind}'")


def binary_labels(
    target_values: np.ndarray, positive_max: float, negative_min: float
) -> np.ndarray:
    """1 if target <= positive_max, 0 if target >= negative_min, DROPPED otherwise or if NaN."""
    out = np.full(target_values.shape, DROPPED, dtype=np.int8)
    out[target_values <= positive_max] = 1
    out[target_values >= negative_min] = 0
    return out

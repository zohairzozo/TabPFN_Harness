"""The gate checks. Each is a pure function: inputs and thresholds in, a CheckResult out.

No check reads a file, a global or the query targets. Thresholds are passed in by the
caller (gate.py), which takes them from the frozen GateConfig. See docs/GATE.md.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

Comparison = Literal[">=", "<=", "within", "== and <="]
Source = Literal["fixed", "provisional", "tuned"]


class CheckResult(BaseModel):
    """One check: what was measured, against what, and whether it passed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    value: float | None  # None when it could not be measured; the check then fails
    threshold: float | tuple[float, float]
    comparison: Comparison
    passed: bool
    reason: str
    source: Source


def min_context_size(n_context: int, threshold: int, source: Source = "fixed") -> CheckResult:
    passed = n_context >= threshold
    return CheckResult(
        name="min_context_size",
        value=float(n_context),
        threshold=float(threshold),
        comparison=">=",
        passed=passed,
        reason=f"{n_context} context rows, need at least {threshold}",
        source=source,
    )


def ood_share(
    distances: np.ndarray, distance_cut: float, limit: float, source: Source
) -> CheckResult:
    """Share of query windows farther from the pool than the cut."""
    d = np.asarray(distances, dtype=float)
    if d.ndim != 1 or len(d) == 0 or not np.isfinite(d).all():
        raise ValueError("need a non-empty 1-D array of finite distances")
    share = float((d > distance_cut).mean())
    return CheckResult(
        name="ood_share",
        value=share,
        threshold=float(limit),
        comparison="<=",
        passed=share <= limit,
        reason=f"{share:.3f} of query windows are far from the pool, limit {limit}",
        source=source,
    )


def near_tipping_support(
    distances: np.ndarray,
    n_near: int,
    k: int,
    distance_cut: float,
    limit: float,
    source: Source = "fixed",
) -> CheckResult:
    """Share of query windows whose k-th nearest near-tipping context row is beyond the cut.

    A context with fewer than k near-tipping rows fails outright (value None): the PFN has
    nothing to say "close to tipping" with.
    """
    d = np.asarray(distances, dtype=float)
    if d.ndim != 1 or len(d) == 0:
        raise ValueError("distances must be a non-empty 1-D array")
    if n_near < k:
        return CheckResult(
            name="near_tipping_support",
            value=None,
            threshold=float(limit),
            comparison="<=",
            passed=False,
            reason=f"only {n_near} near-tipping context rows, need {k}",
            source=source,
        )
    share = float(np.mean(d > distance_cut))
    return CheckResult(
        name="near_tipping_support",
        value=share,
        threshold=float(limit),
        comparison="<=",
        passed=share <= limit,
        reason=f"{share:.3f} of query windows have no near-tipping context row within the cut",
        source=source,
    )


def interval_width_cap(
    lower: np.ndarray, upper: np.ndarray, cap: float, source: Source = "fixed"
) -> CheckResult:
    """Median width of the interval [lower, upper] over the query windows."""
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if lo.shape != hi.shape or lo.ndim != 1 or len(lo) == 0:
        raise ValueError("lower and upper must be 1-D arrays of the same non-zero length")
    width = float(np.median(hi - lo))
    return CheckResult(
        name="interval_width_cap",
        value=width,
        threshold=float(cap),
        comparison="<=",
        passed=width <= cap,
        reason=f"median 90% interval width {width:.3f}, cap {cap}",
        source=source,
    )


def context_calibration(
    coverage: float | None,
    n_trajectories: int,
    min_trajectories: int,
    low: float,
    high: float,
    source: Source,
) -> CheckResult:
    """Coverage of the 90% interval on held-out context trajectories must lie in [low, high]."""
    if coverage is None or n_trajectories < min_trajectories:
        return CheckResult(
            name="context_calibration",
            value=None,
            threshold=(float(low), float(high)),
            comparison="within",
            passed=False,
            reason=f"too few trajectories to calibrate: {n_trajectories} in the context, "
            f"need {min_trajectories}",
            source=source,
        )
    passed = low <= coverage <= high
    return CheckResult(
        name="context_calibration",
        value=float(coverage),
        threshold=(float(low), float(high)),
        comparison="within",
        passed=passed,
        reason=f"held-out context coverage {coverage:.3f}, allowed [{low}, {high}]",
        source=source,
    )


def context_not_degenerate(
    context_y: np.ndarray,
    positive_max: float,
    negative_min: float,
    min_rows: int,
    source: Source = "fixed",
) -> CheckResult:
    """The context needs rows near the transition and rows far from it.

    The value is the smaller of the two counts. A context with a constant target, as in
    the oracle runs on controls, gives a constant prediction that says nothing.
    """
    y = np.asarray(context_y, dtype=float)
    n_near = int((y <= positive_max).sum())
    n_far = int((y >= negative_min).sum())
    value = min(n_near, n_far)
    return CheckResult(
        name="context_not_degenerate",
        value=float(value),
        threshold=float(min_rows),
        comparison=">=",
        passed=value >= min_rows,
        reason=f"{n_near} context rows with target <= {positive_max} and {n_far} with "
        f"target >= {negative_min}, need {min_rows} of each",
        source=source,
    )


def output_integrity(
    n_rows_repaired: int,
    n_rows: int,
    method: str,
    max_repaired_share: float,
    required_method: str,
    source: Source = "fixed",
) -> CheckResult:
    """Few repaired quantile rows, and p_critical from the full predictive distribution."""
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    share = n_rows_repaired / n_rows
    ok_method = method == required_method
    passed = ok_method and share <= max_repaired_share
    reason = f"{share:.3f} of rows had repaired quantiles (limit {max_repaired_share})"
    reason += f"; p_critical method {method}" + ("" if ok_method else f", need {required_method}")
    return CheckResult(
        name="output_integrity",
        value=float(share),
        threshold=float(max_repaired_share),
        comparison="== and <=",
        passed=passed,
        reason=reason,
        source=source,
    )

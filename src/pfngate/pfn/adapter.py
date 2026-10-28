"""The contract every PFN backend follows, and the checks they all share.

A backend is fitted on a context table and then asked about query rows. It answers
with quantiles of the target, a median, and p_critical: the probability that the
target is at or below a threshold. Callers only see PFNBackend and PFNOutput, so a
backend can be swapped without touching the gate or the agent.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

PCriticalMethod = Literal["full_distribution", "quantile_interpolation", "constant_target"]


class PFNMetadata(BaseModel):
    """Where an output came from and how it was made."""

    model_config = ConfigDict(extra="forbid")

    backend: str
    model_version: str
    n_context: int = Field(ge=0)
    fit_seconds: float = Field(ge=0.0)
    predict_seconds: float = Field(ge=0.0)
    quantiles_repaired: bool = False  # the backend returned crossing quantiles, we sorted them
    n_rows_repaired: int = Field(default=0, ge=0)
    p_critical_method: PCriticalMethod
    p_critical_threshold: float
    notes: list[str] = Field(default_factory=list)


class PFNOutput(BaseModel):
    """Predictions for n_query rows. Quantiles are sorted within every row."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    quantile_levels: np.ndarray  # (n_levels,), strictly increasing, inside (0, 1)
    quantiles: np.ndarray  # (n_query, n_levels), non-decreasing along each row
    median: np.ndarray  # (n_query,)
    p_critical: np.ndarray  # (n_query,), inside [0, 1]
    metadata: PFNMetadata

    @field_validator("quantile_levels", "quantiles", "median", "p_critical", mode="before")
    @classmethod
    def _as_read_only_float_array(cls, v: Any) -> np.ndarray:
        arr = np.array(v, dtype=float)  # a copy, so later changes by the caller do not leak in
        arr.flags.writeable = False
        return arr

    @field_serializer("quantile_levels", "quantiles", "median", "p_critical")
    def _to_list(self, v: np.ndarray) -> list[Any]:
        return v.tolist()

    @model_validator(mode="after")
    def _check_shapes_and_values(self) -> PFNOutput:
        lv, q = self.quantile_levels, self.quantiles
        if lv.ndim != 1 or len(lv) == 0:
            raise ValueError("quantile_levels must be a non-empty 1-D array")
        if np.any(lv <= 0.0) or np.any(lv >= 1.0) or np.any(np.diff(lv) <= 0.0):
            raise ValueError("quantile_levels must be strictly increasing and inside (0, 1)")
        if q.ndim != 2 or q.shape[1] != len(lv):
            raise ValueError(f"quantiles must have shape (n_query, {len(lv)}), got {q.shape}")
        for name in ("median", "p_critical"):
            if getattr(self, name).shape != (q.shape[0],):
                raise ValueError(f"{name} must have shape ({q.shape[0]},)")
        if not np.isfinite(q).all() or not np.isfinite(self.median).all():
            raise ValueError("quantiles and median must be finite")
        if np.any(np.diff(q, axis=1) < 0.0):
            raise ValueError("quantiles cross: they must be non-decreasing along each row")
        p = self.p_critical
        if not np.isfinite(p).all() or np.any(p < 0.0) or np.any(p > 1.0):
            raise ValueError("p_critical must be inside [0, 1]")
        return self

    @property
    def n_query(self) -> int:
        return self.quantiles.shape[0]


@runtime_checkable
class PFNBackend(Protocol):
    """Fit on a context table, then predict for query rows."""

    name: str

    def fit(self, context_X: np.ndarray, context_y: np.ndarray) -> None: ...

    def predict(self, query_X: np.ndarray) -> PFNOutput: ...


def check_context(context_X: np.ndarray, context_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(context_X, dtype=float)
    y = np.asarray(context_y, dtype=float)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(X) == 0:
        raise ValueError(f"need context_X (n, d) and context_y (n,), got {X.shape} and {y.shape}")
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ValueError("context contains NaN or infinite values")
    return X, y


def check_query(query_X: np.ndarray, n_features: int) -> np.ndarray:
    X = np.asarray(query_X, dtype=float)
    if X.ndim != 2 or X.shape[1] != n_features or len(X) == 0:
        raise ValueError(f"need query_X of shape (n, {n_features}), got {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("query contains NaN or infinite values")
    return X


def repair_quantiles(quantiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort each row. Returns the sorted array and a boolean per row: was it changed."""
    q = np.asarray(quantiles, dtype=float)
    fixed = np.sort(q, axis=1)
    return fixed, (fixed != q).any(axis=1)


def interpolate_p_critical(
    levels: np.ndarray, quantiles: np.ndarray, threshold: float
) -> np.ndarray:
    """P(target <= threshold) from quantiles, by linear interpolation of the CDF.

    Only the quantile levels are known, so this is coarse: a threshold below the lowest
    quantile gives 0 and one above the highest gives 1, although the real mass there
    is not exactly 0 or 1.
    """
    return np.array([np.interp(threshold, row, levels, left=0.0, right=1.0) for row in quantiles])


def build_output(
    *,
    levels: Sequence[float],
    quantiles: np.ndarray,
    threshold: float,
    backend: str,
    model_version: str,
    n_context: int,
    fit_seconds: float,
    predict_seconds: float,
    p_critical: np.ndarray | None = None,
    p_critical_method: PCriticalMethod = "quantile_interpolation",
    notes: Sequence[str] = (),
) -> PFNOutput:
    """Repair crossing quantiles, derive the median, fill in p_critical, and wrap it all up.

    If p_critical is None it is interpolated from the quantiles and the method is
    recorded as such, whatever p_critical_method says.
    """
    lv = np.asarray(levels, dtype=float)
    q, row_changed = repair_quantiles(quantiles)
    notes = list(notes)
    if row_changed.any():
        notes.append(f"{int(row_changed.sum())} of {len(q)} rows had crossing quantiles; sorted")
    if p_critical is None:
        p, method = interpolate_p_critical(lv, q, threshold), "quantile_interpolation"
        notes.append("p_critical interpolated from the quantiles (coarse at the tails)")
    else:
        p, method = np.clip(np.asarray(p_critical, dtype=float), 0.0, 1.0), p_critical_method
    median = np.array([np.interp(0.5, lv, row) for row in q])
    return PFNOutput(
        quantile_levels=lv,
        quantiles=q,
        median=median,
        p_critical=p,
        metadata=PFNMetadata(
            backend=backend,
            model_version=model_version,
            n_context=n_context,
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
            quantiles_repaired=bool(row_changed.any()),
            n_rows_repaired=int(row_changed.sum()),
            p_critical_method=method,
            p_critical_threshold=threshold,
            notes=notes,
        ),
    )

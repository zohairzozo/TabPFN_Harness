"""Metrics shared by every evaluation: AUROC with a trajectory bootstrap, false alarms, JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight


def balanced_auroc(y: np.ndarray, score: np.ndarray) -> float:
    """AUROC with sklearn 'balanced' sample weights, so both classes weigh the same in total.

    The weights are constant within a class, so this equals the unweighted AUROC. It is
    kept weighted so the metric stays class-balanced if per-window weights are added.
    """
    w = compute_sample_weight("balanced", y)
    return float(roc_auc_score(y, score, sample_weight=w))


def _group_members(groups: np.ndarray) -> list[np.ndarray]:
    codes, _ = pd.factorize(groups)
    order = np.argsort(codes, kind="stable")
    bounds = np.cumsum(np.bincount(codes))[:-1]
    return np.split(order, bounds)


def bootstrap_auroc(
    y: np.ndarray, score: np.ndarray, groups: np.ndarray, *, n_boot: int, seed: int, ci: float
) -> dict[str, Any]:
    """Balanced AUROC with a percentile bootstrap CI that resamples whole trajectories.

    Windows of one trajectory are strongly correlated, so resampling windows would give a
    CI that is far too narrow. A resample that contains only one class is skipped and
    counted in `n_boot_skipped`.
    """
    y, score, groups = np.asarray(y), np.asarray(score, float), np.asarray(groups)
    members = _group_members(groups)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        stats.append(balanced_auroc(yb, score[idx]))
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(stats, [alpha, 1.0 - alpha]) if stats else (np.nan, np.nan)
    return {
        "auroc": balanced_auroc(y, score),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci": ci,
        "n_boot": n_boot,
        "n_boot_skipped": n_boot - len(stats),
        "n_trajectories": len(members),
        "n_positive_windows": int((y == 1).sum()),
        "n_negative_windows": int((y == 0).sum()),
    }


def false_alarm_rate(score: np.ndarray, groups: np.ndarray, threshold: float) -> dict[str, Any]:
    """Share of control windows, and of control trajectories, with a score >= threshold."""
    score, groups = np.asarray(score, float), np.asarray(groups)
    alarm = score >= threshold
    per_traj = pd.Series(alarm).groupby(groups).any()
    return {
        "threshold": threshold,
        "window_rate": float(alarm.mean()),
        "trajectory_rate": float(per_traj.mean()),
        "n_windows": len(score),
        "n_trajectories": len(per_traj),
    }


def alarm_threshold_for_rate(scores: np.ndarray, rate: float) -> float:
    """The smallest threshold at which at most `rate` of these scores raise an alarm.

    Used on validation controls only. An alarm is score >= threshold (as in
    false_alarm_rate), so the threshold is the first float above the (1 - rate) quantile.
    Ties at the quantile therefore do not alarm.
    """
    s = np.asarray(scores, dtype=float)
    if len(s) == 0 or not 0.0 < rate < 1.0:
        raise ValueError("need some scores and a rate inside (0, 1)")
    return float(np.nextafter(np.quantile(s, 1.0 - rate, method="higher"), np.inf))


def _to_json(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"cannot write {type(obj).__name__} to JSON")


def write_json(path: Path, data: dict[str, Any]) -> Path:
    """Write results as indented JSON. NaN is written as null."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_nan_to_none(data), indent=2, default=_to_json, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def _nan_to_none(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, float | np.floating) and np.isnan(obj):
        return None
    return obj

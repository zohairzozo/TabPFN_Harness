"""The impure half of the gate: cached backend calls, context calibration refits, metrics.

The gate itself (pfngate.gate) is pure. This module produces its inputs, which needs the
backend and the context pool, and it scores the gate afterwards, which needs the ground
truth. Nothing here changes a GateResult or a threshold.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pfngate.evals.calibration import (
    calibration_error,
    interval_coverage,
    level_index,
    mean_pinball_loss,
    observed_frequencies,
    pinball_loss,
)
from pfngate.evals.pfn_cache import PFNCache, cache_key, query_hash
from pfngate.gate.config import GateConfig
from pfngate.gate.gate import GateInput, GateResult, evaluate_gate
from pfngate.gate.reference import PoolDistance, near_tipping_distances
from pfngate.pfn import PFNConfig
from pfngate.pfn.adapter import PFNBackend, PFNOutput
from pfngate.pfn.context import ContextSelection, context_hash, query_x


@dataclass
class FitPredict:
    """Fit on a context table and predict a query table, through the disk cache."""

    backend: PFNBackend
    cache: PFNCache
    cfg: PFNConfig
    model_version: str
    data_ver: str
    # The fit and predict time of every call, as the backend first measured it. A call that
    # is answered from the cache adds the time its entry was computed in, so the total is
    # what the same work costs without a cache.
    compute_seconds: float = 0.0

    def __call__(self, context: pd.DataFrame, query: pd.DataFrame) -> tuple[PFNOutput, bool]:
        key = cache_key(
            self.cfg,
            backend=self.backend.name,
            model_version=self.model_version,
            data_ver=self.data_ver,
            context_hash=context_hash(context["row_id"]),
            query_hash_=query_hash(query),
        )
        out = self.cache.get(key)
        if out is not None:
            self.compute_seconds += out.metadata.fit_seconds + out.metadata.predict_seconds
            return out, True
        self.backend.fit(query_x(context), context["target"].to_numpy(float))
        out = self.backend.predict(query_x(query))
        self.cache.put(key, out)
        self.compute_seconds += out.metadata.fit_seconds + out.metadata.predict_seconds
        return out, False


@dataclass(frozen=True)
class CalibrationRefit:
    coverage: float | None  # None if the context has fewer trajectories than folds
    n_trajectories: int
    n_rows: int
    n_folds: int
    n_cache_hits: int
    seconds: float


def trajectory_folds(trajectory_ids: Sequence[str], folds: int, seed: int) -> dict[str, int]:
    """Assign whole trajectories to folds, from the sorted ids and the seed."""
    tids = sorted(set(trajectory_ids))
    order = np.random.default_rng(seed).permutation(len(tids))
    return {tids[j]: i % folds for i, j in enumerate(order)}


def context_calibration_refit(
    fit_predict: FitPredict,
    context: pd.DataFrame,
    *,
    folds: int,
    seed: int,
    interval: tuple[float, float],
    tolerance: float,
) -> CalibrationRefit:
    """Hold out whole context trajectories, refit on the rest, and measure coverage.

    Only context rows (set A) and their targets are used, never the query.
    """
    t0 = time.perf_counter()
    tids = context["trajectory_id"].astype(str)
    n_traj = tids.nunique()
    if n_traj < folds:
        return CalibrationRefit(None, n_traj, 0, 0, 0, time.perf_counter() - t0)
    fold = tids.map(trajectory_folds(tids.tolist(), folds, seed)).to_numpy()
    y, lo, hi, hits = [], [], [], 0
    for i in range(folds):
        out, hit = fit_predict(context[fold != i], context[fold == i])
        hits += hit
        y.append(context.loc[fold == i, "target"].to_numpy(float))
        lo.append(out.quantiles[:, level_index(out.quantile_levels, interval[0])])
        hi.append(out.quantiles[:, level_index(out.quantile_levels, interval[1])])
    y_all = np.concatenate(y)
    coverage = interval_coverage(y_all, np.concatenate(lo), np.concatenate(hi), tolerance)
    return CalibrationRefit(coverage, n_traj, len(y_all), folds, hits, time.perf_counter() - t0)


def gate_one(
    gate_cfg: GateConfig,
    fit_predict: FitPredict,
    pool_distance: PoolDistance,
    selection: ContextSelection,
    query: pd.DataFrame,
    output: PFNOutput,
    *,
    seed: int,
    run_id: str,
) -> tuple[GateResult, CalibrationRefit]:
    """Build the gate input for one run and evaluate the gate. Never reads query targets."""
    refit = context_calibration_refit(
        fit_predict,
        selection.context,
        folds=gate_cfg.checks.context_calibration.folds,
        seed=seed,
        interval=gate_cfg.bad_run.interval,
        tolerance=gate_cfg.bad_run.tolerance,
    )
    tid = str(query["trajectory_id"].iloc[0])
    o, n = gate_cfg.checks.ood_share, gate_cfg.checks.near_tipping_support
    near = near_tipping_distances(
        selection.context,
        query_x(query),
        feature_mean=o.feature_mean,
        feature_std=o.feature_std,
        k=n.k,
        target_max=n.target_max,
    )
    inp = GateInput(
        run_id=run_id,
        trajectory_id=tid,
        strategy=selection.label,
        context_hash=selection.context_hash,
        context_size=selection.size,
        context_y=selection.context["target"].to_numpy(float),
        context_n_trajectories=refit.n_trajectories,
        calibration_coverage=refit.coverage,
        pool_distances=pool_distance.distances(query_x(query), tid),
        near_tipping_distances=near,
        output=output,
    )
    return evaluate_gate(gate_cfg, inp), refit


def ground_truth(query: pd.DataFrame, out: PFNOutput, gate: GateConfig) -> dict[str, Any]:
    """Realised calibration of one run on its own windows. The gate never sees this."""
    y = query["target"].to_numpy(float)
    lv = out.quantile_levels
    lo = out.quantiles[:, level_index(lv, gate.bad_run.interval[0])]
    hi = out.quantiles[:, level_index(lv, gate.bad_run.interval[1])]
    tol = gate.bad_run.tolerance
    cov = interval_coverage(y, lo, hi, tol)
    near = y <= gate.checks.context_not_degenerate.positive_max
    return {
        "coverage_90": cov,
        "bad": cov < gate.bad_run.min_coverage,
        "coverage_90_positive_windows": interval_coverage(y[near], lo[near], hi[near], tol)
        if near.any()
        else None,
        "n_positive_windows": int(near.sum()),
        "calibration_error": calibration_error(y, out.quantiles, lv, tol),
        "observed_frequency": dict(
            zip(
                map(str, lv.tolist()),
                observed_frequencies(y, out.quantiles, lv, tol).tolist(),
                strict=True,
            )
        ),
        "pinball_loss": dict(
            zip(map(str, lv.tolist()), pinball_loss(y, out.quantiles, lv).tolist(), strict=True)
        ),
        "mean_pinball_loss": mean_pinball_loss(y, out.quantiles, lv),
        "used_by_gate": False,
    }


def blocking_metrics(failed: Sequence[Collection[str]], bad: Sequence[bool]) -> dict[str, Any]:
    """Precision and recall of blocking BAD runs, from the failed checks of each run.

    Every run counts. Advisories are not in `failed`, so they never block. Values are None
    when they are undefined.
    """
    if len(failed) != len(bad):
        raise ValueError("need one list of failed checks per run")
    rows = [(bool(f), y) for f, y in zip(failed, bad, strict=True)]
    tp = sum(b and y for b, y in rows)
    n_blocked = sum(b for b, _ in rows)
    n_bad = sum(y for _, y in rows)
    n_good = len(rows) - n_bad
    return {
        "n_counted": len(rows),
        "n_blocked": n_blocked,
        "n_bad": n_bad,
        "n_bad_blocked": tp,
        "precision": tp / n_blocked if n_blocked else None,
        "recall": tp / n_bad if n_bad else None,
        "abstention_good": sum(b and not y for b, y in rows) / n_good if n_good else None,
    }


def bootstrap_blocking(
    failed: Sequence[Collection[str]], bad: Sequence[bool], *, n_boot: int, seed: int, ci: float
) -> dict[str, list[float] | None]:
    """Percentile CIs of precision and recall, resampling whole runs (one trajectory each)."""
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {"precision": [], "recall": []}
    for _ in range(n_boot):
        idx = rng.integers(0, len(bad), len(bad))
        m = blocking_metrics([failed[i] for i in idx], [bad[i] for i in idx])
        for key in draws:
            if m[key] is not None:
                draws[key].append(m[key])
    tail = (1.0 - ci) / 2.0 * 100.0
    return {
        key: [float(np.percentile(v, tail)), float(np.percentile(v, 100.0 - tail))] if v else None
        for key, v in draws.items()
    }


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def gate_summary(records: list[dict[str, Any]], gate: GateConfig, cfg: PFNConfig) -> dict[str, Any]:
    """Counts, abstention, coverage and precision and recall of blocking BAD runs."""
    failed = [tuple(r["gate"]["failed"]) for r in records]
    bad = [bool(r["ground_truth"]["bad"]) for r in records]
    passed = [not f for f in failed]
    cov = [float(r["ground_truth"]["coverage_90"]) for r in records]
    m = blocking_metrics(failed, bad) if records else None
    return {
        "config_sha256": gate.sha256,
        "config_status": gate.status,
        "note": None
        if gate.status == "tuned"
        else "provisional thresholds: these counts are for tuning, not a gate evaluation",
        "n_runs": len(records),
        "n_passed": sum(passed),
        "n_blocked": len(records) - sum(passed),
        "abstention_rate": (len(records) - sum(passed)) / len(records) if records else None,
        "blocked_by_check": dict(Counter(name for f in failed for name in f)),
        "advisories_failed": dict(
            Counter(a["name"] for r in records for a in r["gate"]["advisories"] if not a["passed"])
        ),
        "n_bad": sum(bad),
        "coverage_90_mean_all": _mean(cov),
        "coverage_90_mean_accepted": _mean([c for c, p in zip(cov, passed, strict=True) if p]),
        "bad_share_all": sum(bad) / len(bad) if bad else None,
        "bad_share_accepted": (
            sum(b for b, p in zip(bad, passed, strict=True) if p) / sum(passed)
            if any(passed)
            else None
        ),
        "blocking": m,
        "blocking_ci": bootstrap_blocking(
            failed, bad, n_boot=cfg.bootstrap_n, seed=cfg.bootstrap_seed, ci=cfg.bootstrap_ci
        )
        if records
        else None,
        "precision_blocking_bad": m["precision"] if m else None,
        "recall_blocking_bad": m["recall"] if m else None,
    }


def gate_false_alarms(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Window false alarms on control runs after the gate, per split group.

    An abstained window raises no alarm. `window_rate` counts alerts over all control
    windows; `window_rate_accepted` counts them over the windows of accepted runs only.
    None when the strategy has no alert threshold yet.
    """
    if not records or any(r["decisions"] is None for r in records):
        return None

    def rates(rs: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not rs:
            return None
        alerts = sum(r["decisions"]["alert"] for r in rs)
        windows = sum(r["n_windows"] for r in rs)
        acc = [r for r in rs if not r["gate"]["failed"]]
        acc_windows = sum(r["n_windows"] for r in acc)
        return {
            "n_trajectories": len(rs),
            "n_windows": windows,
            "n_accepted_trajectories": len(acc),
            "abstention_rate": 1.0 - len(acc) / len(rs),
            "window_rate": alerts / windows,
            "window_rate_accepted": alerts / acc_windows if acc_windows else None,
        }

    val = [r for r in records if r["split"] == "validation"]
    held = [r for r in records if r["split"] != "validation"]
    return {"validation": rates(val), "held_out_pooled": rates(held)}

"""Classic early-warning baseline: rising variance and rising lag-1 autocorrelation.

For every window we take the detrended variance and the lag-1 autocorrelation, then the
Kendall tau of each against time over the current window and the K - 1 windows before
it. The alarm score is the mean of the two taus. Only past windows are used, so a
window's score never depends on later data. The first K - 1 windows of a trajectory
have no score (NaN) and are left out of every metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from numpy.lib.stride_tricks import sliding_window_view

from pfngate.features import (
    META_COLUMNS,
    FeatureConfig,
    split_plan,
    target_defined,
    trajectory_id,
    trajectory_windows,
)
from pfngate.features.compute import kendall_tau_vs_time, linear_detrend
from pfngate.features.config import REPO_ROOT
from pfngate.features.windows import window_matrix
from pfngate.scenarios import Scenario, Trajectory
from pfngate.scenarios.config import CommonConfig

DEFAULT_BASELINE_CONFIG = REPO_ROOT / "configs" / "baseline_ews.yaml"
SCORE_COLUMNS: tuple[str, ...] = ("window_variance", "tau_variance_k", "tau_ac1_k", "score")


@dataclass(frozen=True)
class BaselineConfig:
    k_windows: int
    combine: str
    alarm_threshold: float
    splits: tuple[str, ...]
    bootstrap_n: int
    bootstrap_seed: int
    ci: float
    json_path: Path
    figure_path: Path


def load_baseline_config(path: Path = DEFAULT_BASELINE_CONFIG) -> BaselineConfig:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = BaselineConfig(
        k_windows=int(raw["rolling"]["k_windows"]),
        combine=str(raw["combine"]),
        alarm_threshold=float(raw["alarm_threshold"]),
        splits=tuple(raw["splits"]),
        bootstrap_n=int(raw["bootstrap"]["n"]),
        bootstrap_seed=int(raw["bootstrap"]["seed"]),
        ci=float(raw["bootstrap"]["ci"]),
        json_path=REPO_ROOT / raw["output"]["json"],
        figure_path=REPO_ROOT / raw["output"]["figure"],
    )
    if cfg.combine != "mean":
        raise ValueError(f"only combine: mean is implemented, got '{cfg.combine}'")
    if cfg.k_windows < 3:
        raise ValueError("k_windows must be at least 3 for a Kendall tau")
    return cfg


def rolling_tau(v: np.ndarray, k: int) -> np.ndarray:
    """Kendall tau against time of v[i - k + 1 : i + 1], for every i. NaN where i < k - 1."""
    out = np.full(len(v), np.nan)
    if len(v) >= k:
        out[k - 1 :] = kendall_tau_vs_time(sliding_window_view(np.asarray(v, float), k))
    return out


def window_variance(tr: Trajectory, ends: np.ndarray, fc: FeatureConfig) -> np.ndarray:
    """Variance of each linearly detrended window, averaged over channels."""
    per_channel = []
    for c in range(tr.observations.shape[1]):
        r, _ = linear_detrend(window_matrix(tr.observations[:, c], ends, fc.window_length))
        per_channel.append((r * r).mean(axis=-1))
    return np.mean(per_channel, axis=0)


def trajectory_scores(
    tr: Trajectory, fc: FeatureConfig, common: CommonConfig, cfg: BaselineConfig
) -> pd.DataFrame:
    """Window metadata, labels and the baseline score for one trajectory."""
    df = trajectory_windows(tr, fc, common)
    ends = df["window_end"].to_numpy(dtype=int)
    out = df[list(META_COLUMNS)].copy()
    out["window_variance"] = window_variance(tr, ends, fc) if len(ends) else []
    out["tau_variance_k"] = rolling_tau(out["window_variance"].to_numpy(), cfg.k_windows)
    out["tau_ac1_k"] = rolling_tau(df["ac1"].to_numpy(), cfg.k_windows)
    out["score"] = (out["tau_variance_k"] + out["tau_ac1_k"]) / 2.0
    return out


def score_split(
    split: str,
    scenarios: dict[str, Scenario],
    fc: FeatureConfig,
    cfg: BaselineConfig,
    seeds_per_scenario: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Baseline scores for every window of a split, plus the trajectories left out."""
    common = next(iter(scenarios.values())).common
    frames, dropped = [], []
    for sc, seed in split_plan(split, scenarios, seeds_per_scenario):
        tr = sc.generate(seed)
        if not target_defined(sc, tr):
            dropped.append(trajectory_id(sc.scenario_id, seed))
            continue
        frames.append(trajectory_scores(tr, fc, common, cfg))
    return pd.concat(frames, ignore_index=True), dropped

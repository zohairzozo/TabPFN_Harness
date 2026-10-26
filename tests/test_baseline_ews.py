"""Tests for the classic early-warning baseline and the shared metrics."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score

from pfngate.evals.baseline_ews import (
    BaselineConfig,
    load_baseline_config,
    rolling_tau,
    trajectory_scores,
)
from pfngate.evals.metrics import (
    balanced_auroc,
    bootstrap_auroc,
    false_alarm_rate,
    write_json,
)
from pfngate.features import load_feature_config
from pfngate.scenarios import load_scenarios

SCENARIOS = load_scenarios()
COMMON = next(iter(SCENARIOS.values())).common
FC = load_feature_config()
CFG = load_baseline_config()


# Baseline
def test_config_is_read_from_yaml() -> None:
    assert isinstance(CFG, BaselineConfig)
    assert CFG.k_windows == 16 and CFG.combine == "mean"


def test_rolling_tau_matches_scipy_and_is_nan_before_k() -> None:
    v = np.random.default_rng(0).standard_normal(40)
    k = 10
    out = rolling_tau(v, k)
    assert np.isnan(out[: k - 1]).all()
    for i in (k - 1, 20, 39):
        ref = kendalltau(np.arange(k), v[i - k + 1 : i + 1]).statistic
        assert out[i] == pytest.approx(ref)
    assert np.isnan(rolling_tau(v[:5], k)).all()


def test_rolling_tau_of_increasing_series_is_one() -> None:
    assert np.allclose(rolling_tau(np.arange(30.0), 8)[7:], 1.0)


@pytest.mark.parametrize("sid", ["fold_L2_slow", "hopf_L3_fast"])
def test_score_unchanged_when_later_data_is_altered(sid: str) -> None:
    sc = SCENARIOS[sid]
    tr = sc.generate(sc.seeds()[0])
    before = trajectory_scores(tr, FC, COMMON, CFG)
    cut = int(before["window_end"].iloc[len(before) // 2])
    obs = tr.observations.copy()
    obs[cut + 1 :] = 1000.0 * np.random.default_rng(1).standard_normal(obs[cut + 1 :].shape)
    after = trajectory_scores(dataclasses.replace(tr, observations=obs), FC, COMMON, CFG)
    keep = before["window_end"] <= cut
    assert keep.sum() >= CFG.k_windows
    assert np.allclose(before.loc[keep, "score"], after.loc[keep, "score"], equal_nan=True)
    assert not np.allclose(before.loc[~keep, "score"], after.loc[~keep, "score"], equal_nan=True)


def test_score_is_mean_of_the_two_taus() -> None:
    sc = SCENARIOS["fold_L1_slow"]
    df = trajectory_scores(sc.generate(sc.seeds()[0]), FC, COMMON, CFG)
    assert df["score"].isna().sum() == CFG.k_windows - 1
    ok = df["score"].notna()
    expected = (df.loc[ok, "tau_variance_k"] + df.loc[ok, "tau_ac1_k"]) / 2
    assert np.allclose(df.loc[ok, "score"], expected)
    assert df.loc[ok, "score"].between(-1, 1).all()


# Metrics
def test_balanced_auroc_equals_plain_auroc_and_handles_extremes() -> None:
    rng = np.random.default_rng(2)
    y = (rng.random(500) < 0.1).astype(int)
    s = rng.standard_normal(500) + y
    assert balanced_auroc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert balanced_auroc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert balanced_auroc(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_bootstrap_resamples_trajectories_not_windows() -> None:
    # Two trajectories: one all positive, one all negative. A resample of whole
    # trajectories has only one class about half the time. A window resample almost never.
    y = np.array([1] * 50 + [0] * 50)
    s = np.r_[np.ones(50), np.zeros(50)]
    groups = np.array(["a"] * 50 + ["b"] * 50)
    res = bootstrap_auroc(y, s, groups, n_boot=400, seed=0, ci=0.95)
    assert 120 < res["n_boot_skipped"] < 280
    assert res["n_trajectories"] == 2 and res["auroc"] == 1.0


def test_bootstrap_is_deterministic_and_ci_brackets_estimate() -> None:
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(30), 20)
    y = (rng.random(600) < 0.3).astype(int)
    s = rng.standard_normal(600) + 0.8 * y
    a = bootstrap_auroc(y, s, groups, n_boot=200, seed=5, ci=0.9)
    b = bootstrap_auroc(y, s, groups, n_boot=200, seed=5, ci=0.9)
    assert a == b
    assert a["ci_low"] <= a["auroc"] <= a["ci_high"]


def test_false_alarm_rate_by_hand() -> None:
    score = np.array([0.1, 0.6, 0.2, 0.3, 0.5, 0.0])
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    fa = false_alarm_rate(score, groups, threshold=0.5)
    assert fa["window_rate"] == pytest.approx(2 / 6)
    assert fa["trajectory_rate"] == pytest.approx(2 / 3)


def test_write_json_handles_numpy_and_nan(tmp_path: Path) -> None:
    out = write_json(
        tmp_path / "sub" / "r.json",
        {"a": np.float64(0.5), "b": np.int64(3), "c": np.arange(3), "d": float("nan")},
    )
    assert json.loads(out.read_text()) == {"a": 0.5, "b": 3, "c": [0, 1, 2], "d": None}

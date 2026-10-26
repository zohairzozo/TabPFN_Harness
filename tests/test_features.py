"""Tests for windowing, features and per-split datasets in src/pfngate/features."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kendalltau

from pfngate.features import (
    COLUMNS,
    FEATURE_NAMES,
    FeatureConfig,
    build_dataset,
    load_feature_config,
    split_plan,
    trajectory_windows,
    window_features,
)
from pfngate.features.compute import kendall_tau_vs_time
from pfngate.features.windows import window_ends
from pfngate.scenarios import load_scenarios

SCENARIOS = load_scenarios()
COMMON = next(iter(SCENARIOS.values())).common
FC = load_feature_config()
SPLITS = sorted({s.split for s in COMMON.seed_sets.values()})


@pytest.fixture(scope="module")
def small_datasets(tmp_path_factory: pytest.TempPathFactory) -> dict[str, pd.DataFrame]:
    """One seed per scenario for every split, cached in a temp dir (not results/)."""
    tmp: Path = tmp_path_factory.mktemp("cache")
    fc = dataclasses.replace(FC, cache_dir=tmp, summary_dir=tmp)
    return {s: build_dataset(s, seeds_per_scenario=1, feature_config=fc) for s in SPLITS}


# Shapes
def test_window_ends_count_and_spacing() -> None:
    ends = window_ends(1250, FC.window_length, FC.stride)
    assert ends[0] == FC.window_length - 1
    assert np.all(np.diff(ends) == FC.stride)
    assert len(ends) == (1250 - FC.window_length) // FC.stride + 1
    assert len(window_ends(FC.window_length - 1, FC.window_length, FC.stride)) == 0


def test_trajectory_windows_shape_and_metadata() -> None:
    sc = SCENARIOS["hopf_L2_fast"]  # two channels
    tr = sc.generate(1000)
    df = trajectory_windows(tr, FC, COMMON)
    assert list(df.columns) == list(COLUMNS)
    expected = window_ends(len(tr.time), FC.window_length, FC.stride)
    expected = expected[tr.time[expected] <= tr.tip_time]
    assert np.array_equal(df["window_end"].to_numpy(), expected)
    assert (df["trajectory_id"] == "hopf_L2_fast/1000").all()
    assert (df["split"] == "context").all() and (df["family"] == "hopf").all()
    assert np.allclose(df["target"], tr.target[expected])


def test_windows_end_at_or_before_tip(small_datasets: dict[str, pd.DataFrame]) -> None:
    for split, df in small_datasets.items():
        for tid, g in df.groupby("trajectory_id"):
            sid, seed = tid.rsplit("/", 1)
            tip = SCENARIOS[sid].generate(int(seed)).tip_time
            if tip is not None:
                assert g["window_end_time"].max() <= tip, (split, tid)


# No NaNs
def test_no_nans_in_features_or_labels(small_datasets: dict[str, pd.DataFrame]) -> None:
    for split, df in small_datasets.items():
        assert len(df) > 0, split
        assert np.isfinite(df[list(FEATURE_NAMES)].to_numpy()).all(), split
        assert df["target"].notna().all(), split
        assert set(df["label"].unique()) <= {-1, 0, 1}, split


# No future leakage
@pytest.mark.parametrize("sid", ["fold_L2_slow", "hopf_L3_fast", "regime_L2"])
def test_features_unchanged_when_later_data_is_altered(sid: str) -> None:
    sc = SCENARIOS[sid]
    tr = sc.generate(sc.seeds()[0])
    before = trajectory_windows(tr, FC, COMMON)
    cut = int(before["window_end"].iloc[len(before) // 2])  # last index left untouched

    obs = tr.observations.copy()
    rng = np.random.default_rng(0)
    obs[cut + 1 :] = 1000.0 * rng.standard_normal(obs[cut + 1 :].shape) + 50.0
    altered = trajectory_windows(dataclasses.replace(tr, observations=obs), FC, COMMON)

    keep_a = before[before["window_end"] <= cut].reset_index(drop=True)
    keep_b = altered[altered["window_end"] <= cut].reset_index(drop=True)
    assert len(keep_a) > 0
    pd.testing.assert_frame_equal(keep_a, keep_b)
    # Sanity: windows after the cut did change, so the test can detect leakage.
    later = before["window_end"] > cut
    assert not np.allclose(
        before.loc[later, list(FEATURE_NAMES)], altered.loc[later, list(FEATURE_NAMES)]
    )


# No trajectory in two splits
def test_no_trajectory_in_two_splits_plan() -> None:
    owner: dict[tuple[str, int], str] = {}
    for split in SPLITS:
        for sc, seed in split_plan(split, SCENARIOS):
            key = (sc.scenario_id, seed)
            assert key not in owner, f"{key} in {owner.get(key)} and {split}"
            owner[key] = split


def test_no_trajectory_in_two_splits_datasets(small_datasets: dict[str, pd.DataFrame]) -> None:
    ids = {s: set(df["trajectory_id"]) for s, df in small_datasets.items()}
    for a in SPLITS:
        assert (small_datasets[a]["split"] == a).all()
        for b in SPLITS:
            if a < b:
                assert not ids[a] & ids[b], (a, b)


def test_out_of_prior_families_only_in_out_of_prior_split(
    small_datasets: dict[str, pd.DataFrame],
) -> None:
    for split, df in small_datasets.items():
        has_oop = (df["role"] == "out_of_prior").any()
        assert has_oop == (split == "test_out_of_prior"), split
        assert (df["role"] == "control").any(), split


# Feature properties
def test_features_are_scale_free() -> None:
    rng = np.random.default_rng(1)
    w = np.cumsum(rng.standard_normal((5, FC.window_length)), axis=-1)
    a = window_features(w, FC)
    b = window_features(7.0 * w + 3.0, FC)
    for name in FEATURE_NAMES:
        assert np.allclose(a[name], b[name]), name


def test_kendall_tau_matches_scipy() -> None:
    rng = np.random.default_rng(2)
    v = rng.standard_normal((4, 13))
    ours = kendall_tau_vs_time(v)
    ref = [kendalltau(np.arange(13), row).statistic for row in v]
    assert np.allclose(ours, ref)


def test_ac1_detects_slower_recovery() -> None:
    # An AR(1) series with a coefficient closer to 1 should have a higher lag-1 autocorrelation.
    rng = np.random.default_rng(3)
    z = rng.standard_normal((200, FC.window_length))

    def ar1(phi: float) -> np.ndarray:
        x = np.zeros_like(z)
        for t in range(1, z.shape[1]):
            x[:, t] = phi * x[:, t - 1] + z[:, t]
        return x

    assert window_features(ar1(0.9), FC)["ac1"].mean() > window_features(ar1(0.3), FC)["ac1"].mean()


def test_feature_config_is_read_from_yaml() -> None:
    assert isinstance(FC, FeatureConfig)
    assert (FC.window_length, FC.stride) == (128, 16)

"""The real_tipping split. The tests use small synthetic files with the layout of the raw data,
so they run without the real download."""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pfngate.data import REAL_SPLIT
from pfngate.data.daphnia import (
    FAMILY_EXTINCT,
    FAMILY_PERSISTING,
    RawDataMissingError,
    RealConfig,
    build_real_dataset,
    build_trajectories,
    load_real_config,
    surrogate_target,
)
from pfngate.evals.llm_eval import HELD_OUT_LLM_SPLITS, LLM_SPLITS, gate_for_split
from pfngate.evals.run_pfn import CONTROL_FLAGS, HELD_OUT_FLAGS, SPLIT_FLAGS
from pfngate.features import build_dataset
from pfngate.gate import tune
from pfngate.gate.freeze import GateNotFrozenError
from pfngate.pfn.context import select_context

WEEKS = 45
EXTINCT_WEEK = 40
START = pd.Timestamp("2009-01-05")


def write_raw(raw_dir: Path) -> None:
    """Three populations: one extinct, one persisting, one restarted after an early die-off."""
    rng = np.random.default_rng(0)
    ts_rows = []

    def census(pid: str, dates: list[pd.Timestamp], level: np.ndarray) -> None:
        for chamber in (1, 2):
            for d, lv in zip(dates, level, strict=True):
                s = rng.poisson(lv, 3)
                ts_rows.append((pid, chamber, d.strftime("%m/%d/%Y"), *s, 0, 0, 1, 100))

    weeks = [START + pd.Timedelta(weeks=i) for i in range(WEEKS)]
    decay = np.maximum(0, EXTINCT_WEEK - 1.0 * np.arange(WEEKS)).astype(
        float
    )  # dies out at EXTINCT_WEEK
    census("A1", weeks, decay)
    census("B1", weeks, 25 + 5 * rng.random(WEEKS))
    early = [START - pd.Timedelta(weeks=60 - i) for i in range(5)]  # a first, dead population
    census("C1", early + weeks, np.concatenate([[5, 9, 4, 1, 0], 20 + 5 * rng.random(WEEKS)]))
    columns = ["ID", "Subpop", "Date ", "sample1", "sample2", "sample3"]
    columns += ["HoleNum", "HoleSize", "Deteriorating", "FoodLevel"]
    pd.DataFrame(ts_rows, columns=columns).to_csv(raw_dir / "timeseries.csv", index=False)

    end_a = (START + pd.Timedelta(weeks=EXTINCT_WEEK)).strftime("%m/%d/%Y")
    last = (START + pd.Timedelta(weeks=WEEKS - 1)).strftime("%m/%d/%Y")
    start = START.strftime("%m/%d/%Y")
    ext = [
        ("A1", start, end_a, 1, 0, 0, 1, 40, "Extinct during deterioration phase"),
        ("B1", start, last, 0, 0, 0, 0, 400, "Censored constant treatment"),
        ("C1 ", start, last, 0, 0, 0, 0, 400, "Censored constant treatment"),  # trailing space
        ("D1a", start, start, 1, 0, 0, 1, 400, "Extinct during pre-treatment phase"),
    ]
    cols = ["ID", "Start", "End", "Censor", "HoleSize", "HoleNum", "Deteriorating"]
    cols += ["FoodAtExtinct", "Notes"]
    pd.DataFrame(ext, columns=cols).to_csv(raw_dir / "extinctions.csv", index=False)


@pytest.fixture
def real_config(tmp_path: Path) -> RealConfig:
    write_raw(tmp_path)
    return dataclasses.replace(load_real_config(), raw_dir=tmp_path)


def test_the_loader_fails_clearly_when_the_raw_files_are_missing(tmp_path: Path) -> None:
    cfg = dataclasses.replace(load_real_config(), raw_dir=tmp_path / "nothing_here")
    with pytest.raises(RawDataMissingError) as err:
        build_trajectories(cfg.raw_dir, cfg)
    message = str(err.value)
    assert "extinctions.csv" in message and "timeseries.csv" in message
    assert "10.5061/dryad.q3p64" in message and str(cfg.raw_dir) in message


def test_the_loader_names_only_the_file_that_is_missing(tmp_path: Path) -> None:
    write_raw(tmp_path)
    (tmp_path / "timeseries.csv").unlink()
    cfg = dataclasses.replace(load_real_config(), raw_dir=tmp_path)
    with pytest.raises(RawDataMissingError, match="timeseries.csv missing"):
        build_trajectories(cfg.raw_dir, cfg)


def test_labels_are_in_zero_one_and_follow_the_surrogate_rule(real_config: RealConfig) -> None:
    df = build_real_dataset(real_config=real_config)
    assert df["target"].between(0.0, 1.0).all()
    assert set(df["label"].unique()) <= {-1, 0, 1}
    assert (df["split"] == REAL_SPLIT).all()
    extinct = df[df["family"] == FAMILY_EXTINCT]
    persisting = df[df["family"] == FAMILY_PERSISTING]
    assert len(extinct) > 0 and (extinct["label"] == 1).any() and (extinct["label"] == 0).any()
    assert (persisting["target"] == 1.0).all() and (persisting["label"] == 0).all()
    assert extinct["role"].eq("real").all() and persisting["role"].eq("control").all()
    # windows end at or before the extinction, and the target is the fixed-horizon time to it
    t_ext = EXTINCT_WEEK * 7.0
    assert (extinct["window_end_time"] <= t_ext).all()
    expected = np.clip((t_ext - extinct["window_end_time"]) / real_config.horizon_days, 0, 1)
    assert np.allclose(extinct["target"], expected)


def test_the_surrogate_target_is_clipped() -> None:
    t = np.array([0.0, 50.0, 100.0, 200.0, 250.0])
    out = surrogate_target(t, extinction_time=200.0, horizon_days=100.0)
    assert out.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]
    assert surrogate_target(np.array([150.0]), 200.0, 100.0).tolist() == [0.5]


def test_a_restarted_id_uses_only_the_census_from_its_start_date(real_config: RealConfig) -> None:
    trajectories, report = build_trajectories(real_config.raw_dir, real_config)
    c1 = next(t for t in trajectories if t.scenario_id == "daphnia_constant" and t.seed == 2)
    assert len(c1.time) == WEEKS and c1.time[0] == 0.0
    assert report["checks"]["ids_with_census_before_start_dropped"] == ["C1"]
    assert report["ids_without_census_series"] == ["D1a"]
    assert report["checks"]["census_gap_over_14_days"] == []


def test_reading_real_data_needs_the_frozen_gate_and_checks_it_first(
    tmp_path: Path, real_config: RealConfig
) -> None:
    missing = tmp_path / "no_gate.yaml"
    with pytest.raises(GateNotFrozenError):
        build_real_dataset(
            real_config=real_config, gate_config_path=missing, gate_hash_path=missing
        )
    # the gate is checked before the data is read: a missing raw folder does not change that
    bad = dataclasses.replace(real_config, raw_dir=tmp_path / "nothing_here")
    with pytest.raises(GateNotFrozenError):
        build_real_dataset(real_config=bad, gate_config_path=missing, gate_hash_path=missing)


def test_the_context_pool_never_holds_real_data(real_config: RealConfig) -> None:
    pool = build_dataset("context", seeds_per_scenario=1)
    assert (pool["split"] != REAL_SPLIT).all()
    assert not pool["trajectory_id"].str.startswith("daphnia").any()
    real = build_real_dataset(real_config=real_config)
    mixed = pd.concat([pool, real.iloc[:10]], ignore_index=True)
    with pytest.raises(ValueError, match="test only"):
        select_context("random", mixed, pool.iloc[:5], max_context=50)


def test_the_real_split_is_held_out_in_every_entry_point(tmp_path: Path) -> None:
    assert REAL_SPLIT in SPLIT_FLAGS["real"] and REAL_SPLIT in SPLIT_FLAGS["real_controls"]
    assert {"real", "real_controls"} <= HELD_OUT_FLAGS
    assert "real_controls" in CONTROL_FLAGS and "real" not in CONTROL_FLAGS
    assert LLM_SPLITS["real"] == REAL_SPLIT and "real" in HELD_OUT_LLM_SPLITS
    missing = tmp_path / "no_gate.yaml"
    with pytest.raises(GateNotFrozenError):
        gate_for_split("real", missing, missing)  # run_agent and baseline_llm start here


def test_real_data_is_not_a_tuning_input() -> None:
    source = inspect.getsource(tune)
    assert REAL_SPLIT not in source and "daphnia" not in source.lower()


def test_a_trial_run_keeps_the_first_populations_of_each_scenario_and_role(
    real_config: RealConfig,
) -> None:
    full = build_real_dataset(real_config=real_config)
    one = build_real_dataset(real_config=real_config, seeds_per_scenario=1)
    assert set(one["trajectory_id"]) <= set(full["trajectory_id"])
    assert one["trajectory_id"].nunique() == 2  # one extinct population, one persisting
    assert set(one["role"]) == {"real", "control"}

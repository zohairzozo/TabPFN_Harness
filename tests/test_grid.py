"""The real_control split (grid frequency, false alarms only) and the fetch script's rules.
The tests use small synthetic files, so they run without the download."""

from __future__ import annotations

import dataclasses
import importlib.util
import inspect
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from pfngate.data import REAL_CONTROL_SPLIT, REAL_SPLIT, REAL_SPLITS
from pfngate.data.daphnia import RawDataMissingError
from pfngate.data.grid import (
    GridConfig,
    build_real_control_dataset,
    find_raw_file,
    load_grid_config,
    read_slice,
)
from pfngate.evals.llm_eval import HELD_OUT_LLM_SPLITS, LLM_SPLITS, gate_for_split
from pfngate.evals.run_pfn import CONTROL_FLAGS, HELD_OUT_FLAGS, SPLIT_FLAGS
from pfngate.features import build_dataset
from pfngate.gate import tune
from pfngate.gate.freeze import GateNotFrozenError
from pfngate.pfn.context import select_context

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_grid.py"
DAY = 86_400


def load_fetch_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fetch_grid", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FG = load_fetch_script()


def write_slice(root: Path, days: int = 2, nan_runs: tuple[int, ...] = (3, 10)) -> Path:
    folder = root / "data" / "raw" / "grid"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    t = pd.date_range("2019-03-04", periods=days * DAY, freq="s")
    f = 50.0 + 0.02 * rng.standard_normal(len(t)) + 0.01 * np.sin(np.arange(len(t)) / 5000.0)
    pos = 1000
    for n in nan_runs:  # short holes
        f[pos : pos + n] = np.nan
        pos += 5000
    path = folder / "fingrid_2019-03-04_test.csv"
    pd.DataFrame({"time": t, "frequency_hz": f}).to_csv(path, index=False)
    return path


@pytest.fixture
def grid_cfg() -> GridConfig:
    return load_grid_config()


def test_the_loader_fails_clearly_when_the_raw_file_is_missing(
    tmp_path: Path, grid_cfg: GridConfig
) -> None:
    with pytest.raises(RawDataMissingError) as err:
        find_raw_file(grid_cfg, root=tmp_path)
    message = str(err.value)
    assert "scripts/fetch_grid.py" in message and "3744121" in message


def test_every_window_is_negative_and_the_target_is_one(
    tmp_path: Path, grid_cfg: GridConfig
) -> None:
    write_slice(tmp_path)
    df = build_real_control_dataset(grid_config=grid_cfg, root=tmp_path, summary_dir=tmp_path / "s")
    assert (df["label"] == 0).all() and (df["target"] == 1.0).all()
    assert df["target"].between(0.0, 1.0).all()
    assert (df["split"] == REAL_CONTROL_SPLIT).all() and (df["role"] == "control").all()
    assert df["trajectory_id"].nunique() == 2  # one trajectory per day
    summary = (tmp_path / "s" / "real_control_summary.json").read_text(encoding="utf-8")
    assert '"test_only": true' in summary and "not a pre-blackout run" in summary


def test_a_long_nan_hole_stops_the_load(tmp_path: Path) -> None:
    path = write_slice(tmp_path, nan_runs=(100,))
    with pytest.raises(ValueError, match="NaN hole of 100 s"):
        read_slice(path, max_gap=30)


def test_a_day_that_is_not_complete_stops_the_load(tmp_path: Path, grid_cfg: GridConfig) -> None:
    path = write_slice(tmp_path, days=2)
    df = pd.read_csv(path).iloc[: DAY + 5000]  # the second day is cut short
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="not a complete day"):
        build_real_control_dataset(grid_config=grid_cfg, root=tmp_path)


def test_reading_it_needs_the_frozen_gate_and_checks_it_first(
    tmp_path: Path, grid_cfg: GridConfig
) -> None:
    missing = tmp_path / "no_gate.yaml"
    # no raw data either: the gate error comes first
    with pytest.raises(GateNotFrozenError):
        build_real_control_dataset(
            grid_config=grid_cfg, root=tmp_path, gate_config_path=missing, gate_hash_path=missing
        )


def test_the_context_pool_never_holds_real_control_data(
    tmp_path: Path, grid_cfg: GridConfig
) -> None:
    pool = build_dataset("context", seeds_per_scenario=1)
    assert not pool["split"].isin(REAL_SPLITS).any()
    assert not pool["trajectory_id"].str.startswith("grid_").any()
    write_slice(tmp_path)
    real = build_real_control_dataset(grid_config=grid_cfg, root=tmp_path)
    mixed = pd.concat([pool, real.iloc[:10]], ignore_index=True)
    with pytest.raises(ValueError, match="test only"):
        select_context("random", mixed, pool.iloc[:5], max_context=50)


def test_the_split_is_held_out_and_a_control_in_every_entry_point(tmp_path: Path) -> None:
    assert REAL_CONTROL_SPLIT in SPLIT_FLAGS["real_control"]
    assert "real_control" in HELD_OUT_FLAGS and "real_control" in CONTROL_FLAGS
    assert LLM_SPLITS["real_control"] == REAL_CONTROL_SPLIT
    assert "real_control" in HELD_OUT_LLM_SPLITS
    missing = tmp_path / "no_gate.yaml"
    with pytest.raises(GateNotFrozenError):
        gate_for_split("real_control", missing, missing)


def test_real_data_is_not_a_tuning_input() -> None:
    source = inspect.getsource(tune)
    for name in (REAL_SPLIT, REAL_CONTROL_SPLIT, "grid_fingrid"):
        assert name not in source


def test_the_two_real_splits_have_different_names() -> None:
    assert len(set(REAL_SPLITS)) == 2 and dataclasses.is_dataclass(GridConfig)


# The fetch script


def test_the_fetch_script_refuses_every_transnetbw_path() -> None:
    assert FG.allowed_area("Data_cleansed/TransnetBW/2019.zip") is None
    assert FG.allowed_area("Data_converted/TransnetBW/LICENSE.txt") is None
    assert FG.allowed_area("Data_cleansed/Continental/2019.zip") is None
    assert FG.allowed_area("Data_cleansed/Unknown/2019.zip") is None
    assert FG.allowed_area("Data_cleansed/Fingrid/2019.zip") == "fingrid"
    assert FG.allowed_area("Data_cleansed/Nationalgrid/2018.zip") == "nationalgrid"


def test_every_allowed_area_has_a_cc_by_licence_text() -> None:
    assert set(FG.AREA_LICENCE) == {"fingrid", "nationalgrid"}
    assert all("CC BY 4.0" in text for text in FG.AREA_LICENCE.values())


def rows(day: str, outlier: bool = False, n: int = DAY) -> list[tuple[str, float]]:
    t = pd.date_range(day, periods=n, freq="s").strftime("%Y-%m-%d %H:%M:%S")
    out = [(s, 50.0) for s in t]
    if outlier:
        out[100] = (out[100][0], 49.5)  # a large event
    return out


def test_the_stationary_stretch_skips_days_with_an_event_or_missing_rows() -> None:
    stream = (
        rows("2019-01-01")
        + rows("2019-01-02", outlier=True)
        + rows("2019-01-03")
        + rows("2019-01-04", n=DAY - 5)  # not complete
        + rows("2019-01-05")
        + rows("2019-01-06")
        + rows("2019-01-07")
    )
    times, values, days = FG.find_stationary_days(iter(stream), 2, lambda: 0)
    assert days == ["2019-01-05", "2019-01-06"]
    assert len(values) == 2 * DAY and times[0].startswith("2019-01-05")


def test_the_stationary_stretch_stops_at_the_byte_budget() -> None:
    with pytest.raises(RuntimeError, match="byte budget"):
        FG.find_stationary_days(iter(rows("2019-01-01")), 1, lambda: FG.MAX_BYTES + 1)


def test_a_trial_run_keeps_only_the_first_days(tmp_path: Path, grid_cfg: GridConfig) -> None:
    write_slice(tmp_path)
    df = build_real_control_dataset(grid_config=grid_cfg, root=tmp_path, seeds_per_scenario=1)
    assert df["trajectory_id"].nunique() == 1 and (df["label"] == 0).all()

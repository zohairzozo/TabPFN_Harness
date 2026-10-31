"""The split real_control: real power-grid frequency in normal operation.

TEST ONLY, and it exists ONLY to measure false alarms. Every window is labelled negative
(target 1, binary label 0). The data is normal operation of the Nordic grid: it is not a
pre-blackout run and it contains no tipping event. See docs/GRID.md.

Reading it calls assert_frozen() first, it is never in the context pool, and it is never used to
tune the gate.

Source: a slice of the Kruse, Schafer and Witthaut record (Zenodo 3744121, Fingrid files, CC BY
4.0), fetched by scripts/fetch_grid.py into data/raw/grid/ and listed in data/SOURCES.md.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from pfngate.data import REAL_CONTROL_SPLIT
from pfngate.data.daphnia import RawDataMissingError
from pfngate.features.config import FeatureConfig, load_feature_config
from pfngate.features.dataset import COLUMNS, trajectory_id, trajectory_windows
from pfngate.gate import assert_frozen
from pfngate.scenarios import Trajectory, load_scenarios
from pfngate.scenarios.config import DEFAULT_CONFIG_DIR

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GRID_CONFIG = REPO_ROOT / "configs" / "real_control.yaml"
SOURCES = REPO_ROOT / "data" / "SOURCES.md"
SCENARIO_ID = "grid_fingrid"
FAMILY = "real_grid_normal"
SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class GridConfig:
    raw_glob: str
    resample_seconds: int
    max_interpolated_gap_seconds: int
    source: Path  # the YAML file this was read from


def load_grid_config(path: Path = DEFAULT_GRID_CONFIG) -> GridConfig:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = GridConfig(
        raw_glob=str(raw["raw_glob"]),
        resample_seconds=int(raw["resample_seconds"]),
        max_interpolated_gap_seconds=int(raw["max_interpolated_gap_seconds"]),
        source=path,
    )
    if cfg.resample_seconds < 1 or SECONDS_PER_DAY % cfg.resample_seconds:
        raise ValueError("resample_seconds must be a positive divisor of 86400")
    return cfg


def find_raw_file(cfg: GridConfig, root: Path = REPO_ROOT) -> Path:
    files = sorted(root.glob(cfg.raw_glob))
    if not files:
        raise RawDataMissingError(
            f"real control data not found: nothing matches {cfg.raw_glob} under {root}. "
            "Run `python scripts/fetch_grid.py fetch` to fetch the Fingrid slice "
            "(Zenodo record 3744121, CC BY 4.0). The data is not part of this repository and "
            "is never downloaded by the loader."
        )
    if len(files) > 1:
        raise ValueError(f"expected one grid slice, found {len(files)}: {[f.name for f in files]}")
    return files[0]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_slice(path: Path, max_gap: int) -> tuple[pd.Series, int]:
    """The 1-second frequency series, short NaN holes interpolated. Also the count filled."""
    df = pd.read_csv(path, parse_dates=["time"])
    s = pd.Series(df["frequency_hz"].to_numpy(dtype=float), index=pd.DatetimeIndex(df["time"]))
    steps = s.index.to_series().diff().dropna().dt.total_seconds()
    if not (steps == 1.0).all():
        raise ValueError(f"{path.name} is not a regular one-second series")
    is_nan = s.isna()
    if is_nan.any():
        run_id = (is_nan != is_nan.shift()).cumsum()
        longest = int(is_nan.groupby(run_id).sum().max())
        if longest > max_gap:
            raise ValueError(f"a NaN hole of {longest} s is longer than the limit of {max_gap} s")
    n_filled = int(is_nan.sum())
    return s.interpolate(method="linear", limit_area="inside"), n_filled


def daily_trajectories(s: pd.Series, cfg: GridConfig) -> list[Trajectory]:
    """One Trajectory per calendar day: block means of resample_seconds, all windows negative."""
    out: list[Trajectory] = []
    for seed, (day, g) in enumerate(s.groupby(s.index.date)):
        if len(g) != SECONDS_PER_DAY or g.isna().any():
            raise ValueError(f"{day} is not a complete day of one-second values")
        values = g.to_numpy().reshape(-1, cfg.resample_seconds).mean(axis=1)
        time = np.arange(len(values), dtype=float) * cfg.resample_seconds / 60.0  # minutes
        out.append(
            Trajectory(
                scenario_id=SCENARIO_ID,
                family=FAMILY,
                role="control",
                split=REAL_CONTROL_SPLIT,
                seed=seed,
                time=time,
                observations=values[:, None],
                state=values[:, None],
                control_name=None,
                control_parameter=None,
                tip_time=None,
                target=np.ones_like(time),  # no tipping: far from it everywhere
            )
        )
    return out


def build_real_control_dataset(
    *,
    feature_config: FeatureConfig | None = None,
    grid_config: GridConfig | None = None,
    seeds_per_scenario: int | None = None,
    summary_dir: Path | None = None,
    gate_config_path: Path | None = None,
    gate_hash_path: Path | None = None,
    root: Path = REPO_ROOT,
) -> pd.DataFrame:
    """All windows of the real_control split, same columns and feature code as the simulators.

    The gate must be frozen first (the held-out rule of docs/SPLITS.md).
    """
    kwargs = {}
    if gate_config_path is not None and gate_hash_path is not None:
        kwargs = {"config_path": gate_config_path, "hash_path": gate_hash_path}
    assert_frozen(**kwargs)  # before any real data is read

    cfg = grid_config or load_grid_config()
    fc = feature_config or load_feature_config()
    common = next(iter(load_scenarios(DEFAULT_CONFIG_DIR).values())).common
    path = find_raw_file(cfg, root)
    series, n_filled = read_slice(path, cfg.max_interpolated_gap_seconds)
    trajectories = daily_trajectories(series, cfg)
    if seeds_per_scenario is not None:  # a trial subset: the first days
        trajectories = trajectories[:seeds_per_scenario]
    df = pd.concat([trajectory_windows(tr, fc, common) for tr in trajectories], ignore_index=True)
    if (df["label"] != 0).any():
        raise AssertionError("every window of real_control must be negative")
    if summary_dir is not None:
        write_grid_summary(df, trajectories, path, n_filled, cfg, fc, summary_dir)
    return df[list(COLUMNS)]


def write_grid_summary(
    df: pd.DataFrame,
    trajectories: list[Trajectory],
    path: Path,
    n_filled: int,
    cfg: GridConfig,
    fc: FeatureConfig,
    summary_dir: Path,
) -> Path:
    """Counts written by code, so reports read them instead of retyping them."""
    digest = sha256_of(path)
    recorded = SOURCES.exists() and digest in SOURCES.read_text(encoding="utf-8")
    summary: dict[str, Any] = {
        "split": REAL_CONTROL_SPLIT,
        "test_only": True,
        "purpose": "false alarms only: every window is negative. Normal operation, not a "
        "pre-blackout run. See docs/GRID.md.",
        "file": path.name,
        "sha256": digest,
        "sha256_recorded_in_data_sources_md": bool(recorded),
        "resample_seconds": cfg.resample_seconds,
        "window_length": fc.window_length,
        "stride": fc.stride,
        "n_trajectories": len(trajectories),
        "trajectory_ids": [trajectory_id(t.scenario_id, t.seed) for t in trajectories],
        "points_per_trajectory": int(len(trajectories[0].time)),
        "n_windows": len(df),
        "windows_by_label": {str(int(k)): int(v) for k, v in df["label"].value_counts().items()},
        "n_seconds_interpolated": n_filled,
    }
    summary_dir.mkdir(parents=True, exist_ok=True)
    out = summary_dir / f"{REAL_CONTROL_SPLIT}_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return out

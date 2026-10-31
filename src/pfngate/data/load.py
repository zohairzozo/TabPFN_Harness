"""The real trajectories of a real split, for variants that score trajectories directly.

Real data is test only. Every caller has already checked that the gate is frozen (a held-out
split), and the loaders below read it only from data/raw/.
"""

from __future__ import annotations

from pfngate.data import REAL_CONTROL_SPLIT, REAL_SPLIT
from pfngate.data.daphnia import (
    build_trajectories,
    first_per_group,
    load_real_config,
    real_feature_config,
)
from pfngate.data.grid import (
    daily_trajectories,
    find_raw_file,
    load_grid_config,
    read_slice,
)
from pfngate.features.config import FeatureConfig, load_feature_config
from pfngate.scenarios import Trajectory


def real_trajectories(
    split: str, seeds_per_scenario: int | None = None
) -> tuple[list[Trajectory], FeatureConfig]:
    """The trajectories of a real split and the feature settings its windows were cut with."""
    base = load_feature_config()
    if split == REAL_SPLIT:
        cfg = load_real_config()
        trajectories, _ = build_trajectories(cfg.raw_dir, cfg)
        if seeds_per_scenario is not None:
            trajectories = first_per_group(trajectories, seeds_per_scenario)
        return trajectories, real_feature_config(base, cfg)
    if split == REAL_CONTROL_SPLIT:
        grid = load_grid_config()
        series, _ = read_slice(find_raw_file(grid), grid.max_interpolated_gap_seconds)
        trajectories = daily_trajectories(series, grid)
        if seeds_per_scenario is not None:
            trajectories = trajectories[:seeds_per_scenario]
        return trajectories, base
    raise ValueError(f"'{split}' is not a real split")

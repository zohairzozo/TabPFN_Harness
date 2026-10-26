"""Turn trajectories into a table of windows with features and labels, per split."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from pfngate.features.compute import FEATURE_NAMES, window_features
from pfngate.features.config import FeatureConfig, load_feature_config
from pfngate.features.windows import window_ends, window_matrix
from pfngate.scenarios import Scenario, Trajectory, binary_labels, load_scenarios
from pfngate.scenarios.config import DEFAULT_CONFIG_DIR, CommonConfig

META_COLUMNS: tuple[str, ...] = (
    "trajectory_id",
    "scenario_id",
    "family",
    "role",
    "split",
    "seed",
    "window_end",  # index of the last point in the window
    "window_end_time",
    "target",  # regression target at the window end
    "label",  # binary label: 1, 0, or -1 (dropped from the binary task)
)
COLUMNS: tuple[str, ...] = META_COLUMNS + FEATURE_NAMES


def trajectory_id(scenario_id: str, seed: int) -> str:
    return f"{scenario_id}/{seed}"


def trajectory_windows(tr: Trajectory, fc: FeatureConfig, common: CommonConfig) -> pd.DataFrame:
    """One row per window of one trajectory. Windows ending after the tip are left out."""
    ends = window_ends(len(tr.time), fc.window_length, fc.stride)
    if fc.keep_only_before_tip and tr.tip_time is not None:
        ends = ends[tr.time[ends] <= tr.tip_time]
    if len(ends) == 0:
        return pd.DataFrame({c: [] for c in COLUMNS})

    per_channel = [
        window_features(window_matrix(tr.observations[:, c], ends, fc.window_length), fc)
        for c in range(tr.observations.shape[1])
    ]
    target = tr.target[ends]
    data: dict[str, object] = {
        "trajectory_id": trajectory_id(tr.scenario_id, tr.seed),
        "scenario_id": tr.scenario_id,
        "family": tr.family,
        "role": tr.role,
        "split": tr.split,
        "seed": tr.seed,
        "window_end": ends,
        "window_end_time": tr.time[ends],
        "target": target,
        "label": binary_labels(target, common.positive_max, common.negative_min),
    }
    for name in FEATURE_NAMES:
        data[name] = np.mean([f[name] for f in per_channel], axis=0)
    return pd.DataFrame(data, columns=list(COLUMNS))


def split_plan(
    split: str, scenarios: dict[str, Scenario], seeds_per_scenario: int | None = None
) -> list[tuple[Scenario, int]]:
    """Every (scenario, seed) pair that belongs to a split, in a fixed order."""
    plan = []
    for sc in scenarios.values():
        for name in sc.seed_sets:
            seed_set = sc.common.seed_sets[name]
            if seed_set.split == split:
                plan.extend((sc, s) for s in seed_set.seeds()[:seeds_per_scenario])
    if not plan:
        raise ValueError(f"no trajectories for split '{split}'")
    return plan


def _cache_key(
    split: str, fc: FeatureConfig, scenario_dir: Path, ids: Iterable[str], k: int | None
) -> str:
    h = hashlib.sha256()
    for path in [fc.source, *sorted(scenario_dir.glob("*.yaml"))]:
        h.update(path.read_bytes())
    h.update(f"{split}|{fc.cache_version}|{k}|{','.join(ids)}".encode())
    return h.hexdigest()[:16]


def build_dataset(
    split: str,
    *,
    scenario_ids: list[str] | None = None,
    seeds_per_scenario: int | None = None,
    feature_config: FeatureConfig | None = None,
    scenario_dir: Path = DEFAULT_CONFIG_DIR,
    use_cache: bool = True,
) -> pd.DataFrame:
    """All windows of one split as a DataFrame, cached under results/cache/.

    `scenario_ids` and `seeds_per_scenario` select a subset (used by tests). The
    per-split summary in results/datasets/ is only written for a full build.
    """
    fc = feature_config or load_feature_config()
    scenarios = load_scenarios(scenario_dir)
    if scenario_ids is not None:
        scenarios = {k: scenarios[k] for k in scenario_ids}
    common = next(iter(scenarios.values())).common

    key = _cache_key(split, fc, scenario_dir, scenarios, seeds_per_scenario)
    cache_file = fc.cache_dir / f"features_{split}_{key}.pkl"
    if use_cache and cache_file.exists():
        return pd.read_pickle(cache_file)

    frames, dropped = [], []
    for sc, seed in split_plan(split, scenarios, seeds_per_scenario):
        tr = sc.generate(seed)
        if sc.label_kind == "time_to_tip" and tr.tip_time is None:
            dropped.append(trajectory_id(sc.scenario_id, seed))  # no tip: target undefined
            continue
        frames.append(trajectory_windows(tr, fc, common))
    df = pd.concat(frames, ignore_index=True)

    fc.cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache_file)
    if scenario_ids is None and seeds_per_scenario is None:
        write_summary(df, split, dropped, fc, cache_file)
    return df


def write_summary(
    df: pd.DataFrame, split: str, dropped: list[str], fc: FeatureConfig, cache_file: Path
) -> Path:
    """Counts per split, written by code so reports can read them instead of retyping."""
    counts = df.groupby(["family", "label"]).size().unstack(fill_value=0)
    summary = {
        "split": split,
        "n_trajectories": int(df["trajectory_id"].nunique()),
        "n_windows": len(df),
        "dropped_trajectories_no_tip": dropped,
        "windows_by_family_and_label": {
            fam: {str(int(lab)): int(n) for lab, n in row.items()} for fam, row in counts.iterrows()
        },
        "window_length": fc.window_length,
        "stride": fc.stride,
        "features": list(FEATURE_NAMES),
        "cache_file": cache_file.name,
    }
    fc.summary_dir.mkdir(parents=True, exist_ok=True)
    out = fc.summary_dir / f"{split}_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    return out

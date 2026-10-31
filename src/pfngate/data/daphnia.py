"""The split real_tipping: Daphnia magna populations, Drake and Griffen (2010).

Real data is TEST ONLY. It is never part of the context pool and never used to tune the gate.
Reading it calls assert_frozen() first, so it cannot be opened before the gate is frozen.

What is read (docs/DAPHNIA.md has the details and what is unsure):

  extinctions.csv  one row per population: start, end (extinction or censoring date), censor
                   flag (1 = extinct, 0 = still alive at the end), treatment.
  timeseries.csv   weekly census: three counts per chamber, two chambers per population.

One population is one trajectory. Its series is the sum over both chambers of the mean of the
three counts. Time is in days from the first census.

LABELS ARE SURROGATES. For a population that ends in extinction, the target is
clip((t_extinction - t) / horizon_days, 0, 1), the same form as the time-to-tip target in
docs/LABELS.md. A population that persists has target 1 and is a control. These labels are not
the distance to a bifurcation. The extinction time is only known after the fact.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from pfngate.data import REAL_SPLIT
from pfngate.features.config import FeatureConfig, load_feature_config
from pfngate.features.dataset import COLUMNS, trajectory_id, trajectory_windows
from pfngate.gate import assert_frozen
from pfngate.scenarios import Trajectory, load_scenarios
from pfngate.scenarios.config import DEFAULT_CONFIG_DIR

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REAL_CONFIG = REPO_ROOT / "configs" / "real_tipping.yaml"
REQUIRED_FILES: tuple[str, ...] = ("extinctions.csv", "timeseries.csv")
SOURCE = "Drake and Griffen (2010), Dryad doi:10.5061/dryad.q3p64"
FAMILY_EXTINCT = "real_extinction"
FAMILY_PERSISTING = "real_persisting"
ROLE_TIPPING = "real"
ROLE_CONTROL = "control"
MAX_GAP_DAYS = 14  # the census is weekly; a longer gap inside a series is reported
END_MISMATCH_DAYS = 7  # the census is weekly: a larger gap between End and the last census is odd


class RawDataMissingError(FileNotFoundError):
    """The raw Daphnia files are not where configs/real_tipping.yaml says they are."""


@dataclass(frozen=True)
class RealConfig:
    raw_dir: Path
    window_length: int
    stride: int
    sub_length: int
    sub_stride: int
    horizon_days: float
    source: Path  # the YAML file this was read from


def load_real_config(path: Path = DEFAULT_REAL_CONFIG) -> RealConfig:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = RealConfig(
        raw_dir=REPO_ROOT / raw["raw_dir"],
        window_length=int(raw["window"]["length"]),
        stride=int(raw["window"]["stride"]),
        sub_length=int(raw["subwindow"]["length"]),
        sub_stride=int(raw["subwindow"]["stride"]),
        horizon_days=float(raw["horizon_days"]),
        source=path,
    )
    if cfg.horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    if cfg.sub_length > cfg.window_length:
        raise ValueError("subwindow length must not exceed the window length")
    return cfg


def real_feature_config(base: FeatureConfig, cfg: RealConfig) -> FeatureConfig:
    """The simulators' feature settings with the real-data window sizes."""
    return dataclasses.replace(
        base,
        window_length=cfg.window_length,
        stride=cfg.stride,
        sub_length=cfg.sub_length,
        sub_stride=cfg.sub_stride,
        source=cfg.source,
    )


def read_raw(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The extinction table and the census table, with trimmed ids and parsed dates."""
    missing = [name for name in REQUIRED_FILES if not (raw_dir / name).is_file()]
    if missing:
        raise RawDataMissingError(
            f"real data not found: {', '.join(missing)} missing in {raw_dir}. Download "
            f"'data-and-code.zip' from Dryad (doi:10.5061/dryad.q3p64), unzip it so that the "
            f"CSV files are in {raw_dir}, and run again. The data is not part of this "
            "repository and is never downloaded by the code."
        )
    ext = pd.read_csv(raw_dir / "extinctions.csv")
    ts = pd.read_csv(raw_dir / "timeseries.csv")
    ts.columns = [str(c).strip() for c in ts.columns]  # the header has "Date " with a space
    ext["ID"] = ext["ID"].astype(str).str.strip()
    ts["ID"] = ts["ID"].astype(str).str.strip()
    for col in ("Start", "End"):
        ext[col] = pd.to_datetime(ext[col], format="%m/%d/%Y")
    ts["Date"] = pd.to_datetime(ts["Date"], format="%m/%d/%Y")
    return ext, ts


def population_counts(ts: pd.DataFrame) -> dict[str, pd.Series]:
    """Per population, the count per census date: chambers summed, three samples averaged."""
    ts = ts.assign(count=ts[["sample1", "sample2", "sample3"]].mean(axis=1))
    wide = ts.groupby(["ID", "Date"])["count"].sum()
    return {pid: g.droplevel(0).sort_index() for pid, g in wide.groupby(level=0)}


def surrogate_target(time: np.ndarray, extinction_time: float, horizon_days: float) -> np.ndarray:
    """clip((t_extinction - t) / horizon, 0, 1): 1 is far from the extinction, 0 is at it."""
    return np.clip((extinction_time - time) / horizon_days, 0.0, 1.0)


def build_trajectories(raw_dir: Path, cfg: RealConfig) -> tuple[list[Trajectory], dict[str, Any]]:
    """One Trajectory per population that has a census series, and a report about the data."""
    ext, ts = read_raw(raw_dir)
    counts = population_counts(ts)
    info = ext.set_index("ID")
    unknown = sorted(set(counts) - set(info.index))
    if unknown:
        raise ValueError(f"census ids without a row in extinctions.csv: {unknown}")

    trajectories: list[Trajectory] = []
    populations: list[dict[str, Any]] = []
    for seed, pid in enumerate(sorted(counts)):
        row = info.loc[pid]
        # Five ids were started again on Start (after a first population went extinct before the
        # treatment began). Their census file also holds the first population, with a gap of
        # months before the second. Only the population that starts on Start is used.
        full = counts[pid]
        series = full[full.index >= row["Start"]]
        day0 = series.index[0]
        time = (series.index - day0).days.to_numpy(dtype=float)
        values = series.to_numpy(dtype=float)
        extinct = int(row["Censor"]) == 1
        treatment = "deteriorating" if int(row["Deteriorating"]) == 1 else "constant"
        t_ext = float((row["End"] - day0).days) if extinct else None
        target = (
            surrogate_target(time, t_ext, cfg.horizon_days)
            if t_ext is not None
            else np.ones_like(time)
        )
        trajectories.append(
            Trajectory(
                scenario_id=f"daphnia_{treatment}",
                family=FAMILY_EXTINCT if extinct else FAMILY_PERSISTING,
                role=ROLE_TIPPING if extinct else ROLE_CONTROL,
                split=REAL_SPLIT,
                seed=seed,
                time=time,
                observations=values[:, None],
                state=values[:, None],  # no hidden state in real data
                control_name=None,
                control_parameter=None,
                tip_time=t_ext,
                target=target,
            )
        )
        populations.append(
            {
                "id": pid,
                "trajectory_id": trajectory_id(f"daphnia_{treatment}", seed),
                "treatment": treatment,
                "outcome": "extinct" if extinct else "persisting",
                "n_census_points": len(values),
                "n_census_points_before_start_dropped": int(len(full) - len(series)),
                "max_census_gap_days": float(np.max(np.diff(time))) if len(time) > 1 else 0.0,
                "extinction_day": t_ext,
                "last_census_day": float(time[-1]),
                "final_count": float(values[-1]),
                "notes": str(row["Notes"]),
            }
        )
    report = data_report(ext, populations)
    return trajectories, report


def data_report(ext: pd.DataFrame, populations: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts and consistency checks, so that docs/DAPHNIA.md can point at a file."""
    have = {p["id"] for p in populations}
    pop = pd.DataFrame(populations)
    extinct = pop[pop["outcome"] == "extinct"]
    persisting = pop[pop["outcome"] == "persisting"]
    gap = (extinct["last_census_day"] - extinct["extinction_day"]).abs()
    return {
        "source": SOURCE,
        "n_populations_in_extinctions_csv": int(len(ext)),
        "n_populations_with_census_series": len(populations),
        "ids_without_census_series": sorted(set(ext["ID"]) - have),
        "populations_by_treatment_and_outcome": {
            f"{t}/{o}": int(n) for (t, o), n in pop.groupby(["treatment", "outcome"]).size().items()
        },
        "extinction_day": {
            "min": float(extinct["extinction_day"].min()),
            "median": float(extinct["extinction_day"].median()),
            "max": float(extinct["extinction_day"].max()),
        }
        if len(extinct)
        else None,
        "checks": {
            "extinct_with_nonzero_final_count": sorted(extinct[extinct["final_count"] > 0]["id"]),
            "persisting_with_zero_final_count": sorted(
                persisting[persisting["final_count"] <= 0]["id"]
            ),
            f"extinct_end_more_than_{END_MISMATCH_DAYS}_days_from_last_census": sorted(
                extinct[gap > END_MISMATCH_DAYS]["id"]
            ),
            f"census_gap_over_{MAX_GAP_DAYS}_days": sorted(
                pop[pop["max_census_gap_days"] > MAX_GAP_DAYS]["id"]
            ),
            "ids_with_census_before_start_dropped": sorted(
                pop[pop["n_census_points_before_start_dropped"] > 0]["id"]
            ),
        },
        "populations": populations,
    }


def first_per_group(trajectories: list[Trajectory], n: int) -> list[Trajectory]:
    """A trial subset: the first n trajectories (by seed) of each scenario and role."""
    kept: dict[tuple[str, str], int] = {}
    out = []
    for tr in sorted(trajectories, key=lambda t: t.seed):
        key = (tr.scenario_id, tr.role)
        kept[key] = kept.get(key, 0) + 1
        if kept[key] <= n:
            out.append(tr)
    return out


def build_real_dataset(
    *,
    feature_config: FeatureConfig | None = None,
    real_config: RealConfig | None = None,
    seeds_per_scenario: int | None = None,
    summary_dir: Path | None = None,
    gate_config_path: Path | None = None,
    gate_hash_path: Path | None = None,
) -> pd.DataFrame:
    """All windows of the real_tipping split, with the same columns as the simulated splits.

    The gate must be frozen first: this is the held-out rule of docs/SPLITS.md. The simulators'
    feature code is used with the window sizes of configs/real_tipping.yaml.
    """
    kwargs = {}
    if gate_config_path is not None and gate_hash_path is not None:
        kwargs = {"config_path": gate_config_path, "hash_path": gate_hash_path}
    assert_frozen(**kwargs)  # before any real data is read

    cfg = real_config or load_real_config()
    fc = real_feature_config(feature_config or load_feature_config(), cfg)
    common = next(iter(load_scenarios(DEFAULT_CONFIG_DIR).values())).common
    trajectories, report = build_trajectories(cfg.raw_dir, cfg)
    if seeds_per_scenario is not None:
        trajectories = first_per_group(trajectories, seeds_per_scenario)
    frames = [trajectory_windows(tr, fc, common) for tr in trajectories]
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    if summary_dir is not None:
        write_real_summary(df, report, cfg, summary_dir)
    return df[list(COLUMNS)]


def write_real_summary(
    df: pd.DataFrame, report: dict[str, Any], cfg: RealConfig, summary_dir: Path
) -> Path:
    """Counts written by code, so reports read them instead of retyping them."""
    counts = df.groupby(["family", "label"]).size().unstack(fill_value=0)
    summary = {
        "split": REAL_SPLIT,
        "test_only": True,
        "labels_are_surrogates": (
            "target = clip((t_extinction - t) / horizon_days, 0, 1). This is not the true "
            "distance to a bifurcation. See docs/DAPHNIA.md."
        ),
        "horizon_days": cfg.horizon_days,
        "window_length": cfg.window_length,
        "stride": cfg.stride,
        "n_trajectories_with_windows": int(df["trajectory_id"].nunique()),
        "n_windows": len(df),
        "windows_by_family_and_label": {
            fam: {str(int(lab)): int(n) for lab, n in row.items()} for fam, row in counts.iterrows()
        },
        "data": report,
    }
    summary_dir.mkdir(parents=True, exist_ok=True)
    out = summary_dir / f"{REAL_SPLIT}_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return out

"""Load the shared scenario settings from configs/scenarios/_common.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "scenarios"
COMMON_FILE = "_common.yaml"


@dataclass(frozen=True)
class SeedSet:
    """A named, contiguous range of seeds that belongs to one split."""

    name: str
    split: str
    start: int
    count: int

    def seeds(self) -> list[int]:
        return list(range(self.start, self.start + self.count))

    def __contains__(self, seed: object) -> bool:
        return isinstance(seed, int) and self.start <= seed < self.start + self.count


@dataclass(frozen=True)
class CommonConfig:
    """Settings shared by every family: time grid, noise levels, labels, seed sets."""

    dt: float
    observe_every: int
    burn_in: float
    noise_levels: dict[str, dict[str, float]]
    horizon: float
    positive_max: float
    negative_min: float
    seed_sets: dict[str, SeedSet]
    family_files: tuple[str, ...]

    @property
    def obs_dt(self) -> float:
        """Time between two observed points."""
        return self.dt * self.observe_every


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    return data


def load_common(config_dir: Path = DEFAULT_CONFIG_DIR) -> CommonConfig:
    raw = read_yaml(config_dir / COMMON_FILE)
    time = raw["time"]
    labels = raw["labels"]
    seed_sets = {
        name: SeedSet(name=name, split=s["split"], start=int(s["start"]), count=int(s["count"]))
        for name, s in raw["seed_sets"].items()
    }
    return CommonConfig(
        dt=float(time["dt"]),
        observe_every=int(time["observe_every"]),
        burn_in=float(time["burn_in"]),
        noise_levels={
            k: {kk: float(vv) for kk, vv in v.items()} for k, v in raw["noise_levels"].items()
        },
        horizon=float(labels["horizon"]),
        positive_max=float(labels["positive_max"]),
        negative_min=float(labels["negative_min"]),
        seed_sets=seed_sets,
        family_files=tuple(raw["family_files"]),
    )

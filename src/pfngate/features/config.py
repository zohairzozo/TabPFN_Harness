"""Load windowing and feature settings from configs/features.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FEATURE_CONFIG = REPO_ROOT / "configs" / "features.yaml"


@dataclass(frozen=True)
class FeatureConfig:
    window_length: int
    stride: int
    keep_only_before_tip: bool
    detrend: str
    sub_length: int
    sub_stride: int
    low_frequency_fraction: float
    channels: str
    cache_dir: Path
    cache_version: int
    summary_dir: Path
    source: Path  # the YAML file this was read from, part of the cache key


def load_feature_config(path: Path = DEFAULT_FEATURE_CONFIG) -> FeatureConfig:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = FeatureConfig(
        window_length=int(raw["window"]["length"]),
        stride=int(raw["window"]["stride"]),
        keep_only_before_tip=bool(raw["window"]["keep_only_before_tip"]),
        detrend=str(raw["detrend"]),
        sub_length=int(raw["subwindow"]["length"]),
        sub_stride=int(raw["subwindow"]["stride"]),
        low_frequency_fraction=float(raw["spectrum"]["low_frequency_fraction"]),
        channels=str(raw["channels"]),
        cache_dir=REPO_ROOT / raw["cache"]["dir"],
        cache_version=int(raw["cache"]["version"]),
        summary_dir=REPO_ROOT / raw["summary_dir"],
        source=path,
    )
    if cfg.detrend != "linear":
        raise ValueError(f"only linear detrending is implemented, got '{cfg.detrend}'")
    if cfg.channels != "mean":
        raise ValueError(f"only channels: mean is implemented, got '{cfg.channels}'")
    if cfg.sub_length > cfg.window_length:
        raise ValueError("subwindow length must not exceed the window length")
    return cfg

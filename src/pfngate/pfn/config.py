"""Load PFN backend settings from configs/pfn.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from pfngate.features.config import REPO_ROOT

DEFAULT_PFN_CONFIG = REPO_ROOT / "configs" / "pfn.yaml"


@dataclass(frozen=True)
class PFNConfig:
    backend: str
    model_version: str
    device: str
    seed: int
    quantile_levels: tuple[float, ...]
    critical_threshold: float
    max_context: int
    knn_neighbors: int
    balanced_near_share: float
    target_false_alarm_rate: float
    bootstrap_n: int
    bootstrap_seed: int
    bootstrap_ci: float
    client_enabled: bool
    source: Path


def load_pfn_config(path: Path = DEFAULT_PFN_CONFIG) -> PFNConfig:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = PFNConfig(
        backend=str(raw["backend"]),
        model_version=str(raw["model_version"]),
        device=str(raw["device"]),
        seed=int(raw["seed"]),
        quantile_levels=tuple(float(q) for q in raw["quantile_levels"]),
        critical_threshold=float(raw["critical_threshold"]),
        max_context=int(raw["context"]["max_context"]),
        knn_neighbors=int(raw["context"]["knn_neighbors"]),
        balanced_near_share=float(raw["context"]["balanced_near_share"]),
        target_false_alarm_rate=float(raw["evaluation"]["target_false_alarm_rate"]),
        bootstrap_n=int(raw["evaluation"]["bootstrap"]["n"]),
        bootstrap_seed=int(raw["evaluation"]["bootstrap"]["seed"]),
        bootstrap_ci=float(raw["evaluation"]["bootstrap"]["ci"]),
        client_enabled=bool(raw["tabpfn_client"]["enabled"]),
        source=path,
    )
    levels = cfg.quantile_levels
    if not levels or any(not 0.0 < q < 1.0 for q in levels):
        raise ValueError("quantile_levels must be non-empty and inside (0, 1)")
    if any(b <= a for a, b in zip(levels, levels[1:], strict=False)):
        raise ValueError("quantile_levels must be strictly increasing")
    if not 0.0 < cfg.target_false_alarm_rate < 1.0:
        raise ValueError("target_false_alarm_rate must be inside (0, 1)")
    if cfg.max_context < 1 or cfg.knn_neighbors < 1:
        raise ValueError("max_context and knn_neighbors must be at least 1")
    if not 0.0 < cfg.balanced_near_share < 1.0:
        raise ValueError("balanced_near_share must be inside (0, 1)")
    return cfg

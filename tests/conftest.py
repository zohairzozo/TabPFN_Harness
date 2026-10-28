"""Shared fixtures: a small slice of the real scenario data and test gate configs."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pfngate.evals.pfn_cache import data_version
from pfngate.features import FeatureConfig, build_dataset, load_feature_config
from pfngate.gate.build import build_gate_config, write_gate_config
from pfngate.gate.freeze import write_frozen_hash
from pfngate.pfn import FakeBackend


@pytest.fixture(scope="session")
def small_data(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One seed per scenario: the context pool and the validation split, in a temp cache."""
    tmp = tmp_path_factory.mktemp("small")
    fc = dataclasses.replace(load_feature_config(), cache_dir=tmp / "features", summary_dir=tmp)
    return {
        "fc": fc,
        "pool": build_dataset("context", seeds_per_scenario=1, feature_config=fc),
        "val": build_dataset("validation", seeds_per_scenario=1, feature_config=fc),
    }


def write_test_gate(
    directory: Path,
    pool: pd.DataFrame,
    fc: FeatureConfig,
    *,
    status: str = "tuned",
    model_version: str = FakeBackend.model_version,
    max_context: int = 200,
    alert: dict[str, float | None] | None = None,
    freeze: bool = True,
) -> tuple[Path, Path]:
    """A gate config for the fake backend, built from `pool`, optionally frozen."""
    directory.mkdir(parents=True, exist_ok=True)
    applicability = {
        "backend": "fake",
        "model_version": model_version,
        "package_version": "-",
        "max_context": max_context,
        "data_version": data_version(fc),
    }
    data = build_gate_config(
        pool,
        applicability=applicability,
        strategies=("random", "knn"),
        seed=0,
        status=status,  # type: ignore[arg-type]
        n_sample=2000,
    )
    if alert is not None:
        data["alert_thresholds"] = alert
    config, digest = directory / "gate.yaml", directory / "gate.frozen.sha256"
    write_gate_config(data, config)
    if freeze:
        write_frozen_hash(config, digest)
    return config, digest


@pytest.fixture(scope="session")
def make_gate() -> Callable[..., tuple[Path, Path]]:
    return write_test_gate

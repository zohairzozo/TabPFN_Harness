"""A disk cache of PFN fit-and-predict results, keyed by context hash.

TabPFN's fit is lazy: it only stores the context. The cost is in predict, and predict
depends on the query as well. So the cache cannot hold the fit alone. It holds the whole
PFNOutput for one (context, query) pair, and the file name starts with the context hash.
The key also covers everything else that changes the answer: the backend, its model
version and package version, the seed, the quantile levels, the threshold, and a hash of
the feature and scenario configs (the context hash only names rows, not their values).

A cache hit skips the fit and the predict. The stored metadata keeps the seconds of the
run that made the result, so latency reports are not changed by caching.
"""

from __future__ import annotations

import hashlib
import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from pfngate.features import FEATURE_NAMES
from pfngate.features.config import REPO_ROOT, FeatureConfig
from pfngate.pfn import PFNConfig, PFNOutput
from pfngate.scenarios.config import DEFAULT_CONFIG_DIR

DEFAULT_PFN_CACHE_DIR = REPO_ROOT / "results" / "cache" / "pfn"


def data_version(fc: FeatureConfig, scenario_dir: Path = DEFAULT_CONFIG_DIR) -> str:
    """Hash of the configs that decide the feature values in the pool and the query."""
    h = hashlib.sha256()
    for path in [fc.source, *sorted(scenario_dir.glob("*.yaml"))]:
        h.update(path.read_bytes())
    h.update(str(fc.cache_version).encode())
    return h.hexdigest()[:16]


def query_hash(query: pd.DataFrame) -> str:
    """Hash of the query feature values and window positions."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(query[list(FEATURE_NAMES)].to_numpy(float)).tobytes())
    h.update(query["window_end"].to_numpy(np.int64).tobytes())
    return h.hexdigest()[:16]


def package_version(backend: str) -> str:
    name = {"tabpfn_local": "tabpfn", "tabpfn_client": "tabpfn-client"}.get(backend)
    if name is None:
        return "-"
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def cache_key(
    cfg: PFNConfig,
    *,
    backend: str,
    model_version: str,
    data_ver: str,
    context_hash: str,
    query_hash_: str,
) -> str:
    """File stem: the context hash first, then a hash of everything else."""
    rest = json.dumps(
        {
            "backend": backend,
            "model_version": model_version,
            "package_version": package_version(backend),
            "device": cfg.device,
            "seed": cfg.seed,
            "levels": list(cfg.quantile_levels),
            "threshold": cfg.critical_threshold,
            "data_version": data_ver,
            "query": query_hash_,
        },
        sort_keys=True,
    )
    return f"{context_hash}_{hashlib.sha256(rest.encode()).hexdigest()[:16]}"


class PFNCache:
    def __init__(self, directory: Path = DEFAULT_PFN_CACHE_DIR, enabled: bool = True) -> None:
        self.directory = directory
        self.enabled = enabled

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> PFNOutput | None:
        path = self._path(key)
        if not self.enabled or not path.exists():
            return None
        return PFNOutput.model_validate_json(path.read_text(encoding="utf-8"))

    def put(self, key: str, output: PFNOutput) -> None:
        if not self.enabled:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self._path(key).with_suffix(".tmp")
        tmp.write_text(output.model_dump_json(), encoding="utf-8")
        os.replace(tmp, self._path(key))  # a run stopped half way never leaves a broken file

"""TabPFN through the hosted tabpfn-client service. Off by default."""

from __future__ import annotations

import inspect
import time
from importlib.metadata import version
from typing import Any

import numpy as np

from pfngate.pfn.adapter import PFNOutput, build_output, check_context, check_query
from pfngate.pfn.config import PFNConfig, load_pfn_config


class ClientDisabledError(RuntimeError):
    """The remote client was used while tabpfn_client.enabled is false in configs/pfn.yaml."""


class TabPFNClient:
    """WARNING: THIS BACKEND SENDS YOUR DATA OFF THIS MACHINE.

    Every context row and every query row is uploaded to a remote server. Do not use it
    with data you may not share. It is disabled by default: set tabpfn_client.enabled to
    true in configs/pfn.yaml to allow it, and only after reading the terms of the service.

    NOT TESTED AGAINST THE REAL PACKAGE. tabpfn-client is not installed here (checked
    2026-10-27), so its API has not been read from the installed package, as CLAUDE.md
    requires. Before first use, run pip show tabpfn-client and read
    help(tabpfn_client.TabPFNRegressor). At fit time this class only checks that
    predict accepts output_type and quantiles, and refuses to run otherwise. The remote
    service picks the model, so model_version is recorded as "remote-default".
    p_critical is interpolated from the quantiles, because no distribution is assumed.
    """

    name = "tabpfn_client"

    def __init__(self, config: PFNConfig | None = None) -> None:
        self.cfg = config or load_pfn_config()
        self._reg: Any = None
        self._n_context = 0
        self._n_features = 0
        self._fit_seconds = 0.0

    def _import_client(self) -> Any:
        if not self.cfg.client_enabled:
            raise ClientDisabledError(
                "tabpfn_client is disabled. It uploads data to a remote server. "
                "Set tabpfn_client.enabled: true in configs/pfn.yaml to allow it."
            )
        try:
            import tabpfn_client
        except ImportError as exc:
            raise RuntimeError("tabpfn-client is not installed (uv add tabpfn-client)") from exc
        return tabpfn_client

    def fit(self, context_X: np.ndarray, context_y: np.ndarray) -> None:
        client = self._import_client()
        X, y = check_context(context_X, context_y)
        params = inspect.signature(client.TabPFNRegressor.predict).parameters
        if not {"output_type", "quantiles"} <= set(params):
            raise RuntimeError(
                "the installed tabpfn-client predict() has a different signature; "
                "read help(tabpfn_client.TabPFNRegressor.predict) and update this adapter"
            )
        reg = client.TabPFNRegressor()
        t0 = time.perf_counter()
        reg.fit(X, y)
        self._fit_seconds = time.perf_counter() - t0
        self._reg, self._n_context, self._n_features = reg, X.shape[0], X.shape[1]

    def predict(self, query_X: np.ndarray) -> PFNOutput:
        if self._reg is None:
            self._import_client()  # raises the right error if the client is off or missing
            raise RuntimeError("call fit before predict")
        X = check_query(query_X, self._n_features)
        levels = [float(q) for q in self.cfg.quantile_levels]
        t0 = time.perf_counter()
        quantiles = self._reg.predict(X, output_type="quantiles", quantiles=levels)
        return build_output(
            levels=levels,
            quantiles=np.stack(quantiles, axis=1),
            threshold=self.cfg.critical_threshold,
            backend=self.name,
            model_version="remote-default",
            n_context=self._n_context,
            fit_seconds=self._fit_seconds,
            predict_seconds=time.perf_counter() - t0,
            notes=[f"tabpfn-client {version('tabpfn-client')}; data was sent to a remote server"],
        )

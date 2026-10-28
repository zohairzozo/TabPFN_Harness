"""A backend for tests: no model, deterministic output, optional crossing quantiles."""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import norm

from pfngate.pfn.adapter import PFNOutput, build_output, check_context, check_query
from pfngate.pfn.config import PFNConfig, load_pfn_config


class FakeBackend:
    """Two deterministic modes, so tests can predict the output.

    mode="fixed" (default): quantiles are a fixed function of the query rows and ignore
    the context. The centre of each row is a sigmoid of the mean of its features, which
    lies in (0, 1). Quantiles are normal quantiles around that centre, clipped to [0, 1].
    With cross_quantiles=True the columns are reversed, to exercise the repair step.
    p_critical is interpolated from the quantiles.

    mode="neighbours": quantiles are the empirical quantiles of the targets of the
    `neighbours` nearest context rows (features standardised with the context mean and
    sd), and p_critical is the share of those targets at or below the threshold. It
    learns from its context, so the gate tests can tell a good context from a bad one.
    interval_scale < 1 pulls every quantile towards the median: an overconfident model.
    """

    name = "fake"
    model_version = "fake-1"

    def __init__(
        self,
        config: PFNConfig | None = None,
        *,
        spread: float = 0.1,
        cross_quantiles: bool = False,
        mode: Literal["fixed", "neighbours"] = "fixed",
        neighbours: int = 30,
        interval_scale: float = 1.0,
    ) -> None:
        self.cfg = config or load_pfn_config()
        self.spread = spread
        self.cross_quantiles = cross_quantiles
        self.mode = mode
        self.neighbours = neighbours
        self.interval_scale = interval_scale
        if mode == "neighbours":  # a different model: the cache key must differ too
            self.model_version = f"fake-nn{neighbours}-x{interval_scale:g}-1"
        self._X = np.empty((0, 0))
        self._y = np.empty(0)
        self._fitted = False
        self._n_context = 0
        self._n_features = 0
        self._fit_seconds = 0.0

    def fit(self, context_X: np.ndarray, context_y: np.ndarray) -> None:
        t0 = time.perf_counter()
        X, y = check_context(context_X, context_y)
        self._n_context, self._n_features = X.shape[0], X.shape[1]
        self._X, self._y = X, y
        self._fitted = True
        self._fit_seconds = time.perf_counter() - t0

    def predict(self, query_X: np.ndarray) -> PFNOutput:
        if not self._fitted:
            raise RuntimeError("call fit before predict")
        t0 = time.perf_counter()
        X = check_query(query_X, self._n_features)
        if self.mode == "neighbours":
            return self._predict_neighbours(X, t0)
        centre = 1.0 / (1.0 + np.exp(-X.mean(axis=1)))
        z = norm.ppf(np.asarray(self.cfg.quantile_levels))
        q = np.clip(centre[:, None] + self.spread * z[None, :], 0.0, 1.0)
        if self.cross_quantiles:
            q = q[:, ::-1]
        return build_output(
            levels=self.cfg.quantile_levels,
            quantiles=q,
            threshold=self.cfg.critical_threshold,
            backend=self.name,
            model_version=self.model_version,
            n_context=self._n_context,
            fit_seconds=self._fit_seconds,
            predict_seconds=time.perf_counter() - t0,
        )

    def _predict_neighbours(self, X: np.ndarray, t0: float) -> PFNOutput:
        mean, sd = self._X.mean(axis=0), self._X.std(axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        k = min(self.neighbours, len(self._X))
        _, idx = cKDTree((self._X - mean) / sd).query((X - mean) / sd, k=k)
        t = self._y[idx.reshape(len(X), k)]
        levels = np.asarray(self.cfg.quantile_levels)
        q = np.quantile(t, levels, axis=1).T
        med = np.median(t, axis=1)[:, None]
        q = np.clip(med + self.interval_scale * (q - med), 0.0, 1.0)
        return build_output(
            levels=self.cfg.quantile_levels,
            quantiles=q,
            threshold=self.cfg.critical_threshold,
            backend=self.name,
            model_version=self.model_version,
            n_context=self._n_context,
            fit_seconds=self._fit_seconds,
            predict_seconds=time.perf_counter() - t0,
            p_critical=(t <= self.cfg.critical_threshold).mean(axis=1),
            p_critical_method="full_distribution",
        )

"""TabPFN regression on this machine, through the installed tabpfn package (CPU)."""

from __future__ import annotations

import time
from importlib.metadata import version
from typing import Any

import numpy as np

from pfngate.pfn.adapter import (
    PCriticalMethod,
    PFNOutput,
    build_output,
    check_context,
    check_query,
)
from pfngate.pfn.config import PFNConfig, load_pfn_config


class TabPFNLocal:
    """Wraps tabpfn.TabPFNRegressor. API read from tabpfn 9.2.0 on 2026-10-27.

    The model version, device and seed come from configs/pfn.yaml. The v3.x weights are
    gated and non-commercial: accept the licence and set TABPFN_TOKEN as
    scripts/smoke_tabpfn.py explains. Errors from the package (licence, token, the
    5000-row CPU limit) are passed on unchanged. Nothing is set around them.

    Quantiles come from predict(output_type="full", quantiles=levels). That call also
    returns the predictive distribution (criterion and logits), so p_critical is the
    CDF of that distribution at the threshold, not an interpolation.

    fit_seconds includes loading the weights, so the first fit in a process is slower.
    """

    name = "tabpfn_local"

    def __init__(self, config: PFNConfig | None = None) -> None:
        self.cfg = config or load_pfn_config()
        self._reg: Any = None
        self._n_context = 0
        self._n_features = 0
        self._fit_seconds = 0.0

    def fit(self, context_X: np.ndarray, context_y: np.ndarray) -> None:
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion

        X, y = check_context(context_X, context_y)
        reg = TabPFNRegressor.create_default_for_version(
            ModelVersion(self.cfg.model_version),
            device=self.cfg.device,
            random_state=self.cfg.seed,
        )
        t0 = time.perf_counter()
        reg.fit(X, y)
        self._fit_seconds = time.perf_counter() - t0
        self._reg, self._n_context, self._n_features = reg, X.shape[0], X.shape[1]

    def predict(self, query_X: np.ndarray) -> PFNOutput:
        if self._reg is None:
            raise RuntimeError("call fit before predict")
        X = check_query(query_X, self._n_features)
        levels = [float(q) for q in self.cfg.quantile_levels]  # tabpfn requires python floats
        t0 = time.perf_counter()
        out = self._reg.predict(X, output_type="full", quantiles=levels)
        p_critical, method, notes = self._p_critical(out, len(X))
        seconds = time.perf_counter() - t0
        return build_output(
            levels=levels,
            quantiles=np.stack(out["quantiles"], axis=1),
            threshold=self.cfg.critical_threshold,
            backend=self.name,
            model_version=self.cfg.model_version,
            n_context=self._n_context,
            fit_seconds=self._fit_seconds,
            predict_seconds=seconds,
            p_critical=p_critical,
            p_critical_method=method,
            notes=[f"tabpfn {version('tabpfn')}, device {self.cfg.device}", *notes],
        )

    def _p_critical(
        self, out: dict[str, Any], n_query: int
    ) -> tuple[np.ndarray | None, PCriticalMethod, list[str]]:
        """P(target <= threshold) per row, from the full predictive distribution if present."""
        thr = self.cfg.critical_threshold
        if getattr(self._reg, "is_constant_target_", False):
            # The package then returns no usable distribution, only the constant.
            hit = float(self._reg.constant_value_ <= thr)
            return np.full(n_query, hit), "constant_target", ["context target was constant"]
        if "criterion" in out and "logits" in out:
            import torch

            ys = torch.tensor([thr], dtype=torch.float64)
            cdf = out["criterion"].cdf(out["logits"], ys)
            return cdf.reshape(-1).detach().cpu().numpy(), "full_distribution", []
        return None, "quantile_interpolation", ["full predictive distribution not exposed"]

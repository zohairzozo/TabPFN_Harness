"""Placeholder for TipPFN. It does nothing: the weights and code are not public."""

from __future__ import annotations

import numpy as np

from pfngate.pfn.adapter import PFNOutput

# Assumption, not verified: 99 evenly spaced levels 0.01 ... 0.99.
EXPECTED_QUANTILE_LEVELS: tuple[float, ...] = tuple(k / 100 for k in range(1, 100))


class TipPFNBackend:
    """Stub for TipPFN (kausable GmbH, arXiv 2605.12308). It always raises NotImplementedError.

    This is NOT a reimplementation and does not reproduce TipPFN. It marks where the real
    backend would plug in. The description below is our understanding from the project
    brief. The paper was not read for this stub, so treat every line as unconfirmed.

    Expected input (as we understand it):
      - A time-indexed feature table for the series being scored: one row per time step
        or window, each carrying its time index.
      - Zero to three context episodes: example series from other systems, which the
        model reads in context instead of being trained on them.

    Expected output (as we understand it):
      - 99 quantiles of the relative distance to criticality for each row. The target is
        presumably 0 at the tipping point and larger further away.

    Uncertain, to be checked against the paper before any real implementation:
      - The exact feature set and how time is encoded.
      - Whether a context episode carries labels, and in what form.
      - How relative distance is normalised, and its range. Does it stay in [0, 1]?
      - Whether the 99 levels are 0.01 ... 0.99 (see EXPECTED_QUANTILE_LEVELS).
      - Whether the paper's model is a regressor over rows or reads whole series at once.

    The PFNBackend protocol takes a context table (X, y). A context episode is richer than
    that, so the real backend may need an extended fit signature. A mapping from 99 quantiles
    to the 5 levels in configs/pfn.yaml, and to p_critical, would also be needed.
    """

    name = "tippfn"

    def fit(self, context_X: np.ndarray, context_y: np.ndarray) -> None:
        raise NotImplementedError(
            "TipPFN is not publicly available. This stub only documents the expected "
            "interface. See the class docstring."
        )

    def predict(self, query_X: np.ndarray) -> PFNOutput:
        raise NotImplementedError(
            "TipPFN is not publicly available. This stub only documents the expected "
            "interface. See the class docstring."
        )

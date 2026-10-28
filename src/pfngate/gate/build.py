"""Write configs/gate.yaml: the design constants of docs/GATE.md plus the pool reference.

The file starts with status "provisional". Its tuned thresholds (ood_share limit and the
context_calibration range) then hold the loosest values
of the pre-registered grid, and the alert thresholds are empty. A provisional file can
gate validation runs, which is what tuning needs, but it cannot be frozen, so held-out
evaluation stays refused until the gate is tuned on validation runs. Only set A (the pool)
is read here: its features, and its targets for the near-tipping cut (pool labels, which a
context may use).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pfngate.features import FEATURE_NAMES
from pfngate.gate.config import BAD_RUN_INTERVAL, BAD_RUN_MIN_COVERAGE, GateStatus
from pfngate.gate.reference import fit_near_tipping_reference, fit_pool_reference

# Pre-registered in docs/GATE.md. The tuner may only pick from these grids.
TUNING_GRID: dict[str, tuple[float, ...]] = {
    "ood_share_limit": (0.05, 0.1, 0.2, 0.3),
    "calibration_low": (0.70, 0.75, 0.80, 0.85),
    "calibration_high": (0.97, 0.99, 1.00),
}
LOOSEST = {"ood_share_limit": 0.3, "calibration_low": 0.70, "calibration_high": 1.00}
POSITIVE_MAX = 0.2  # binary positive zone, docs/LABELS.md
NEAR_TIPPING_K = 5
NEAR_TIPPING_LIMIT = 0.5  # advisory only, not tuned (docs/GATE.md, round 3)

HEADER = """\
# Gate thresholds. Written by scripts/build_gate_config.py from docs/GATE.md and the
# context pool (seed set A). Do not edit by hand. Only a file with status "tuned" may be
# frozen (configs/gate.frozen.sha256), and test, out-of-prior and real-data evaluation
# refuse to run until it is.
"""


def build_gate_config(
    pool: pd.DataFrame,
    *,
    applicability: dict[str, Any],
    strategies: tuple[str, ...],
    seed: int,
    status: GateStatus = "provisional",
    k: int = 10,
    quantile: float = 0.99,
    n_sample: int = 5000,
) -> dict[str, Any]:
    """The full gate config as a plain dict, ready to dump to YAML."""
    ref = fit_pool_reference(pool, k=k, quantile=quantile, n_sample=n_sample, seed=seed)
    near = fit_near_tipping_reference(
        pool,
        feature_mean=ref.feature_mean,
        feature_std=ref.feature_std,
        k=NEAR_TIPPING_K,
        target_max=POSITIVE_MAX,
        quantile=quantile,
        n_sample=n_sample,
        seed=seed,
    )
    return {
        "status": status,
        "bad_run": {
            "interval": list(BAD_RUN_INTERVAL),
            "min_coverage": BAD_RUN_MIN_COVERAGE,
            "tolerance": 1.0e-6,
        },
        "checks": {
            "min_context_size": {"threshold": 100},
            "ood_share": {
                "limit": LOOSEST["ood_share_limit"],
                "k": ref.k,
                "quantile": ref.quantile,
                "n_sample": ref.n_sample,
                "distance_cut": ref.distance_cut,
                "feature_names": list(FEATURE_NAMES),
                "feature_mean": list(ref.feature_mean),
                "feature_std": list(ref.feature_std),
                "n_pool_rows": ref.n_pool_rows,
                "n_pool_trajectories": ref.n_pool_trajectories,
            },
            "near_tipping_support": {
                "role": "advisory",
                "limit": NEAR_TIPPING_LIMIT,
                "k": near.k,
                "target_max": near.target_max,
                "quantile": near.quantile,
                "n_sample": near.n_sample,
                "distance_cut": near.distance_cut,
                "n_near_rows": near.n_near_rows,
            },
            "interval_width_cap": {"threshold": 0.6, "role": "advisory"},
            "context_calibration": {
                "low": LOOSEST["calibration_low"],
                "high": LOOSEST["calibration_high"],
                "folds": 5,
            },
            "context_not_degenerate": {
                "min_rows": 10,
                "positive_max": POSITIVE_MAX,
                "negative_min": 0.6,
            },
            "output_integrity": {
                "max_repaired_share": 0.01,
                "required_method": "full_distribution",
            },
        },
        "alert_thresholds": {s: None for s in strategies},
        "applicability": {**applicability, "strategies": list(strategies)},
    }


def write_gate_config(data: dict[str, Any], path: Path) -> Path:
    """Dump with LF line endings, so the bytes (and the hash) are the same on every OS."""
    text = HEADER + yaml.safe_dump(data, sort_keys=False, default_flow_style=None, width=100)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path

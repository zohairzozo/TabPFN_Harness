"""A summary of the query windows: what the agent's describe_query tool returns, and the text
the LLM-only baseline reads. Features only. No target, no family, no trajectory id."""

from __future__ import annotations

from typing import Any

import pandas as pd

from pfngate.features import FEATURE_NAMES
from pfngate.pfn.context import query_x

TREND_NOTE = "trend is the mean of the last third of the windows minus the mean of the first third"


def r4(value: float) -> float:
    return round(float(value), 4)


def third(n: int) -> int:
    return max(1, n // 3)


def summarise_query(query: pd.DataFrame) -> dict[str, Any]:
    """For each feature: mean, spread and trend over the windows, in time order."""
    x = query_x(query.sort_values("window_end", kind="stable"))
    n, k = len(x), third(len(x))
    features = {
        name: {
            "mean": r4(x[:, j].mean()),
            "std": r4(x[:, j].std()),
            "trend": r4(x[-k:, j].mean() - x[:k, j].mean()),
        }
        for j, name in enumerate(FEATURE_NAMES)
    }
    return {"n_windows": n, "features": features, "note": TREND_NOTE}


def summary_text(summary: dict[str, Any]) -> str:
    """The same summary as plain text, one line per feature."""
    lines = [
        f"The query has {summary['n_windows']} windows, in time order.",
        f"For each feature: its mean, its spread (std) and its trend. {summary['note']}.",
    ]
    for name, s in summary["features"].items():
        lines.append(f"- {name}: mean {s['mean']:.4f}, std {s['std']:.4f}, trend {s['trend']:+.4f}")
    return "\n".join(lines)

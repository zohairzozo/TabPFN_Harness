"""Run-level metrics for the LLM variants, computed the same way for every one of them.

An LLM variant gives one answer per query trajectory: alert, no_alert or (for the gated agent)
abstain, and sometimes a score. The truth for a run is the label of its LAST window, because
the answer is about the end of the series: positive (near tipping), negative (far from it) or
middle (the band the binary task drops, docs/LABELS.md). Runs in the middle band are left out
of every rate. An abstention is not an alert.

All rates have a percentile bootstrap CI that resamples whole runs. These are run-level
metrics. The PFN runs of run_pfn.py are scored per window, so the two sets of numbers are not
the same quantity and must not be compared as if they were.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pfngate.evals.metrics import bootstrap_auroc

Decision = Literal["alert", "no_alert", "abstain"]
EndLabel = Literal["positive", "negative", "middle"]


@dataclass(frozen=True)
class RunOutcome:
    trajectory_id: str
    role: str  # "in_prior", "out_of_prior" or "control"
    end_label: EndLabel
    decision: Decision
    score: float | None  # a probability-like alert score. None if the variant gave none.


def end_label(label: int) -> EndLabel:
    """The binary label of the last window: 1 near tipping, 0 far, -1 the middle band."""
    return {1: "positive", 0: "negative"}.get(int(label), "middle")  # type: ignore[return-value]


def alert_score(decision: str, confidence: float) -> float:
    """The probability of an alert that a decision with a confidence stands for."""
    return confidence if decision == "alert" else 1.0 - confidence


def _rate(
    hit: np.ndarray, mask: np.ndarray, *, n_boot: int, seed: int, ci: float
) -> dict[str, Any]:
    """The share of `hit` among the runs in `mask`, with a bootstrap CI over runs."""
    n = int(mask.sum())
    if n == 0:
        return {"value": None, "n": 0, "ci": None}
    idx = np.random.default_rng(seed).integers(0, len(hit), size=(n_boot, len(hit)))
    in_mask = mask[idx]
    counts = in_mask.sum(axis=1)
    ok = counts > 0
    shares = (hit[idx] & in_mask).sum(axis=1)[ok] / counts[ok]
    tail = (1.0 - ci) / 2.0 * 100.0
    interval = [float(np.percentile(shares, tail)), float(np.percentile(shares, 100 - tail))]
    return {"value": float(hit[mask].mean()), "n": n, "ci": interval if len(shares) else None}


def source_metrics(
    outcomes: list[RunOutcome], *, n_boot: int, seed: int, ci: float
) -> dict[str, Any]:
    """Counts, rates and AUROC of one source of decisions, over its runs."""
    kw = {"n_boot": n_boot, "seed": seed, "ci": ci}
    n = len(outcomes)
    if n == 0:
        return {"n_runs": 0}
    alert = np.array([o.decision == "alert" for o in outcomes])
    label = np.array([o.end_label for o in outcomes])
    positive, negative = label == "positive", label == "negative"
    control = np.array([o.role == "control" for o in outcomes])
    decided = positive | negative
    out: dict[str, Any] = {
        "n_runs": n,
        "n_positive_end": int(positive.sum()),
        "n_negative_end": int(negative.sum()),
        "n_middle_end": int((label == "middle").sum()),
        "n_control_runs": int(control.sum()),
        "decisions": {
            d: sum(o.decision == d for o in outcomes) for d in ("alert", "no_alert", "abstain")
        },
        "abstention": _rate(
            np.array([o.decision == "abstain" for o in outcomes]), np.ones(n, dtype=bool), **kw
        ),  # fmt: skip
        "recall_end": _rate(alert, positive, **kw),  # alerts among runs that end near tipping
        "false_alarm_end": _rate(alert, negative, **kw),  # alerts among runs that end far
        "control_alert": _rate(alert, control, **kw),  # alerts among control runs
        "precision_end": _rate(positive, alert & decided, **kw),  # right alerts among alerts
    }
    scored = [(o.end_label == "positive", o.score) for o in outcomes
              if o.end_label != "middle" and o.score is not None]  # fmt: skip
    y = np.array([s[0] for s in scored], dtype=float)
    out["n_scored"] = len(scored)
    if len(scored) and 0 < y.sum() < len(y):
        a = bootstrap_auroc(y, np.array([s[1] for s in scored]), np.arange(len(scored)), **kw)
        out["auroc_end"] = {"auroc": a["auroc"], "ci_low": a["ci_low"], "ci_high": a["ci_high"]}
    else:
        out["auroc_end"] = None  # needs runs of both classes
    return out

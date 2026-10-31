"""The per-variant numbers of the comparison table, from the runs of one evaluation.

Every run of every variant is a dict with the same shape:

  decision      the variant's final decision for the end of the series
                (alert, no_alert or abstain)
  accepted      True or False if a gate ruled on the run, else None
  pfn_quality   how good the PFN's output was on the run's final context: coverage_90,
                calibration_error and bad (coverage below 0.80). None if the variant has no PFN
  timing        llm_seconds, pfn_seconds and total_seconds of the run
  record        the variant's own record, which holds n_tool_calls for variants with tools
  ground_truth  scenario_id, role and the label of the last window

The summary has the latency (mean, median and p95), the tool calls per verdict, how often the
three seeds of a scenario agree (a pass^3 style number), and the PFN quality, including how many
BAD runs the gate blocked. See docs/GATE.md for what a BAD run is.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

K_SEEDS = 3  # the number of seeds per scenario that must agree


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def latency_block(seconds: list[float]) -> dict[str, Any]:
    if not seconds:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    a = np.asarray(seconds, dtype=float)
    return {
        "n": len(a),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def pass_k(runs: list[dict[str, Any]], k: int = K_SEEDS) -> dict[str, Any]:
    """In how many scenarios do all k seeds give the same decision?

    The decisions that count are alert, no_alert and abstain, so a scenario where the gate
    abstains on every seed agrees too. A scenario with fewer or more than k runs is left out
    and counted in n_incomplete. This measures consistency, not correctness.
    """
    by_scenario: dict[str, list[str]] = defaultdict(list)
    for r in runs:
        by_scenario[r["ground_truth"]["scenario_id"]].append(r["decision"])
    complete = {s: d for s, d in by_scenario.items() if len(d) == k}
    agree = sorted(s for s, d in complete.items() if len(set(d)) == 1)
    return {
        "k": k,
        "n_scenarios": len(complete),
        "n_agree": len(agree),
        "share": len(agree) / len(complete) if complete else None,
        "n_incomplete": len(by_scenario) - len(complete),
        "decisions": {s: complete[s] for s in sorted(complete)},
    }


def pfn_block(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Coverage, calibration error and the BAD runs, over the runs that have a PFN output."""
    scored = [r for r in runs if r.get("pfn_quality")]
    if not scored:
        return None
    gated = any(r["accepted"] is not None for r in scored)
    accepted = [r for r in scored if r["accepted"] is True]
    bad = [r for r in scored if r["pfn_quality"]["bad"]]
    blocked = [r for r in bad if r["accepted"] is False]
    return {
        "n_runs": len(scored),
        "n_without_pfn_output": len(runs) - len(scored),
        "gated": gated,
        "coverage_90_mean_all": _mean([r["pfn_quality"]["coverage_90"] for r in scored]),
        "coverage_90_mean_accepted": _mean([r["pfn_quality"]["coverage_90"] for r in accepted])
        if gated
        else None,
        "n_accepted": len(accepted) if gated else None,
        "calibration_error_mean": _mean([r["pfn_quality"]["calibration_error"] for r in scored]),
        "n_bad": len(bad),
        "n_bad_blocked": len(blocked) if gated else None,
        "bad_blocked_share": (len(blocked) / len(bad)) if gated and bad else None,
    }


def summarise_runs(runs: list[dict[str, Any]], *, has_tools: bool) -> dict[str, Any]:
    """The numbers of one evaluation: latency, tool calls, pass^k and the PFN quality."""
    calls = [r["record"]["n_tool_calls"] for r in runs] if has_tools else []
    return {
        "n_runs": len(runs),
        "latency_seconds": latency_block([r["timing"]["total_seconds"] for r in runs]),
        "llm_seconds_mean": _mean([r["timing"]["llm_seconds"] for r in runs]),
        "pfn_seconds_mean": _mean([r["timing"]["pfn_seconds"] for r in runs]),
        "tool_calls": {"n": len(calls), "mean": _mean(calls), "max": max(calls) if calls else None}
        if has_tools
        else None,
        "pass_k": pass_k(runs),
        "pfn": pfn_block(runs),
    }

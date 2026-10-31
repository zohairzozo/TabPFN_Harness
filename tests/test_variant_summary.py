"""Tests for the per-variant summary numbers, with values worked out by hand."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pfngate.evals.variant_summary import latency_block, pass_k, pfn_block, summarise_runs


def run(
    scenario: str,
    decision: str,
    *,
    seconds: float = 1.0,
    accepted: bool | None = None,
    coverage: float | None = None,
    bad: bool = False,
    cal: float = 0.1,
    calls: int | None = None,
) -> dict[str, Any]:
    quality = None
    if coverage is not None:
        quality = {"coverage_90": coverage, "bad": bad, "calibration_error": cal}
    record = {} if calls is None else {"n_tool_calls": calls}
    return {
        "decision": decision,
        "accepted": accepted,
        "pfn_quality": quality,
        "record": record,
        "ground_truth": {"scenario_id": scenario},
        "timing": {
            "llm_seconds": seconds / 2,
            "pfn_seconds": seconds / 2,
            "total_seconds": seconds,
        },
    }


def test_latency_percentiles_by_hand() -> None:
    values = [float(i) for i in range(1, 101)]  # 1 ... 100
    b = latency_block(values)
    assert b["n"] == 100 and b["mean"] == 50.5 and b["max"] == 100.0
    assert b["p50"] == pytest.approx(np.percentile(values, 50)) == pytest.approx(50.5)
    assert b["p95"] == pytest.approx(95.05)  # linear interpolation between the 95th and 96th
    assert latency_block([]) == {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    assert latency_block([2.0])["p95"] == 2.0


def test_pass_k_counts_scenarios_where_all_three_seeds_agree() -> None:
    runs = [run("a", "alert")] * 3  # agree
    runs += [run("b", "alert"), run("b", "alert"), run("b", "no_alert")]  # disagree
    runs += [run("c", "abstain")] * 3  # agree: an abstention is a decision
    runs += [run("d", "alert"), run("d", "alert")]  # incomplete: two seeds only
    p = pass_k(runs)
    assert p["k"] == 3 and p["n_scenarios"] == 3 and p["n_agree"] == 2 and p["n_incomplete"] == 1
    assert p["share"] == pytest.approx(2 / 3)
    assert p["decisions"] == {"a": ["alert"] * 3, "b": ["alert", "alert", "no_alert"],
                              "c": ["abstain"] * 3}  # fmt: skip
    assert pass_k([run("d", "alert")])["share"] is None  # no complete scenario: no share
    assert pass_k([])["n_scenarios"] == 0


def test_pass_k_is_about_agreement_not_correctness() -> None:
    wrong_but_consistent = [run("a", "no_alert")] * 3  # whatever the truth is
    assert pass_k(wrong_but_consistent)["share"] == 1.0


def test_pfn_block_by_hand_for_a_gated_variant() -> None:
    runs = [
        run("a", "alert", accepted=True, coverage=0.90, cal=0.10),
        run("b", "alert", accepted=True, coverage=0.80, cal=0.20),
        run("c", "abstain", accepted=False, coverage=0.50, bad=True, cal=0.30),  # bad, blocked
        run("d", "alert", accepted=True, coverage=0.70, bad=True, cal=0.40),  # bad, not blocked
        run("e", "abstain", accepted=None),  # no context, so no PFN output
    ]
    b = pfn_block(runs)
    assert b is not None and b["gated"] is True
    assert b["n_runs"] == 4 and b["n_without_pfn_output"] == 1
    assert b["coverage_90_mean_all"] == pytest.approx((0.9 + 0.8 + 0.5 + 0.7) / 4)
    assert b["coverage_90_mean_accepted"] == pytest.approx((0.9 + 0.8 + 0.7) / 3)
    assert b["calibration_error_mean"] == pytest.approx((0.1 + 0.2 + 0.3 + 0.4) / 4)
    assert (b["n_bad"], b["n_bad_blocked"], b["bad_blocked_share"]) == (2, 1, 0.5)


def test_pfn_block_for_an_ungated_variant_has_no_blocking() -> None:
    runs = [run("a", "alert", coverage=0.9, bad=False), run("b", "alert", coverage=0.6, bad=True)]
    b = pfn_block(runs)
    assert b is not None and b["gated"] is False
    assert b["n_bad"] == 1 and b["n_bad_blocked"] is None and b["bad_blocked_share"] is None
    assert b["coverage_90_mean_accepted"] is None and b["n_accepted"] is None
    assert pfn_block([run("a", "alert")]) is None  # no PFN at all


def test_no_bad_run_means_no_blocked_share() -> None:
    b = pfn_block([run("a", "alert", accepted=True, coverage=0.9)])
    assert b is not None and b["n_bad"] == 0 and b["bad_blocked_share"] is None


def test_summarise_runs_tool_calls_and_timing() -> None:
    runs = [run("a", "alert", seconds=2.0, calls=4), run("a", "alert", seconds=4.0, calls=6)]
    s = summarise_runs(runs, has_tools=True)
    assert s["tool_calls"] == {"n": 2, "mean": 5.0, "max": 6}
    assert s["latency_seconds"]["mean"] == 3.0 and s["llm_seconds_mean"] == 1.5
    assert s["pfn_seconds_mean"] == 1.5 and s["n_runs"] == 2
    assert summarise_runs(runs, has_tools=False)["tool_calls"] is None  # classic and the oracle

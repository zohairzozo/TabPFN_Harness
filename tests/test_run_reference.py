"""Tests for the reference variants (classic baseline and oracle) and the per-run scoring that
every variant shares. The PFN is the fake backend, on one seed per scenario of the real data."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from agent_support import (
    AGENT_CFG,
    AGENT_TOOL_STEPS,
    NAIVE_FINAL,
    NAIVE_TOOL_STEPS,
    VERDICT,
    GateFactory,
    StatelessScript,
    build_setup,
    make_harness,
    neighbours,
)
from pfngate.evals import run_agent, run_reference
from pfngate.evals.llm_eval import final_context_quality, load_split_queries, result_path
from pfngate.evals.run_reference import oracle_alert_threshold
from pfngate.features.config import REPO_ROOT
from pfngate.gate import GateNotFrozenError
from pfngate.pfn import load_pfn_config

PFN = dataclasses.replace(load_pfn_config(), bootstrap_n=50)


@pytest.fixture(scope="module")
def env(small_data: dict[str, Any], make_gate: GateFactory, tmp_path_factory: Any) -> Any:
    tmp = tmp_path_factory.mktemp("reference")
    pool, fc = small_data["pool"], small_data["fc"]
    version = neighbours().model_version
    config, digest = make_gate(tmp / "gate", pool, fc, model_version=version,
                               alert={"random": 0.5, "knn": 0.5})  # fmt: skip
    prov, prov_digest = make_gate(tmp / "prov", pool, fc, model_version=version,
                                  status="provisional", freeze=False)  # fmt: skip
    return {**small_data, "config": config, "digest": digest, "prov": prov,
            "prov_digest": prov_digest}  # fmt: skip


def kw(env: Any, tmp: Path, gate: str = "config") -> dict[str, Any]:
    return {
        "cfg": PFN, "feature_config": env["fc"], "seeds_per_scenario": 1,
        "pool_seeds_per_scenario": 1, "results_dir": tmp, "gate_config_path": env[gate],
        "gate_hash_path": env["digest" if gate == "config" else "prov_digest"], "progress": False,
    }  # fmt: skip


def oracle(env: Any, tmp: Path, split: str = "val", **extra: Any) -> dict[str, Any]:
    return run_reference.evaluate("oracle", split, backend=neighbours(), alert_threshold=0.5,
                                  max_context=200, **kw(env, tmp), **extra)  # fmt: skip


# The classic baseline
def test_the_classic_baseline_alerts_when_the_last_window_score_reaches_the_threshold(
    env: Any, tmp_path: Path
) -> None:
    result = run_reference.evaluate("classic", "val", limit=2, **kw(env, tmp_path))
    assert result["method"] == "classic_ews" and result["model"] is None
    assert result["n_runs"] == 4 and result["settings"]["alarm_threshold"] == 0.5
    for run in result["runs"]:
        rec = run["record"]
        expected = "alert" if rec["score"] is not None and rec["score"] >= 0.5 else "no_alert"
        assert rec["decision"] == run["decision"] == expected
        assert run["pfn_quality"] is None and run["accepted"] is None
        assert run["timing"]["llm_seconds"] == 0.0
        assert run["timing"]["total_seconds"] == pytest.approx(rec["seconds"])
    s = result["summary"]
    assert s["tool_calls"] is None and s["pfn"] is None and s["latency_seconds"]["n"] == 4
    assert "classic_ews" in result["metrics"] and result["usage"] is None
    assert result_path(tmp_path, "classic_ews", "val").exists()


def test_the_classic_baseline_has_no_score_before_k_windows_so_it_does_not_alert(
    env: Any,
) -> None:
    import numpy as np
    import pandas as pd

    from pfngate.evals.baseline_ews import load_baseline_config

    cfg = load_baseline_config()
    q = pd.DataFrame({"trajectory_id": ["t/1"] * 3, "window_end": [10, 20, 30]})
    undefined = pd.DataFrame({"window_end": [10, 20, 30], "score": [np.nan, np.nan, np.nan]})
    record, decision, score = run_reference.classic_run(q, undefined, 0.1, cfg)
    assert (decision, score, record["score_defined"]) == ("no_alert", None, False)
    high = pd.DataFrame({"window_end": [30, 10, 20], "score": [0.9, 0.1, 0.2]})  # unsorted
    assert run_reference.classic_run(q, high, 0.1, cfg)[1:] == ("alert", 0.9)
    with pytest.raises(ValueError, match="last window"):
        run_reference.classic_run(q, high.assign(window_end=[31, 11, 21]), 0.1, cfg)


# The oracle
def test_the_oracle_uses_a_family_matched_context_and_no_gate_or_llm(
    env: Any, tmp_path: Path
) -> None:
    result = oracle(env, tmp_path, limit=2)
    assert result["method"] == "oracle" and result["oracle"] is True and result["model"] is None
    assert result["n_runs"] == 4 and result["skipped"] == []
    for run in result["runs"]:
        rec = run["record"]
        assert run["decision"] == ("alert" if rec["score"] >= 0.5 else "no_alert")
        assert run["accepted"] is None  # no gate
        q = run["pfn_quality"]
        assert 0.0 <= q["coverage_90"] <= 1.0 and 0.0 <= q["calibration_error"] <= 1.0
        assert q["last_window_p_critical"] == pytest.approx(rec["score"])
        assert run["timing"]["llm_seconds"] == 0.0 and run["timing"]["pfn_seconds"] >= 0.0
    p = result["summary"]["pfn"]
    assert p["gated"] is False and p["n_bad_blocked"] is None and p["n_runs"] == 4
    assert result["settings"]["strategy"] == "family_matched"


def test_the_oracle_cannot_run_on_a_family_that_is_not_in_the_pool(
    env: Any, tmp_path: Path
) -> None:
    result = oracle(env, tmp_path, split="ood", limit=2)
    skipped = {s["trajectory_id"].split("_L")[0].split("_N")[0] for s in result["skipped"]}
    assert skipped == {"rate", "noise"}  # every out-of-prior tipping run
    assert all("no pool rows" in s["reason"] for s in result["skipped"])
    assert {r["ground_truth"]["role"] for r in result["runs"]} == {"control"}
    assert result["n_runs"] == 2 and len(result["skipped"]) == 2 and result["held_out"] is True
    assert result["metrics"]["oracle"]["auroc_end"] is None  # only controls: no positive run


def test_the_oracle_threshold_is_read_from_the_results_it_came_from(tmp_path: Path) -> None:
    (tmp_path / run_reference.ORACLE_THRESHOLD_FILE).write_text(
        json.dumps({"false_alarms": {"thresholds": {"p_critical": 0.25}}})
    )
    assert oracle_alert_threshold(tmp_path) == 0.25
    with pytest.raises(FileNotFoundError, match="run_pfn"):
        oracle_alert_threshold(tmp_path / "missing")
    real = oracle_alert_threshold(REPO_ROOT / "results")  # the validation-derived value
    saved = json.loads((REPO_ROOT / "results" / run_reference.ORACLE_THRESHOLD_FILE).read_text())
    assert real == saved["false_alarms"]["thresholds"]["p_critical"]


@pytest.mark.parametrize("variant", ["classic", "oracle"])
def test_held_out_reference_runs_need_a_frozen_gate(env: Any, tmp_path: Path, variant: str) -> None:
    extra = {"backend": neighbours(), "alert_threshold": 0.5} if variant == "oracle" else {}
    with pytest.raises(GateNotFrozenError):
        run_reference.evaluate(variant, "test", **extra, **kw(env, tmp_path, gate="prov"))
    assert not (tmp_path / "traces").exists()


# The scoring the LLM variants share
def test_the_agent_and_naive_evaluations_score_the_pfn_on_their_final_context(
    env: Any, tmp_path: Path
) -> None:
    kwargs = {"agent_cfg": AGENT_CFG, "max_context": 200, "backend": neighbours(), "limit": 2,
              **kw(env, tmp_path)}  # fmt: skip
    agent_model = StatelessScript(AGENT_TOOL_STEPS, VERDICT).model()
    naive_model = StatelessScript(NAIVE_TOOL_STEPS, NAIVE_FINAL).model()
    gated = run_agent.evaluate("agent", "val", model=agent_model, **kwargs)
    naive = run_agent.evaluate("naive", "val", model=naive_model, **kwargs)
    for result, calls in ((gated, 4.0), (naive, 2.0)):
        s = result["summary"]
        assert s["tool_calls"]["mean"] == calls and s["pfn"]["n_runs"] == result["n_runs"]
        for run in result["runs"]:
            t = run["timing"]
            assert t["total_seconds"] == pytest.approx(t["llm_seconds"] + t["pfn_seconds"])
            assert t["llm_seconds"] == run["record"]["llm_seconds"] > 0.0
            assert 0.0 <= run["pfn_quality"]["coverage_90"] <= 1.0
    # Only the gated agent has accepted runs, and they are the runs whose gate passed.
    assert all(r["accepted"] is None for r in naive["runs"])
    assert [r["accepted"] for r in gated["runs"]] == [
        r["record"]["final"]["gate_passed"] for r in gated["runs"]
    ]
    assert gated["summary"]["pfn"]["gated"] is True and naive["summary"]["pfn"]["gated"] is False


def test_the_final_context_must_be_the_one_the_run_used(
    small_data: dict[str, Any], make_gate: GateFactory, tmp_path: Path
) -> None:
    setup = build_setup(small_data, make_gate, tmp_path / "setup")
    q = load_split_queries("val", setup["fc"], 1, 1)[0]
    harness = make_harness(setup, neighbours(), tmp_path)
    with pytest.raises(ValueError, match="not the one the run used"):
        final_context_quality(harness, q, "knn", 200, "0" * 16)

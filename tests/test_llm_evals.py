"""Tests for the evaluation entry points of the LLM variants (run_agent and baseline_llm).

Scripted fake models and the fake PFN backend, on one seed per scenario of the real data.
Held-out splits need a tuned and frozen gate, so the runs here use one in a temp folder.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agent_support import (
    AGENT_CFG,
    AGENT_TOOL_STEPS,
    COST_PER_CALL,
    LLM_ONLY_FINAL,
    MODEL_NAME,
    NAIVE_FINAL,
    NAIVE_TOOL_STEPS,
    VERDICT,
    GateFactory,
    StatelessScript,
    neighbours,
)
from pfngate.agent.config import load_agent_config
from pfngate.agent.replay import LLMStore, RecordReplayModel
from pfngate.evals import baseline_llm, llm_eval, run_agent
from pfngate.evals.llm_eval import (
    DEFAULT_GATE_CONFIG,
    HELD_OUT_LLM_SPLITS,
    build_llm_model,
    load_split_queries,
    result_path,
)
from pfngate.evals.trace import TRACE_FIELDS, read_trace
from pfngate.gate import GateNotApplicableError, GateNotFrozenError, load_gate_config
from pfngate.pfn import load_pfn_config

PFN = dataclasses.replace(load_pfn_config(), bootstrap_n=50)
SCRIPTS = {
    "agent": lambda: StatelessScript(AGENT_TOOL_STEPS, VERDICT),
    "naive": lambda: StatelessScript(NAIVE_TOOL_STEPS, NAIVE_FINAL),
    "llm_only": lambda: StatelessScript([], LLM_ONLY_FINAL),
}
METHOD = {"agent": "agent", "naive": "naive_agent", "llm_only": "llm_only"}


def exploding() -> FunctionModel:
    def boom(messages: list[Any], info: AgentInfo) -> Any:
        raise AssertionError("the model was called")

    return FunctionModel(boom, model_name=MODEL_NAME)


@pytest.fixture(scope="module")
def env(small_data: dict[str, Any], make_gate: GateFactory, tmp_path_factory: Any) -> Any:
    tmp = tmp_path_factory.mktemp("llm_evals")
    pool, fc = small_data["pool"], small_data["fc"]
    version = neighbours().model_version
    alert = {"random": 0.5, "knn": 0.5}
    config, digest = make_gate(tmp / "gate", pool, fc, model_version=version, alert=alert)
    prov, prov_digest = make_gate(tmp / "prov", pool, fc, model_version=version,
                                  status="provisional", freeze=False)  # fmt: skip
    return {**small_data, "config": config, "digest": digest, "prov": prov,
            "prov_digest": prov_digest}  # fmt: skip


def call(
    which: str, split: str, env: Any, tmp: Path, model: Any, *, gate: str = "config", **extra: Any
) -> dict[str, Any]:
    """Run one evaluation. `gate` is "config" (tuned and frozen) or "prov" (not frozen)."""
    common: dict[str, Any] = {
        "feature_config": env["fc"],
        "agent_cfg": AGENT_CFG,
        "pfn_cfg": PFN,
        "seeds_per_scenario": 1,
        "results_dir": tmp,
        "gate_config_path": env[gate],
        "gate_hash_path": env["digest" if gate == "config" else "prov_digest"],
        "progress": False,
        "model": model,
    }
    if which == "llm_only":
        return baseline_llm.evaluate(split, **common, **extra)
    common["cfg"] = common.pop("pfn_cfg")
    defaults = {"max_context": 200, "pool_seeds_per_scenario": 1}
    return run_agent.evaluate(
        "agent" if which == "agent" else "naive",
        split,
        backend=neighbours(),
        **{**defaults, **common, **extra},
    )


def n_validation(env: Any) -> int:
    return len(load_split_queries("val", env["fc"], 1, None))


# What an evaluation reports
def test_the_agent_evaluation_scores_the_final_decision_and_the_recommendation(
    env: Any, tmp_path: Path
) -> None:
    n = n_validation(env)
    result = call("agent", "val", env, tmp_path, SCRIPTS["agent"]().model())
    assert result["method"] == "agent" and result["n_runs"] == n and result["held_out"] is False
    assert set(result["metrics"]) == {"agent_final", "agent_recommendation"}
    asked, final = result["metrics"]["agent_recommendation"], result["metrics"]["agent_final"]
    assert asked["decisions"] == {"alert": n, "no_alert": 0, "abstain": 0}  # the script says alert
    assert asked["control_alert"]["value"] == 1.0 and asked["recall_end"]["value"] == 1.0
    assert sum(final["decisions"].values()) == n
    g = result["gate"]
    assert g["n_gate_passed"] + g["n_gate_blocked"] + g["n_no_context"] == n
    assert g["agent_alerts_blocked_by_gate"] == final["decisions"]["abstain"]
    u = result["usage"]  # 5 model calls per run: 4 tools and the answer
    assert u["model_requests"] == 5 * n and u["tokens_in"] == 500 * n
    assert u["cost_usd_approx"] == pytest.approx(5 * n * COST_PER_CALL)
    assert u["stop_reasons"]["completed"] == n and u["replayed_requests"] == 0
    # The truth sits next to each record and the record does not hold it.
    for run in result["runs"]:
        assert run["ground_truth"]["used_by_variant"] is False
        assert "end_label" not in json.dumps(run["record"])
    saved = json.loads(result_path(tmp_path, "agent", "val").read_text())
    assert saved["n_runs"] == n and set(saved["metrics"]) == set(result["metrics"])
    rows = read_trace(tmp_path / "traces" / result["trace"])
    assert {tuple(r) for r in rows} == {TRACE_FIELDS}
    assert sum(r["tool"] == "llm_call" for r in rows) == 5 * n
    assert sum(r["tool"] == "gate" for r in rows) == g["n_gate_passed"] + g["n_gate_blocked"]


def test_the_naive_evaluation_has_no_gate_and_one_source(env: Any, tmp_path: Path) -> None:
    n = n_validation(env)
    result = call("naive", "val", env, tmp_path, SCRIPTS["naive"]().model())
    assert result["method"] == "naive_agent" and result["gate"] is None
    assert set(result["metrics"]) == {"naive_agent"}
    assert result["metrics"]["naive_agent"]["decisions"]["abstain"] == 0  # it cannot abstain
    assert result["usage"]["model_requests"] == 3 * n  # two tools and the answer
    names = {r["tool"] for r in read_trace(tmp_path / "traces" / result["trace"])}
    assert "gate" not in names and "context_calibration" not in names
    assert result_path(tmp_path, "naive_agent", "val").exists()


def test_the_baseline_evaluation_has_one_call_per_run_and_a_score(env: Any, tmp_path: Path) -> None:
    n = n_validation(env)
    result = call("llm_only", "val", env, tmp_path, SCRIPTS["llm_only"]().model())
    assert result["method"] == "llm_only" and result["usage"]["model_requests"] == n
    m = result["metrics"]["llm_only"]
    assert m["decisions"]["alert"] == n and m["n_scored"] > 0
    assert m["auroc_end"]["auroc"] == pytest.approx(0.5)  # one probability for every run
    assert result_path(tmp_path, "llm_only", "val").exists()


def test_the_truth_of_a_run_is_the_label_of_its_last_window(env: Any, tmp_path: Path) -> None:
    result = call("llm_only", "val", env, tmp_path, SCRIPTS["llm_only"]().model())
    by_id = {
        str(q["trajectory_id"].iloc[0]): q for q in load_split_queries("val", env["fc"], 1, None)
    }
    names = {1: "positive", 0: "negative", -1: "middle"}
    seen = set()
    for run in result["runs"]:
        q = by_id[run["record"]["trajectory_id"]]
        last = q.loc[q["window_end"].idxmax()]
        truth = run["ground_truth"]
        assert truth["end_target"] == float(last["target"])
        assert truth["end_label"] == names[int(last["label"])]
        assert truth["role"] == str(last["role"])
        seen.add((truth["role"] == "control", truth["end_label"]))
    assert (True, "negative") in seen and (False, "positive") in seen  # both kinds of run


def test_the_limit_picks_whole_scenarios_of_both_kinds_spread_over_the_families(
    env: Any, tmp_path: Path
) -> None:
    result = call("llm_only", "val", env, tmp_path, SCRIPTS["llm_only"]().model(), limit=3)
    roles = [r["ground_truth"]["role"] for r in result["runs"]]
    assert result["limit"] == 3 and result["n_runs"] == 6  # 3 tipping and 3 control scenarios
    assert roles.count("control") == 3 and roles.count("in_prior") == 3
    families = {
        r["ground_truth"]["family"]
        for r in result["runs"]
        if r["ground_truth"]["role"] != "control"
    }
    assert len(families) == 3  # not three scenarios of one family
    assert (
        len({r["ground_truth"]["family"] for r in result["runs"]}) == 6
    )  # and three control kinds


# Record, then replay a whole evaluation with the model unplugged
@pytest.mark.parametrize("which", ["agent", "naive", "llm_only"])
def test_an_evaluation_replays_identically_without_the_model(
    env: Any, tmp_path: Path, which: str
) -> None:
    cassettes = tmp_path / "llm"
    recorder = RecordReplayModel(SCRIPTS[which]().model(), mode="record", directory=cassettes)
    recorded = call(which, "val", env, tmp_path / "rec", recorder, limit=2)
    before = {p.name: p.read_bytes() for p in cassettes.glob("*.json")}
    assert before and recorder.recorded_calls == len(before)

    replayer = RecordReplayModel(exploding(), mode="replay", directory=cassettes)
    replayed = call(which, "val", env, tmp_path / "rep", replayer, limit=2)
    assert {p.name: p.read_bytes() for p in cassettes.glob("*.json")} == before
    assert (replayed["llm_mode"], recorded["llm_mode"]) == ("replay", "record")
    u = replayed["usage"]
    assert u["replayed_requests"] == u["model_requests"] > 0 and replayer.live_calls == 0

    def comparable(r: dict[str, Any]) -> Any:
        runs = json.loads(json.dumps(r["runs"]))
        for run in runs:
            run["record"].pop("run_id"), run["record"].pop("n_replayed_requests")
            # The PFN time is measured afresh in each folder. The recorded LLM latency must match.
            run["timing"] = {"llm_seconds": run["timing"]["llm_seconds"]}
            if run["record"].get("gate_result"):
                run["record"]["gate_result"].pop("run_id")
        return r["metrics"], runs, {k: v for k, v in r["usage"].items() if k != "replayed_requests"}

    assert comparable(replayed) == comparable(recorded)


def test_a_missing_cassette_stops_an_evaluation(env: Any, tmp_path: Path) -> None:
    from pfngate.agent.replay import ReplayMissError

    model = RecordReplayModel(exploding(), mode="replay", directory=tmp_path / "empty")
    with pytest.raises(ReplayMissError):
        call("agent", "val", env, tmp_path / "run", model, limit=2)


# The held-out splits
@pytest.mark.parametrize("which", ["agent", "naive", "llm_only"])
def test_a_held_out_split_refuses_a_gate_that_is_not_frozen_before_any_model_call(
    env: Any, tmp_path: Path, which: str
) -> None:
    assert HELD_OUT_LLM_SPLITS == {"test", "ood", "real", "real_control"}  # real: test only
    for split in ("test", "ood"):
        with pytest.raises(GateNotFrozenError):
            call(which, split, env, tmp_path, exploding(), gate="prov")
    assert not (tmp_path / "traces").exists()  # nothing was read, nothing was written


@pytest.mark.parametrize("which", ["agent", "naive", "llm_only"])
def test_a_held_out_split_runs_with_a_frozen_gate_and_records_it(
    env: Any, tmp_path: Path, which: str
) -> None:
    result = call(which, "test", env, tmp_path, SCRIPTS[which]().model(), limit=2)
    assert result["held_out"] is True and result["dataset_splits"] == ["test_in_prior"]
    assert result["n_runs"] == 4 and result_path(tmp_path, METHOD[which], "test").exists()
    if which != "llm_only":  # the gate's hash is in the result, as in run_pfn
        assert result["frozen_gate_sha256"] == load_gate_config(env["config"]).sha256


@pytest.mark.parametrize("which", ["agent", "naive"])
def test_a_held_out_split_refuses_settings_the_gate_was_not_tuned_for(
    env: Any, tmp_path: Path, which: str
) -> None:
    with pytest.raises(GateNotApplicableError):
        call(which, "test", env, tmp_path, exploding(), max_context=100)


def test_the_agent_needs_a_tuned_gate_on_validation_too(env: Any, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tuned gate"):
        call("agent", "val", env, tmp_path, SCRIPTS["agent"]().model(), gate="prov", limit=1)


# --llm-mode
def test_the_default_llm_mode_is_replay_only_if_the_cache_has_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = load_agent_config()
    monkeypatch.setattr("pfngate.agent.replay.load_dotenv", lambda *a, **k: None)  # keep .env out
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-not-a-key")
    assert build_llm_model(cfg, debug=False, mode=None, directory=tmp_path / "none").mode == "live"
    LLMStore(tmp_path).put("a" * 64, "m:n", ModelResponse(parts=[]), [])
    monkeypatch.delenv("OPENROUTER_API_KEY")
    replay = build_llm_model(cfg, debug=False, mode=None, directory=tmp_path)
    assert replay.mode == "replay" and replay.model_ref == cfg.model  # no key was needed
    assert build_llm_model(cfg, debug=True, mode="replay", directory=tmp_path).model_ref == (
        cfg.debug_model
    )


@pytest.mark.parametrize("module", [run_agent, baseline_llm])
def test_the_llm_mode_flag_reaches_the_evaluation(
    module: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []
    committed: list[Path] = []

    def fake(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append({"args": args, **kwargs})
        usage = {"model_requests": 0, "replayed_requests": 0, "cost_usd_approx": 0.0}
        return {"n_runs": 0, "llm_mode": "x", "usage": usage, "method": "m"}

    monkeypatch.setattr(module, "evaluate", fake)
    monkeypatch.setattr(module, "assert_committed", committed.append)
    base = ["--results-dir", str(tmp_path)]
    module.main(["--split", "val", *base])
    assert (
        calls[-1]["llm_mode"] is None and calls[-1]["debug"] is False
    )  # evaluate picks the default
    assert committed == []  # validation is not held out
    module.main(["--split", "val", "--llm-mode", "record", "--debug", "--limit", "2", *base])
    assert (
        calls[-1]["llm_mode"] == "record" and calls[-1]["debug"] is True and calls[-1]["limit"] == 2
    )
    module.main(["--split", "test", "--llm-mode", "replay", *base])
    assert calls[-1]["llm_mode"] == "replay" and committed == [DEFAULT_GATE_CONFIG]
    if module is run_agent:
        module.main(["--variant", "naive", "--split", "ood", *base])
        assert calls[-1]["args"][0] == "naive" and committed == [DEFAULT_GATE_CONFIG] * 2
    assert llm_eval.LLM_SPLITS == {"val": "validation", "test": "test_in_prior",
                                   "ood": "test_out_of_prior",
                                   "real": "real_tipping",
                                   "real_control": "real_control"}  # fmt: skip

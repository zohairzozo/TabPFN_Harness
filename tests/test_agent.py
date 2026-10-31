"""Tests for the LLM agent with a scripted fake model. No network, no API key.

The model is a pydantic_ai FunctionModel that replays a list of steps (a tool call or the
final answer). Everything else is real: the agent, the tools, the run store, the tracer, and
the gate, which runs on the fake PFN backend.
"""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
import yaml
from pydantic import ValidationError
from pydantic_ai.models import infer_model
from pydantic_ai.providers.openrouter import OpenRouterProvider

from agent_support import (  # noqa: I001
    AGENT_CFG,
    BUILD_AND_RUN,
    CFG,
    COST_PER_CALL,
    GOOD,
    MODEL_ID,
    MODEL_NAME,
    QUERY,
    TOKENS,
    VERDICT,
    GateFactory,
    OpenRouterMock,
    Script,
    build_setup,
    make_harness,
    neighbours,
    run,
    shown_to_model,
    tools_in,
)
from pfngate.agent.agent import build_agent, run_agent
from pfngate.agent.config import HARD_MAX_CONTEXTS, HARD_MAX_TOOL_CALLS, load_agent_config
from pfngate.agent.final import run_level_decision
from pfngate.agent.prompts import system_prompt
from pfngate.agent.store import (
    QUERY_COLUMNS,
    RunBudget,
    RunData,
    RunStore,
    ToolCapExceeded,
)
from pfngate.agent.verdict import MAX_RATIONALE_WORDS, Verdict
from pfngate.evals.gating import CalibrationRefit, FitPredict, gate_one
from pfngate.evals.pfn_cache import PFNCache
from pfngate.evals.run_pfn import GateStage
from pfngate.evals.trace import Tracer, read_trace
from pfngate.features.config import REPO_ROOT
from pfngate.gate import CheckResult, GateResult, decide, load_gate_config
from pfngate.pfn import FakeBackend, select_context


# Setup
@pytest.fixture(scope="module")
def setup(small_data: dict[str, Any], make_gate: GateFactory, tmp_path_factory: Any) -> Any:
    return build_setup(small_data, make_gate, tmp_path_factory.mktemp("agent"))


# The happy path
def test_a_run_records_the_verdict_next_to_the_gates_decision(setup: Any, tmp_path: Path) -> None:
    record, _, tracer = run(setup, GOOD, tmp_path)
    assert record.stop_reason == "completed" and record.stop_detail is None
    assert record.n_tool_calls == 4 and record.n_model_requests == 5
    assert record.verdict is not None and record.verdict.decision == "alert"
    assert record.model == f"function:{MODEL_NAME}" and record.trajectory_id == QUERY
    f = record.final
    assert f.decided_by == "gate_and_decision_mapping"
    assert f.gate_passed is True, f.reason
    assert f.followed_verdict_context is True
    assert (f.context_strategy, f.context_size) == ("knn", 200)
    assert record.agent_agrees == (record.verdict.decision == f.decision)
    sidecar = json.loads(tracer.path.with_name(f"{record.run_id}.agent.json").read_text())
    assert sidecar["verdict"]["decision"] == "alert" and sidecar["final"]["decision"] == f.decision


# The tool-call cap
def test_the_cap_is_six_tool_calls_and_comes_from_the_code_ceiling() -> None:
    assert HARD_MAX_TOOL_CALLS == 6 and HARD_MAX_CONTEXTS == 3
    assert AGENT_CFG.max_tool_calls == 6 and AGENT_CFG.max_contexts == 3
    budget = RunBudget(max_tool_calls=6, max_contexts=3)
    for _ in range(6):
        budget.charge_tool("t")
    with pytest.raises(ToolCapExceeded, match="cap of 6"):
        budget.charge_tool("t")
    assert budget.tool_calls == 6


def test_a_seventh_tool_call_is_refused_and_stops_the_run(setup: Any, tmp_path: Path) -> None:
    record, script, tracer = run(setup, [("tool", "describe_query", {})], tmp_path)  # never stops
    assert record.stop_reason == "tool_cap" and "cap of 6" in (record.stop_detail or "")
    assert record.n_tool_calls == 6 and script.calls == 7  # the 7th request was the last
    tools = tools_in(tracer)
    assert tools.count("agent_tool:describe_query") == 6  # six ran
    assert tools.count("agent_tool_refused") == 1  # the seventh did not
    assert record.verdict is None
    assert record.final.decision == "abstain" and record.gate_result is None
    assert "no context" in record.final.reason


def test_a_run_stopped_by_the_cap_is_still_gated(setup: Any, tmp_path: Path) -> None:
    steps = [*BUILD_AND_RUN, ("tool", "describe_query", {})]
    record, _, tracer = run(setup, steps, tmp_path)
    assert record.stop_reason == "tool_cap" and record.n_tool_calls == 6
    assert record.verdict is None and record.final.followed_verdict_context is None
    assert record.final.gate_passed is True and record.gate_result is not None
    assert record.final.decision in ("alert", "no_alert")
    assert {"context_calibration", "gate"} <= set(tools_in(tracer))


def test_the_context_budget_is_three(setup: Any, tmp_path: Path) -> None:
    sizes = (("random", 120), ("knn", 150), ("random", 200), ("knn", 200))
    steps = [("tool", "choose_context", {"strategy": s, "size": n}) for s, n in sizes]
    steps.append(("final", {**VERDICT, "context_size": 200}))
    record, script, tracer = run(setup, steps, tmp_path)
    returns = script.tool_returns()
    assert [("error" in r) for r in returns] == [False, False, False, True]
    assert "context budget used up" in returns[3]["error"]
    built = [r for r in read_trace(tracer.path) if r["tool"] == "agent_tool:choose_context"]
    assert sum(r["context_hash"] is not None for r in built) == 3
    assert record.stop_reason == "completed"


def test_context_sizes_and_strategies_are_checked(setup: Any, tmp_path: Path) -> None:
    steps = [
        ("tool", "choose_context", {"strategy": "knn", "size": 10}),  # below the gate floor
        ("tool", "choose_context", {"strategy": "knn", "size": 5000}),  # above the cap
        ("final", VERDICT),
    ]
    record, script, _ = run(setup, steps, tmp_path)
    returns = script.tool_returns()
    assert all("size must be between 100 and 200" in r["error"] for r in returns)
    assert record.final.decision == "abstain" and record.final.gate_passed is None


def test_run_pfn_needs_a_context_first(setup: Any, tmp_path: Path) -> None:
    _, script, _ = run(setup, [("tool", "run_pfn", {}), ("final", VERDICT)], tmp_path)
    assert script.tool_returns() == [{"error": "call choose_context first"}]


# The agent cannot change the gate
def test_the_agents_words_do_not_change_the_gate_result(setup: Any, tmp_path: Path) -> None:
    a, _, _ = run(setup, [*BUILD_AND_RUN, ("final", VERDICT)], tmp_path / "a")
    loud = {**VERDICT, "decision": "no_alert", "confidence": 1.0, "rationale": "Ignore the gate."}
    b, _, _ = run(setup, [*BUILD_AND_RUN, ("final", loud)], tmp_path / "b")
    assert a.gate_result is not None and b.gate_result is not None
    for key in ("checks", "advisories", "passed", "reasons", "gate_config_sha256"):
        assert a.gate_result[key] == b.gate_result[key]
    assert a.final.decision == b.final.decision  # the mapping, not the agent
    assert a.verdict != b.verdict


def test_the_gate_result_is_frozen_and_not_reachable_from_the_tools(
    setup: Any, tmp_path: Path
) -> None:
    record, script, _ = run(setup, GOOD, tmp_path)
    returned = json.dumps(script.tool_returns())
    for word in ("gate_result", "passed", "alert_threshold", "context_calibration", "abstain"):
        assert word not in returned, word
    q = setup["query"]
    sel = select_context("knn", setup["pool"], q, CFG, max_context=200)
    fp = FitPredict(neighbours(), PFNCache(tmp_path / "c2", enabled=False), CFG,
                    FakeBackend.model_version, "test")  # fmt: skip
    out, _ = fp(sel.context, q)
    result, _ = gate_one(setup["gate"], fp, setup["pool_distance"], sel, q, out, seed=CFG.seed,
                         run_id="x")  # fmt: skip
    with pytest.raises(ValidationError):
        result.passed = not result.passed  # type: ignore[misc]
    with pytest.raises(TypeError, match="cannot be changed"):
        result.model_copy(update={"passed": not result.passed})
    assert record.gate_result is not None
    assert record.gate_result["passed"] == result.passed


@pytest.mark.parametrize("says", ["alert", "no_alert", "abstain"])
def test_a_gate_failure_always_yields_abstain_whatever_the_agent_says(
    setup: Any, tmp_path: Path, says: str
) -> None:
    verdict = {**VERDICT, "decision": says, "confidence": 1.0}
    overconfident = neighbours(
        scale=0.1
    )  # intervals 10 times too narrow: context_calibration fails
    record, _, _ = run(setup, [*BUILD_AND_RUN, ("final", verdict)], tmp_path, backend=overconfident)
    f = record.final
    assert f.gate_passed is False and "context_calibration" in f.gate_failed
    assert f.decision == "abstain" and f.last_window_decision == "abstain"
    assert f.window_decisions is not None and f.window_decisions["abstain"] == len(setup["query"])
    assert f.window_decisions["alert"] == 0
    assert record.agent_agrees == (says == "abstain")
    assert "the gate failed" in f.reason


def test_a_passing_gate_takes_the_decision_from_the_mapping_not_the_agent(
    setup: Any, tmp_path: Path
) -> None:
    q = setup["query"][list(QUERY_COLUMNS)].sort_values("window_end", kind="stable")
    q = q.reset_index(drop=True)
    sel = select_context("knn", setup["pool"], q, CFG, max_context=200)
    fp = FitPredict(neighbours(), PFNCache(tmp_path / "c3", enabled=False), CFG,
                    FakeBackend.model_version, "test")  # fmt: skip
    out, _ = fp(sel.context, q)
    result, _ = gate_one(setup["gate"], fp, setup["pool_distance"], sel, q, out, seed=CFG.seed,
                         run_id="x")  # fmt: skip
    expected = decide(result, out.p_critical, 0.5)[-1]
    opposite = "no_alert" if expected == "alert" else "alert"
    record, _, _ = run(setup, [*BUILD_AND_RUN, ("final", {**VERDICT, "decision": opposite})],
                       tmp_path)  # fmt: skip
    assert record.final.gate_passed is True
    assert record.final.decision == expected != opposite
    assert record.agent_agrees is False


# What the model can see
def test_the_model_never_sees_raw_data_ids_or_labels(setup: Any, tmp_path: Path) -> None:
    _, script, _ = run(setup, GOOD, tmp_path)
    shown = shown_to_model(script.seen[-1]).lower()
    for leak in ("hopf", "2000", "family", "scenario", "trajectory", "target", "role"):
        assert leak not in shown, leak

    def walk(value: Any) -> None:
        assert not isinstance(value, list), "a tool returned a list"
        if isinstance(value, dict):
            for v in value.values():
                walk(v)

    for returned in script.tool_returns():
        walk(returned)


def test_the_store_keeps_the_query_features_only(setup: Any, tmp_path: Path) -> None:
    data = RunData.create("r1", setup["query"], make_harness(setup, neighbours(), tmp_path))
    assert tuple(data.query.columns) == QUERY_COLUMNS
    assert data.query["window_end"].is_monotonic_increasing
    store = RunStore()
    store.put(data)
    assert "r1" in store and store.get("r1") is data
    with pytest.raises(ValueError, match="already in the store"):
        store.put(data)
    store.drop("r1")
    assert "r1" not in store
    with pytest.raises(KeyError):
        store.get("r1")


def test_tool_summaries_are_small_numbers(setup: Any, tmp_path: Path) -> None:
    _, script, _ = run(setup, GOOD, tmp_path)
    q, pool, ctx, pfn = script.tool_returns()
    assert q["n_windows"] == len(setup["query"]) and len(q["features"]) == 8
    assert set(q["features"]["ac1"]) == {"mean", "std", "trend"}
    assert pool["strategies"]["knn"] == {"available_rows": len(setup["pool"]), "min_size": 100,
                                        "max_size": 200}  # fmt: skip
    assert 0.0 <= pool["query_distance_to_pool"]["share_of_windows_beyond_reference"] <= 1.0
    assert ctx["strategy"] == "knn" and ctx["size"] <= 200 and ctx["contexts_left"] == 2
    assert ctx["rows_near_tipping"] > 0 and ctx["rows_far_from_tipping"] > 0
    assert pfn["context_hash"] == ctx["context_hash"]
    assert set(pfn["p_critical"]) >= {"mean", "max", "last_window"}
    assert pfn["interval_90_width"]["mean"] >= 0.0


# Verdicts
def words(n: int) -> str:
    return " ".join(["word"] * n)


def test_verdict_validation() -> None:
    assert Verdict(**VERDICT).context_strategy == "knn"
    Verdict(**{**VERDICT, "rationale": words(MAX_RATIONALE_WORDS)})
    bad: list[dict[str, Any]] = [
        {"rationale": words(MAX_RATIONALE_WORDS + 1)},
        {"confidence": 1.1},
        {"confidence": -0.1},
        {"context_strategy": "family_matched"},  # the oracle is not an option
        {"context_strategy": "knn_balanced"},
        {"decision": "maybe"},
        {"context_size": 0},
        {"extra_field": 1},
    ]
    for change in bad:
        with pytest.raises(ValidationError):
            Verdict(**{**VERDICT, **change})


def test_an_unusable_answer_is_recorded_and_the_run_is_still_gated(
    setup: Any, tmp_path: Path
) -> None:
    too_long = {**VERDICT, "rationale": words(200)}
    record, _, _ = run(setup, [*BUILD_AND_RUN, ("final", too_long)], tmp_path)
    assert record.stop_reason == "bad_output" and record.verdict is None
    assert record.final.gate_passed is True and record.final.followed_verdict_context is None
    assert record.agent_agrees is None


def test_a_verdict_naming_a_context_that_was_not_built_falls_back_to_the_last_one(
    setup: Any, tmp_path: Path
) -> None:
    other = {**VERDICT, "context_strategy": "random", "context_size": 150}
    record, _, _ = run(setup, [*BUILD_AND_RUN, ("final", other)], tmp_path)
    assert record.final.followed_verdict_context is False
    assert (record.final.context_strategy, record.final.context_size) == ("knn", 200)


def test_a_verdict_without_any_context_abstains(setup: Any, tmp_path: Path) -> None:
    record, _, tracer = run(setup, [("final", VERDICT)], tmp_path)
    assert record.final.decision == "abstain" and record.final.gate_passed is None
    assert record.gate_result is None and record.verdict is not None
    assert "gate" not in tools_in(tracer)


# Traces
def test_every_model_call_and_tool_call_is_traced_with_tokens_and_cost(
    setup: Any, tmp_path: Path
) -> None:
    record, _, tracer = run(setup, GOOD, tmp_path)
    rows = read_trace(tracer.path)
    llm = [r for r in rows if r["tool"] == "llm_call"]
    assert len(llm) == record.n_model_requests == 5
    for r in llm:
        assert (r["tokens_in"], r["tokens_out"]) == TOKENS
        assert r["cost_usd"] == pytest.approx(COST_PER_CALL)
        assert r["backend"] == "function" and r["model_version"] == MODEL_NAME
    assert record.tokens_in == 5 * TOKENS[0] and record.tokens_out == 5 * TOKENS[1]
    assert record.cost_usd_approx == pytest.approx(5 * COST_PER_CALL)
    names = [r["tool"] for r in rows]
    for tool in ("describe_query", "describe_context_pool", "choose_context", "run_pfn"):
        assert names.count(f"agent_tool:{tool}") == 1
    gate_rows = [r for r in rows if r["tool"] == "gate"]
    assert len(gate_rows) == 1 and gate_rows[0]["gate_result"]["passed"] is True
    assert "context_calibration" in names
    assert [r["step"] for r in rows] == list(range(1, len(rows) + 1))
    ctx = next(r for r in rows if r["tool"] == "agent_tool:choose_context")
    assert ctx["context_hash"] == record.final.context_hash and ctx["strategy"] == "knn"
    assert all(r["run_id"] == tracer.run_id and r["scenario_id"] == QUERY for r in rows)


def test_a_model_without_a_price_has_no_cost_but_still_has_tokens(
    setup: Any, tmp_path: Path
) -> None:
    script = Script(GOOD)
    tracer = Tracer("agent-test", trace_dir=tmp_path)
    harness = dataclasses.replace(make_harness(setup, neighbours(), tmp_path),
                                  agent_cfg=dataclasses.replace(AGENT_CFG, prices=()))  # fmt: skip
    record = run_agent(build_agent(AGENT_CFG), harness, setup["query"], tracer, RunStore(),
                       model=script.model())  # fmt: skip
    llm = [r for r in read_trace(tracer.path) if r["tool"] == "llm_call"]
    assert all(r["cost_usd"] is None and r["tokens_in"] == TOKENS[0] for r in llm)
    assert record.cost_usd_approx is None and record.tokens_in > 0


def test_a_run_needs_a_tuned_gate(setup: Any, make_gate: GateFactory, tmp_path: Path) -> None:
    config, _ = make_gate(tmp_path / "prov", setup["pool"], setup["fc"], status="provisional",
                          freeze=False)  # fmt: skip
    with pytest.raises(ValueError, match="tuned gate"):
        run(setup, GOOD, tmp_path / "run", gate=load_gate_config(config))


# The run-level decision
def gate_stage(passed: bool, decisions: tuple[str, ...] | None) -> GateStage:
    check = CheckResult(name="c", value=1.0, threshold=1.0, comparison=">=", passed=passed,
                        reason="why", source="fixed")  # fmt: skip
    result = GateResult(
        run_id="r", trajectory_id="t", strategy="knn", context_hash="h", checks=(check,),
        advisories=(), passed=passed, reasons=() if passed else ("c: why",),
        gate_config_sha256="0" * 64, gate_config_status="tuned", gate_code_version="0",
    )  # fmt: skip
    refit = CalibrationRefit(0.9, 10, 100, 5, 0, 0.0)
    return GateStage(result, refit, decisions, 0.5 if decisions else None, 0.0)  # type: ignore[arg-type]


def test_the_run_level_decision_table() -> None:
    assert run_level_decision(gate_stage(False, ("abstain",) * 3))[0] == "abstain"
    assert run_level_decision(gate_stage(True, ("alert", "no_alert", "alert")))[0] == "alert"
    assert run_level_decision(gate_stage(True, ("alert", "alert", "no_alert")))[0] == "no_alert"
    decision, reason = run_level_decision(gate_stage(True, None))
    assert decision == "abstain" and "no alert threshold" in reason
    # An alert in an early window does not count: the decision is about the end.
    assert run_level_decision(gate_stage(True, ("alert", "no_alert")))[0] == "no_alert"


# The configuration and the code rules
def test_the_prompt_states_the_enforced_limits_and_the_gate() -> None:
    text = system_prompt(AGENT_CFG.max_tool_calls, AGENT_CFG.max_contexts)
    assert "at most 3 contexts" in text and "at most 6 tool calls" in text
    assert "at most 80 words" in text
    assert "deterministic gate" in text and "do not argue" in text
    assert 'result is "abstain", whatever you recommend' in text


def test_models_and_prices_come_from_the_yaml(tmp_path: Path) -> None:
    cfg = load_agent_config()
    assert cfg.model == "openrouter:nvidia/nemotron-3.5-super-vl:free"
    assert cfg.model_for(debug=True) == "openrouter:nvidia/nemotron-3.5-lightning:free"
    assert cfg.model_for(debug=False) == cfg.model
    assert cfg.cost_usd(cfg.model, 1_000_000, 1_000_000) == 0.0  # a free model: known, zero
    assert cfg.cost_usd("openrouter:other/model:free", 10, 10) is None  # not in the table
    paid = dataclasses.replace(cfg, prices=(("some/paid-model", 2.0, 10.0),))
    assert paid.cost_usd("openrouter:some/paid-model", 1_000_000, 1_000_000) == pytest.approx(12.0)
    assert paid.cost_usd("some/paid-model", 1_000_000, 0) == pytest.approx(2.0)
    raw = yaml.safe_load((REPO_ROOT / "configs" / "agent.yaml").read_text(encoding="utf-8"))
    assert raw["prices"]["approximate"] is True
    for path in (REPO_ROOT / "src" / "pfngate" / "agent").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not MODEL_ID.search(node.value.lower()), (path.name, node.value[:40])


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        (None, "temperature", 0.7, "temperature must be 0"),
        ("limits", "max_tool_calls", 7, "max_tool_calls"),
        ("limits", "max_contexts", 4, "max_contexts"),
        ("limits", "max_tool_calls", 0, "max_tool_calls"),
        (None, "model", "nvidia/nemotron-3.5-super-vl:free", "provider:model"),  # no provider
        (None, "debug_model", "openrouter:", "provider:model"),
    ],
)
def test_a_loosened_agent_config_is_refused(
    tmp_path: Path, section: str | None, key: str, value: Any, message: str
) -> None:
    raw = yaml.safe_load((REPO_ROOT / "configs" / "agent.yaml").read_text(encoding="utf-8"))
    (raw[section] if section else raw)[key] = value
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_agent_config(path)


def test_prices_must_be_marked_approximate(tmp_path: Path) -> None:
    raw = yaml.safe_load((REPO_ROOT / "configs" / "agent.yaml").read_text(encoding="utf-8"))
    raw["prices"]["approximate"] = False
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="approximate"):
        load_agent_config(path)


def test_agent_code_cannot_load_write_or_freeze_the_gate() -> None:
    banned = {"load_gate_config", "write_frozen_hash", "write_gate_config", "build_gate_config",
              "assert_frozen", "assert_committed"}  # fmt: skip
    for path in (REPO_ROOT / "src" / "pfngate" / "agent").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                assert not {a.name for a in node.names} & banned, path.name
                assert node.module not in {"pfngate.gate.tune", "pfngate.gate.freeze"}, path.name
            if isinstance(node, ast.Name | ast.Attribute):
                name = node.id if isinstance(node, ast.Name) else node.attr
                assert name not in banned, (path.name, name)


def test_agent_code_never_reads_a_target() -> None:
    for path in (REPO_ROOT / "src" / "pfngate" / "agent").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in ("ground_truth", 'query["target"]', "query['target']", "coverage_90"):
            assert name not in text, (path.name, name)


@pytest.mark.parametrize(
    "model_string", [load_agent_config().model, load_agent_config().debug_model]
)
def test_the_real_openrouter_path_with_the_yaml_models_and_prices(
    setup: Any, tmp_path: Path, model_string: str
) -> None:
    cfg = load_agent_config()
    mock = OpenRouterMock([*GOOD[:1], *BUILD_AND_RUN, ("final", VERDICT)])
    tracer = Tracer("agent-test", trace_dir=tmp_path)
    harness = dataclasses.replace(make_harness(setup, neighbours(), tmp_path), agent_cfg=cfg)
    record = run_agent(build_agent(cfg), harness, setup["query"], tracer, RunStore(),
                       model=mock.model(model_string))  # fmt: skip
    tracer.close()

    assert record.model == model_string and record.stop_reason == "completed"
    assert record.verdict == Verdict(**VERDICT)  # parsed from the final_result tool call
    assert record.n_model_requests == 4 and record.n_tool_calls == 3
    # Temperature 0 is on the wire in every request, so it is applied.
    name = model_string.split(":", 1)[1]  # the id OpenRouter uses, which holds a ":" itself
    assert record.temperature_requested == 0.0 and record.temperature_applied is True
    for body in mock.bodies:
        assert body["model"] == name and body["temperature"] == 0.0
    assert set(mock.urls) == {"https://openrouter.ai/api/v1/chat/completions"}
    first = mock.bodies[0]
    offered = {t["function"]["name"] for t in first["tools"]}
    assert offered - {"final_result"} == {"describe_query", "describe_context_pool",
                                          "choose_context", "run_pfn"}  # fmt: skip
    answer = next(t for t in first["tools"] if t["function"]["name"] == "final_result")
    assert set(answer["function"]["parameters"]["properties"]) == set(Verdict.model_fields)
    # No raw data and no label reach the provider, in any request.
    sent = json.dumps(mock.bodies).lower()
    for leak in ("hopf", "2000", "family", "scenario", "target"):
        assert leak not in sent, leak
    assert "at most 6 tool calls" in sent and "deterministic gate" in sent
    # Tokens come from the response and the cost from configs/agent.yaml: free, so zero.
    llm = [r for r in read_trace(tracer.path) if r["tool"] == "llm_call"]
    assert all(r["backend"] == "openrouter" and r["model_version"] == name for r in llm)
    assert all((r["tokens_in"], r["tokens_out"]) == TOKENS for r in llm)
    assert all(r["cost_usd"] == 0.0 for r in llm) and record.cost_usd_approx == 0.0
    paid = dataclasses.replace(cfg, prices=((name, 2.0, 10.0),))
    again = run_agent(
        build_agent(paid), dataclasses.replace(harness, agent_cfg=paid), setup["query"],
        Tracer("agent-test", trace_dir=tmp_path / "paid"), RunStore(),
        model=OpenRouterMock([*GOOD[:1], *BUILD_AND_RUN, ("final", VERDICT)]).model(model_string),
    )  # fmt: skip
    assert again.cost_usd_approx == pytest.approx(4 * (TOKENS[0] * 2.0 + TOKENS[1] * 10.0) / 1e6)


def test_a_provider_error_is_not_turned_into_an_agent_decision(setup: Any, tmp_path: Path) -> None:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(401, json={"error": {"message": "No auth credentials found",
                                                    "code": 401}})  # fmt: skip

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(refuse))

    def provider(name: str) -> OpenRouterProvider:
        return OpenRouterProvider(api_key="dummy-key", http_client=client)

    model = infer_model(load_agent_config().model, provider_factory=provider)
    tracer = Tracer("agent-test", trace_dir=tmp_path)
    harness = make_harness(setup, neighbours(), tmp_path)
    store = RunStore()
    with pytest.raises(Exception, match="401|No auth credentials"):
        run_agent(build_agent(AGENT_CFG), harness, setup["query"], tracer, store, model=model)
    assert tracer.run_id not in store  # the run's data is dropped even on an error


def test_the_code_and_config_carry_no_anthropic_setup() -> None:
    files = [*(REPO_ROOT / "src").rglob("*.py"), *(REPO_ROOT / "configs").glob("*.yaml"),
             REPO_ROOT / "pyproject.toml"]  # fmt: skip
    for path in files:
        assert "anthropic" not in path.read_text(encoding="utf-8").lower(), path.name


def test_no_tool_has_an_empty_input_schema(setup: Any, tmp_path: Path) -> None:
    # A tool with no input at all made the free OpenRouter model send
    # {"additionalProperties": "{}"} on every call, so no run ever got a tool result.
    _, script, _ = run(setup, GOOD, tmp_path)
    assert set(script.schemas) == {
        "describe_query",
        "describe_context_pool",
        "choose_context",
        "run_pfn",
    }
    for name, schema in script.schemas.items():
        assert schema["properties"], name


def test_a_tool_accepts_an_optional_reason(setup: Any, tmp_path: Path) -> None:
    steps = [
        ("tool", "describe_query", {"reason": "see what kind of series this is"}),
        ("final", VERDICT),
    ]
    _, script, tracer = run(setup, steps, tmp_path)
    assert "agent_tool:describe_query" in tools_in(tracer)
    assert script.tool_returns()  # the call went through and returned a summary

"""Tests for the LLM-only baseline and the naive agent, with scripted fake models.

No network and no key. The PFN is the fake backend, and the baseline has no PFN at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from agent_support import (
    AGENT_CFG,
    AGENT_TOOL_STEPS,
    CFG,
    COST_PER_CALL,
    LLM_ONLY_FINAL,
    NAIVE_FINAL,
    NAIVE_TOOL_STEPS,
    TOKENS,
    VERDICT,
    GateFactory,
    StatelessScript,
    build_setup,
    make_harness,
    neighbours,
    shown_to_model,
)
from pfngate.agent.agent import build_agent, run_agent
from pfngate.agent.naive import NaiveVerdict, build_naive_agent, run_naive_agent
from pfngate.agent.prompts import naive_system_prompt
from pfngate.agent.store import RunStore
from pfngate.agent.summary import summarise_query
from pfngate.evals.baseline_llm import LLMOnlyAnswer, build_llm_only_agent, run_llm_only
from pfngate.evals.trace import TRACE_FIELDS, Tracer, read_trace
from pfngate.pfn import select_context

LEAKS = ("hopf", "2000", "family", "scenario", "trajectory", "target")


@pytest.fixture(scope="module")
def setup(small_data: dict[str, Any], make_gate: GateFactory, tmp_path_factory: Any) -> Any:
    return build_setup(small_data, make_gate, tmp_path_factory.mktemp("variants"))


def llm_only(setup: Any, tmp: Path, script: StatelessScript) -> tuple[Any, Tracer]:
    tracer = Tracer("llm-only-test", trace_dir=tmp)
    record = run_llm_only(build_llm_only_agent(), setup["query"], tracer, cfg=AGENT_CFG,
                          model=script.model())  # fmt: skip
    tracer.close()
    return record, tracer


def naive(
    setup: Any, tmp: Path, script: StatelessScript, backend: Any = None
) -> tuple[Any, Tracer]:
    tracer = Tracer("naive-test", trace_dir=tmp)
    harness = make_harness(setup, backend or neighbours(), tmp)
    record = run_naive_agent(build_naive_agent(AGENT_CFG), harness, setup["query"], tracer,
                             RunStore(), model=script.model())  # fmt: skip
    tracer.close()
    return record, tracer


# The LLM-only baseline
def test_the_baseline_reads_only_a_text_summary_and_has_no_tools(
    setup: Any, tmp_path: Path
) -> None:
    script = StatelessScript([], LLM_ONLY_FINAL)
    record, _ = llm_only(setup, tmp_path, script)
    assert script.offered_tools == [set()]  # no tools, so nothing to call
    shown = shown_to_model(script.seen[-1])
    summary = summarise_query(setup["query"])
    for name, s in summary["features"].items():
        assert f"{name}: mean {s['mean']:.4f}, std {s['std']:.4f}, trend {s['trend']:+.4f}" in shown
    assert f"The query has {summary['n_windows']} windows" in shown
    for leak in LEAKS:
        assert leak not in shown.lower(), leak
    assert record.stop_reason == "completed" and record.n_model_requests == 1
    assert record.decision == "alert" and record.answer == LLMOnlyAnswer(**LLM_ONLY_FINAL)
    assert record.defaulted is False and record.consistent is True
    assert record.n_tool_calls == 0 and record.model == f"function:{script.model().model_name}"


def test_the_baseline_has_no_abstain_and_a_probability_in_range() -> None:
    assert LLMOnlyAnswer(decision="no_alert", probability=0.0).probability == 0.0
    for change in ({"decision": "abstain"}, {"probability": 1.2}, {"probability": -0.1},
                   {"extra": 1}):  # fmt: skip
        with pytest.raises(ValidationError):
            LLMOnlyAnswer(**{**LLM_ONLY_FINAL, **change})


def test_a_baseline_decision_that_disagrees_with_its_probability_is_flagged(
    setup: Any, tmp_path: Path
) -> None:
    record, _ = llm_only(
        setup, tmp_path, StatelessScript([], {"decision": "alert", "probability": 0.2})
    )
    assert record.decision == "alert" and record.consistent is False  # kept, and flagged


def test_an_unusable_baseline_answer_counts_as_no_alert_and_says_so(
    setup: Any, tmp_path: Path
) -> None:
    record, _ = llm_only(
        setup, tmp_path, StatelessScript([], {"decision": "alert", "probability": 2.0})
    )
    assert record.stop_reason == "bad_output" and record.answer is None
    assert record.decision == "no_alert" and record.defaulted is True and record.consistent is None


def test_the_baseline_trace_has_the_same_fields_and_costs_as_the_agents(
    setup: Any, tmp_path: Path
) -> None:
    record, tracer = llm_only(setup, tmp_path / "b", StatelessScript([], LLM_ONLY_FINAL))
    rows = read_trace(tracer.path)
    assert [r["tool"] for r in rows] == ["llm_call"]
    assert tuple(rows[0]) == TRACE_FIELDS  # the same fields, in the same order
    assert (rows[0]["tokens_in"], rows[0]["tokens_out"]) == TOKENS
    assert rows[0]["cost_usd"] == pytest.approx(COST_PER_CALL)
    assert record.cost_usd_approx == pytest.approx(COST_PER_CALL)
    agent_trace = Tracer("agent-test", trace_dir=tmp_path / "a")
    harness = make_harness(setup, neighbours(), tmp_path / "a")
    script = StatelessScript(AGENT_TOOL_STEPS, VERDICT)
    run_agent(build_agent(AGENT_CFG), harness, setup["query"], agent_trace, RunStore(),
              model=script.model())  # fmt: skip
    agent_trace.close()
    assert {tuple(r) for r in read_trace(agent_trace.path)} == {TRACE_FIELDS}


# The naive agent
def test_the_naive_agent_has_a_random_context_no_gate_and_no_abstain(
    setup: Any, tmp_path: Path
) -> None:
    script = StatelessScript(NAIVE_TOOL_STEPS, NAIVE_FINAL)
    record, tracer = naive(setup, tmp_path, script)
    # It cannot choose a context or look at the pool, and cannot answer "abstain".
    assert script.offered_tools[0] == {"describe_query", "run_pfn"}
    decision_schema = script.answer_schemas[0][0]["properties"]["decision"]
    assert set(decision_schema["enum"]) == {"alert", "no_alert"}
    assert set(NaiveVerdict.model_fields) == {"decision", "rationale", "confidence"}
    # The context is random and was built by the harness at the largest size.
    sel = select_context("random", setup["pool"], setup["query"], CFG, max_context=200)
    assert record.context_strategy == "random" and record.context_hash == sel.context_hash
    assert record.context_size == sel.size
    # No gate ran, so the decision is the agent's own.
    tools = {r["tool"] for r in read_trace(tracer.path)}
    assert tools == {"llm_call", "agent_tool:describe_query", "agent_tool:run_pfn"}
    assert record.decision == "alert" == record.verdict.decision  # type: ignore[union-attr]
    assert record.defaulted is False and record.ran_pfn is True and record.n_tool_calls == 2
    assert "gate_result" not in record.model_dump()


def test_a_failed_gate_would_not_stop_the_naive_agent_but_stops_the_agent(
    setup: Any, tmp_path: Path
) -> None:
    overconfident = neighbours(scale=0.1)  # the gate fails on calibration
    gated_tracer = Tracer("agent-test", trace_dir=tmp_path / "g")
    gated = run_agent(
        build_agent(AGENT_CFG), make_harness(setup, overconfident, tmp_path / "g"), setup["query"],
        gated_tracer, RunStore(), model=StatelessScript(AGENT_TOOL_STEPS, VERDICT).model(),
    )  # fmt: skip
    naive_record, _ = naive(setup, tmp_path / "n", StatelessScript(NAIVE_TOOL_STEPS, NAIVE_FINAL),
                            overconfident)  # fmt: skip
    assert gated.verdict is not None and gated.verdict.decision == "alert"
    assert gated.final.gate_passed is False and gated.final.decision == "abstain"
    assert naive_record.decision == "alert"  # nothing stood in its way


def test_the_naive_agent_cannot_abstain_and_a_missing_answer_is_no_alert(
    setup: Any, tmp_path: Path
) -> None:
    with pytest.raises(ValidationError):
        NaiveVerdict(**{**NAIVE_FINAL, "decision": "abstain"})
    asking = StatelessScript(NAIVE_TOOL_STEPS, {**NAIVE_FINAL, "decision": "abstain"})
    record, _ = naive(setup, tmp_path, asking)
    assert record.stop_reason == "bad_output" and record.verdict is None
    assert record.decision == "no_alert" and record.defaulted is True


def test_the_naive_agent_keeps_the_tool_cap(setup: Any, tmp_path: Path) -> None:
    script = StatelessScript([("describe_query", {})] * 10, NAIVE_FINAL)
    record, tracer = naive(setup, tmp_path, script)
    assert record.stop_reason == "tool_cap" and record.n_tool_calls == 6
    assert [r["tool"] for r in read_trace(tracer.path)].count("agent_tool_refused") == 1
    assert record.decision == "no_alert" and record.defaulted is True


def test_the_naive_agent_never_sees_ids_or_labels(setup: Any, tmp_path: Path) -> None:
    script = StatelessScript(NAIVE_TOOL_STEPS, NAIVE_FINAL)
    naive(setup, tmp_path, script)
    shown = shown_to_model(script.seen[-1]).lower()
    for leak in LEAKS:
        assert leak not in shown, leak
    prompt = naive_system_prompt(AGENT_CFG.max_tool_calls)
    assert 'There is no "abstain"' in prompt and "Nothing runs after you" in prompt
    assert "gate" not in prompt.lower()  # the naive agent is not told about a gate

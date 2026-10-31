"""The naive agent: an ablation of the agent, to show what the choice of context, the gate and
the abstain option add.

It differs from the agent in three ways:
  - the context is random and is built by the harness before the agent starts, at the largest
    allowed size. The agent cannot choose it (it has no choose_context tool);
  - no gate runs after it: its verdict is the decision;
  - its verdict has no abstain option.

It keeps the agent's tools to describe the query and to run the PFN, the same cap of 6 tool
calls, and the same tracing. A run that gives no usable answer counts as "no_alert", and the
record says so (defaulted), because this variant has no way to decline.
"""

from __future__ import annotations

import json
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_ai import Agent, Tool
from pydantic_ai.models import Model

from pfngate.agent.agent import resolve_model
from pfngate.agent.calls import tracing_hooks
from pfngate.agent.config import AgentConfig, load_agent_config
from pfngate.agent.prompts import USER_PROMPT, naive_system_prompt
from pfngate.agent.runner import LLMOutcome, StopReason, run_llm
from pfngate.agent.store import (
    AgentDeps,
    AgentHarness,
    RunBudget,
    RunData,
    RunStore,
    TriedContext,
)
from pfngate.agent.tools import describe_query, run_pfn
from pfngate.agent.verdict import MAX_RATIONALE_WORDS
from pfngate.evals.trace import Tracer
from pfngate.pfn.context import select_context

NAIVE_TOOLS = tuple(Tool(f, sequential=True) for f in (describe_query, run_pfn))  # see tools.TOOLS


class NaiveVerdict(BaseModel):
    """The answer of the naive agent. It is the decision: nothing checks it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision: Literal["alert", "no_alert"] = Field(
        description="Your decision for the end of the series."
    )
    rationale: str = Field(description=f"At most {MAX_RATIONALE_WORDS} words.")
    confidence: float = Field(ge=0.0, le=1.0, description="Your confidence, between 0 and 1.")

    @field_validator("rationale")
    @classmethod
    def _short_rationale(cls, v: str) -> str:
        n = len(v.split())
        if n > MAX_RATIONALE_WORDS:
            raise ValueError(f"rationale has {n} words, the limit is {MAX_RATIONALE_WORDS}")
        return v


class NaiveRunRecord(BaseModel):
    """One run of the naive agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    trajectory_id: str
    model: str
    temperature_requested: float
    temperature_applied: bool
    stop_reason: StopReason
    stop_detail: str | None
    n_tool_calls: int
    n_model_requests: int
    n_replayed_requests: int
    llm_seconds: float
    tokens_in: int
    tokens_out: int
    cost_usd_approx: float | None
    verdict: NaiveVerdict | None
    decision: Literal["alert", "no_alert"]  # the verdict, or no_alert if there was none
    defaulted: bool  # True: no usable answer, so the decision is the default no_alert
    context_strategy: Literal["random"] = "random"
    context_size: int
    context_hash: str
    ran_pfn: bool  # did the agent look at the PFN's predictions at all?


def build_naive_agent(cfg: AgentConfig | None = None) -> Agent[AgentDeps, NaiveVerdict]:
    cfg = cfg or load_agent_config()
    return Agent(
        None,
        output_type=NaiveVerdict,
        deps_type=AgentDeps,
        system_prompt=naive_system_prompt(cfg.max_tool_calls),
        tools=NAIVE_TOOLS,
        capabilities=[tracing_hooks()],
        defer_model_check=True,
        name="pfngate-naive-agent",
    )


def run_naive_agent(
    agent: Agent[AgentDeps, NaiveVerdict],
    harness: AgentHarness,
    query: pd.DataFrame,
    tracer: Tracer,
    store: RunStore,
    *,
    model: Model | str | None = None,
    run_id: str | None = None,
    write_sidecar: bool = True,
) -> NaiveRunRecord:
    """Run the naive agent on one query trajectory. No gate. One tracer per run, unless
    run ids are given."""
    cfg = harness.agent_cfg
    resolved = resolve_model(cfg.model if model is None else model)
    run_id = run_id or tracer.run_id
    data = RunData.create(run_id, query, harness)
    store.put(data)
    sel = select_context(
        "random", harness.pool, data.query, harness.pfn_cfg, max_context=harness.max_context
    )
    data.tried.append(TriedContext(sel, "random", harness.max_context))
    data.current = data.tried[-1]
    deps = AgentDeps(
        run_id=run_id,
        scenario_id=data.scenario_id,
        tracer=tracer,
        agent_cfg=cfg,
        model_system=resolved.system,
        model_name=resolved.model_name,
        store=store,
        budget=RunBudget(cfg.max_tool_calls, cfg.max_contexts),
    )
    try:
        outcome: LLMOutcome[NaiveVerdict] = run_llm(
            agent, USER_PROMPT, model=resolved, deps=deps, cfg=cfg
        )
        ran_pfn = sel.context_hash in data.outputs
    finally:
        store.drop(run_id)
    verdict = outcome.output
    record = NaiveRunRecord(
        run_id=run_id,
        trajectory_id=data.scenario_id,
        model=f"{deps.model_system}:{deps.model_name}",
        temperature_requested=cfg.temperature,
        temperature_applied=outcome.temperature_applied,
        stop_reason=outcome.stop,
        stop_detail=outcome.detail,
        n_tool_calls=deps.budget.tool_calls,
        n_model_requests=deps.model_requests,
        n_replayed_requests=deps.replayed_requests,
        llm_seconds=deps.llm_seconds,
        tokens_in=deps.tokens_in,
        tokens_out=deps.tokens_out,
        cost_usd_approx=deps.cost_usd,
        verdict=verdict,
        decision="no_alert" if verdict is None else verdict.decision,
        defaulted=verdict is None,
        context_size=sel.size,
        context_hash=sel.context_hash,
        ran_pfn=ran_pfn,
    )
    if write_sidecar:
        path = tracer.path.with_name(f"{run_id}.naive.json")
        text = json.dumps(record.model_dump(mode="json"), allow_nan=False, indent=2)
        path.write_text(text + "\n", encoding="utf-8")
    return record

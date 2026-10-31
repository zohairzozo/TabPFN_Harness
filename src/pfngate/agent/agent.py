"""The LLM agent: it chooses how to build the context and recommends a decision.

Flow of one run (run_agent):
  1. The query goes into a server-side store under the run id. The model gets tools that
     return summaries, never rows.
  2. The agent calls at most 6 tools (enforced in code) and returns a Verdict.
  3. The gate runs on the final context, in code, and the decision comes from the gate and
     the decision mapping. The Verdict is kept next to it. See pfngate.agent.final.

The agent cannot change a GateResult or a threshold: it holds no reference to either. A run
that hits a cap or returns an unusable answer is stopped, recorded, and still gated.
Network and provider errors are not caught: they must not look like an agent decision.

PydanticAI is 2.x here. Model strings are "provider:model" (pydantic_ai.models.infer_model).
"""

from __future__ import annotations

import json

import pandas as pd
import pydantic_ai
from pydantic_ai import Agent
from pydantic_ai.models import Model

from pfngate.agent.calls import tracing_hooks
from pfngate.agent.config import AgentConfig, load_agent_config
from pfngate.agent.final import AgentRunRecord, FinalOutcome, finalize
from pfngate.agent.prompts import USER_PROMPT, system_prompt
from pfngate.agent.replay import make_model
from pfngate.agent.runner import LLMOutcome, run_llm
from pfngate.agent.store import (
    AgentDeps,
    AgentHarness,
    RunBudget,
    RunData,
    RunStore,
)
from pfngate.agent.tools import TOOLS
from pfngate.agent.verdict import Verdict
from pfngate.evals.trace import Tracer
from pfngate.gate import GateResult

pydantic_ai.BANNER_ENABLED = False  # the start-up banner goes to stdout


def build_agent(cfg: AgentConfig | None = None) -> Agent[AgentDeps, Verdict]:
    """The PydanticAI agent. The model is given at run time, so building it needs no key."""
    cfg = cfg or load_agent_config()
    return Agent(
        None,
        output_type=Verdict,
        deps_type=AgentDeps,
        system_prompt=system_prompt(cfg.max_tool_calls, cfg.max_contexts),
        tools=TOOLS,
        capabilities=[tracing_hooks()],
        defer_model_check=True,
        name="pfngate-context-agent",
    )


def resolve_model(model: Model | str) -> Model:
    """A model object as it is, or a "provider:model" string turned into a live model.

    Keys come from the environment or from .env. They are never printed. To record or replay,
    build the model with pfngate.agent.replay.make_model and pass it in.
    """
    return make_model(model, mode="live") if isinstance(model, str) else model


def run_agent(
    agent: Agent[AgentDeps, Verdict],
    harness: AgentHarness,
    query: pd.DataFrame,
    tracer: Tracer,
    store: RunStore,
    *,
    model: Model | str | None = None,
    run_id: str | None = None,
    write_sidecar: bool = True,
) -> AgentRunRecord:
    """Run the agent on one query trajectory, then the gate. One tracer per run.

    `query` may carry any columns: only the trajectory id, the window position and the
    features enter the store. The gate must be tuned, and for held-out data the caller must
    have checked it with assert_frozen, as run_pfn does. The record is returned and also
    written next to the trace as <run_id>.agent.json.
    """
    cfg = harness.agent_cfg
    if harness.gate_cfg.status != "tuned":
        raise ValueError("agent runs need a tuned gate (configs/gate.yaml with status tuned)")
    resolved = resolve_model(cfg.model if model is None else model)
    run_id = run_id or tracer.run_id  # one tracer can serve many runs, with their own ids
    data = RunData.create(run_id, query, harness)
    store.put(data)
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
        outcome = run_llm(agent, USER_PROMPT, model=resolved, deps=deps, cfg=cfg)
        verdict = outcome.output
        final, gate_result = finalize(data, deps, verdict)
    finally:
        store.drop(run_id)

    record = _record(deps, data, verdict, final, gate_result, outcome)
    if write_sidecar:
        path = tracer.path.with_name(f"{run_id}.agent.json")
        text = json.dumps(record.model_dump(mode="json"), allow_nan=False, indent=2)
        path.write_text(text + "\n", encoding="utf-8")
    return record


def _record(
    deps: AgentDeps,
    data: RunData,
    verdict: Verdict | None,
    final: FinalOutcome,
    gate_result: GateResult | None,
    outcome: LLMOutcome[Verdict],
) -> AgentRunRecord:
    return AgentRunRecord(
        run_id=deps.run_id,
        trajectory_id=data.scenario_id,
        model=f"{deps.model_system}:{deps.model_name}",
        temperature_requested=data.harness.agent_cfg.temperature,
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
        final=final,
        agent_agrees=None if verdict is None else verdict.decision == final.decision,
        gate_result=None if gate_result is None else gate_result.model_dump(mode="json"),
    )

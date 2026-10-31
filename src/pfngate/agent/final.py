"""After the agent: the gate runs on the final context, and the gate decides.

The agent's verdict is a recommendation. Here the same gate code as run_pfn --gate runs on the
final context and its predictions, and the decision comes from the gate and the decision
mapping (pfngate.gate.decide). The agent's own recommendation is stored next to it, never
instead of it. Nothing in this module reads a target or a label of the query.

Run-level decision. The gate maps each window to abstain, alert or no_alert. A run needs one
answer, so it is the answer for the end of the series: abstain if the gate failed (every
window abstains), otherwise the decision of the last window.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from pfngate.agent.runner import StopReason
from pfngate.agent.store import AgentDeps, RunData, TriedContext
from pfngate.agent.verdict import Verdict
from pfngate.evals.run_pfn import GateStage, gate_and_trace
from pfngate.gate import Decision, GateResult


class FinalOutcome(BaseModel):
    """The decision that counts, and how it was reached."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision: Decision
    decided_by: Literal["gate_and_decision_mapping"] = "gate_and_decision_mapping"
    reason: str
    gate_passed: bool | None  # None: no context, so no gate run
    gate_failed: tuple[str, ...]
    alert_threshold: float | None
    window_decisions: dict[str, int] | None
    last_window_decision: Decision | None
    last_window_p_critical: float | None  # the PFN's p_critical at the last window
    context_strategy: str | None
    context_size: int | None
    context_hash: str | None
    # None: no verdict. False: the verdict named no context that was built.
    followed_verdict_context: bool | None


class AgentRunRecord(BaseModel):
    """One run: the agent's recommendation next to the final decision. Written as JSON."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    trajectory_id: str
    model: str
    temperature_requested: float
    # False when the provider layer dropped the setting (PydanticAI warns when it does).
    # Then temperature 0 was asked for but not applied.
    temperature_applied: bool
    stop_reason: StopReason
    stop_detail: str | None
    n_tool_calls: int
    n_model_requests: int
    n_replayed_requests: int  # model calls served from the replay cache
    llm_seconds: float  # latency of the model calls, as recorded for a replay
    tokens_in: int
    tokens_out: int
    cost_usd_approx: float | None  # from the approximate price table in configs/agent.yaml
    verdict: Verdict | None  # the agent's recommendation
    final: FinalOutcome  # the gate's decision
    agent_agrees: bool | None
    gate_result: dict[str, Any] | None


def pick_context(data: RunData, verdict: Verdict | None) -> tuple[TriedContext | None, bool | None]:
    """The context to gate: the one the verdict names among those built, else the last built."""
    if verdict is not None:
        for t in reversed(data.tried):
            if t.strategy == verdict.context_strategy and verdict.context_size in (
                t.size,
                t.selection.size,
            ):
                return t, True
    return data.current, (False if verdict is not None and data.current is not None else None)


def run_level_decision(stage: GateStage) -> tuple[Decision, str]:
    """abstain if the gate failed, else the decision of the last window. See the module doc."""
    if not stage.result.passed:
        return "abstain", "the gate failed: " + "; ".join(stage.result.reasons)
    if stage.decisions is None:
        return "abstain", "the gate passed, but this strategy has no alert threshold"
    last = stage.decisions[-1]
    return last, f"the gate passed, so the decision is that of the last window ({last})"


def finalize(
    data: RunData, deps: AgentDeps, verdict: Verdict | None
) -> tuple[FinalOutcome, GateResult | None]:
    """Run the gate on the final context and take the decision from it."""
    tried, followed = pick_context(data, verdict)
    if tried is None:
        outcome = FinalOutcome(
            decision="abstain",
            reason="the agent built no context, so there is nothing to gate",
            gate_passed=None,
            gate_failed=(),
            alert_threshold=None,
            window_decisions=None,
            last_window_decision=None,
            last_window_p_critical=None,
            context_strategy=None,
            context_size=None,
            context_hash=None,
            followed_verdict_context=followed,
        )
        return outcome, None

    h, sel = data.harness, tried.selection
    out = data.outputs.get(sel.context_hash)
    if out is None:  # the agent never ran this context: the harness does, through the cache
        t0 = time.perf_counter()
        out, hit = h.fit_predict(sel.context, data.query)
        data.outputs[sel.context_hash] = out
        deps.log(
            tool="pfn_cache_hit" if hit else "pfn_predict",
            latency_s=time.perf_counter() - t0,
            context_hash=sel.context_hash,
            context_size=sel.size,
            strategy=sel.label,
            backend=out.metadata.backend,
            model_version=out.metadata.model_version,
        )
    stage = gate_and_trace(
        h.gate_cfg, h.fit_predict, h.pool_distance, deps.tracer, sel, data.query, out, h.pfn_cfg
    )
    decision, reason = run_level_decision(stage)
    counts = (
        None
        if stage.decisions is None
        else {d: stage.decisions.count(d) for d in ("abstain", "alert", "no_alert")}
    )
    outcome = FinalOutcome(
        decision=decision,
        reason=reason,
        gate_passed=stage.result.passed,
        gate_failed=stage.result.failed,
        alert_threshold=stage.alert_threshold,
        window_decisions=counts,
        last_window_decision=None if stage.decisions is None else stage.decisions[-1],
        last_window_p_critical=float(out.p_critical[-1]),
        context_strategy=sel.strategy,
        context_size=sel.size,
        context_hash=sel.context_hash,
        followed_verdict_context=followed,
    )
    return outcome, stage.result

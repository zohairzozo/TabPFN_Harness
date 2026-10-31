"""What every LLM run shares: its deps and the hook that traces each model call.

The agent, the naive agent and the LLM-only baseline all use LLMDeps and tracing_hooks(), so
their traces have the same fields. One trace step is written per model call, with the tokens
and an approximate cost in USD from the price table in configs/agent.yaml. A call that came
from the replay cache is written as "llm_replay" instead of "llm_call". It keeps the tokens,
the cost and the latency of the call as they were recorded, so the cost and the latency of
a run are known offline too.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import ModelRequestContext

from pfngate.agent.config import AgentConfig
from pfngate.evals.trace import Tracer


@dataclass(kw_only=True)
class LLMDeps:
    """What PydanticAI hands to every tool and hook. It holds ids and counters, no data."""

    run_id: str
    scenario_id: str  # the trajectory id. It goes to the local trace, never to the model.
    tracer: Tracer
    agent_cfg: AgentConfig
    model_system: str  # provider, for the trace
    model_name: str
    lock: threading.RLock = field(default_factory=threading.RLock)
    # Totals over the model calls of the run. The cost is None once one call had no price.
    model_requests: int = 0
    replayed_requests: int = 0
    llm_seconds: float = 0.0  # latency of the model calls, as recorded for a replay
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float | None = 0.0

    def log(self, **fields: Any) -> None:
        """Append one trace step for this run. Serialised, because tools can run in threads."""
        with self.lock:
            self.tracer.log_step(scenario_id=self.scenario_id, **fields)


def tracing_hooks() -> Hooks[LLMDeps]:
    """One trace step per model call, with tokens and an approximate cost in USD."""
    hooks: Hooks[LLMDeps] = Hooks()

    @hooks.on.model_request
    async def trace_model_call(
        ctx: RunContext[LLMDeps],
        request_context: ModelRequestContext,
        handler: Callable[[ModelRequestContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        deps, model = ctx.deps, request_context.model
        replays_before = getattr(model, "replayed_calls", 0)
        t0 = time.perf_counter()
        response = await handler(request_context)
        replayed = getattr(model, "replayed_calls", 0) > replays_before
        recorded = getattr(model, "last_latency_s", None)  # None if the model does not record
        seconds = recorded if recorded is not None else time.perf_counter() - t0
        tokens_in, tokens_out = response.usage.input_tokens, response.usage.output_tokens
        cost = deps.agent_cfg.cost_usd(deps.model_name, tokens_in, tokens_out)
        with deps.lock:
            deps.model_requests += 1
            deps.replayed_requests += int(replayed)
            deps.llm_seconds += seconds
            deps.tokens_in += tokens_in
            deps.tokens_out += tokens_out
            deps.cost_usd = None if cost is None or deps.cost_usd is None else deps.cost_usd + cost
        deps.log(
            tool="llm_replay" if replayed else "llm_call",
            latency_s=seconds,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            backend=deps.model_system,
            model_version=deps.model_name,
        )
        return response

    return hooks

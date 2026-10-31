"""Run one PydanticAI agent and say how it stopped. Shared by every LLM variant.

The agent, the naive agent and the LLM-only baseline all call run_llm(). It requests
temperature 0, checks whether the provider dropped it (PydanticAI warns when it does, and
the replay cache repeats that warning), and turns the three ways an agent can fail into a
stop reason. Anything else, such as a network or provider error, is not caught: it must not
look like an answer of the model.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from pfngate.agent.calls import LLMDeps
from pfngate.agent.config import AgentConfig
from pfngate.agent.replay import is_sampling_warning
from pfngate.agent.store import ToolCapExceeded

StopReason = Literal["completed", "tool_cap", "request_cap", "bad_output"]

T = TypeVar("T")
D = TypeVar("D", bound=LLMDeps)


@dataclass(frozen=True)
class LLMOutcome(Generic[T]):  # noqa: UP046  (this project supports Python 3.11)
    output: T | None  # None if the run stopped before a usable answer
    stop: StopReason
    detail: str | None
    temperature_applied: bool  # False if the provider dropped the temperature setting


def run_llm(
    agent: Agent[D, T], prompt: str, *, model: Model, deps: D, cfg: AgentConfig
) -> LLMOutcome[T]:
    output: T | None = None
    stop: StopReason = "completed"
    detail: str | None = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = agent.run_sync(
                prompt,
                model=model,
                deps=deps,
                usage_limits=UsageLimits(request_limit=cfg.max_model_requests),
                model_settings={"temperature": cfg.temperature},
            )
            output = result.output
        except ToolCapExceeded as exc:
            stop, detail = "tool_cap", str(exc)
        except UsageLimitExceeded as exc:
            stop, detail = "request_cap", str(exc)
        except UnexpectedModelBehavior as exc:
            stop, detail = "bad_output", str(exc)
    applied = not any(is_sampling_warning(w.message) for w in caught)
    for w in caught:  # every other warning is shown as usual
        if not is_sampling_warning(w.message):
            warnings.warn(w.message, w.category, stacklevel=2)
    return LLMOutcome(output, stop, detail, applied)

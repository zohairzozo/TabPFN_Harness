"""The typed answer of the LLM agent. It is a recommendation, never the final decision."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_RATIONALE_WORDS = 80
# The strategies an agent may pick. family_matched is an oracle and is not an option.
AgentStrategy = Literal["random", "knn"]
AGENT_STRATEGIES: tuple[str, ...] = ("random", "knn")


class Verdict(BaseModel):
    """What the agent recommends. The gate and the decision mapping take the final decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision: Literal["alert", "no_alert", "abstain"] = Field(
        description="Your recommendation for the end of the series."
    )
    context_strategy: AgentStrategy = Field(
        description="The strategy of the context you want the gate to check."
    )
    context_size: int = Field(
        ge=1, description="The size of the context you want the gate to check."
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

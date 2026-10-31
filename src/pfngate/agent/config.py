"""Settings of the LLM agent, read from configs/agent.yaml.

The model names and the price table live in the YAML, not in code. Loading refuses a
temperature other than 0 (CLAUDE.md) and limits above the ceilings written here, so a
loosened YAML fails before a run starts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from pfngate.features.config import REPO_ROOT

DEFAULT_AGENT_CONFIG = REPO_ROOT / "configs" / "agent.yaml"

# Ceilings that the YAML may lower but not raise. A test pins them.
HARD_MAX_TOOL_CALLS = 6
HARD_MAX_CONTEXTS = 3


@dataclass(frozen=True)
class AgentConfig:
    model: str
    debug_model: str
    temperature: float
    max_tool_calls: int
    max_contexts: int
    max_model_requests: int
    # (model name, input price, output price) in USD per million tokens. Approximate.
    prices: tuple[tuple[str, float, float], ...]
    source: Path

    def model_for(self, debug: bool) -> str:
        return self.debug_model if debug else self.model

    def price_for(self, model_name: str) -> tuple[float, float] | None:
        """Prices for a "provider:model" string or a bare model name, or None."""
        table = {name: (p_in, p_out) for name, p_in, p_out in self.prices}
        if model_name in table:
            return table[model_name]
        return table.get(model_name.split(":", 1)[-1])

    def cost_usd(self, model_name: str, tokens_in: int, tokens_out: int) -> float | None:
        """Estimated cost of one call, or None when the model is not in the price table."""
        price = self.price_for(model_name)
        if price is None:
            return None
        return (tokens_in * price[0] + tokens_out * price[1]) / 1_000_000


def load_agent_config(path: Path = DEFAULT_AGENT_CONFIG) -> AgentConfig:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw["prices"].get("approximate") is not True:
        raise ValueError("prices must be marked approximate: true")
    cfg = AgentConfig(
        model=str(raw["model"]),
        debug_model=str(raw["debug_model"]),
        temperature=float(raw["temperature"]),
        max_tool_calls=int(raw["limits"]["max_tool_calls"]),
        max_contexts=int(raw["limits"]["max_contexts"]),
        max_model_requests=int(raw["limits"]["max_model_requests"]),
        prices=tuple(
            (str(name), float(p["input"]), float(p["output"]))
            for name, p in sorted(raw["prices"]["models"].items())
        ),
        source=path,
    )
    problems = []
    if cfg.temperature != 0.0:
        problems.append("temperature must be 0")
    if not 1 <= cfg.max_tool_calls <= HARD_MAX_TOOL_CALLS:
        problems.append(f"max_tool_calls must be in 1..{HARD_MAX_TOOL_CALLS}")
    if not 1 <= cfg.max_contexts <= HARD_MAX_CONTEXTS:
        problems.append(f"max_contexts must be in 1..{HARD_MAX_CONTEXTS}")
    if cfg.max_model_requests < 2:
        problems.append("max_model_requests must be at least 2")
    for ref in (cfg.model, cfg.debug_model):
        provider, _, name = ref.partition(":")  # OpenRouter ids hold ":" and "/" after it
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", provider) or not name:
            problems.append(f"model '{ref}' must be 'provider:model'")
    if any(p_in < 0 or p_out < 0 for _, p_in, p_out in cfg.prices):
        problems.append("prices must not be negative")
    if problems:
        raise ValueError("agent config refused: " + "; ".join(problems))
    return cfg

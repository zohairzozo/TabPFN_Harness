"""Helpers shared by the agent tests: a scripted model, a mock OpenRouter endpoint, and a harness.

The scripted model is a pydantic_ai FunctionModel that replays a list of steps (a tool call or
the final answer). The OpenRouter mock answers in the chat-completions format that OpenRouter
uses, so a real OpenRouterModel can run against it with no network and no key.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import infer_model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.usage import RequestUsage

from pfngate.agent.agent import build_agent, run_agent
from pfngate.agent.config import load_agent_config
from pfngate.agent.final import AgentRunRecord
from pfngate.agent.store import AgentHarness, RunStore
from pfngate.evals.gating import FitPredict
from pfngate.evals.pfn_cache import PFNCache
from pfngate.evals.trace import Tracer, read_trace
from pfngate.gate import load_gate_config
from pfngate.gate.reference import PoolDistance
from pfngate.pfn import FakeBackend, load_pfn_config

CFG = load_pfn_config()
QUERY = "hopf_L2_fast/2000"  # a validation trajectory the fake backend handles well
MODEL_NAME = "scripted-1"
TOKENS = (100, 20)  # tokens in and out of every scripted model call
AGENT_CFG = dataclasses.replace(load_agent_config(), prices=((MODEL_NAME, 1.0, 2.0),))
COST_PER_CALL = (TOKENS[0] * 1.0 + TOKENS[1] * 2.0) / 1_000_000
GateFactory = Callable[..., tuple[Path, Path]]
MODEL_ID = re.compile(r"nvidia/[a-z0-9]|(anthropic|openai|openrouter):[a-z0-9]")  # a model id

VERDICT = {
    "decision": "alert",
    "context_strategy": "knn",
    "context_size": 200,
    "rationale": "The series looks close to tipping.",
    "confidence": 0.7,
}
BUILD_AND_RUN = [
    ("tool", "choose_context", {"strategy": "knn", "size": 200}),
    ("tool", "run_pfn", {}),
]
GOOD = [("tool", "describe_query", {}), ("tool", "describe_context_pool", {}), *BUILD_AND_RUN,
        ("final", VERDICT)]  # fmt: skip


class Script:
    """A scripted model. The last step repeats when the script runs out."""

    def __init__(self, steps: list[tuple[Any, ...]]) -> None:
        self.steps = steps
        self.calls = 0
        self.seen: list[list[Any]] = []  # the messages the model was shown, per call
        self.schemas: dict[str, dict[str, Any]] = {}  # tool name -> JSON schema of its input

    def respond(self, messages: list[Any], info: AgentInfo) -> ModelResponse:
        self.seen.append(list(messages))
        self.schemas = {t.name: t.parameters_json_schema for t in info.function_tools}
        step = self.steps[min(self.calls, len(self.steps) - 1)]
        self.calls += 1
        if step[0] == "tool":
            part = ToolCallPart(step[1], step[2])
        else:
            part = ToolCallPart(info.output_tools[0].name, step[1])
        usage = RequestUsage(input_tokens=TOKENS[0], output_tokens=TOKENS[1])
        return ModelResponse(parts=[part], usage=usage)

    def model(self) -> FunctionModel:
        return FunctionModel(self.respond, model_name=MODEL_NAME)

    def tool_returns(self) -> list[Any]:
        """Everything the tools returned to the model, in order."""
        return [
            p.content
            for m in self.seen[-1]
            for p in getattr(m, "parts", [])
            if isinstance(p, ToolReturnPart)
        ]


def build_setup(small_data: dict[str, Any], make_gate: GateFactory, tmp: Path) -> Any:
    """The gate, the pool distance and one validation query, for one test module."""
    alert = {"random": 0.5, "knn": 0.5}
    config, _ = make_gate(tmp / "frozen", small_data["pool"], small_data["fc"], alert=alert)
    gate = load_gate_config(config)
    ref = gate.checks.ood_share
    pool_distance = PoolDistance(small_data["pool"], ref.feature_mean, ref.feature_std, ref.k)
    val = small_data["val"]
    query = val[val["trajectory_id"] == QUERY].reset_index(drop=True)
    assert len(query) > 0 and "target" in query.columns and "family" in query.columns
    return {**small_data, "gate": gate, "pool_distance": pool_distance, "query": query}


def make_harness(setup: Any, backend: FakeBackend, tmp: Path, gate: Any = None) -> AgentHarness:
    fit_predict = FitPredict(backend, PFNCache(tmp / "cache", enabled=False), CFG,
                             backend.model_version, "test")  # fmt: skip
    return AgentHarness(
        pool=setup["pool"],
        gate_cfg=gate or setup["gate"],
        pool_distance=setup["pool_distance"],
        fit_predict=fit_predict,
        pfn_cfg=CFG,
        agent_cfg=AGENT_CFG,
        max_context=200,
    )


def neighbours(scale: float = 1.0) -> FakeBackend:
    return FakeBackend(CFG, mode="neighbours", interval_scale=scale)


def run(
    setup: Any,
    steps: list[tuple[Any, ...]],
    tmp: Path,
    *,
    backend: FakeBackend | None = None,
    gate: Any = None,
) -> tuple[AgentRunRecord, Script, Tracer]:
    script = Script(steps)
    tracer = Tracer("agent-test", trace_dir=tmp)
    harness = make_harness(setup, backend or neighbours(), tmp, gate)
    record = run_agent(build_agent(AGENT_CFG), harness, setup["query"], tracer, RunStore(),
                       model=script.model())  # fmt: skip
    tracer.close()
    return record, script, tracer


def tools_in(tracer: Tracer) -> list[str]:
    return [r["tool"] for r in read_trace(tracer.path)]


# The real OpenRouter path, against a mock transport. No network and no key: the transport
# answers in the chat-completions format OpenRouter uses, so the model string, the request, the
# parsing of tool calls and of the structured answer, the token counts and the price table are
# all real.
class OpenRouterMock:
    """A mock OpenRouter endpoint. A step is a tool call, or the final answer, which a
    tool-calling model gives by calling the final_result tool."""

    def __init__(self, steps: list[tuple[Any, ...]]) -> None:
        self.steps = steps
        self.bodies: list[dict[str, Any]] = []
        self.urls: list[str] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        self.urls.append(str(request.url))
        n = len(self.bodies)
        step = self.steps[min(n - 1, len(self.steps) - 1)]
        name, args = (step[1], step[2]) if step[0] == "tool" else ("final_result", step[1])
        call = {"id": f"call_{n}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}}  # fmt: skip
        message = {"role": "assistant", "content": None, "tool_calls": [call]}
        return httpx2.Response(200, json={
            "id": f"gen-{n}", "provider": "MockProvider", "object": "chat.completion",
            "created": 1, "model": body["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": TOKENS[0], "completion_tokens": TOKENS[1],
                      "total_tokens": sum(TOKENS)},
        })  # fmt: skip

    def model(self, model_string: str) -> Any:
        client = httpx2.AsyncClient(transport=httpx2.MockTransport(self.handler))

        def provider(name: str) -> OpenRouterProvider:
            return OpenRouterProvider(api_key="dummy-key", http_client=client)

        return infer_model(model_string, provider_factory=provider)


class StatelessScript:
    """A scripted model for many runs: its next step depends on how many tool results it has
    seen in the conversation, not on how many calls it has made. It also records what it was
    offered (the tools and the answer schema) and what it was shown."""

    def __init__(self, tool_steps: list[tuple[str, dict[str, Any]]], final: dict[str, Any]) -> None:
        self.tool_steps, self.final = tool_steps, final
        self.calls = 0
        self.offered_tools: list[set[str]] = []
        self.answer_schemas: list[list[dict[str, Any]]] = []
        self.seen: list[list[Any]] = []

    def respond(self, messages: list[Any], info: AgentInfo) -> ModelResponse:
        self.calls += 1
        self.seen.append(list(messages))
        self.offered_tools.append({t.name for t in info.function_tools})
        self.answer_schemas.append([t.parameters_json_schema for t in info.output_tools])
        n = sum(isinstance(p, ToolReturnPart) for m in messages for p in getattr(m, "parts", []))
        if n < len(self.tool_steps):
            part = ToolCallPart(*self.tool_steps[n])
        else:
            part = ToolCallPart(info.output_tools[0].name, self.final)
        usage = RequestUsage(input_tokens=TOKENS[0], output_tokens=TOKENS[1])
        return ModelResponse(parts=[part], usage=usage)

    def model(self) -> FunctionModel:
        return FunctionModel(self.respond, model_name=MODEL_NAME)


AGENT_TOOL_STEPS = [
    ("describe_query", {}),
    ("describe_context_pool", {}),
    ("choose_context", {"strategy": "knn", "size": 200}),
    ("run_pfn", {}),
]
NAIVE_TOOL_STEPS = [("describe_query", {}), ("run_pfn", {})]
NAIVE_FINAL = {"decision": "alert", "confidence": 0.8, "rationale": "The features rise."}
LLM_ONLY_FINAL = {"decision": "alert", "probability": 0.9}


def shown_to_model(messages: list[Any]) -> str:
    """What the model was shown, without the random parts.

    The messages carry timestamps and random ids. A leak check that looks for a number such as
    "2000" must not match those by chance.
    """
    text = ModelMessagesTypeAdapter.dump_json(messages).decode()
    return re.sub(r'"(timestamp|tool_call_id|run_id|id)":"[^"]*"', "", text)

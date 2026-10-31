"""Tests for the record/replay cache of LLM responses (pfngate.agent.replay).

Two layers. A small PydanticAI agent with a scripted model tests the cache itself. The real
agent, tools and gate, run against a mock OpenRouter endpoint, test a whole run that is
recorded and then replayed with the network locked.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import socket
import time
import warnings
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agent_support import (
    BUILD_AND_RUN,
    GOOD,
    VERDICT,
    GateFactory,
    OpenRouterMock,
    build_setup,
    make_harness,
    neighbours,
)
from pfngate.agent.agent import build_agent, run_agent
from pfngate.agent.config import load_agent_config
from pfngate.agent.replay import (
    DEFAULT_LLM_CACHE,
    LLMStore,
    NetworkBlockedError,
    RecordReplayModel,
    ReplayMissError,
    add_llm_arguments,
    default_llm_mode,
    is_sampling_warning,
    make_model,
    offline_model,
    request_key,
    resolve_llm_mode,
)
from pfngate.agent.store import RunStore
from pfngate.evals.trace import Tracer, read_trace
from pfngate.features.config import REPO_ROOT

MODEL = load_agent_config().model
LOOPBACK = {"127.0.0.1", "::1", "localhost"}


# A scripted model that counts its calls
class Inner:
    """Calls a tool once, then answers. `exploding` makes any call a test failure."""

    def __init__(self, *, exploding: bool = False, answer: str = "the sum is 3") -> None:
        self.calls, self.exploding, self.answer = 0, exploding, answer

    def respond(self, messages: list[Any], info: AgentInfo) -> ModelResponse:
        self.calls += 1
        assert not self.exploding, "the wrapped model was called"
        returned = any(
            isinstance(p, ToolReturnPart) for m in messages for p in getattr(m, "parts", [])
        )
        part = TextPart(self.answer) if returned else ToolCallPart("add", {"a": 1, "b": 2})
        return ModelResponse(parts=[part])

    def model(self) -> FunctionModel:
        return FunctionModel(self.respond, model_name="inner-1")


def wrap(inner: Inner, mode: str, directory: Path) -> RecordReplayModel:
    return RecordReplayModel(inner.model(), mode=mode, directory=directory)  # type: ignore[arg-type]


def run_text(
    model: Any,
    *,
    system: str = "You are a test.",
    tool_doc: str = "Add two numbers.",
    user: str = "hello",
    temperature: float = 0.0,
    bump: int = 0,
) -> tuple[str, list[dict[str, Any]]]:
    """Run a one-tool agent. Returns the answer and its model responses as plain dicts."""
    agent = Agent(None, output_type=str, system_prompt=system, defer_model_check=True)

    def add(a: int, b: int) -> dict[str, int]:
        return {"sum": a + b + bump}

    add.__doc__ = tool_doc
    agent.tool_plain(add)
    result = agent.run_sync(user, model=model, model_settings={"temperature": temperature})
    responses = [m for m in result.all_messages() if isinstance(m, ModelResponse)]
    dumped = ModelMessagesTypeAdapter.dump_python(responses, mode="json")
    for d in dumped:  # the agent stamps these on every run
        d.pop("run_id"), d.pop("conversation_id")
    return result.output, dumped


def files(directory: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(directory.glob("*.json"))}


def sha(data: dict[str, bytes]) -> str:
    return hashlib.sha256(json.dumps({k: v.hex() for k, v in data.items()}).encode()).hexdigest()


# Record and replay
def test_replay_gives_the_recorded_answer_and_never_calls_the_model(tmp_path: Path) -> None:
    recorder = wrap(Inner(), "record", tmp_path)
    answer, responses = run_text(recorder)
    assert answer == "the sum is 3" and len(files(tmp_path)) == 2  # the tool call, the answer
    assert (recorder.live_calls, recorder.recorded_calls, recorder.replayed_calls) == (2, 2, 0)

    never = Inner(exploding=True)
    replayer = wrap(never, "replay", tmp_path)
    assert run_text(replayer) == (answer, responses)
    assert never.calls == 0
    assert (replayer.live_calls, replayer.recorded_calls, replayer.replayed_calls) == (0, 0, 2)


def test_replay_is_byte_for_byte_deterministic(tmp_path: Path) -> None:
    recorded = wrap(Inner(), "record", tmp_path)
    _, first = run_text(recorded)
    before = files(tmp_path)

    outputs = []
    for _ in range(3):
        outputs.append(run_text(wrap(Inner(exploding=True), "replay", tmp_path)))
    assert all(o == (outputs[0][0], outputs[0][1]) for o in outputs)
    assert outputs[0][1] == first  # the replayed responses are the recorded ones
    assert files(tmp_path) == before  # replay writes nothing and changes nothing

    # The same bytes at the level of one request, and the file itself is stable JSON.
    messages = [ModelRequest(parts=[UserPromptPart("hi")])]
    params, settings = ModelRequestParameters(), {"temperature": 0.0}
    direct = [
        asyncio.run(
            wrap(Inner(exploding=i > 0), mode, tmp_path / "direct").request(
                messages, settings, params
            )
        )  # fmt: skip
        for i, mode in enumerate(["record", "replay", "replay"])
    ]
    raw = [ModelMessagesTypeAdapter.dump_json([r]) for r in direct]
    assert raw[0] == raw[1] == raw[2]
    for name, content in files(tmp_path / "direct").items():
        text = content.decode("utf-8")
        assert "\r" not in text and text.endswith("}\n")
        assert (
            json.dumps(json.loads(text), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            == text
        )
        assert name == json.loads(text)["key"] + ".json"


def test_an_entry_is_written_once_and_record_mode_uses_it(tmp_path: Path) -> None:
    run_text(wrap(Inner(answer="first"), "record", tmp_path))
    before = files(tmp_path)
    again = Inner(answer="second")
    answer, _ = run_text(wrap(again, "record", tmp_path))
    assert answer == "first" and again.calls == 0  # all hits: the model is not called
    assert files(tmp_path) == before


def test_the_store_never_changes_an_entry(tmp_path: Path) -> None:
    store = LLMStore(tmp_path)
    first = ModelResponse(parts=[TextPart("first")])
    store.put("k" * 64, "function:m", first, [])
    before = files(tmp_path)
    store.put("k" * 64, "function:m", ModelResponse(parts=[TextPart("second")]), ["warned"])
    assert files(tmp_path) == before
    stored = store.get("k" * 64)
    assert stored is not None and stored.sampling_warnings == [] and stored.latency_s is None
    assert stored.response.parts[0].content == "first"  # type: ignore[union-attr]


SAMPLING = "Sampling parameters ['temperature'] are not supported by 'inner-1'. Ignored."


class Warns(Inner):
    """An inner model that warns while it answers, as PydanticAI does for 5.5 models."""

    def respond(self, messages: list[Any], info: AgentInfo) -> ModelResponse:
        warnings.warn(SAMPLING, UserWarning, stacklevel=2)
        return super().respond(messages, info)


def test_the_warning_of_the_live_call_is_stored_and_replayed(tmp_path: Path) -> None:
    with warnings.catch_warnings(record=True) as live:
        warnings.simplefilter("always")
        run_text(wrap(Warns(), "record", tmp_path))
    assert any(is_sampling_warning(w.message) for w in live)  # recording does not hide it
    stored = [json.loads(c)["sampling_warnings"] for c in files(tmp_path).values()]
    assert stored == [[SAMPLING], [SAMPLING]]

    with warnings.catch_warnings(record=True) as replayed:
        warnings.simplefilter("always")
        run_text(wrap(Inner(exploding=True), "replay", tmp_path))  # this model cannot warn
    assert [str(w.message) for w in replayed if is_sampling_warning(w.message)] == [SAMPLING] * 2


def test_no_warning_is_invented_for_a_call_that_had_none(tmp_path: Path) -> None:
    run_text(wrap(Inner(), "record", tmp_path))
    with warnings.catch_warnings(record=True) as replayed:
        warnings.simplefilter("always")
        run_text(wrap(Inner(exploding=True), "replay", tmp_path))
    assert not any(is_sampling_warning(w.message) for w in replayed)


class Sleepy(Inner):
    """An inner model that takes a while to answer, as a real one does."""

    def respond(self, messages: list[Any], info: AgentInfo) -> ModelResponse:
        time.sleep(0.06)
        return super().respond(messages, info)


def test_the_latency_of_the_live_call_is_stored_and_replayed(tmp_path: Path) -> None:
    messages = [ModelRequest(parts=[UserPromptPart("hi")])]
    params, settings = ModelRequestParameters(), {"temperature": 0.0}
    recorder = wrap(Sleepy(), "record", tmp_path)
    asyncio.run(recorder.request(messages, settings, params))
    before = files(tmp_path)
    stored = json.loads(next(iter(files(tmp_path).values())))["latency_s"]
    assert recorder.last_latency_s == stored and stored >= 0.06  # measured, and written down

    replayer = wrap(Inner(exploding=True), "replay", tmp_path)
    t0 = time.perf_counter()
    asyncio.run(replayer.request(messages, settings, params))
    assert time.perf_counter() - t0 < 0.05  # the replay itself is fast
    assert replayer.last_latency_s == stored  # but it reports the recorded latency

    live = wrap(Sleepy(), "live", tmp_path)
    asyncio.run(live.request(messages, settings, params))
    assert live.last_latency_s is not None and live.last_latency_s >= 0.06
    assert files(tmp_path) == before  # live mode writes nothing


def test_an_entry_without_a_latency_replays_with_none(tmp_path: Path) -> None:
    store = LLMStore(tmp_path)
    store.put("k" * 64, "function:m", ModelResponse(parts=[TextPart("x")]), [])  # an older entry
    assert json.loads(store.path("k" * 64).read_text())["latency_s"] is None
    stored = store.get("k" * 64)
    assert stored is not None and stored.latency_s is None


def test_live_mode_neither_reads_nor_writes_the_cache(tmp_path: Path) -> None:
    run_text(wrap(Inner(), "record", tmp_path))
    before = files(tmp_path)
    live = wrap(Inner(answer="fresh"), "live", tmp_path)
    assert run_text(live)[0] == "fresh"  # the cached "the sum is 3" is ignored
    assert (live.live_calls, live.replayed_calls, live.recorded_calls) == (2, 0, 0)
    assert files(tmp_path) == before


# A missing key fails loudly
def test_replay_with_a_missing_key_fails_loudly_and_writes_nothing(tmp_path: Path) -> None:
    never = Inner(exploding=True)
    with pytest.raises(ReplayMissError) as err:
        run_text(wrap(never, "replay", tmp_path))
    text = str(err.value)
    assert "no recorded LLM response" in text and "never calls the network" in text
    assert "--llm-mode record" in text and "function:inner-1" in text
    assert any(len(w) == 64 for w in text.replace(",", " ").replace(")", " ").split())  # the key
    assert never.calls == 0 and not list(tmp_path.glob("*"))


@pytest.mark.parametrize(
    "change",
    [
        {"system": "Another prompt."},
        {"user": "goodbye"},
        {"temperature": 0.5},
        {"tool_doc": "Add them."},
    ],  # fmt: skip
)
def test_a_changed_prompt_tool_or_setting_is_a_miss_not_a_stale_replay(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    run_text(wrap(Inner(), "record", tmp_path))
    with pytest.raises(ReplayMissError):
        run_text(wrap(Inner(exploding=True), "replay", tmp_path), **change)


def test_a_changed_tool_result_misses_on_the_next_request_only(tmp_path: Path) -> None:
    run_text(wrap(Inner(), "record", tmp_path))
    replayer = wrap(Inner(exploding=True), "replay", tmp_path)
    with pytest.raises(ReplayMissError):
        run_text(replayer, bump=1)  # the first request is the same, the second is not
    assert replayer.replayed_calls == 1


def test_the_key_ignores_ids_and_times_and_sees_the_model_name(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    run_text(wrap(Inner(), "record", a))
    run_text(wrap(Inner(), "record", b))  # another run: new ids, timestamps and tool-call ids
    assert set(files(a)) == set(files(b))
    messages = [ModelRequest(parts=[UserPromptPart("hi")])]
    args = (messages, {"temperature": 0.0}, ModelRequestParameters())
    assert request_key("m:one", *args) != request_key("m:two", *args)
    assert request_key("m:one", *args) == request_key("m:one", *args)


def test_a_corrupt_entry_is_refused(tmp_path: Path) -> None:
    store = LLMStore(tmp_path)
    key = "a" * 64
    store.path(key).write_text(json.dumps({"key": "b" * 64, "version": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a cache entry"):
        store.get(key)


def test_streaming_is_not_cached(tmp_path: Path) -> None:
    async def stream() -> None:
        async with wrap(Inner(), "replay", tmp_path).request_stream([], None, None):  # type: ignore[arg-type]
            pass

    with pytest.raises(NotImplementedError, match="streaming"):
        asyncio.run(stream())


# The default mode and the command line
def test_the_default_mode_is_replay_only_when_a_cache_exists(tmp_path: Path) -> None:
    assert default_llm_mode(tmp_path / "none") == "live"
    assert default_llm_mode(tmp_path) == "live"  # exists, but empty
    run_text(wrap(Inner(), "record", tmp_path))
    assert default_llm_mode(tmp_path) == "replay"
    assert default_llm_mode.__defaults__ == (DEFAULT_LLM_CACHE,)
    assert DEFAULT_LLM_CACHE == REPO_ROOT / "results" / "cache" / "llm"


def test_the_llm_mode_flag(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    add_llm_arguments(parser)
    assert parser.parse_args([]).llm_mode is None
    assert resolve_llm_mode(None, tmp_path) == "live"
    run_text(wrap(Inner(), "record", tmp_path))
    assert resolve_llm_mode(parser.parse_args([]).llm_mode, tmp_path) == "replay"
    for mode in ("live", "record", "replay"):
        assert parser.parse_args(["--llm-mode", mode]).llm_mode == mode
        assert resolve_llm_mode(mode, tmp_path) == mode
    with pytest.raises(SystemExit):
        parser.parse_args(["--llm-mode", "cached"])


# A whole run: recorded against the mock endpoint, replayed with the network locked
@pytest.fixture(scope="module")
def setup(small_data: dict[str, Any], make_gate: GateFactory, tmp_path_factory: Any) -> Any:
    return build_setup(small_data, make_gate, tmp_path_factory.mktemp("replay"))


STEPS = [*GOOD[:1], *BUILD_AND_RUN, ("final", VERDICT)]


def run_full(setup: Any, tmp: Path, model: Any) -> tuple[Any, Tracer]:
    cfg = load_agent_config()
    tracer = Tracer("replay-test", trace_dir=tmp / "traces")
    harness = dataclasses.replace(make_harness(setup, neighbours(), tmp), agent_cfg=cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        record = run_agent(build_agent(cfg), harness, setup["query"], tracer, RunStore(),
                           model=model)  # fmt: skip
    tracer.close()
    return record, tracer


def comparable(record: Any) -> dict[str, Any]:
    d = record.model_dump(mode="json")
    d.pop("run_id"), d.pop("n_replayed_requests")
    d["gate_result"].pop("run_id")
    return d


class LockedNetwork:
    """Patches the socket module: any connection that is not to this machine is recorded and
    refused, and so is any name lookup. asyncio itself talks to localhost, which is allowed."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.attempts: list[Any] = []
        self.local_connects = 0
        real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

        def host_of(address: Any) -> Any:
            return address[0] if isinstance(address, tuple) else address

        def connect(sock: socket.socket, address: Any) -> Any:
            if host_of(address) not in LOOPBACK:
                self.attempts.append(address)
                raise OSError(f"network blocked by the test: {address}")
            self.local_connects += 1
            return real_connect(sock, address)

        def connect_ex(sock: socket.socket, address: Any) -> Any:
            if host_of(address) not in LOOPBACK:
                self.attempts.append(address)
                return 10013
            return real_connect_ex(sock, address)

        real_getaddrinfo = socket.getaddrinfo

        def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
            if host not in LOOPBACK and host is not None:
                self.attempts.append(("dns", host))
                raise OSError(f"name lookup blocked by the test: {host}")
            return real_getaddrinfo(host, *args, **kwargs)

        monkeypatch.setattr(socket.socket, "connect", connect)
        monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
        monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


@pytest.mark.parametrize(
    ("model_string", "temperature_applied"),
    [(load_agent_config().model, True), (load_agent_config().debug_model, True)],
)
def test_a_whole_run_replays_identically_with_the_network_locked(
    setup: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_string: str,
    temperature_applied: bool,
) -> None:
    cassettes = tmp_path / "llm"
    mock = OpenRouterMock(STEPS)
    recorder = RecordReplayModel(mock.model(model_string), mode="record", directory=cassettes)
    recorded, rec_tracer = run_full(setup, tmp_path / "rec", recorder)
    assert recorded.stop_reason == "completed" and recorded.n_replayed_requests == 0
    assert recorded.temperature_applied is temperature_applied
    assert len(mock.bodies) == 4 and len(files(cassettes)) == 4
    stored = files(cassettes)

    net = LockedNetwork(monkeypatch)
    with pytest.raises(OSError, match="blocked by the test"):  # the trap is live
        socket.create_connection(("openrouter.ai", 443), timeout=1)
    net.attempts.clear()

    replayed, rep_tracer = run_full(
        setup, tmp_path / "rep", make_model(model_string, mode="replay", directory=cassettes)
    )
    again, _ = run_full(
        setup, tmp_path / "rep2", make_model(model_string, mode="replay", directory=cassettes)
    )
    assert net.attempts == []  # nothing left this machine, and no name was looked up
    assert len(mock.bodies) == 4  # the mock endpoint saw no new request
    assert files(cassettes) == stored  # and the cache did not change by one byte
    assert replayed.n_replayed_requests == replayed.n_model_requests == 4
    assert comparable(replayed) == comparable(recorded) == comparable(again)
    assert replayed.temperature_applied is temperature_applied  # the flag survives replay

    rows = {k: read_trace(t.path) for k, t in (("rec", rec_tracer), ("rep", rep_tracer))}
    calls = {k: [r for r in v if r["tool"].startswith("llm_")] for k, v in rows.items()}
    assert [r["tool"] for r in calls["rec"]] == ["llm_call"] * 4
    assert [r["tool"] for r in calls["rep"]] == ["llm_replay"] * 4
    for live, replay in zip(calls["rec"], calls["rep"], strict=True):
        for field in ("tokens_in", "tokens_out", "cost_usd", "backend", "model_version"):
            assert live[field] == replay[field]  # the cost of the run is known offline too


def test_a_replay_miss_in_a_whole_run_is_an_error_not_an_agent_answer(
    setup: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    net = LockedNetwork(monkeypatch)
    model = make_model(MODEL, mode="replay", directory=tmp_path / "empty")
    with pytest.raises(ReplayMissError, match="no recorded LLM response"):
        run_full(setup, tmp_path / "run", model)
    assert net.attempts == []


def test_the_offline_model_cannot_reach_the_network_even_if_called_directly() -> None:
    model = offline_model(MODEL)
    agent = Agent(model, output_type=str)
    with pytest.raises(Exception) as err:  # noqa: B017
        agent.run_sync("hi")
    chain, seen = err.value, []
    while chain is not None:
        seen.append(str(chain))
        chain = chain.__cause__ or chain.__context__
    assert any("replay mode must not call the network" in s for s in seen)
    assert NetworkBlockedError.__name__ in " ".join(seen) or any("replay mode" in s for s in seen)


def test_replay_needs_no_api_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    model = make_model(MODEL, mode="replay", directory=tmp_path)
    assert model.model_ref == MODEL and model.mode == "replay"
    with pytest.raises(ValueError, match="openrouter"):
        offline_model("openai:gpt-5")


def test_how_a_tool_is_scheduled_is_not_part_of_the_cache_key() -> None:
    from pydantic_ai.tools import ToolDefinition

    def params(sequential: bool) -> ModelRequestParameters:
        tool = ToolDefinition(
            name="t", parameters_json_schema={"type": "object"}, sequential=sequential
        )
        return ModelRequestParameters(function_tools=[tool])

    assert request_key("m:one", [], None, params(True)) == request_key(
        "m:one", [], None, params(False)
    )


def test_the_agents_tools_run_one_at_a_time() -> None:
    from pfngate.agent.naive import NAIVE_TOOLS
    from pfngate.agent.tools import TOOLS

    assert all(tool.sequential for tool in (*TOOLS, *NAIVE_TOOLS))

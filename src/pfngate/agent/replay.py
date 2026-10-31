"""Record and replay of LLM responses, so a demo runs with no network and no key.

RecordReplayModel wraps a PydanticAI model and has three modes:

  live    every call goes to the model. Nothing is read from the cache or written to it.
  record  a call whose key is in the cache is answered from it. Any other call goes to the
          model, and its response is stored. This needs the model's API key.
  replay  every call is answered from the cache. A key that is missing raises
          ReplayMissError. The wrapped model is never called, and it is built with a dummy key
          and a transport that raises on any request, so no network call can be made.

A response is stored under results/cache/llm/<key>.json. The key is the SHA-256 of the model
name, the system prompt, all messages (which carry the earlier tool calls and tool results),
the tool and answer definitions, and the model settings. The last two go beyond what the
model "sees as input", on purpose: change a tool description or a temperature and an old
response would answer a different question, so it must not be replayed.

Run ids, conversation ids, timestamps and generated tool-call ids differ between runs and
are left out of the key. Tool-call ids are renumbered by order of appearance.

A response is written once and never changed, as JSON with sorted keys and LF line endings
(see .gitattributes), so a replay reads the same bytes every time.

Each entry also stores how long the live call took. A replay reports that latency, not its own
(which is about zero), so the latency of a run is known offline too.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx2  # installed with the openai package. Used for the transport that blocks calls.
from dotenv import load_dotenv
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters, infer_model
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.settings import ModelSettings

from pfngate.features.config import REPO_ROOT

LLMMode = Literal["live", "record", "replay"]
LLM_MODES: tuple[str, ...] = ("live", "record", "replay")
DEFAULT_LLM_CACHE = REPO_ROOT / "results" / "cache" / "llm"
KEY_VERSION = 1

# Fields that differ between two identical runs. They are not part of the key.
_VOLATILE = frozenset(
    {
        "timestamp",
        "run_id",
        "conversation_id",
        "provider_response_id",
        "provider_details",
        "provider_url",
        "usage",
        "metadata",
        "state",
        "history_digest",
        "workspace_ref",
        "failed_attempts",
        "id",
    }
)


class ReplayMissError(RuntimeError):
    """Replay mode found no recorded response for a request. The network is never tried."""


class NetworkBlockedError(RuntimeError):
    """Something tried to reach the network through a model that is only allowed to replay."""


def is_sampling_warning(message: object) -> bool:
    """PydanticAI warns when a model does not take sampling settings and drops them."""
    text = str(message)
    return "Sampling parameters" in text and "temperature" in text


# The key
def _strip(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k not in _VOLATILE}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def _renumber(value: Any, ids: dict[str, str]) -> Any:
    """Replace each tool-call id by call_0, call_1, ... in order of first appearance."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k == "tool_call_id" and isinstance(v, str):
                out[k] = ids.setdefault(v, f"call_{len(ids)}")
            else:
                out[k] = _renumber(v, ids)
        return out
    if isinstance(value, list):
        return [_renumber(v, ids) for v in value]
    return value


def canonical_request(
    model_ref: str,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    params: ModelRequestParameters,
) -> dict[str, Any]:
    """Everything that decides the answer, without what changes from run to run."""
    dumped = ModelMessagesTypeAdapter.dump_python(messages, mode="json")
    definitions = {k: v for k, v in dataclasses.asdict(params).items() if not k.startswith("_")}
    for tool in definitions.get("function_tools", []):
        # How a tool is scheduled does not change what the model sees, so the key always says
        # "not sequential". (The tools now run one at a time; responses recorded before that
        # change were keyed with False, and stay valid.)
        tool["sequential"] = False
    return {
        "version": KEY_VERSION,
        "model": model_ref,
        "messages": _renumber(_strip(dumped), {}),
        "definitions": definitions,
        "settings": dict(model_settings or {}),
    }


def request_key(
    model_ref: str,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    params: ModelRequestParameters,
) -> str:
    canonical = canonical_request(model_ref, messages, model_settings, params)
    text = json.dumps(canonical, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# The store
@dataclass(frozen=True)
class StoredResponse:
    response: ModelResponse
    sampling_warnings: list[str]  # what the live call warned about, to be warned again
    latency_s: float | None  # how long the live call took. None for an older entry.


class LLMStore:
    """One JSON file per response, written once."""

    def __init__(self, directory: Path = DEFAULT_LLM_CACHE) -> None:
        self.directory = directory

    def path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def has_entries(self) -> bool:
        return self.directory.is_dir() and any(self.directory.glob("*.json"))

    def get(self, key: str) -> StoredResponse | None:
        """The stored response, its sampling warnings and its latency, or None."""
        path = self.path(key)
        if not path.exists():
            return None
        entry = json.loads(path.read_bytes())
        if entry.get("key") != key or entry.get("version") != KEY_VERSION:
            raise ValueError(f"{path} is not a cache entry for key {key}: delete it and record")
        response = ModelMessagesTypeAdapter.validate_python([entry["response"]])[0]
        if not isinstance(response, ModelResponse):
            raise ValueError(f"{path} does not hold a model response")
        return StoredResponse(response, list(entry["sampling_warnings"]), entry.get("latency_s"))

    def put(
        self,
        key: str,
        model_ref: str,
        response: ModelResponse,
        sampling_warnings: list[str],
        latency_s: float | None = None,
    ) -> None:
        path = self.path(key)
        if path.exists():  # an entry is never changed
            return
        entry = {
            "version": KEY_VERSION,
            "key": key,
            "model": model_ref,
            "recorded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "sampling_warnings": sampling_warnings,
            "latency_s": latency_s,
            "response": ModelMessagesTypeAdapter.dump_python([response], mode="json")[0],
        }
        text = json.dumps(entry, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes((text + "\n").encode("utf-8"))  # bytes: the same file on every OS
        os.replace(tmp, path)


# The wrapper
class RecordReplayModel(WrapperModel):
    """A model that records or replays another model's responses. See the module doc."""

    def __init__(
        self, wrapped: Model, *, mode: LLMMode = "replay", directory: Path = DEFAULT_LLM_CACHE
    ) -> None:
        super().__init__(wrapped)
        if mode not in LLM_MODES:
            raise ValueError(f"mode must be one of {LLM_MODES}, got '{mode}'")
        self.mode: LLMMode = mode
        self.store = LLMStore(directory)
        self.live_calls = 0  # calls that reached the wrapped model
        self.replayed_calls = 0  # calls answered from the cache
        self.recorded_calls = 0  # responses written to the cache
        # The latency of the last call: measured if it was live, as recorded if it was replayed
        # (None for an entry without one).
        self.last_latency_s: float | None = None

    @property
    def model_ref(self) -> str:
        return f"{self.wrapped.system}:{self.wrapped.model_name}"

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        if self.mode == "live":
            self.live_calls += 1
            t0 = time.perf_counter()
            response = await self.wrapped.request(
                messages, model_settings, model_request_parameters
            )
            self.last_latency_s = time.perf_counter() - t0
            return response

        key = request_key(self.model_ref, messages, model_settings, model_request_parameters)
        hit = self.store.get(key)
        if hit is not None:
            for text in hit.sampling_warnings:  # the live call warned: so does the replay
                warnings.warn(text, UserWarning, stacklevel=2)
            self.replayed_calls += 1
            self.last_latency_s = hit.latency_s
            return hit.response
        if self.mode == "replay":
            raise ReplayMissError(
                f"no recorded LLM response for this request (key {key}, model {self.model_ref}, "
                f"{len(messages)} messages) in {self.store.directory}. Replay never calls the "
                "network. Record it with --llm-mode record (this needs the API key), or check "
                "that the prompt, the tools and the settings are the same as when it was recorded."
            )

        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            response = await self.wrapped.request(
                messages, model_settings, model_request_parameters
            )
        latency = time.perf_counter() - t0
        for w in caught:  # shown as usual
            warnings.warn(w.message, w.category, stacklevel=2)
        sampling = [str(w.message) for w in caught if is_sampling_warning(w.message)]
        self.store.put(key, self.model_ref, response, sampling, latency)
        self.last_latency_s = latency
        self.live_calls += 1
        self.recorded_calls += 1
        return response

    @asynccontextmanager
    async def request_stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        """Streaming is not cached. It is only allowed in live mode."""
        if self.mode != "live":
            raise NotImplementedError("the LLM cache does not support streaming requests")
        async with self.wrapped.request_stream(*args, **kwargs) as stream:
            yield stream


# Building a model
def _no_network(request: httpx2.Request) -> httpx2.Response:
    raise NetworkBlockedError(f"replay mode must not call the network: {request.url}")


def offline_model(ref: str) -> Model:
    """A model that has the real model's profile but cannot reach the network.

    It gets a dummy key and a transport that raises on any request. The replay wrapper never
    calls it. This is the second lock on the door, after the wrapper itself.
    """
    provider, _, _ = ref.partition(":")
    if provider != "openrouter":
        raise ValueError(
            f"replay without a key is built for 'openrouter:' models, not '{ref}'. "
            "Pass a model object to RecordReplayModel for other providers."
        )
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(_no_network))

    def factory(_: str) -> OpenRouterProvider:
        return OpenRouterProvider(api_key="replay-only-not-a-key", http_client=client)

    return infer_model(ref, provider_factory=factory)


def make_model(
    ref: str, *, mode: LLMMode, directory: Path = DEFAULT_LLM_CACHE
) -> RecordReplayModel:
    """The model for an "openrouter:<model id>" string in a mode. The key, OPENROUTER_API_KEY,
    comes from the environment or from .env, and is never printed. Replay needs none."""
    if mode == "replay":
        inner = offline_model(ref)
    else:
        load_dotenv(REPO_ROOT / ".env")
        inner = infer_model(ref)
    return RecordReplayModel(inner, mode=mode, directory=directory)


# The command line
def default_llm_mode(directory: Path = DEFAULT_LLM_CACHE) -> LLMMode:
    """replay if the cache has entries, else live."""
    return "replay" if LLMStore(directory).has_entries() else "live"


def add_llm_arguments(parser: Any) -> None:
    """--llm-mode and --llm-cache-dir, for every evaluation entry point that calls an LLM."""
    parser.add_argument(
        "--llm-mode",
        choices=LLM_MODES,
        default=None,
        help="live: call the model. record: call it and store the responses. replay: use "
        "the stored responses only, with no network and no key. Default: replay if the "
        "cache has entries, else live.",
    )
    parser.add_argument("--llm-cache-dir", type=Path, default=DEFAULT_LLM_CACHE)


def resolve_llm_mode(mode: str | None, directory: Path) -> LLMMode:
    return default_llm_mode(directory) if mode is None else mode  # type: ignore[return-value]

"""Server-side data of the agent runs, keyed by run id, and the budget of each run.

The model never sees an array. The tools read from a RunData found by run id, and return
small summaries. The query table that enters the store is cut down to the trajectory id, the
window position and the features: no target, no family, no scenario name. So no tool can
return them, whatever it is asked.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import pandas as pd

from pfngate.agent.calls import LLMDeps
from pfngate.agent.config import AgentConfig
from pfngate.evals.gating import FitPredict
from pfngate.features import FEATURE_NAMES
from pfngate.gate import GateConfig
from pfngate.gate.reference import PoolDistance
from pfngate.pfn import PFNConfig
from pfngate.pfn.adapter import PFNOutput
from pfngate.pfn.context import ContextSelection

QUERY_COLUMNS: tuple[str, ...] = ("trajectory_id", "window_end", *FEATURE_NAMES)


def sanitise_query(query: pd.DataFrame) -> pd.DataFrame:
    """Cut a query table down to QUERY_COLUMNS, in time order. One trajectory only."""
    missing = set(QUERY_COLUMNS) - set(query.columns)
    if missing:
        raise ValueError(f"query is missing columns {sorted(missing)}")
    if query["trajectory_id"].astype(str).nunique() != 1:
        raise ValueError("query must come from exactly one trajectory")
    return (
        query[list(QUERY_COLUMNS)].sort_values("window_end", kind="stable").reset_index(drop=True)
    )


class ToolCapExceeded(RuntimeError):
    """The agent asked for a tool call over the cap. The run is stopped, not continued."""


@dataclass
class RunBudget:
    """What one run may use. Counted under the lock of its AgentDeps."""

    max_tool_calls: int
    max_contexts: int
    tool_calls: int = 0
    contexts_built: int = 0

    def charge_tool(self, name: str) -> None:
        if self.tool_calls >= self.max_tool_calls:
            raise ToolCapExceeded(
                f"tool call cap of {self.max_tool_calls} reached, '{name}' was refused"
            )
        self.tool_calls += 1

    @property
    def contexts_left(self) -> int:
        return self.max_contexts - self.contexts_built


@dataclass(frozen=True, eq=False)
class AgentHarness:
    """Everything a run needs from outside, fixed before the agent starts."""

    pool: pd.DataFrame
    gate_cfg: GateConfig  # tuned and frozen: the caller checks that, as run_pfn does
    pool_distance: PoolDistance
    fit_predict: FitPredict
    pfn_cfg: PFNConfig
    agent_cfg: AgentConfig
    max_context: int


@dataclass(frozen=True, eq=False)
class TriedContext:
    selection: ContextSelection
    strategy: str  # as the agent asked for it
    size: int  # as the agent asked for it. A knn context can come out smaller.


@dataclass
class RunData:
    run_id: str
    scenario_id: str  # the trajectory id. It goes to the local trace, never to the model.
    query: pd.DataFrame  # QUERY_COLUMNS only, sorted by window_end
    harness: AgentHarness
    tried: list[TriedContext] = field(default_factory=list)
    outputs: dict[str, PFNOutput] = field(default_factory=dict)  # by context hash
    current: TriedContext | None = None

    @classmethod
    def create(cls, run_id: str, query: pd.DataFrame, harness: AgentHarness) -> RunData:
        clean = sanitise_query(query)
        return cls(
            run_id=run_id,
            scenario_id=str(clean["trajectory_id"].iloc[0]),
            query=clean,
            harness=harness,
        )


class RunStore:
    """Run data by run id. A run id is never reused."""

    def __init__(self) -> None:
        self._runs: dict[str, RunData] = {}
        self._lock = threading.Lock()

    def put(self, data: RunData) -> None:
        with self._lock:
            if data.run_id in self._runs:
                raise ValueError(f"run id {data.run_id} is already in the store")
            self._runs[data.run_id] = data

    def get(self, run_id: str) -> RunData:
        with self._lock:
            return self._runs[run_id]

    def drop(self, run_id: str) -> None:
        with self._lock:
            self._runs.pop(run_id, None)

    def __contains__(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._runs


@dataclass(kw_only=True)
class AgentDeps(LLMDeps):
    """The deps of a tool-using run: the base, plus the store and the budget."""

    store: RunStore
    budget: RunBudget

    @property
    def data(self) -> RunData:
        return self.store.get(self.run_id)

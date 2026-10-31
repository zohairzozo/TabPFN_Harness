"""The four tools of the agent. Each returns a small summary, never rows or arrays.

Every call goes through call_tool(): it charges the run budget, refuses a call over the cap
by raising ToolCapExceeded, times the call and writes one trace step. The functions at the
bottom are what PydanticAI shows to the model. Their docstrings are the tool descriptions.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
from pydantic_ai import RunContext, Tool

from pfngate.agent.store import AgentDeps, ToolCapExceeded, TriedContext
from pfngate.agent.summary import r4 as _r
from pfngate.agent.summary import summarise_query
from pfngate.agent.summary import third as _third
from pfngate.agent.verdict import AGENT_STRATEGIES, AgentStrategy
from pfngate.evals.calibration import level_index
from pfngate.features import FEATURE_NAMES
from pfngate.pfn.adapter import PFNOutput
from pfngate.pfn.context import query_x, select_context

ToolResult = tuple[dict[str, Any], dict[str, Any]]  # what the model sees, trace fields


def call_tool(
    deps: AgentDeps, name: str, fn: Callable[..., ToolResult], *args: Any
) -> dict[str, Any]:
    """Run one tool under the budget. The 7th call of a run (cap 6) raises ToolCapExceeded."""
    with deps.lock:
        try:
            deps.budget.charge_tool(name)
        except ToolCapExceeded:
            deps.log(tool="agent_tool_refused", latency_s=0.0)
            raise
        t0 = time.perf_counter()
        result, trace_fields = fn(deps, *args)
        deps.log(tool=f"agent_tool:{name}", latency_s=time.perf_counter() - t0, **trace_fields)
        return result


# Implementations. They read from the store and return (summary, trace fields).
def _describe_query(deps: AgentDeps) -> ToolResult:
    return summarise_query(deps.data.query), {}


def _describe_context_pool(deps: AgentDeps) -> ToolResult:
    data = deps.data
    h = data.harness
    gate = h.gate_cfg.checks
    near_max = gate.context_not_degenerate.positive_max
    far_min = gate.context_not_degenerate.negative_min
    pool = h.pool
    lo, hi = gate.min_context_size.threshold, h.max_context
    p05, p50, p95 = np.percentile(pool[list(FEATURE_NAMES)].to_numpy(float), [5, 50, 95], axis=0)
    ranges = {
        name: {"p05": _r(p05[j]), "p50": _r(p50[j]), "p95": _r(p95[j])}
        for j, name in enumerate(FEATURE_NAMES)
    }
    dist = h.pool_distance.distances(query_x(data.query), data.scenario_id)
    cut = gate.ood_share.distance_cut
    target = pool["target"].to_numpy(float)
    available = {"available_rows": len(pool), "min_size": lo, "max_size": hi}
    return {
        "pool_rows": len(pool),
        "pool_trajectories": int(pool["trajectory_id"].nunique()),
        "pool_share_near_tipping": _r((target <= near_max).mean()),
        "pool_share_far_from_tipping": _r((target >= far_min).mean()),
        "strategies": {"random": available, "knn": available},
        "feature_ranges": ranges,
        "query_distance_to_pool": {
            "median": _r(np.median(dist)),
            "p90": _r(np.percentile(dist, 90)),
            "pool_reference_p99": _r(cut),
            "share_of_windows_beyond_reference": _r((dist > cut).mean()),
        },
    }, {}


def _choose_context(deps: AgentDeps, strategy: str, size: int) -> ToolResult:
    data, budget = deps.data, deps.budget
    h = data.harness
    lo, hi = h.gate_cfg.checks.min_context_size.threshold, h.max_context
    if strategy not in AGENT_STRATEGIES:
        return {"error": f"strategy must be one of {list(AGENT_STRATEGIES)}"}, {}
    if not lo <= size <= hi:
        return {"error": f"size must be between {lo} and {hi}"}, {}
    if budget.contexts_left <= 0:
        return {"error": f"context budget used up: at most {budget.max_contexts} contexts"}, {}
    budget.contexts_built += 1
    sel = select_context(strategy, h.pool, data.query, h.pfn_cfg, max_context=size)
    tried = TriedContext(sel, strategy, size)
    data.tried.append(tried)
    data.current = tried
    gate = h.gate_cfg.checks.context_not_degenerate
    target = sel.context["target"].to_numpy(float)
    return {
        "context_hash": sel.context_hash,
        "size": sel.size,
        "strategy": strategy,
        "rows_near_tipping": int((target <= gate.positive_max).sum()),
        "rows_far_from_tipping": int((target >= gate.negative_min).sum()),
        "contexts_left": budget.contexts_left,
    }, {"context_hash": sel.context_hash, "context_size": sel.size, "strategy": sel.label}


def _summarise(out: PFNOutput, interval: tuple[float, float]) -> dict[str, Any]:
    lv, q = out.quantile_levels, out.quantiles
    width = q[:, level_index(lv, interval[1])] - q[:, level_index(lv, interval[0])]
    k = _third(len(width))
    med, p = out.median, out.p_critical
    return {
        "n_windows": len(p),
        "median_prediction": {
            "mean": _r(med.mean()),
            "first_third_mean": _r(med[:k].mean()),
            "last_third_mean": _r(med[-k:].mean()),
            "last_window": _r(med[-1]),
        },
        "interval_90_width": {
            "mean": _r(width.mean()),
            "max": _r(width.max()),
            "last_window": _r(width[-1]),
        },
        "p_critical": {
            "mean": _r(p.mean()),
            "max": _r(p.max()),
            "last_window": _r(p[-1]),
            "last_third_mean": _r(p[-k:].mean()),
            "share_of_windows_above_half": _r((p > 0.5).mean()),
        },
    }


def _run_pfn(deps: AgentDeps) -> ToolResult:
    data = deps.data
    tried = data.current
    if tried is None:
        return {"error": "call choose_context first"}, {}
    sel = tried.selection
    out, hit = data.harness.fit_predict(sel.context, data.query)
    data.outputs[sel.context_hash] = out
    md = out.metadata
    result = _summarise(out, data.harness.gate_cfg.bad_run.interval)
    result["context_hash"] = sel.context_hash
    return result, {
        "context_hash": sel.context_hash,
        "context_size": sel.size,
        "strategy": sel.label,
        "backend": md.backend,
        "model_version": md.model_version,
    }


# What PydanticAI shows to the model. The docstrings are the tool descriptions.
# The tools that need no input still take an optional `reason`. A tool with an empty schema
# made the free OpenRouter model send {"additionalProperties": "{}"}, which failed validation
# on every call. The reason is not used.
def describe_query(ctx: RunContext[AgentDeps], reason: str = "") -> dict[str, Any]:
    """Summary statistics of the query windows: for each feature, the mean, the spread and
    the trend over the series. Use it to see what kind of series this is.

    Args:
        reason: optional, a few words on why you call this.
    """
    return call_tool(ctx.deps, "describe_query", _describe_query)


def describe_context_pool(ctx: RunContext[AgentDeps], reason: str = "") -> dict[str, Any]:
    """The pool that contexts are built from: how many rows each strategy can use, the sizes
    you may ask for, the range of each feature, and how far the query sits from the pool.

    Args:
        reason: optional, a few words on why you call this.
    """
    return call_tool(ctx.deps, "describe_context_pool", _describe_context_pool)


def choose_context(
    ctx: RunContext[AgentDeps], strategy: AgentStrategy, size: int
) -> dict[str, Any]:
    """Build a context and make it the current one. Returns its hash, its real size and how
    many of its rows are near and far from tipping. This counts as one of your contexts.

    Args:
        strategy: "random" or "knn".
        size: how many rows the context should have.
    """
    return call_tool(ctx.deps, "choose_context", _choose_context, strategy, size)


def run_pfn(ctx: RunContext[AgentDeps], reason: str = "") -> dict[str, Any]:
    """Fit the model on the current context and predict the query windows. Returns summaries
    of the predicted median, the width of the 90% interval and p_critical (the probability
    that the system is near tipping), not the rows.

    Args:
        reason: optional, a few words on why you call this.
    """
    return call_tool(ctx.deps, "run_pfn", _run_pfn)


# The tools change the shared run state (the chosen context), so they must run one at a time in
# the order the model asked for them. Run at the same time, run_pfn could go before
# choose_context, and a replay would then differ from the recording.
TOOLS = tuple(
    Tool(f, sequential=True)
    for f in (describe_query, describe_context_pool, choose_context, run_pfn)
)
__all__ = ["TOOLS", "call_tool"]

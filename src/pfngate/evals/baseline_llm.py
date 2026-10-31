"""The LLM-only baseline: the same model reads a text summary of the query's window features
and answers "alert" or "no_alert" with a probability. No tools, no PFN, no gate.

It shows what the language model knows about early-warning signals on its own, so the agent's
result can be compared with it. It is traced the same way as the agent (one step per model
call, with tokens and an approximate cost) and scored with the same run-level metrics.

    python -m pfngate.evals.baseline_llm --split val --llm-mode replay

The summary is the one the agent's describe_query tool returns: feature means, spreads and
trends, with no target, family or trajectory id. Results go to results/llm_only_<split>.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model

from pfngate.agent.agent import resolve_model
from pfngate.agent.calls import LLMDeps, tracing_hooks
from pfngate.agent.config import AgentConfig, load_agent_config
from pfngate.agent.prompts import llm_only_system_prompt, llm_only_user_prompt
from pfngate.agent.replay import DEFAULT_LLM_CACHE
from pfngate.agent.runner import StopReason, run_llm
from pfngate.agent.store import sanitise_query
from pfngate.agent.summary import summarise_query, summary_text
from pfngate.evals.agent_metrics import RunOutcome, source_metrics
from pfngate.evals.llm_eval import (
    DEFAULT_GATE_CONFIG,
    DEFAULT_GATE_HASH,
    DEFAULT_RESULTS_DIR,
    HELD_OUT_LLM_SPLITS,
    LLM_SPLITS,
    add_common_arguments,
    build_llm_model,
    gate_for_split,
    ground_truth,
    load_split_queries,
    outcome,
    result_path,
    timing,
    usage_totals,
    write_result,
)
from pfngate.evals.trace import Tracer
from pfngate.evals.variant_summary import summarise_runs
from pfngate.features import FeatureConfig, load_feature_config
from pfngate.gate.freeze import assert_committed
from pfngate.pfn import PFNConfig, load_pfn_config

NAME = "llm_only"


class LLMOnlyAnswer(BaseModel):
    """The answer of the LLM-only baseline. It is the decision: nothing checks it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision: Literal["alert", "no_alert"] = Field(
        description="Is the end of the series close to tipping?"
    )
    probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Your probability that the end of the series is close to tipping.",
    )


class LLMOnlyRecord(BaseModel):
    """One run of the baseline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    trajectory_id: str
    model: str
    temperature_requested: float
    temperature_applied: bool
    stop_reason: StopReason
    stop_detail: str | None
    n_tool_calls: int = 0  # it has no tools
    n_model_requests: int
    n_replayed_requests: int
    llm_seconds: float
    tokens_in: int
    tokens_out: int
    cost_usd_approx: float | None
    answer: LLMOnlyAnswer | None
    decision: Literal["alert", "no_alert"]  # the answer, or no_alert if there was none
    defaulted: bool  # True: no usable answer, so the decision is the default no_alert
    consistent: bool | None  # does the decision match the probability (alert iff p >= 0.5)?


def build_llm_only_agent() -> Agent[LLMDeps, LLMOnlyAnswer]:
    """An agent with no tools. The model is given at run time, so building it needs no key."""
    return Agent(
        None,
        output_type=LLMOnlyAnswer,
        deps_type=LLMDeps,
        system_prompt=llm_only_system_prompt(),
        capabilities=[tracing_hooks()],
        defer_model_check=True,
        name="pfngate-llm-only",
    )


def run_llm_only(
    agent: Agent[LLMDeps, LLMOnlyAnswer],
    query: pd.DataFrame,
    tracer: Tracer,
    *,
    cfg: AgentConfig,
    model: Model | str | None = None,
    run_id: str | None = None,
    write_sidecar: bool = True,
) -> LLMOnlyRecord:
    """Ask the model about one query trajectory. Only the feature summary reaches it."""
    resolved = resolve_model(cfg.model if model is None else model)
    q = sanitise_query(query)  # trajectory id, window position and features only
    deps = LLMDeps(
        run_id=run_id or tracer.run_id,
        scenario_id=str(q["trajectory_id"].iloc[0]),
        tracer=tracer,
        agent_cfg=cfg,
        model_system=resolved.system,
        model_name=resolved.model_name,
    )
    prompt = llm_only_user_prompt(summary_text(summarise_query(q)))
    result = run_llm(agent, prompt, model=resolved, deps=deps, cfg=cfg)
    answer = result.output
    record = LLMOnlyRecord(
        run_id=deps.run_id,
        trajectory_id=deps.scenario_id,
        model=f"{deps.model_system}:{deps.model_name}",
        temperature_requested=cfg.temperature,
        temperature_applied=result.temperature_applied,
        stop_reason=result.stop,
        stop_detail=result.detail,
        n_model_requests=deps.model_requests,
        n_replayed_requests=deps.replayed_requests,
        llm_seconds=deps.llm_seconds,
        tokens_in=deps.tokens_in,
        tokens_out=deps.tokens_out,
        cost_usd_approx=deps.cost_usd,
        answer=answer,
        decision="no_alert" if answer is None else answer.decision,
        defaulted=answer is None,
        consistent=None
        if answer is None
        else (answer.probability >= 0.5) == (answer.decision == "alert"),
    )
    if write_sidecar:
        path = tracer.path.with_name(f"{deps.run_id}.llm_only.json")
        text = json.dumps(record.model_dump(mode="json"), allow_nan=False, indent=2)
        path.write_text(text + "\n", encoding="utf-8")
    return record


def evaluate(
    split: str,
    *,
    model: Model | None = None,
    agent_cfg: AgentConfig | None = None,
    pfn_cfg: PFNConfig | None = None,
    feature_config: FeatureConfig | None = None,
    seeds_per_scenario: int | None = None,
    limit: int | None = None,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    trace_dir: Path | None = None,
    gate_config_path: Path = DEFAULT_GATE_CONFIG,
    gate_hash_path: Path = DEFAULT_GATE_HASH,
    llm_mode: str | None = None,
    llm_cache_dir: Path = DEFAULT_LLM_CACHE,
    debug: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Run the baseline on every trajectory of a split, write the result JSON, return it."""
    if split not in LLM_SPLITS:
        raise ValueError(f"split must be one of {list(LLM_SPLITS)}")
    gate_for_split(split, gate_config_path, gate_hash_path)  # held-out: frozen gate first
    cfg = agent_cfg or load_agent_config()
    pfn = pfn_cfg or load_pfn_config()
    resolved = model or build_llm_model(cfg, debug=debug, mode=llm_mode, directory=llm_cache_dir)
    queries = load_split_queries(
        split, feature_config or load_feature_config(), seeds_per_scenario, limit
    )
    tracer = Tracer(f"{NAME}-{split}", trace_dir=trace_dir or results_dir / "traces")
    agent = build_llm_only_agent()
    runs: list[dict[str, Any]] = []
    outcomes: list[RunOutcome] = []
    t_start = time.perf_counter()
    for i, q in enumerate(queries):
        record = run_llm_only(
            agent, q, tracer, cfg=cfg, model=resolved, run_id=f"{tracer.run_id}.{i:04d}",
            write_sidecar=False,
        )  # fmt: skip
        truth = ground_truth(q)
        score = None if record.answer is None else record.answer.probability
        outcomes.append(outcome(record.model_dump(), truth, record.decision, score))
        runs.append({
            "record": record.model_dump(mode="json"), "ground_truth": truth,
            "decision": record.decision, "accepted": None, "pfn_quality": None,
            "timing": timing(record.llm_seconds, 0.0),
        })  # fmt: skip
        if progress:
            print(
                f"[{i + 1}/{len(queries)}] {record.trajectory_id} -> {record.decision}", flush=True
            )
    tracer.close()

    records = [r["record"] for r in runs]
    result: dict[str, Any] = {
        "method": NAME,
        "split": split,
        "dataset_splits": [LLM_SPLITS[split]],
        "held_out": split in HELD_OUT_LLM_SPLITS,
        "model": records[0]["model"] if records else f"{resolved.system}:{resolved.model_name}",
        "llm_mode": getattr(resolved, "mode", None),
        "debug_model": debug,
        "seeds_per_scenario": seeds_per_scenario,
        "limit": limit,
        "n_runs": len(runs),
        "usage": usage_totals(records),
        "summary": summarise_runs(runs, has_tools=True),  # no tools: every run made 0 calls
        "metrics": {
            NAME: source_metrics(
                outcomes, n_boot=pfn.bootstrap_n, seed=pfn.bootstrap_seed, ci=pfn.bootstrap_ci
            )
        },
        "trace": tracer.path.name,
        "wall_seconds": time.perf_counter() - t_start,
        "scoring": "run level: the label of the last window of each run (agent_metrics.py)",
        "runs": runs,
    }
    write_result(results_dir, NAME, split, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    if args.split in HELD_OUT_LLM_SPLITS:
        assert_committed(DEFAULT_GATE_CONFIG)  # docs/SPLITS.md: committed before C and D
    result = evaluate(
        args.split,
        seeds_per_scenario=args.seeds_per_scenario,
        limit=args.limit,
        results_dir=args.results_dir,
        llm_mode=args.llm_mode,
        llm_cache_dir=args.llm_cache_dir,
        debug=args.debug,
    )
    u = result["usage"]
    print(
        f"done: {result['n_runs']} runs, mode {result['llm_mode']}, "
        f"{u['model_requests']} model calls ({u['replayed_requests']} replayed), "
        f"cost about ${u['cost_usd_approx']} (approximate)"
    )
    print(f"wrote {result_path(args.results_dir, NAME, args.split)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

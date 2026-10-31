"""Shared parts of the evaluation entry points that call an LLM.

Entry points: python -m pfngate.evals.run_agent (the agent, and the naive agent) and
python -m pfngate.evals.baseline_llm (the LLM-only baseline).

A split here is a whole dataset split, tipping runs and control runs together, because the
run-level metrics (agent_metrics.py) need both: every tipping run ends near tipping and every
control run ends far from it. The test and out-of-prior splits are held out: they need the
gate to be tuned, frozen and committed first, for every variant, so no held-out run is looked
at before the freeze.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pfngate.agent.config import AgentConfig
from pfngate.agent.replay import (
    DEFAULT_LLM_CACHE,
    RecordReplayModel,
    add_llm_arguments,
    make_model,
    resolve_llm_mode,
)
from pfngate.agent.store import AgentHarness, sanitise_query
from pfngate.data import REAL_CONTROL_SPLIT, REAL_SPLIT
from pfngate.evals.agent_metrics import RunOutcome, end_label
from pfngate.evals.gating import ground_truth as pfn_ground_truth
from pfngate.evals.metrics import write_json
from pfngate.features import FeatureConfig, build_dataset
from pfngate.features.config import REPO_ROOT
from pfngate.gate import GateConfig, assert_frozen, load_gate_config
from pfngate.gate.config import DEFAULT_GATE_CONFIG, DEFAULT_GATE_HASH
from pfngate.pfn.adapter import PFNOutput
from pfngate.pfn.context import select_context

DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
LLM_SPLITS: dict[str, str] = {
    "val": "validation",
    "test": "test_in_prior",
    "ood": "test_out_of_prior",
    "real": REAL_SPLIT,  # real Daphnia data, test only (docs/DAPHNIA.md)
    "real_control": REAL_CONTROL_SPLIT,  # real grid data, false alarms only (docs/GRID.md)
}
HELD_OUT_LLM_SPLITS: frozenset[str] = frozenset({"test", "ood", "real", "real_control"})


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split", required=True, choices=list(LLM_SPLITS))
    add_llm_arguments(parser)
    parser.add_argument(
        "--debug", action="store_true", help="use debug_model from configs/agent.yaml (cheaper)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="a trial run: N scenarios of each kind (tipping and control), spread evenly over "
        "the scenarios, with all their seeds",
    )
    parser.add_argument(
        "--seeds-per-scenario",
        type=int,
        default=None,
        help="use only the first N seeds of each scenario (default: all). Report it if used.",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)


def select_scenarios(queries: list[pd.DataFrame], limit: int | None) -> list[pd.DataFrame]:
    """A trial sample: `limit` scenarios of each kind, tipping and control, with all their seeds.

    The scenarios are spread evenly over the ordered list, so that several families appear.
    Taking the first N trajectories instead would drop every control run (they come last) and
    cut the seeds of a scenario apart, and the three seeds of a scenario are needed together.
    """
    if limit is None:
        return queries
    if limit < 1:
        raise ValueError("limit must be at least 1")
    kinds: dict[bool, list[str]] = {False: [], True: []}  # control or not: scenario ids in order
    for q in queries:
        ids = kinds[q["role"].iloc[0] == "control"]
        if str(q["scenario_id"].iloc[0]) not in ids:
            ids.append(str(q["scenario_id"].iloc[0]))
    keep: set[str] = set()
    for ids in kinds.values():
        if ids:
            picks = np.round(np.linspace(0, len(ids) - 1, num=min(limit, len(ids)))).astype(int)
            keep.update(ids[i] for i in np.unique(picks))
    return [q for q in queries if str(q["scenario_id"].iloc[0]) in keep]


def load_split_queries(
    split: str, fc: FeatureConfig, seeds_per_scenario: int | None, limit: int | None
) -> list[pd.DataFrame]:
    """One table of windows per trajectory, tipping and control, in a fixed order."""
    df = build_dataset(LLM_SPLITS[split], seeds_per_scenario=seeds_per_scenario, feature_config=fc)
    queries = [g.reset_index(drop=True) for _, g in df.groupby("trajectory_id", sort=False)]
    return select_scenarios(queries, limit)


def gate_for_split(split: str, config_path: Path, hash_path: Path) -> GateConfig:
    """The gate. A held-out split refuses to start unless it is tuned and frozen."""
    if split in HELD_OUT_LLM_SPLITS:
        return assert_frozen(config_path, hash_path)
    return load_gate_config(config_path)


def build_llm_model(
    cfg: AgentConfig, *, debug: bool, mode: str | None, directory: Path = DEFAULT_LLM_CACHE
) -> RecordReplayModel:
    """The model from configs/agent.yaml, in the --llm-mode (default: replay if a cache exists)."""
    return make_model(
        cfg.model_for(debug), mode=resolve_llm_mode(mode, directory), directory=directory
    )


def ground_truth(query: pd.DataFrame) -> dict[str, Any]:
    """What the end of the series really is. Only the evaluation reads it, never a variant."""
    last = query.sort_values("window_end", kind="stable").iloc[-1]
    return {
        "scenario_id": str(last["scenario_id"]),
        "role": str(last["role"]),
        "family": str(last["family"]),
        "split": str(last["split"]),
        "end_target": float(last["target"]),
        "end_label": end_label(int(last["label"])),
        "used_by_variant": False,
    }


def pfn_quality(query: pd.DataFrame, out: PFNOutput, gate_cfg: GateConfig) -> dict[str, Any]:
    """How good the PFN's output was on one run, against the truth of its windows.

    `query` must have a target column and be in the order of the windows of `out`. Only the
    evaluation calls this. The gate never does, and neither does any variant.
    """
    gt = pfn_ground_truth(query, out, gate_cfg)
    return {
        "coverage_90": gt["coverage_90"],
        "bad": bool(gt["bad"]),  # coverage below 0.80 (docs/GATE.md)
        "calibration_error": gt["calibration_error"],
        "coverage_90_positive_windows": gt["coverage_90_positive_windows"],
        "n_windows": len(query),
        "last_window_p_critical": float(out.p_critical[-1]),
    }


def final_context_quality(
    harness: AgentHarness, query: pd.DataFrame, strategy: str, size: int, context_hash: str
) -> dict[str, Any]:
    """pfn_quality() of a run's final context.

    The context is rebuilt from its strategy and size, checked against its hash, and predicted
    through the cache the run already filled, so this costs no new fit.
    """
    q = sanitise_query(query)
    sel = select_context(strategy, harness.pool, q, harness.pfn_cfg, max_context=size)
    if sel.context_hash != context_hash:
        raise ValueError(f"the context of {strategy}/{size} is not the one the run used")
    out, _ = harness.fit_predict(sel.context, q)
    ordered = query.sort_values("window_end", kind="stable").reset_index(drop=True)
    return pfn_quality(ordered, out, harness.gate_cfg)


def timing(llm_seconds: float, pfn_seconds: float) -> dict[str, float]:
    """The latency of one run: the model calls (as recorded) plus the PFN's fit and predict
    time (as first measured, gate refits included). Tool code and the gate's checks are small."""
    return {
        "llm_seconds": llm_seconds,
        "pfn_seconds": pfn_seconds,
        "total_seconds": llm_seconds + pfn_seconds,
    }


def outcome(
    record: dict[str, Any], truth: dict[str, Any], decision: str, score: float | None
) -> RunOutcome:
    return RunOutcome(
        trajectory_id=record["trajectory_id"],
        role=truth["role"],
        end_label=truth["end_label"],
        decision=decision,  # type: ignore[arg-type]
        score=score,
    )


def usage_totals(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Tokens and cost over the runs. The cost is as recorded, so a replay reports it too."""
    costs = [r["cost_usd_approx"] for r in records]
    return {
        "n_runs": len(records),
        "model_requests": sum(r["n_model_requests"] for r in records),
        "replayed_requests": sum(r["n_replayed_requests"] for r in records),
        "tokens_in": sum(r["tokens_in"] for r in records),
        "tokens_out": sum(r["tokens_out"] for r in records),
        "llm_seconds": sum(r["llm_seconds"] for r in records),
        "cost_usd_approx": None if any(c is None for c in costs) else sum(costs),
        "cost_note": "approximate: prices in configs/agent.yaml, as recorded",
        "temperature_applied_runs": sum(r["temperature_applied"] for r in records),
        "stop_reasons": {
            s: sum(r["stop_reason"] == s for r in records)
            for s in ("completed", "tool_cap", "request_cap", "bad_output")
        },
    }


def result_path(results_dir: Path, name: str, split: str) -> Path:
    return results_dir / f"{name}_{split}.json"


def write_result(results_dir: Path, name: str, split: str, payload: dict[str, Any]) -> Path:
    path = result_path(results_dir, name, split)
    write_json(path, payload)
    return path


__all__ = [
    "select_scenarios",
    "DEFAULT_GATE_CONFIG",
    "DEFAULT_GATE_HASH",
    "DEFAULT_RESULTS_DIR",
    "HELD_OUT_LLM_SPLITS",
    "LLM_SPLITS",
    "add_common_arguments",
    "build_llm_model",
    "gate_for_split",
    "ground_truth",
    "load_split_queries",
    "outcome",
    "final_context_quality",
    "pfn_quality",
    "result_path",
    "timing",
    "usage_totals",
    "write_result",
]

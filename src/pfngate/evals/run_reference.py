"""The two reference variants that use no LLM, scored like the LLM variants.

    python -m pfngate.evals.run_reference --variant classic --split test
    python -m pfngate.evals.run_reference --variant oracle --split test

classic  the classic early-warning baseline (configs/baseline_ews.yaml). The run's score is the
         baseline score of its last window, and it alerts if that is at or above the alarm
         threshold in the config. Results go to results/classic_ews_<split>.json.
oracle   the family_matched oracle: a context from the family of the query, which a real run
         would not know, and no gate. The run's score is p_critical of its last window, and it
         alerts if that is at or above the threshold that run_pfn set from validation controls
         (read from results/pfn_family_matched_controls.json). It cannot run on a family that is
         not in the pool, so out-of-prior tipping runs are skipped. Results go to
         results/oracle_<split>.json.

Each uses the same trajectories (the same --limit and seeds), the same truth (the label of the
last window) and the same result format as the LLM variants (run_agent.py, baseline_llm.py), so
they fill one comparison table. The test and out-of-prior splits need the gate to be tuned,
frozen and committed first, as for every other variant.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pfngate.data import REAL_SPLITS
from pfngate.data.load import real_trajectories
from pfngate.evals.agent_metrics import RunOutcome, source_metrics
from pfngate.evals.baseline_ews import BaselineConfig, load_baseline_config, trajectory_scores
from pfngate.evals.gating import FitPredict
from pfngate.evals.llm_eval import (
    DEFAULT_GATE_CONFIG,
    DEFAULT_GATE_HASH,
    DEFAULT_RESULTS_DIR,
    HELD_OUT_LLM_SPLITS,
    LLM_SPLITS,
    gate_for_split,
    ground_truth,
    load_split_queries,
    outcome,
    pfn_quality,
    result_path,
    timing,
    write_result,
)
from pfngate.evals.pfn_cache import PFNCache, data_version, package_version
from pfngate.evals.trace import Tracer
from pfngate.evals.variant_summary import summarise_runs
from pfngate.features import (
    FeatureConfig,
    build_dataset,
    load_feature_config,
    split_plan,
    trajectory_id,
)
from pfngate.features.config import REPO_ROOT
from pfngate.gate.freeze import assert_committed
from pfngate.pfn import PFNConfig, load_pfn_config, make_backend
from pfngate.pfn.adapter import PFNBackend
from pfngate.pfn.context import NoMatchingFamilyError, select_context
from pfngate.scenarios import load_scenarios

VARIANTS = {"classic": "classic_ews", "oracle": "oracle"}
ORACLE_THRESHOLD_FILE = "pfn_family_matched_controls.json"


def oracle_alert_threshold(results_dir: Path) -> float:
    """The p_critical alert threshold run_pfn set for the oracle from validation controls only."""
    path = results_dir / ORACLE_THRESHOLD_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run run_pfn --strategy family_matched first")
    return float(
        json.loads(path.read_text(encoding="utf-8"))["false_alarms"]["thresholds"]["p_critical"]
    )


def classic_run(
    q: pd.DataFrame, scores: pd.DataFrame, seconds: float, cfg: BaselineConfig
) -> tuple[dict[str, Any], str, float | None]:
    """The record, the decision and the score of one classic baseline run."""
    last = scores.sort_values("window_end", kind="stable").iloc[-1]
    if int(last["window_end"]) != int(q["window_end"].max()):
        raise ValueError("the baseline and the dataset disagree about the last window")
    score = None if np.isnan(last["score"]) else float(last["score"])
    decision = "alert" if score is not None and score >= cfg.alarm_threshold else "no_alert"
    record = {
        "trajectory_id": str(q["trajectory_id"].iloc[0]),
        "n_windows": len(q),
        "score": score,
        "score_defined": score is not None,  # the first K - 1 windows have no score
        "alarm_threshold": cfg.alarm_threshold,
        "decision": decision,
        "seconds": seconds,
    }
    return record, decision, score


def evaluate(
    variant: str,
    split: str,
    *,
    backend: PFNBackend | None = None,
    cfg: PFNConfig | None = None,
    feature_config: FeatureConfig | None = None,
    baseline_cfg: BaselineConfig | None = None,
    alert_threshold: float | None = None,
    max_context: int | None = None,
    seeds_per_scenario: int | None = None,
    pool_seeds_per_scenario: int | None = None,
    limit: int | None = None,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    cache_dir: Path | None = None,
    trace_dir: Path | None = None,
    gate_config_path: Path = DEFAULT_GATE_CONFIG,
    gate_hash_path: Path = DEFAULT_GATE_HASH,
    progress: bool = True,
) -> dict[str, Any]:
    """Run a reference variant on every selected trajectory of a split and write the result."""
    if variant not in VARIANTS or split not in LLM_SPLITS:
        raise ValueError(
            f"variant must be one of {list(VARIANTS)}, split one of {list(LLM_SPLITS)}"
        )
    held_out = split in HELD_OUT_LLM_SPLITS
    gate_cfg = gate_for_split(split, gate_config_path, gate_hash_path)  # before any data is read
    pfn = cfg or load_pfn_config()
    fc = feature_config or load_feature_config()
    name = VARIANTS[variant]
    queries = load_split_queries(split, fc, seeds_per_scenario, limit)
    tracer = Tracer(f"{name}-{split}", trace_dir=trace_dir or results_dir / "traces")
    runs: list[dict[str, Any]] = []
    outcomes: list[RunOutcome] = []
    skipped: list[dict[str, str]] = []
    settings: dict[str, Any] = {}
    t_start = time.perf_counter()

    if variant == "classic":
        bcfg = baseline_cfg or load_baseline_config()
        scenarios = load_scenarios()
        common = next(iter(scenarios.values())).common
        wanted = {str(q["trajectory_id"].iloc[0]): q for q in queries}
        settings = {"alarm_threshold": bcfg.alarm_threshold, "k_windows": bcfg.k_windows}
        simulated = {}
        score_fc = fc  # the settings the windows were cut with (shorter for the Daphnia data)
        if LLM_SPLITS[split] in REAL_SPLITS:
            real, score_fc = real_trajectories(LLM_SPLITS[split], seeds_per_scenario)
            simulated = {trajectory_id(t.scenario_id, t.seed): t for t in real}
        else:
            for sc, seed in split_plan(LLM_SPLITS[split], scenarios, seeds_per_scenario):
                if (tid := trajectory_id(sc.scenario_id, seed)) in wanted:
                    simulated[tid] = sc.generate(seed)
        for i, (tid, q) in enumerate(wanted.items()):
            t0 = time.perf_counter()
            scores = trajectory_scores(simulated[tid], score_fc, common, bcfg)
            seconds = time.perf_counter() - t0
            tracer.log_step(scenario_id=tid, tool="classic_ews_score", latency_s=seconds)
            record, decision, score = classic_run(q, scores, seconds, bcfg)
            truth = ground_truth(q)
            outcomes.append(outcome(record, truth, decision, score))
            runs.append(
                {
                    "record": record,
                    "ground_truth": truth,
                    "decision": decision,
                    "accepted": None,
                    "pfn_quality": None,
                    "timing": timing(0.0, seconds),
                }
            )
            if progress:
                print(f"[{i + 1}/{len(wanted)}] {tid} -> {decision}", flush=True)
    else:
        backend = backend or make_backend(pfn)
        cap = pfn.max_context if max_context is None else max_context
        threshold = (
            oracle_alert_threshold(results_dir) if alert_threshold is None else alert_threshold
        )
        model_version = getattr(backend, "model_version", pfn.model_version)
        data_ver = data_version(fc)
        settings = {
            "backend": backend.name,
            "model_version": model_version,
            "package_version": package_version(backend.name),
            "max_context": cap,
            "data_version": data_ver,
            "strategy": "family_matched",
            "alert_threshold": threshold,
        }
        pool = build_dataset(
            "context", seeds_per_scenario=pool_seeds_per_scenario, feature_config=fc
        )
        fit_predict = FitPredict(
            backend,
            PFNCache(cache_dir or results_dir / "cache" / "pfn", enabled=True),
            pfn,
            model_version,
            data_ver,
        )
        for i, q in enumerate(queries):
            q = q.sort_values("window_end", kind="stable").reset_index(drop=True)
            tid = str(q["trajectory_id"].iloc[0])
            try:
                sel = select_context("family_matched", pool, q, pfn, max_context=cap)
            except NoMatchingFamilyError as exc:
                skipped.append({"trajectory_id": tid, "reason": str(exc)})
                continue
            before = fit_predict.compute_seconds
            out, hit = fit_predict(sel.context, q)
            seconds = fit_predict.compute_seconds - before
            tracer.log_pfn_step(tid, sel, out, "pfn_cache_hit" if hit else "pfn_predict")
            last_p = float(out.p_critical[-1])
            decision = "alert" if last_p >= threshold else "no_alert"
            record = {
                "trajectory_id": tid,
                "n_windows": len(q),
                "context_hash": sel.context_hash,
                "context_size": sel.size,
                "score": last_p,
                "alert_threshold": threshold,
                "decision": decision,
            }
            truth = ground_truth(q)
            outcomes.append(outcome(record, truth, decision, last_p))
            runs.append(
                {
                    "record": record,
                    "ground_truth": truth,
                    "decision": decision,
                    "accepted": None,
                    "pfn_quality": pfn_quality(q, out, gate_cfg),
                    "timing": timing(0.0, seconds),
                }
            )
            if progress:
                print(f"[{i + 1}/{len(queries)}] {tid} -> {decision}", flush=True)
    tracer.close()

    result: dict[str, Any] = {
        "method": name,
        "split": split,
        "dataset_splits": [LLM_SPLITS[split]],
        "held_out": held_out,
        "model": None,  # no LLM
        "llm_mode": None,
        "oracle": variant == "oracle",
        "settings": settings,
        "gate_config_sha256": gate_cfg.sha256,
        "frozen_gate_sha256": gate_cfg.sha256 if held_out else None,
        "seeds_per_scenario": seeds_per_scenario,
        "limit": limit,
        "n_runs": len(runs),
        "skipped": skipped,
        "usage": None,
        "summary": summarise_runs(runs, has_tools=False),
        "metrics": {
            name: source_metrics(
                outcomes, n_boot=pfn.bootstrap_n, seed=pfn.bootstrap_seed, ci=pfn.bootstrap_ci
            )
        },
        "trace": tracer.path.name,
        "wall_seconds": time.perf_counter() - t_start,
        "scoring": "run level: the label of the last window of each run (agent_metrics.py)",
        "runs": runs,
    }
    write_result(results_dir, name, split, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    pfn = load_pfn_config()
    parser.add_argument("--variant", required=True, choices=list(VARIANTS))
    parser.add_argument("--split", required=True, choices=list(LLM_SPLITS))
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="a trial run: N scenarios of each kind, with all their seeds",
    )
    parser.add_argument("--seeds-per-scenario", type=int, default=None)
    parser.add_argument(
        "--max-context",
        type=int,
        default=None,
        help=f"cap on context rows (default {pfn.max_context})",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args(argv)
    if args.split in HELD_OUT_LLM_SPLITS:
        assert_committed(DEFAULT_GATE_CONFIG)  # docs/SPLITS.md: committed before C and D
    result = evaluate(
        args.variant,
        args.split,
        max_context=args.max_context,
        seeds_per_scenario=args.seeds_per_scenario,
        limit=args.limit,
        results_dir=args.results_dir,
    )
    print(f"done: {result['n_runs']} runs, {len(result['skipped'])} skipped")
    print(f"wrote {result_path(args.results_dir, result['method'], args.split)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["REPO_ROOT", "VARIANTS", "evaluate", "main"]

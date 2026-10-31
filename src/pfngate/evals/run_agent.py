"""Evaluate the LLM agent, or its naive variant, on a split.

    python -m pfngate.evals.run_agent --variant agent --split val --llm-mode replay
    python -m pfngate.evals.run_agent --variant naive --split val --llm-mode replay

agent  chooses the context (random or knn, and the size), recommends a decision, and then the
       gate decides. Written to results/agent_<split>.json. Both the final decision (the gate's)
       and the agent's own recommendation are scored.
naive  the same agent with a random context chosen by the harness, no gate and no abstain
       option. Written to results/naive_agent_<split>.json.

Both use the same PFN backend, cache and frozen gate as run_pfn --gate. The test and
out-of-prior splits need the gate tuned, frozen and committed. --llm-mode is live, record or
replay (default: replay if results/cache/llm has entries, else live). Replay needs no key and
makes no network call.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path
from typing import Any

from pfngate.agent.agent import build_agent, run_agent
from pfngate.agent.config import AgentConfig, load_agent_config
from pfngate.agent.naive import build_naive_agent, run_naive_agent
from pfngate.agent.replay import DEFAULT_LLM_CACHE
from pfngate.agent.store import AgentHarness, RunStore
from pfngate.agent.verdict import AGENT_STRATEGIES
from pfngate.evals.agent_metrics import RunOutcome, alert_score, source_metrics
from pfngate.evals.gating import FitPredict
from pfngate.evals.llm_eval import (
    DEFAULT_GATE_CONFIG,
    DEFAULT_GATE_HASH,
    DEFAULT_RESULTS_DIR,
    HELD_OUT_LLM_SPLITS,
    LLM_SPLITS,
    add_common_arguments,
    build_llm_model,
    final_context_quality,
    gate_for_split,
    ground_truth,
    load_split_queries,
    outcome,
    result_path,
    timing,
    usage_totals,
    write_result,
)
from pfngate.evals.pfn_cache import PFNCache, data_version, package_version
from pfngate.evals.run_pfn import pool_distance_for
from pfngate.evals.trace import Tracer
from pfngate.evals.variant_summary import summarise_runs
from pfngate.features import FeatureConfig, build_dataset, load_feature_config
from pfngate.gate import assert_applicable
from pfngate.gate.freeze import assert_committed
from pfngate.pfn import PFNConfig, load_pfn_config, make_backend
from pfngate.pfn.adapter import PFNBackend

VARIANTS = {"agent": "agent", "naive": "naive_agent"}


def gate_block(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """How often the gate stood between the agent and the decision. Agent runs only."""
    records = [r["record"] for r in runs]
    finals = [r["final"] for r in records]
    verdicts = [r["verdict"] for r in records]
    return {
        "n_runs": len(records),
        "n_gate_passed": sum(f["gate_passed"] is True for f in finals),
        "n_gate_blocked": sum(f["gate_passed"] is False for f in finals),
        "n_no_context": sum(f["gate_passed"] is None for f in finals),
        "blocked_by_check": dict(Counter(c for f in finals for c in f["gate_failed"])),
        "agent_alerts_blocked_by_gate": sum(
            v is not None and v["decision"] == "alert" and f["decision"] == "abstain"
            for v, f in zip(verdicts, finals, strict=True)
        ),
        "agent_agrees_with_final": sum(r["agent_agrees"] is True for r in records),
        "agent_disagrees_with_final": sum(r["agent_agrees"] is False for r in records),
        "no_verdict": sum(v is None for v in verdicts),
        "verdict_named_a_built_context": sum(f["followed_verdict_context"] is True for f in finals),
        "final_strategy": dict(Counter(f["context_strategy"] for f in finals)),
    }


def evaluate(
    variant: str,
    split: str,
    *,
    backend: PFNBackend | None = None,
    model: Any = None,
    agent_cfg: AgentConfig | None = None,
    cfg: PFNConfig | None = None,
    feature_config: FeatureConfig | None = None,
    max_context: int | None = None,
    seeds_per_scenario: int | None = None,
    pool_seeds_per_scenario: int | None = None,
    limit: int | None = None,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    cache_dir: Path | None = None,
    trace_dir: Path | None = None,
    gate_config_path: Path = DEFAULT_GATE_CONFIG,
    gate_hash_path: Path = DEFAULT_GATE_HASH,
    llm_mode: str | None = None,
    llm_cache_dir: Path = DEFAULT_LLM_CACHE,
    debug: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Run a variant on every trajectory of a split, write the result JSON, return it."""
    if variant not in VARIANTS or split not in LLM_SPLITS:
        raise ValueError(
            f"variant must be one of {list(VARIANTS)}, split one of {list(LLM_SPLITS)}"
        )
    held_out = split in HELD_OUT_LLM_SPLITS
    gate_cfg = gate_for_split(split, gate_config_path, gate_hash_path)  # before any data is read
    pfn = cfg or load_pfn_config()
    cap = pfn.max_context if max_context is None else max_context
    fc = feature_config or load_feature_config()
    backend = backend or make_backend(pfn)
    model_version = getattr(backend, "model_version", pfn.model_version)
    data_ver = data_version(fc)
    settings = {
        "backend": backend.name,
        "model_version": model_version,
        "package_version": package_version(backend.name),
        "max_context": cap,
        "data_version": data_ver,
    }
    if held_out:
        for strategy in AGENT_STRATEGIES:
            assert_applicable(gate_cfg, {**settings, "strategy": strategy}, oracle=False)

    acfg = agent_cfg or load_agent_config()
    resolved = model or build_llm_model(acfg, debug=debug, mode=llm_mode, directory=llm_cache_dir)
    pool = build_dataset("context", seeds_per_scenario=pool_seeds_per_scenario, feature_config=fc)
    harness = AgentHarness(
        pool=pool,
        gate_cfg=gate_cfg,
        pool_distance=pool_distance_for(gate_cfg, pool),
        fit_predict=FitPredict(
            backend,
            PFNCache(cache_dir or results_dir / "cache" / "pfn", enabled=True),
            pfn,
            model_version,
            data_ver,
        ),
        pfn_cfg=pfn,
        agent_cfg=acfg,
        max_context=cap,
    )

    name = VARIANTS[variant]
    queries = load_split_queries(split, fc, seeds_per_scenario, limit)
    tracer = Tracer(f"{name}-{split}", trace_dir=trace_dir or results_dir / "traces")
    agent = build_agent(acfg) if variant == "agent" else build_naive_agent(acfg)
    store = RunStore()
    runs: list[dict[str, Any]] = []
    outcomes: dict[str, list[RunOutcome]] = {}
    t_start = time.perf_counter()
    for i, q in enumerate(queries):
        kwargs: dict[str, Any] = {
            "model": resolved, "run_id": f"{tracer.run_id}.{i:04d}", "write_sidecar": False,
        }  # fmt: skip
        truth = ground_truth(q)
        pfn_before = harness.fit_predict.compute_seconds
        if variant == "agent":
            rec = run_agent(agent, harness, q, tracer, store, **kwargs)
            d, f_, v = rec.model_dump(mode="json"), rec.final, rec.verdict
            gated = outcome(
                d, truth, f_.decision, f_.last_window_p_critical if f_.gate_passed else None
            )
            asked = outcome(d, truth, v.decision if v else "abstain",
                            alert_score(v.decision, v.confidence) if v else None)  # fmt: skip
            outcomes.setdefault("agent_final", []).append(gated)
            outcomes.setdefault("agent_recommendation", []).append(asked)
            shown, accepted = f_.decision, f_.gate_passed
            context = (f_.context_strategy, f_.context_size, f_.context_hash)
        else:
            rec = run_naive_agent(agent, harness, q, tracer, store, **kwargs)
            d, v = rec.model_dump(mode="json"), rec.verdict
            score = None if v is None else alert_score(v.decision, v.confidence)
            outcomes.setdefault(name, []).append(outcome(d, truth, rec.decision, score))
            shown, accepted = rec.decision, None
            context = (rec.context_strategy, rec.context_size, rec.context_hash)
        pfn_seconds = harness.fit_predict.compute_seconds - pfn_before  # before the scoring below
        quality = final_context_quality(harness, q, *context) if context[2] else None
        runs.append({
            "record": d, "ground_truth": truth, "decision": shown, "accepted": accepted,
            "pfn_quality": quality, "timing": timing(rec.llm_seconds, pfn_seconds),
        })  # fmt: skip
        if progress:
            print(f"[{i + 1}/{len(queries)}] {rec.trajectory_id} -> {shown}", flush=True)
    tracer.close()

    records = [r["record"] for r in runs]
    result: dict[str, Any] = {
        "method": name,
        "split": split,
        "dataset_splits": [LLM_SPLITS[split]],
        "held_out": held_out,
        "model": records[0]["model"] if records else f"{resolved.system}:{resolved.model_name}",
        "llm_mode": getattr(resolved, "mode", None),
        "debug_model": debug,
        "settings": settings,
        "gate_config_sha256": gate_cfg.sha256,
        "frozen_gate_sha256": gate_cfg.sha256 if held_out else None,
        "seeds_per_scenario": seeds_per_scenario,
        "limit": limit,
        "n_runs": len(runs),
        "usage": usage_totals(records),
        "metrics": {
            k: source_metrics(
                v, n_boot=pfn.bootstrap_n, seed=pfn.bootstrap_seed, ci=pfn.bootstrap_ci
            )
            for k, v in outcomes.items()
        },
        "gate": gate_block(runs) if variant == "agent" else None,
        "summary": summarise_runs(runs, has_tools=True),
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
    parser.add_argument("--variant", choices=list(VARIANTS), default="agent")
    add_common_arguments(parser)
    pfn = load_pfn_config()
    parser.add_argument(
        "--max-context",
        type=int,
        default=None,
        help=f"cap on context rows (default {pfn.max_context}, from configs/pfn.yaml)",
    )
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
    print(f"wrote {result_path(args.results_dir, result['method'], args.split)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run a PFN backend on the scenario windows with a context strategy, and score it.

    python -m pfngate.evals.run_pfn --strategy knn --split val --max-context 500

For every trajectory in the split: pick a context with the strategy, fit once, predict all
the windows of the trajectory in one batch, and trace every call. Two scores are computed
per window: -median of the predicted target (low target means close to the tip) and
p_critical (the probability that the target is at or below the threshold in
configs/pfn.yaml). Each is scored with the balanced AUROC and a bootstrap CI resampled over
trajectories, the same code as the classic baseline.

Splits (the dataset splits are in docs/SPLITS.md):
  val       validation split, tipping trajectories
  test      test_in_prior split, tipping trajectories
  ood       test_out_of_prior split, tipping trajectories
  controls  the negative controls of validation, test_in_prior and test_out_of_prior
  val_controls  the negative controls of validation only (for tuning the gate)

Controls exist in every split, but they are run here, once, so that the alarm threshold can
be chosen from the VALIDATION controls and then applied unchanged to the others. The AUROC
splits (val, test, ood) therefore hold tipping trajectories only.

family_matched is an ORACLE. It reads the family of the query. Its results are labelled as
an oracle and are an upper bound, not a method.

Results go to results/pfn_<strategy>_<split>.json. The trace goes to results/traces/. Fit
and predict results are cached in results/cache/pfn/ by context hash (see pfn_cache.py).

Held-out splits (test, ood, controls) refuse to start unless the gate is tuned and frozen:
configs/gate.yaml must match configs/gate.frozen.sha256 (pfngate.gate.freeze). From the
command line the file must also be committed. Validation runs need neither: they are the
data the gate is tuned on.

--gate runs the gate (docs/GATE.md) on every run: context calibration refits on held-out
context trajectories, the distance of every query window to the pool, and the checks with
the thresholds in configs/gate.yaml. The GateResult goes into the result JSON and the
trace. With --gate on a held-out split, the run settings must also match the settings the
gate was tuned for. The realised coverage of each run (its ground truth) is recorded next
to the gate result for scoring the gate; the gate never sees it.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pfngate.data import REAL_CONTROL_SPLIT, REAL_SPLIT
from pfngate.evals.gating import (
    CalibrationRefit,
    FitPredict,
    gate_false_alarms,
    gate_one,
    gate_summary,
    ground_truth,
)
from pfngate.evals.metrics import (
    alarm_threshold_for_rate,
    bootstrap_auroc,
    false_alarm_rate,
    write_json,
)
from pfngate.evals.pfn_cache import PFNCache, data_version, package_version
from pfngate.evals.trace import Tracer
from pfngate.features import FeatureConfig, build_dataset, load_feature_config
from pfngate.features.config import REPO_ROOT
from pfngate.gate import (
    Decision,
    GateConfig,
    GateResult,
    assert_applicable,
    assert_frozen,
    decide,
    load_gate_config,
)
from pfngate.gate.config import DEFAULT_GATE_CONFIG, DEFAULT_GATE_HASH
from pfngate.gate.freeze import assert_committed
from pfngate.gate.reference import PoolDistance
from pfngate.pfn import PFNConfig, load_pfn_config, make_backend
from pfngate.pfn.adapter import PFNBackend, PFNOutput
from pfngate.pfn.context import (
    ORACLE_STRATEGIES,
    STRATEGIES,
    ContextSelection,
    NoMatchingFamilyError,
    select_context,
)

SPLIT_FLAGS: dict[str, tuple[str, ...]] = {
    "val": ("validation",),
    "test": ("test_in_prior",),
    "ood": ("test_out_of_prior",),
    "controls": ("validation", "test_in_prior", "test_out_of_prior"),
    "val_controls": ("validation",),
    # Real data is test only (docs/DAPHNIA.md, docs/GRID.md).
    "real": (REAL_SPLIT,),  # Daphnia populations that go extinct
    "real_controls": ("validation", REAL_SPLIT),  # Daphnia populations that persist
    "real_control": ("validation", REAL_CONTROL_SPLIT),  # grid frequency, normal operation
}
HELD_OUT_FLAGS: frozenset[str] = frozenset(
    {"test", "ood", "controls", "real", "real_controls", "real_control"}
)
CONTROL_FLAGS: frozenset[str] = frozenset(
    {"controls", "val_controls", "real_controls", "real_control"}
)
SCORES: tuple[str, ...] = ("neg_median", "p_critical")
SCORE_NOTES = {
    "neg_median": "minus the median of the predicted target; higher means closer to the tip",
    "p_critical": "predicted probability that the target is at or below critical_threshold",
}
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


def result_path(results_dir: Path, strategy: str, split_flag: str) -> Path:
    return results_dir / f"pfn_{strategy}_{split_flag}.json"


def load_queries(
    split_flag: str, fc: FeatureConfig, seeds_per_scenario: int | None
) -> list[pd.DataFrame]:
    """One DataFrame of windows per trajectory, tipping or control depending on the flag."""
    controls = split_flag in CONTROL_FLAGS
    queries = []
    for split in SPLIT_FLAGS[split_flag]:
        df = build_dataset(split, seeds_per_scenario=seeds_per_scenario, feature_config=fc)
        df = df[(df["role"] == "control") == controls]
        queries.extend(g.reset_index(drop=True) for _, g in df.groupby("trajectory_id", sort=False))
    return queries


def window_table(q: pd.DataFrame, out: PFNOutput) -> pd.DataFrame:
    if out.n_query != len(q):
        raise ValueError(f"{out.n_query} predictions for {len(q)} windows")
    table = q[["trajectory_id", "scenario_id", "family", "role", "split", "label", "target"]].copy()
    table["neg_median"] = -out.median
    table["p_critical"] = out.p_critical
    return table


def auroc_blocks(win: pd.DataFrame, cfg: PFNConfig) -> dict[str, Any]:
    """Balanced AUROC with a trajectory bootstrap CI, for each score, overall and per family."""
    labelled = win[win["label"] >= 0]  # the middle band is dropped from the binary task
    subsets = {"tipping_only": labelled}
    subsets.update({f"family:{f}": g for f, g in labelled.groupby("family")})
    out: dict[str, Any] = {}
    for name, g in subsets.items():
        out[name] = {}
        for score in SCORES:
            if g["label"].nunique() < 2:
                out[name][score] = None
                continue
            res = bootstrap_auroc(
                g["label"].to_numpy(),
                g[score].to_numpy(),
                g["trajectory_id"].to_numpy(),
                n_boot=cfg.bootstrap_n,
                seed=cfg.bootstrap_seed,
                ci=cfg.bootstrap_ci,
            )
            res["role"] = str(g["role"].iloc[0]) if name.startswith("family:") else None
            out[name][score] = res
    return out


def false_alarm_blocks(win: pd.DataFrame, cfg: PFNConfig) -> dict[str, Any]:
    """False alarms on controls. The threshold comes from validation controls only."""
    val = win[win["split"] == "validation"]
    if val.empty:
        raise ValueError("the controls run needs validation controls to set the threshold")
    held_out = win[win["split"] != "validation"]

    def rates(df: pd.DataFrame, score: str, thr: float) -> dict[str, Any]:
        groups = {"all_controls": df, **dict(iter(df.groupby("family")))}
        return {
            name: false_alarm_rate(g[score].to_numpy(), g["trajectory_id"].to_numpy(), thr)
            for name, g in groups.items()
        }

    out: dict[str, Any] = {
        "rule": "threshold from validation controls only, then applied unchanged to the rest",
        "target_window_false_alarm_rate": cfg.target_false_alarm_rate,
        "thresholds": {},
        "scores": {},
    }
    for score in SCORES:
        thr = alarm_threshold_for_rate(val[score].to_numpy(), cfg.target_false_alarm_rate)
        out["thresholds"][score] = thr
        per_split = {s: rates(g, score, thr) for s, g in win.groupby("split")}
        out["scores"][score] = {
            "splits": per_split,
            "held_out_pooled": rates(held_out, score, thr) if len(held_out) else None,
        }
    return out


def timing_block(records: list[dict[str, Any]], total_seconds: float) -> dict[str, Any]:
    """Latency per trajectory is fit plus predict of the run that computed the result."""
    if not records:
        return {"total_wall_seconds": total_seconds, "n_trajectories": 0}
    lat = np.array([r["latency_seconds"] for r in records])
    return {
        "total_wall_seconds": total_seconds,
        "n_trajectories": len(records),
        "latency_seconds_median": float(np.median(lat)),
        "latency_seconds_mean": float(lat.mean()),
        "latency_seconds_max": float(lat.max()),
        "fit_seconds_median": float(np.median([r["fit_seconds"] for r in records])),
        "predict_seconds_median": float(np.median([r["predict_seconds"] for r in records])),
        "windows_per_trajectory_median": float(np.median([r["n_windows"] for r in records])),
    }


def pool_distance_for(gate_cfg: GateConfig, pool: pd.DataFrame) -> PoolDistance:
    """The pool distance with the transform stored in the gate. The pool must be the same."""
    ref = gate_cfg.checks.ood_share
    n_traj = pool["trajectory_id"].nunique()
    if (len(pool), n_traj) != (ref.n_pool_rows, ref.n_pool_trajectories):
        raise ValueError(
            f"the pool has {len(pool)} rows in {n_traj} trajectories, but the gate reference "
            f"was built on {ref.n_pool_rows} rows in {ref.n_pool_trajectories}"
        )
    return PoolDistance(pool, ref.feature_mean, ref.feature_std, ref.k)


def decision_counts(decisions: tuple[str, ...] | None) -> dict[str, int] | None:
    if decisions is None:
        return None
    return {d: decisions.count(d) for d in ("abstain", "alert", "no_alert")}


def run(
    strategy: str,
    split_flag: str,
    *,
    max_context: int | None = None,
    backend_name: str | None = None,
    backend: PFNBackend | None = None,
    gate: bool = False,
    gate_config_path: Path = DEFAULT_GATE_CONFIG,
    gate_hash_path: Path = DEFAULT_GATE_HASH,
    seeds_per_scenario: int | None = None,
    pool_seeds_per_scenario: int | None = None,
    feature_config: FeatureConfig | None = None,
    cfg: PFNConfig | None = None,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    cache_dir: Path | None = None,
    trace_dir: Path | None = None,
    use_cache: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    """Run one strategy on one split, write the result JSON, and return it.

    `backend` replaces the backend named in the config (tests pass a FakeBackend).
    """
    if strategy not in STRATEGIES or split_flag not in SPLIT_FLAGS:
        raise ValueError(f"strategy must be one of {STRATEGIES}, split one of {list(SPLIT_FLAGS)}")
    held_out = split_flag in HELD_OUT_FLAGS
    # Before anything is read or written: held-out data needs a frozen gate.
    frozen = assert_frozen(gate_config_path, gate_hash_path) if held_out else None
    gate_cfg = (frozen or load_gate_config(gate_config_path)) if gate else None

    cfg = cfg or load_pfn_config()
    if backend_name is not None:
        cfg = dataclasses.replace(cfg, backend=backend_name)
    cap = cfg.max_context if max_context is None else max_context
    fc = feature_config or load_feature_config()
    backend = backend or make_backend(cfg)
    cache = PFNCache(cache_dir or results_dir / "cache" / "pfn", enabled=use_cache)
    data_ver = data_version(fc)
    model_version = getattr(backend, "model_version", cfg.model_version)
    settings = {
        "backend": backend.name,
        "model_version": model_version,
        "package_version": package_version(backend.name),
        "max_context": cap,
        "data_version": data_ver,
        "strategy": strategy,
    }
    if gate_cfg is not None and held_out:
        assert_applicable(gate_cfg, settings, oracle=strategy in ORACLE_STRATEGIES)
    fit_predict = FitPredict(backend, cache, cfg, model_version, data_ver)

    pool = build_dataset("context", seeds_per_scenario=pool_seeds_per_scenario, feature_config=fc)
    pool_distance = pool_distance_for(gate_cfg, pool) if gate_cfg is not None else None
    queries = load_queries(split_flag, fc, seeds_per_scenario)
    tracer = Tracer(
        f"{backend.name}-{strategy}-{split_flag}", trace_dir=trace_dir or results_dir / "traces"
    )

    tables, records, skipped = [], [], []
    t_start = time.perf_counter()
    for i, q in enumerate(queries, start=1):
        tid = str(q["trajectory_id"].iloc[0])
        t0 = time.perf_counter()
        try:
            sel = select_context(strategy, pool, q, cfg, max_context=cap)
        except NoMatchingFamilyError as exc:
            skipped.append({"trajectory_id": tid, "reason": str(exc)})
            continue
        tracer.log_step(
            scenario_id=tid,  # the trajectory id: scenario id and seed
            tool="select_context",
            latency_s=time.perf_counter() - t0,
            context_hash=sel.context_hash,
            context_size=sel.size,
            strategy=sel.label,
        )

        t1 = time.perf_counter()
        out, hit = fit_predict(sel.context, q)
        tracer.log_step(
            scenario_id=tid,
            tool="pfn_cache_hit" if hit else "pfn_predict",
            latency_s=time.perf_counter() - t1,
            context_hash=sel.context_hash,
            context_size=sel.size,
            strategy=sel.label,
            backend=out.metadata.backend,
            model_version=out.metadata.model_version,
        )

        tables.append(window_table(q, out))
        md = out.metadata
        gate_block = (
            gate_run(gate_cfg, fit_predict, pool_distance, tracer, sel, q, out, cfg)
            if gate_cfg is not None and pool_distance is not None
            else {}
        )
        records.append(
            {
                "trajectory_id": tid,
                "scenario_id": str(q["scenario_id"].iloc[0]),
                "family": str(q["family"].iloc[0]),
                "split": str(q["split"].iloc[0]),
                "n_windows": len(q),
                "context_size": sel.size,
                "context_hash": sel.context_hash,
                "fit_seconds": md.fit_seconds,
                "predict_seconds": md.predict_seconds,
                "latency_seconds": md.fit_seconds + md.predict_seconds,
                "cache_hit": hit,
                **gate_block,
            }
        )
        if progress:
            done = time.perf_counter() - t_start
            eta = done / i * (len(queries) - i)
            print(
                f"[{i}/{len(queries)}] {tid} ctx={sel.size} "
                f"{'cache' if hit else f'{time.perf_counter() - t1:.1f}s'} "
                f"elapsed {done / 60:.1f} min, about {eta / 60:.1f} min left",
                flush=True,
            )
    tracer.close()
    total = time.perf_counter() - t_start

    win = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    result: dict[str, Any] = {
        "method": "pfn",
        "backend": backend.name,
        "strategy": strategy,
        "oracle": strategy in ORACLE_STRATEGIES,
        "split": split_flag,
        "dataset_splits": list(SPLIT_FLAGS[split_flag]),
        "model_version": model_version,
        "package_version": package_version(backend.name),
        "device": cfg.device,
        "seed": cfg.seed,
        "critical_threshold": cfg.critical_threshold,
        "quantile_levels": list(cfg.quantile_levels),
        "max_context": cap,
        "max_context_config_default": cfg.max_context,
        "seeds_per_scenario": seeds_per_scenario,
        "bootstrap": {"n": cfg.bootstrap_n, "seed": cfg.bootstrap_seed, "ci": cfg.bootstrap_ci},
        "scores": SCORE_NOTES,
        "pool": {"n_rows": len(pool), "n_trajectories": int(pool["trajectory_id"].nunique())},
        "versions": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy")},
        "n_trajectories": len(records),
        "n_windows": len(win),
        "n_cache_hits": sum(r["cache_hit"] for r in records),
        "skipped": skipped,
        "timing": timing_block(records, total),
        "trajectories": records,
        "trace_file": tracer.path.relative_to(REPO_ROOT).as_posix()
        if tracer.path.is_relative_to(REPO_ROOT)
        else tracer.path.as_posix(),
        "frozen_gate_sha256": frozen.sha256 if frozen is not None else None,
        "gate": None,
    }
    if split_flag in CONTROL_FLAGS:
        result["false_alarms"] = false_alarm_blocks(win, cfg) if len(win) else None
    else:
        result["auroc"] = auroc_blocks(win, cfg) if len(win) else None
    if gate_cfg is not None and records:
        result["gate"] = gate_summary(records, gate_cfg, cfg)
        if split_flag in CONTROL_FLAGS:
            result["gate"]["false_alarms"] = gate_false_alarms(records)
        else:
            accepted = {r["trajectory_id"] for r in records if not r["gate"]["failed"]}
            acc = win[win["trajectory_id"].isin(accepted)]
            result["gate"]["auroc_accepted"] = auroc_blocks(acc, cfg) if len(acc) else None
    write_json(result_path(results_dir, strategy, split_flag), result)
    return result


@dataclasses.dataclass(frozen=True)
class GateStage:
    """The gate's verdict on one run, its refit, and the decision of each window."""

    result: GateResult
    refit: CalibrationRefit
    decisions: tuple[Decision, ...] | None  # None: no alert threshold for the strategy yet
    alert_threshold: float | None
    seconds: float


def gate_and_trace(
    gate_cfg: GateConfig,
    fit_predict: FitPredict,
    pool_distance: PoolDistance,
    tracer: Tracer,
    sel: ContextSelection,
    q: pd.DataFrame,
    out: PFNOutput,
    cfg: PFNConfig,
) -> GateStage:
    """Run the gate on one run and write its trace steps. Never reads the query targets.

    The agent calls this too, with a query that has no target or label columns, so both
    paths use the same gate code.
    """
    tid = str(q["trajectory_id"].iloc[0])
    t0 = time.perf_counter()
    result, refit = gate_one(
        gate_cfg, fit_predict, pool_distance, sel, q, out, seed=cfg.seed, run_id=tracer.run_id
    )
    tracer.log_step(
        scenario_id=tid,
        tool="context_calibration",
        latency_s=refit.seconds,
        context_hash=sel.context_hash,
        context_size=sel.size,
        strategy=sel.label,
        backend=out.metadata.backend,
        model_version=out.metadata.model_version,
    )
    dump = result.model_dump(mode="json")
    tracer.log_step(
        scenario_id=tid,
        tool="gate",
        latency_s=time.perf_counter() - t0 - refit.seconds,
        context_hash=sel.context_hash,
        context_size=sel.size,
        strategy=sel.label,
        gate_result=dump,
    )
    thr = gate_cfg.alert_threshold(sel.strategy)
    decisions = decide(result, out.p_critical, thr) if thr is not None else None
    return GateStage(result, refit, decisions, thr, time.perf_counter() - t0)


def gate_run(
    gate_cfg: GateConfig,
    fit_predict: FitPredict,
    pool_distance: PoolDistance,
    tracer: Tracer,
    sel: ContextSelection,
    q: pd.DataFrame,
    out: PFNOutput,
    cfg: PFNConfig,
) -> dict[str, Any]:
    """Gate one run, trace it, and return the per-trajectory fields for the result JSON."""
    stage = gate_and_trace(gate_cfg, fit_predict, pool_distance, tracer, sel, q, out, cfg)
    result, refit, decisions = stage.result, stage.refit, stage.decisions
    thr = stage.alert_threshold
    dump = result.model_dump(mode="json")
    return {
        "gate": {**dump, "failed": list(result.failed)},
        "decisions": decision_counts(decisions),
        "decisions_note": None if thr is not None else "no alert threshold for this strategy yet",
        "context_calibration": {
            "coverage": refit.coverage,
            "n_trajectories": refit.n_trajectories,
            "n_rows": refit.n_rows,
            "n_folds": refit.n_folds,
            "n_cache_hits": refit.n_cache_hits,
            "seconds": refit.seconds,
        },
        "gate_seconds": stage.seconds,
        "ground_truth": ground_truth(q, out, gate_cfg),
        # Per-window scores of control runs: the tuner sets the alert threshold from them.
        "p_critical": out.p_critical.tolist() if q["role"].iloc[0] == "control" else None,
    }


def main(argv: list[str] | None = None) -> int:
    cfg = load_pfn_config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--strategy", required=True, choices=STRATEGIES)
    p.add_argument("--split", required=True, choices=list(SPLIT_FLAGS))
    p.add_argument(
        "--max-context",
        type=int,
        default=None,
        help=f"cap on context rows (default {cfg.max_context}, from configs/pfn.yaml)",
    )
    p.add_argument("--backend", default=None, help=f"default {cfg.backend}, from configs/pfn.yaml")
    p.add_argument(
        "--seeds-per-scenario",
        type=int,
        default=None,
        help="use only the first N seeds of each scenario (default: all). Report it if used.",
    )
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--gate", action="store_true", help="run the gate on every run")
    args = p.parse_args(argv)
    if args.split in HELD_OUT_FLAGS:
        assert_committed(DEFAULT_GATE_CONFIG)  # docs/SPLITS.md: committed before C, D, real

    result = run(
        args.strategy,
        args.split,
        max_context=args.max_context,
        backend_name=args.backend,
        seeds_per_scenario=args.seeds_per_scenario,
        results_dir=args.results_dir,
        use_cache=not args.no_cache,
        gate=args.gate,
    )
    t = result["timing"]
    print(
        f"done: {result['n_trajectories']} trajectories, {result['n_windows']} windows, "
        f"{result['n_cache_hits']} cache hits, {len(result['skipped'])} skipped, "
        f"{t['total_wall_seconds'] / 60:.1f} min"
    )
    if result["gate"] is not None:
        g = result["gate"]
        print(f"gate ({g['config_status']}): {g['n_passed']} passed, {g['n_blocked']} blocked")
    print(f"wrote {result_path(args.results_dir, args.strategy, args.split)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

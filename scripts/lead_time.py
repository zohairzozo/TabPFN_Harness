"""Write results/lead_time.json: how early the PFN alarms come, at the frozen alert thresholds.

The run-level table scores one verdict at the end of the series. Early warning is a different
question: how long before the tip does the first alarm come? For each held-out tipping
trajectory this takes the first window whose p_critical is at or above the strategy's alert
threshold (the value frozen in configs/gate.yaml: nothing is tuned here) and reports the time
from that window to the tip. A trajectory with no alarm before the tip is a miss.

It also reports the share of held-out control trajectories with an alarm in any window.

It uses the same context selection, cache and backend settings as run_pfn, so with the PFN cache
present it only reads. The gate must be frozen first (assert_frozen runs before any data).

Usage: python scripts/lead_time.py [--results-dir results]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))  # run from a checkout without installing

from pfngate.evals.gating import FitPredict  # noqa: E402
from pfngate.evals.metrics import write_json  # noqa: E402
from pfngate.evals.pfn_cache import PFNCache, data_version  # noqa: E402
from pfngate.evals.run_pfn import load_queries  # noqa: E402
from pfngate.features import build_dataset, load_feature_config  # noqa: E402
from pfngate.gate import assert_frozen  # noqa: E402
from pfngate.pfn import load_pfn_config, make_backend  # noqa: E402
from pfngate.pfn.context import select_context  # noqa: E402
from pfngate.scenarios import load_scenarios  # noqa: E402

STRATEGIES = ("random", "knn")
TIPPING_FLAGS = ("test", "ood")
CONTROL_FLAG = "controls"
RULE = (
    "alarm = a window whose p_critical is at or above the strategy alert threshold frozen in "
    "configs/gate.yaml; lead time = tip time minus the end time of the first alarm window; "
    "a trajectory with no alarm before the tip is a miss"
)


def first_alarm(p_critical: np.ndarray, threshold: float) -> int | None:
    """The index of the first window at or above the threshold, or None."""
    hits = np.flatnonzero(np.asarray(p_critical, dtype=float) >= threshold)
    return int(hits[0]) if len(hits) else None


def lead_time(
    times: np.ndarray, p_critical: np.ndarray, tip_time: float, thr: float
) -> float | None:
    """Time from the first alarm window to the tip, or None if no window alarms before it."""
    i = first_alarm(p_critical, thr)
    if i is None or times[i] > tip_time:
        return None
    return float(tip_time - times[i])


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Detection share and lead times over the rows, and the same over the accepted rows."""

    def block(sub: list[dict[str, Any]]) -> dict[str, Any]:
        leads = [r["lead_time"] for r in sub if r["lead_time"] is not None]
        out: dict[str, Any] = {
            "n_runs": len(sub),
            "n_detected": len(leads),
            "detection_rate": len(leads) / len(sub) if sub else None,
        }
        if leads:
            q25, q50, q75 = np.percentile(leads, [25, 50, 75])
            out.update({"lead_median": float(q50), "lead_q25": float(q25), "lead_q75": float(q75)})
        else:
            out.update({"lead_median": None, "lead_q25": None, "lead_q75": None})
        return out

    return {
        "all_runs": block(rows),
        "gate_accepted_runs": block([r for r in rows if r["accepted"]]),
    }


def summarise_controls(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The share of control trajectories with an alarm in any window."""

    def block(sub: list[dict[str, Any]]) -> dict[str, Any]:
        n_alarm = sum(r["any_alarm"] for r in sub)
        return {
            "n_runs": len(sub),
            "n_with_alarm": n_alarm,
            "share": n_alarm / len(sub) if sub else None,
        }

    return {
        "all_runs": block(rows),
        "gate_accepted_runs": block([r for r in rows if r["accepted"]]),
    }


def accepted_ids(results_dir: Path, strategy: str, flag: str) -> set[str]:
    """Trajectory ids that the gate accepted, from the gated PFN result file."""
    path = results_dir / f"pfn_{strategy}_{flag}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run run_pfn with --gate first")
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {t["trajectory_id"] for t in doc["trajectories"] if not t["gate"]["failed"]}


def seed_of(trajectory_id: str) -> int:
    """The seed at the end of a trajectory id such as fold_L1_slow/3000."""
    return int(trajectory_id.rsplit("/", 1)[-1])


def build(results_dir: Path, cache_dir: Path | None = None) -> dict[str, Any]:
    """Compute the lead-time results for every strategy and held-out split."""
    gate_cfg = assert_frozen()  # held-out data: the gate must be frozen before anything is read
    cfg = load_pfn_config()
    fc = load_feature_config()
    backend = make_backend(cfg)
    ref = json.loads((results_dir / "pfn_random_test.json").read_text(encoding="utf-8"))
    cap, seeds = ref["max_context"], ref["seeds_per_scenario"]
    model_version = getattr(backend, "model_version", cfg.model_version)
    fit_predict = FitPredict(
        backend,
        PFNCache(cache_dir or results_dir / "cache" / "pfn", enabled=True),
        cfg,
        model_version,
        data_version(fc),
    )
    pool = build_dataset("context", feature_config=fc)
    scenarios = load_scenarios()
    out: dict[str, Any] = {
        "rule": RULE,
        "frozen_gate_sha256": gate_cfg.sha256,
        "alert_thresholds": {},
        "window_span_time": None,
        "tipping": {},
        "controls": {},
    }
    for strategy in STRATEGIES:
        thr = gate_cfg.alert_threshold(strategy)
        if thr is None:
            continue
        out["alert_thresholds"][strategy] = thr
        for flag in (*TIPPING_FLAGS, CONTROL_FLAG):
            accepted = accepted_ids(results_dir, strategy, flag)
            rows: list[dict[str, Any]] = []
            for q in load_queries(flag, fc, seeds):
                if flag == CONTROL_FLAG and q["split"].iloc[0] == "validation":
                    continue  # validation controls set the thresholds: not a held-out result
                tid = str(q["trajectory_id"].iloc[0])
                sel = select_context(strategy, pool, q, cfg, max_context=cap)
                scores, _ = fit_predict(sel.context, q)
                p, times = scores.p_critical, q["window_end_time"].to_numpy(float)
                row = {"trajectory_id": tid, "accepted": tid in accepted}
                if flag == CONTROL_FLAG:
                    row["any_alarm"] = first_alarm(p, thr) is not None
                else:
                    tr = scenarios[str(q["scenario_id"].iloc[0])].generate(seed_of(tid))
                    row["lead_time"] = lead_time(times, p, float(tr.tip_time), thr)
                    if out["window_span_time"] is None:
                        dt = float(tr.time[1] - tr.time[0])
                        out["window_span_time"] = dt * (fc.window_length - 1)
                rows.append(row)
            if flag == CONTROL_FLAG:
                out["controls"][strategy] = summarise_controls(rows)
            else:
                out["tipping"].setdefault(flag, {})[strategy] = summarise(rows)
    return out


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: write results/lead_time.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--cache-dir", type=Path, default=None, help="the PFN cache to read")
    args = parser.parse_args(argv)
    result = build(args.results_dir, args.cache_dir)
    path = args.results_dir / "lead_time.json"
    write_json(path, result)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

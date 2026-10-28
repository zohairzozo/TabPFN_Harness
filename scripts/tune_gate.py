"""Tune the gate on in-prior validation runs only, then write and freeze configs/gate.yaml.

    python scripts/tune_gate.py

Reads only results/pfn_{random,knn}_{val,val_controls}.json (gated runs made with the
provisional configs/gate.yaml). The oracle is not tuned on. Prints the precision and
recall trade-off over the grid, writes the tuned configs/gate.yaml, freezes it
(configs/gate.frozen.sha256) and writes results/gate_tuning.json. See docs/GATE.md,
"Tuning protocol".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pfngate.features.config import REPO_ROOT
from pfngate.gate.config import DEFAULT_GATE_CONFIG, DEFAULT_GATE_HASH
from pfngate.gate.tune import (
    MIN_RUNS,
    MORE_RUNS_HELP,
    TooFewRunsError,
    TuningStoppedError,
    tune,
)
from pfngate.pfn import load_pfn_config


def fmt(v: float | None) -> str:
    return "  n/a" if v is None else f"{v:5.3f}"


def print_tradeoff(record: dict[str, Any]) -> None:
    chosen = record["chosen"] or {}
    rows = sorted(record["points"], key=lambda r: (r["feasible"], r["f1"] or 0.0),
                  reverse=True)  # fmt: skip
    print("\nood_limit  cal_low  cal_high  blocked  bad_blocked  precision  recall     f1  "
          "abstain_good  feasible")  # fmt: skip
    for r in rows:
        mark = " <- chosen" if all(r[k] == chosen.get(k) for k in
                                   ("ood_share_limit", "calibration_low",
                                    "calibration_high")) else ""  # fmt: skip
        print(
            f"{r['ood_share_limit']:9.2f}  {r['calibration_low']:7.2f}  "
            f"{r['calibration_high']:8.2f}  {r['n_blocked']:7d}  {r['n_bad_blocked']:11d}  "
            f"{fmt(r['precision']):>9}  {fmt(r['recall']):>6}  {fmt(r['f1']):>5}  "
            f"{fmt(r['abstention_good']):>12}  {'yes' if r['feasible'] else 'no':>8}{mark}"
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    args = p.parse_args()
    cfg = load_pfn_config()
    out = args.results_dir / "gate_tuning.json"
    try:
        record = tune(
            args.results_dir,
            DEFAULT_GATE_CONFIG,
            DEFAULT_GATE_HASH,
            seed=cfg.seed,
            target_false_alarm_rate=cfg.target_false_alarm_rate,
            out_path=out,
        )
    except TooFewRunsError as exc:
        bar = "!" * 78
        print(f"{bar}\nNOT TUNED: {exc}.\nThe gate needs at least {MIN_RUNS} validation "
              f"runs. Nothing was written.\n\n{MORE_RUNS_HELP}\n{bar}")  # fmt: skip
        return 2
    except TuningStoppedError as exc:
        stopped = json.loads(out.read_text(encoding="utf-8"))
        if stopped.get("points"):
            print_tradeoff(stopped)
        print(f"\nNOT TUNED: {exc}. See {out}. configs/gate.yaml was not changed.")
        return 3

    print(f"{record['n_runs']} validation runs ({record['n_trajectories']} trajectories), "
          f"{record['n_bad']} BAD. By strategy and split: "
          f"{record['n_runs_by_strategy_and_split']}")  # fmt: skip
    print_tradeoff(record)
    c, o = record["chosen"], record["out_of_fold"]
    print(f"\nchosen: ood_share limit {c['ood_share_limit']}, context_calibration "
          f"[{c['calibration_low']}, {c['calibration_high']}]")  # fmt: skip
    print(f"in-sample: precision {fmt(c['precision'])}, recall {fmt(c['recall'])}, "
          f"F1 {fmt(c['f1'])}, "
          f"abstention {fmt(c['abstention_rate'])}")  # fmt: skip
    print(f"out-of-fold: precision {fmt(o['precision'])}, recall {fmt(o['recall'])} "
          f"({o['n_infeasible_folds']} infeasible folds)")  # fmt: skip
    for s, a in record["alert_thresholds"].items():
        print(f"alert threshold {s}: {a['threshold']} from {a['n_accepted_control_runs']} of "
              f"{a['n_control_runs']} accepted validation control runs")  # fmt: skip
    print(f"\nwrote {DEFAULT_GATE_CONFIG} (tuned), {DEFAULT_GATE_HASH} and {out}")
    print("Commit configs/gate.yaml and configs/gate.frozen.sha256 before any held-out run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

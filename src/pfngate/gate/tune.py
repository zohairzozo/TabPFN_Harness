"""Tune the gate thresholds on in-prior validation runs only (docs/GATE.md, Tuning protocol).

Inputs are the gated validation runs written by run_pfn: results/pfn_<strategy>_val.json
(tipping) and results/pfn_<strategy>_val_controls.json (controls), made with the
provisional configs/gate.yaml, for the deployable strategies only. The oracle
(family_matched) is not tuned on (docs/GATE.md, decision 9). No other result file is
opened. Every trajectory in them must belong to the validation split, and every run must
have been gated with the current provisional file, because the check values depend on its
pool reference.

The grid, the minimum precision, the objective and the tie-break are fixed in
docs/GATE.md and in pfngate.gate.build.TUNING_GRID. The tuner only picks a point.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pfngate.evals.gating import blocking_metrics, trajectory_folds
from pfngate.evals.metrics import alarm_threshold_for_rate, write_json
from pfngate.gate.build import TUNING_GRID, write_gate_config
from pfngate.gate.config import GateConfig, load_gate_config
from pfngate.gate.freeze import write_frozen_hash

# The round 2 tuning set: the best precision and recall of the three rounds (PROGRESS.md).
# No oracle (decision 9). knn_balanced is a context strategy, but not part of the gate.
TUNING_STRATEGIES: tuple[str, ...] = ("random", "knn")
VALIDATION_FLAGS: tuple[str, ...] = ("val", "val_controls")
MIN_RUNS = 30  # below this, refuse to tune at all
MIN_BAD = 20  # docs/GATE.md stop rule
MAX_BAD_SHARE = 0.5  # docs/GATE.md stop rule: a majority of BAD runs is a model problem
# Decision 3 set a minimum precision of 0.6. You removed it on 2026-10-28 after three
# stopped rounds: the tuner now maximises F1 with no floor (docs/GATE.md, decision 12).
MIN_PRECISION: float | None = None
CV_FOLDS = 5
FIXED_CHECKS: tuple[str, ...] = (  # blocking checks with fixed thresholds
    "min_context_size",
    "context_not_degenerate",
    "output_integrity",
)

MORE_RUNS_HELP = """\
How to get more validation runs:
  1. Validation seeds are seed set B in configs/scenarios/_common.yaml
     (B: {start: 2000, count: 10, split: validation}). Raise `count` for more seeds.
  2. Run each tuning strategy S (random, knn) on both validation
     splits with the gate, without --seeds-per-scenario (or with a larger value):
       python -m pfngate.evals.run_pfn --strategy S --split val --max-context 200 --gate
       python -m pfngate.evals.run_pfn --strategy S --split val_controls --max-context 200 --gate
  3. Run scripts/tune_gate.py again."""


class TooFewRunsError(RuntimeError):
    """Fewer validation runs than MIN_RUNS. Nothing was written."""


class TuningStoppedError(RuntimeError):
    """A docs/GATE.md stop rule fired. results/gate_tuning.json says why; no gate written."""


@dataclass(frozen=True)
class RunRecord:
    """What the tuner needs from one gated validation run."""

    strategy: str
    flag: str
    trajectory_id: str
    ood_share: float
    calibration: float | None
    fixed_failed: tuple[str, ...]
    bad: bool
    coverage_90: float
    n_windows: int
    p_critical: tuple[float, ...] | None  # control runs only


@dataclass(frozen=True)
class GridPoint:
    ood_share_limit: float
    calibration_low: float
    calibration_high: float


def grid_points() -> list[GridPoint]:
    g = TUNING_GRID
    return [
        GridPoint(a, b, c)
        for a, b, c in itertools.product(
            g["ood_share_limit"], g["calibration_low"], g["calibration_high"]
        )
    ]


def looseness(p: GridPoint) -> tuple[float, float, float]:
    """Larger is looser: a higher ood limit, a lower calibration low, a higher high."""
    return (p.ood_share_limit, -p.calibration_low, p.calibration_high)


def failed_checks(r: RunRecord, p: GridPoint) -> tuple[str, ...]:
    """The checks this run fails at grid point p, in the gate's check order."""
    failed = set(r.fixed_failed)
    if r.ood_share > p.ood_share_limit:
        failed.add("ood_share")
    if r.calibration is None or not p.calibration_low <= r.calibration <= p.calibration_high:
        failed.add("context_calibration")
    order = ("min_context_size", "ood_share", "context_calibration", "context_not_degenerate",
             "output_integrity")  # fmt: skip
    return tuple(c for c in order if c in failed)


def evaluate_point(records: Sequence[RunRecord], p: GridPoint) -> dict[str, Any]:
    failed = [failed_checks(r, p) for r in records]
    m = blocking_metrics(failed, [r.bad for r in records])
    m["abstention_rate"] = sum(bool(f) for f in failed) / len(records) if records else None
    p_, r_ = m["precision"], m["recall"]
    m["f1"] = 2 * p_ * r_ / (p_ + r_) if p_ is not None and r_ is not None and p_ + r_ else None
    return m


def choose(
    records: Sequence[RunRecord], points: Sequence[GridPoint], min_precision: float | None
) -> tuple[GridPoint | None, list[dict[str, Any]]]:
    """Highest F1 of blocking BAD runs, among points whose F1 is defined and whose precision
    reaches min_precision (no floor when None). Ties: higher precision, then lower
    abstention on good runs, then the looser point (docs/GATE.md, decision 12)."""
    rows = []
    for p in points:
        m = evaluate_point(records, p)
        feasible = m["f1"] is not None and (
            min_precision is None or m["precision"] >= min_precision
        )
        rows.append({**asdict(p), **m, "feasible": feasible})
    feasible = [(p, r) for p, r in zip(points, rows, strict=True) if r["feasible"]]
    if not feasible:
        return None, rows

    def key(item: tuple[GridPoint, dict[str, Any]]) -> tuple[Any, ...]:
        p, r = item
        return (r["f1"], r["precision"], -(r["abstention_good"] or 0.0), looseness(p))

    return max(feasible, key=key)[0], rows


def out_of_fold(
    records: Sequence[RunRecord],
    points: Sequence[GridPoint],
    min_precision: float | None,
    *,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    """Tune on 4 folds, apply to the 5th. Folds group runs by trajectory id, so the runs of
    one trajectory under different strategies stay together."""
    fold_of = trajectory_folds([r.trajectory_id for r in records], folds, seed)
    loosest = max(points, key=looseness)
    failed, bad, chosen, infeasible = [], [], [], 0
    for k in range(folds):
        train = [r for r in records if fold_of[r.trajectory_id] != k]
        test = [r for r in records if fold_of[r.trajectory_id] == k]
        p, _ = choose(train, points, min_precision)
        if p is None:
            infeasible += 1
            p = loosest
        chosen.append(asdict(p))
        failed += [failed_checks(r, p) for r in test]
        bad += [r.bad for r in test]
    m = blocking_metrics(failed, bad)
    return {
        **m,
        "folds": folds,
        "n_infeasible_folds": infeasible,
        "note": "a fold with no feasible point used the loosest point" if infeasible else None,
        "chosen_per_fold": chosen,
    }


def alert_thresholds(
    records: Sequence[RunRecord], p: GridPoint, strategies: Sequence[str], rate: float
) -> dict[str, dict[str, Any]]:
    """Per strategy, from the windows of validation control runs the gate accepts."""
    out = {}
    for s in strategies:
        controls = [r for r in records if r.strategy == s and r.flag == "val_controls"]
        accepted = [r for r in controls if not failed_checks(r, p)]
        scores = [v for r in accepted for v in (r.p_critical or ())]
        out[s] = {
            "threshold": alarm_threshold_for_rate(np.array(scores), rate) if scores else None,
            "target_window_false_alarm_rate": rate,
            "n_control_runs": len(controls),
            "n_accepted_control_runs": len(accepted),
            "n_windows": len(scores),
            "note": None if scores else "no accepted validation control run: no threshold",
        }
    return out


def _record(run: dict[str, Any], t: dict[str, Any], flag: str) -> RunRecord:
    checks = {c["name"]: c for c in t["gate"]["checks"]}
    return RunRecord(
        strategy=run["strategy"],
        flag=flag,
        trajectory_id=t["trajectory_id"],
        ood_share=float(checks["ood_share"]["value"]),
        calibration=checks["context_calibration"]["value"],
        fixed_failed=tuple(c for c in FIXED_CHECKS if not checks[c]["passed"]),
        bad=bool(t["ground_truth"]["bad"]),
        coverage_90=float(t["ground_truth"]["coverage_90"]),
        n_windows=int(t["n_windows"]),
        p_critical=tuple(t["p_critical"]) if t.get("p_critical") is not None else None,
    )


def validation_files(results_dir: Path) -> list[Path]:
    """The only files the tuner reads."""
    return [results_dir / f"pfn_{s}_{f}.json" for s in TUNING_STRATEGIES for f in VALIDATION_FLAGS]


def load_validation_runs(results_dir: Path, gate: GateConfig) -> list[RunRecord]:
    missing = [p.name for p in validation_files(results_dir) if not p.exists()]
    if missing:
        raise TooFewRunsError(f"missing gated validation runs: {', '.join(missing)}")
    a = gate.applicability
    records = []
    for path in validation_files(results_dir):
        run = json.loads(path.read_text(encoding="utf-8"))
        flag = run["split"]
        problems = []
        if flag not in VALIDATION_FLAGS or run["dataset_splits"] != ["validation"]:
            problems.append(f"split {flag} {run['dataset_splits']} is not validation only")
        if run.get("gate") is None:
            problems.append("was not run with --gate")
        elif run["gate"]["config_sha256"] != gate.sha256:
            problems.append("was gated with another configs/gate.yaml than the current one")
        if (run["backend"], run["model_version"], run["max_context"]) != (
            a.backend,
            a.model_version,
            a.max_context,
        ):
            problems.append("backend, model version or max_context differ from the gate")
        if any(t["split"] != "validation" for t in run["trajectories"]):
            problems.append("holds a trajectory outside the validation split")
        if problems:
            raise ValueError(f"{path.name}: " + "; ".join(problems))
        records += [_record(run, t, flag) for t in run["trajectories"]]
    return records


def tune(
    results_dir: Path,
    gate_config_path: Path,
    gate_hash_path: Path,
    *,
    seed: int,
    target_false_alarm_rate: float,
    out_path: Path,
) -> dict[str, Any]:
    """Run the protocol. On success write the tuned gate, its hash and the tuning record."""
    if gate_hash_path.exists():
        raise FileExistsError(f"{gate_hash_path} exists: the gate is already frozen")
    gate = load_gate_config(gate_config_path)
    if gate.status != "provisional":
        raise ValueError(f"{gate_config_path} has status '{gate.status}', expected provisional")
    records = load_validation_runs(results_dir, gate)
    if len(records) < MIN_RUNS:
        raise TooFewRunsError(f"only {len(records)} validation runs, need at least {MIN_RUNS}")

    points = grid_points()
    n_bad = sum(r.bad for r in records)
    by = {f"{s}/{f}": sum(r.strategy == s and r.flag == f for r in records)
          for s in TUNING_STRATEGIES for f in VALIDATION_FLAGS}  # fmt: skip
    record: dict[str, Any] = {
        "tuned_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "stopped",
        "reason": None,
        "input_gate_sha256": gate.sha256,
        "output_gate_sha256": None,
        "input_files": [p.name for p in validation_files(results_dir)],
        "strategies": list(TUNING_STRATEGIES),
        "n_runs": len(records),
        "n_runs_by_strategy_and_split": by,
        "n_trajectories": len({r.trajectory_id for r in records}),
        "n_bad": n_bad,
        "bad_share": n_bad / len(records),
        "rules": {
            "min_runs": MIN_RUNS,
            "min_bad": MIN_BAD,
            "max_bad_share": MAX_BAD_SHARE,
            "min_precision": MIN_PRECISION,
            "objective": "highest F1 of blocking BAD runs, no minimum precision "
            "(decision 12, 2026-10-28)",
            "tie_break": "higher precision, then lower abstention on good runs, then the "
            "looser point",
            "not_tuned_on": "the oracle (family_matched), docs/GATE.md decision 9",
            "advisory": "interval_width_cap is reported, never blocks (decision 7)",
        },
        "grid": {k: list(v) for k, v in TUNING_GRID.items()},
        "points": [],
        "chosen": None,
        "out_of_fold": None,
        "alert_thresholds": None,
        "runs": [{k: v for k, v in asdict(r).items() if k != "p_critical"} for r in records],
    }
    if n_bad < MIN_BAD or n_bad > MAX_BAD_SHARE * len(records):
        record["reason"] = (
            f"{n_bad} BAD runs out of {len(records)}: need at least {MIN_BAD} and at most "
            f"{MAX_BAD_SHARE:.0%} (docs/GATE.md stop rule)"
        )
        write_json(out_path, record)
        raise TuningStoppedError(record["reason"])

    chosen, rows = choose(records, points, MIN_PRECISION)
    record["points"] = rows
    if chosen is None:
        record["reason"] = (
            "no grid point has a defined F1"
            if MIN_PRECISION is None
            else f"no grid point reaches precision {MIN_PRECISION}"
        )
        write_json(out_path, record)
        raise TuningStoppedError(record["reason"])

    record["chosen"] = {**asdict(chosen), **evaluate_point(records, chosen)}
    record["out_of_fold"] = out_of_fold(records, points, MIN_PRECISION, folds=CV_FOLDS,
                                        seed=seed)  # fmt: skip
    alerts = alert_thresholds(records, chosen, TUNING_STRATEGIES, target_false_alarm_rate)
    record["alert_thresholds"] = alerts

    raw = yaml.safe_load(gate_config_path.read_text(encoding="utf-8"))
    raw["status"] = "tuned"
    raw["checks"]["ood_share"]["limit"] = chosen.ood_share_limit
    raw["checks"]["context_calibration"]["low"] = chosen.calibration_low
    raw["checks"]["context_calibration"]["high"] = chosen.calibration_high
    raw["alert_thresholds"] = {s: alerts[s]["threshold"] for s in raw["alert_thresholds"]}
    write_gate_config(raw, gate_config_path)
    tuned = load_gate_config(gate_config_path)  # floors and shapes checked again
    record["output_gate_sha256"] = write_frozen_hash(gate_config_path, gate_hash_path)
    assert record["output_gate_sha256"] == tuned.sha256
    record["status"] = "tuned"
    write_json(out_path, record)
    return record

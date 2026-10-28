"""Tests for the gate tuner: the objective, the stop rules, and that it reads validation only."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from pfngate.gate import assert_frozen, load_gate_config
from pfngate.gate.tune import (
    MIN_PRECISION,
    MIN_RUNS,
    TUNING_STRATEGIES,
    GridPoint,
    RunRecord,
    TooFewRunsError,
    TuningStoppedError,
    _record,
    alert_thresholds,
    choose,
    failed_checks,
    grid_points,
    out_of_fold,
    tune,
    validation_files,
)

GateFactory = Callable[..., tuple[Path, Path]]
CHECKS = ("min_context_size", "ood_share", "context_calibration", "context_not_degenerate",
          "output_integrity")  # fmt: skip


def rec(i: int, *, cal: float | None = 0.9, ood: float = 0.0, bad: bool = False,
        fixed: tuple[str, ...] = (), strategy: str = "random", flag: str = "val",
        p: tuple[float, ...] | None = None) -> RunRecord:  # fmt: skip
    return RunRecord(strategy, flag, f"s/{i}", ood, cal, fixed, bad, 0.9, 10, p)


# The objective
def test_failed_checks_follow_the_grid_point() -> None:
    p = GridPoint(0.1, 0.8, 0.99)
    assert failed_checks(rec(0), p) == ()
    assert failed_checks(rec(0, ood=0.2), p) == ("ood_share",)
    assert failed_checks(rec(0, cal=None), p) == ("context_calibration",)
    assert failed_checks(rec(0, cal=1.0), p) == ("context_calibration",)  # above high
    both = rec(0, cal=0.5, fixed=("output_integrity", "min_context_size"))
    assert failed_checks(both, p) == ("min_context_size", "context_calibration",
                                      "output_integrity")  # fmt: skip


def test_grid_is_the_pre_registered_one() -> None:
    pts = grid_points()
    assert len(pts) == 48
    assert {p.ood_share_limit for p in pts} == {0.05, 0.1, 0.2, 0.3}
    assert {p.calibration_low for p in pts} == {0.70, 0.75, 0.80, 0.85}
    assert {p.calibration_high for p in pts} == {0.97, 0.99, 1.00}


def test_highest_f1_wins_and_ties_go_to_the_loosest_point() -> None:
    runs = [rec(i, cal=0.72, bad=True) for i in range(10)]
    runs += [rec(10 + i, cal=0.83, bad=True) for i in range(5)]
    runs += [rec(20 + i) for i in range(30)]
    runs += [rec(60 + i, cal=0.82) for i in range(3)]
    p, rows = choose(runs, grid_points(), 0.6)
    # Only low = 0.85 catches all 15 BAD runs (and 3 good ones: precision 15/18).
    assert p == GridPoint(0.3, 0.85, 1.0)
    row = next(r for r in rows if (r["ood_share_limit"], r["calibration_low"],
                                   r["calibration_high"]) == (0.3, 0.85, 1.0))  # fmt: skip
    assert row["recall"] == 1.0 and row["precision"] == pytest.approx(15 / 18)


def test_an_optional_minimum_precision_still_binds() -> None:
    runs = [rec(i, cal=0.72, bad=True) for i in range(10)]
    runs += [rec(10 + i, cal=0.83, bad=True) for i in range(4)]
    runs += [rec(20 + i) for i in range(30)]
    runs += [rec(60 + i, cal=0.84) for i in range(10)]  # low = 0.85 would block these too
    p, rows = choose(runs, grid_points(), 0.6)
    # low 0.85 has recall 1 but precision 14/24 < 0.6. Next best: recall 10/14 at low
    # 0.75 or 0.80, no good run blocked; the tie goes to the looser 0.75.
    assert p is not None and p.calibration_low == 0.75
    assert all(not r["feasible"] for r in rows if r["calibration_low"] == 0.85)


def test_a_failed_width_advisory_does_not_block_and_every_run_counts() -> None:
    t = traj("s/2000", cal=0.9, bad=True)
    t["gate"]["advisories"] = [{"name": "interval_width_cap", "value": 0.9, "passed": False}]
    r = _record({"strategy": "random"}, t, "val")
    assert r.fixed_failed == () and failed_checks(r, GridPoint(0.3, 0.7, 1.0)) == ()
    runs = [rec(i, cal=0.72, bad=True) for i in range(5)] + [rec(10 + i, bad=True)
                                                             for i in range(50)]  # fmt: skip
    p, rows = choose(runs, grid_points(), 0.6)
    assert p is not None
    assert all(row["n_counted"] == 55 for row in rows)
    assert max(row["recall"] for row in rows) == pytest.approx(5 / 55)


def test_the_oracle_is_not_a_tuning_strategy() -> None:
    assert TUNING_STRATEGIES == ("random", "knn")
    assert MIN_PRECISION is None  # decision 12: no floor


def test_without_a_floor_a_low_precision_point_can_win() -> None:
    runs = [rec(i, cal=0.72, bad=True) for i in range(10)]
    runs += [rec(10 + i, cal=0.72) for i in range(20)]  # good runs that look the same
    runs += [rec(40 + i) for i in range(30)]
    p, rows = choose(runs, grid_points(), None)
    assert p is not None and p.calibration_low == 0.75  # the loosest point that blocks
    row = next(r for r in rows if r["calibration_low"] == 0.75 and r["ood_share_limit"] == 0.3
               and r["calibration_high"] == 1.0)  # fmt: skip
    assert row["precision"] == pytest.approx(1 / 3) and row["f1"] == pytest.approx(0.5)
    assert choose(runs, grid_points(), 0.6)[0] is None  # the old floor would stop here


def test_no_feasible_point_returns_none() -> None:
    runs = [rec(i, bad=True) for i in range(5)] + [rec(10 + i, cal=0.5) for i in range(20)]
    p, _ = choose(runs, grid_points(), 0.6)  # only good runs can ever be blocked
    assert p is None


def test_out_of_fold_keeps_trajectories_together() -> None:
    runs = [rec(i % 20, cal=0.72 if i % 3 == 0 else 0.9, bad=i % 3 == 0, strategy=s)
            for s in ("random", "knn") for i in range(20)]  # fmt: skip
    res = out_of_fold(runs, grid_points(), 0.6, folds=5, seed=0)
    assert len(res["chosen_per_fold"]) == 5 and res["recall"] == 1.0
    assert res["n_counted"] == len(runs)


def test_alert_threshold_uses_accepted_validation_controls_only() -> None:
    p = GridPoint(0.3, 0.7, 1.0)
    ok = rec(0, flag="val_controls", p=tuple(i / 100 for i in range(100)))
    blocked = rec(1, cal=None, flag="val_controls", p=(0.99,) * 100)
    tipping = rec(2, flag="val", p=None)
    out = alert_thresholds([ok, blocked, tipping], p, ("random", "knn"), 0.05)
    a = out["random"]
    assert a["n_control_runs"] == 2 and a["n_accepted_control_runs"] == 1
    assert a["n_windows"] == 100 and 0.94 < a["threshold"] < 0.99
    assert out["knn"]["threshold"] is None and "no accepted" in out["knn"]["note"]


# The whole protocol on synthetic result files
def traj(tid: str, *, cal: float, bad: bool, split: str = "validation",
         p: list[float] | None = None) -> dict[str, Any]:  # fmt: skip
    values = {"ood_share": 0.0, "context_calibration": cal}
    return {
        "trajectory_id": tid, "split": split, "n_windows": 50,
        "gate": {"checks": [{"name": c, "value": values.get(c, 1.0), "passed": True}
                            for c in CHECKS]},
        "ground_truth": {"bad": bad, "coverage_90": 0.6 if bad else 0.9},
        "p_critical": p,
    }  # fmt: skip


def write_runs(directory: Path, sha: str, *, n_val: int = 20, n_bad: int = 10,
               n_ctrl: int = 5, split: str = "validation") -> None:  # fmt: skip
    directory.mkdir(parents=True, exist_ok=True)
    # An oracle file gated with another config: the tuner must not even open it.
    oracle = {"gate": {"config_sha256": "0" * 64}, "trajectories": [{"split": "test_in_prior"}]}
    (directory / "pfn_family_matched_val.json").write_text(json.dumps(oracle))
    for s in ("random", "knn"):
        for flag in ("val", "val_controls"):
            if flag == "val":
                ts = [traj(f"{s}{i}/2000", cal=0.65 if i < n_bad else 0.9, bad=i < n_bad,
                           split=split) for i in range(n_val)]  # fmt: skip
            else:
                ts = [traj(f"c{s}{i}/2000", cal=0.9, bad=False,
                           p=[j / 50 for j in range(50)]) for i in range(n_ctrl)]  # fmt: skip
            run = {"strategy": s, "split": flag, "dataset_splits": ["validation"],
                   "backend": "fake", "model_version": "fake-1", "max_context": 200,
                   "gate": {"config_sha256": sha}, "trajectories": ts}  # fmt: skip
            (directory / f"pfn_{s}_{flag}.json").write_text(json.dumps(run))


@pytest.fixture()
def provisional(small_data: dict[str, Any], make_gate: GateFactory, tmp_path: Path) -> Any:
    config, digest = make_gate(tmp_path / "gate", small_data["pool"], small_data["fc"],
                               status="provisional", freeze=False)  # fmt: skip
    return config, digest, load_gate_config(config).sha256


def do_tune(tmp_path: Path, config: Path, digest: Path) -> dict[str, Any]:
    return tune(tmp_path / "results", config, digest, seed=0, target_false_alarm_rate=0.05,
                out_path=tmp_path / "results" / "gate_tuning.json")  # fmt: skip


def test_tune_writes_a_frozen_tuned_gate_and_the_record(tmp_path: Path, provisional: Any) -> None:
    config, digest, sha = provisional
    write_runs(tmp_path / "results", sha)
    record = do_tune(tmp_path, config, digest)
    gate = assert_frozen(config, digest)  # tuned, and the hash matches
    assert gate.status == "tuned" and record["output_gate_sha256"] == gate.sha256
    c = record["chosen"]
    assert (c["ood_share_limit"], c["calibration_low"], c["calibration_high"]) == (0.3, 0.7, 1.0)
    assert (gate.checks.ood_share.limit, gate.checks.context_calibration.low) == (0.3, 0.7)
    assert gate.alert_threshold("random") is not None
    saved = json.loads((tmp_path / "results" / "gate_tuning.json").read_text())
    assert saved["n_runs"] == 50 and saved["n_bad"] == 20 and saved["status"] == "tuned"
    assert saved["strategies"] == ["random", "knn"]
    assert saved["grid"]["calibration_low"] == [0.7, 0.75, 0.8, 0.85]
    assert saved["input_files"] == [p.name for p in validation_files(tmp_path / "results")]
    assert len(saved["points"]) == 48 and saved["out_of_fold"]["folds"] == 5
    with pytest.raises(FileExistsError):
        do_tune(tmp_path, config, digest)  # frozen: never tuned twice


def test_too_few_runs_writes_nothing(tmp_path: Path, provisional: Any) -> None:
    config, digest, sha = provisional
    before = config.read_bytes()
    write_runs(tmp_path / "results", sha, n_val=3, n_bad=1, n_ctrl=1)  # 8 runs
    assert 8 < MIN_RUNS
    with pytest.raises(TooFewRunsError, match="need at least 30"):
        do_tune(tmp_path, config, digest)
    assert config.read_bytes() == before and not digest.exists()
    assert not (tmp_path / "results" / "gate_tuning.json").exists()


def test_too_few_bad_runs_stops_and_says_why(tmp_path: Path, provisional: Any) -> None:
    config, digest, sha = provisional
    before = config.read_bytes()
    write_runs(tmp_path / "results", sha, n_bad=2)  # 4 BAD runs
    with pytest.raises(TuningStoppedError, match="BAD runs"):
        do_tune(tmp_path, config, digest)
    assert config.read_bytes() == before and not digest.exists()
    saved = json.loads((tmp_path / "results" / "gate_tuning.json").read_text())
    assert saved["status"] == "stopped" and saved["chosen"] is None


def test_a_majority_of_bad_runs_stops(tmp_path: Path, provisional: Any) -> None:
    config, digest, sha = provisional
    write_runs(tmp_path / "results", sha, n_bad=20, n_ctrl=1)  # 40 of 42 BAD
    with pytest.raises(TuningStoppedError, match="at most 50%"):
        do_tune(tmp_path, config, digest)


def test_tuner_refuses_runs_outside_validation(tmp_path: Path, provisional: Any) -> None:
    config, digest, sha = provisional
    write_runs(tmp_path / "results", sha, split="test_in_prior")
    with pytest.raises(ValueError, match="outside the validation split"):
        do_tune(tmp_path, config, digest)


def test_tuner_refuses_runs_gated_with_another_config(tmp_path: Path, provisional: Any) -> None:
    config, digest, _ = provisional
    write_runs(tmp_path / "results", "0" * 64)
    with pytest.raises(ValueError, match="another configs/gate.yaml"):
        do_tune(tmp_path, config, digest)


def test_tuner_reads_only_validation_files(tmp_path: Path) -> None:
    names = [p.name for p in validation_files(tmp_path)]
    assert names == ["pfn_random_val.json", "pfn_random_val_controls.json",
                     "pfn_knn_val.json", "pfn_knn_val_controls.json"]  # fmt: skip


def test_missing_validation_runs_are_reported(tmp_path: Path, provisional: Any) -> None:
    config, digest, _ = provisional
    with pytest.raises(TooFewRunsError, match="missing gated validation runs"):
        do_tune(tmp_path, config, digest)

"""Tests for scripts/make_summary.py: numbers come from the result files, flags from code."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_summary.py"


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_summary", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MS = load_module()


def au(value: float, lo: float, hi: float) -> dict[str, Any]:
    return {"auroc": value, "ci_low": lo, "ci_high": hi}


def baseline_json() -> dict[str, Any]:
    split = lambda a: {"auroc": {"tipping_only": a, "family:fold": a}}  # noqa: E731
    return {
        "splits": {
            "validation": split(au(0.58, 0.55, 0.60)),
            "test_in_prior": split(au(0.57, 0.55, 0.59)),
            "test_out_of_prior": split(au(0.52, 0.48, 0.56)),
        },
        "false_alarms_at_validation_threshold": {
            "target_window_false_alarm_rate": 0.05,
            "threshold": 0.64,
            "splits": {
                "validation": {"all_controls": {"window_rate": 0.049, "trajectory_rate": 1.0}}
            },
            "held_out_pooled": {"all_controls": {"window_rate": 0.045, "trajectory_rate": 0.97}},
        },
    }


def pfn_json(
    strategy: str, flag: str, a: dict[str, Any] | None, cap: int = 200, seeds: int | None = 3
) -> dict[str, Any]:
    run: dict[str, Any] = {
        "strategy": strategy, "split": flag, "max_context": cap, "max_context_config_default": 2000,
        "seeds_per_scenario": seeds, "n_trajectories": 5, "n_windows": 100, "n_cache_hits": 0,
        "skipped": [], "model_version": "v3.5", "package_version": "9.2.0",
        "timing": {"total_wall_seconds": 600.0},
        "trajectories": [{"latency_seconds": s} for s in (4.0, 8.0, 6.0)],
    }  # fmt: skip
    if flag == "controls":
        rates = lambda w: {"all_controls": {"window_rate": w, "trajectory_rate": 1.0}}  # noqa: E731
        block = {"splits": {"validation": rates(0.04)}, "held_out_pooled": rates(0.09)}
        run["false_alarms"] = {
            "target_window_false_alarm_rate": 0.05,
            "thresholds": {"p_critical": 0.3, "neg_median": -0.7},
            "scores": {"p_critical": block, "neg_median": block},
        }
    else:
        sub = {"p_critical": a, "neg_median": a}
        run["auroc"] = {"tipping_only": sub, "family:fold": sub}
    return run


@pytest.fixture()
def results_dir(tmp_path: Path) -> Path:
    (tmp_path / "baseline_ews.json").write_text(json.dumps(baseline_json()))
    (tmp_path / "timing_tabpfn.json").write_text(
        json.dumps({"n_test": 200, "timings": {"2000": {"predict_seconds_median": 40.8}}})
    )
    for flag, a in [("val", au(0.70, 0.65, 0.75)), ("test", au(0.55, 0.45, 0.65)),
                    ("ood", au(0.99, 0.97, 1.0)), ("controls", None)]:  # fmt: skip
        (tmp_path / f"pfn_random_{flag}.json").write_text(json.dumps(pfn_json("random", flag, a)))
    return tmp_path


def test_table_has_the_requested_rows_and_columns(results_dir: Path) -> None:
    text = MS.build_summary(MS.Results(results_dir))
    header = next(line for line in text.splitlines() if line.startswith("| Method |"))
    for col in ("in-prior val", "in-prior test", "out-of-prior", "False alarms", "latency"):
        assert col in header
    for row in ("Classic EWS baseline", "TabPFN random", "TabPFN knn",
                "TabPFN family_matched (ORACLE)"):  # fmt: skip
        assert f"| {row} |" in text


def test_numbers_come_from_the_files_and_missing_runs_say_so(results_dir: Path) -> None:
    text = MS.build_summary(MS.Results(results_dir))
    row = next(line for line in text.splitlines() if line.startswith("| TabPFN random |"))
    assert "0.70 [0.65, 0.75]" in row and "0.090" in row  # AUROC and held-out false alarms
    assert "| 6.0 |" in row  # median of the three latencies 4, 8, 6
    knn = next(line for line in text.splitlines() if line.startswith("| TabPFN knn |"))
    assert knn.count("not run") >= 4
    base = next(line for line in text.splitlines() if line.startswith("| Classic EWS baseline |"))
    assert "0.58 [0.55, 0.60]" in base and "0.045" in base and "not measured" in base


def test_flags_are_computed_from_the_numbers(results_dir: Path) -> None:
    text = MS.build_summary(MS.Results(results_dir))
    row = next(line for line in text.splitlines() if line.startswith("| TabPFN random |"))
    assert "0.55 [0.45, 0.65] †" in row  # CI includes 0.5
    assert "0.70 [0.65, 0.75] †" not in row
    assert "0.99 [0.97, 1.00] ⚠" in row  # above 0.98: suspicious
    assert "usually means leakage" in text
    assert "above the baseline (CIs do not overlap)" in text  # val: 0.65 > 0.60
    assert "not clearly different from the baseline (CIs overlap)" in text  # test


def test_reduced_runs_are_stated_with_the_measured_reason(results_dir: Path) -> None:
    text = MS.build_summary(MS.Results(results_dir))
    assert "max_context was reduced" in text and "from the default 2000 to 200" in text
    assert "41 s" in text  # 40.8 s from timing_tabpfn.json, rounded by the script
    assert "first 3 seeds" in text


def test_no_reduction_note_when_the_run_used_the_defaults(results_dir: Path) -> None:
    for flag, a in [("val", au(0.7, 0.6, 0.8)), ("controls", None)]:
        run = pfn_json("random", flag, a, cap=2000, seeds=None)
        (results_dir / f"pfn_random_{flag}.json").write_text(json.dumps(run))
    for flag in ("test", "ood"):
        (results_dir / f"pfn_random_{flag}.json").unlink()
    assert "Reduced runs" not in MS.build_summary(MS.Results(results_dir))


def test_oracle_on_out_of_prior_is_reported_as_not_applicable(results_dir: Path) -> None:
    run = pfn_json("family_matched", "ood", None)
    run.update(
        auroc=None, n_trajectories=0, skipped=[{"reason": "no pool rows in family 'x' for y"}]
    )
    (results_dir / "pfn_family_matched_ood.json").write_text(json.dumps(run))
    text = MS.build_summary(MS.Results(results_dir))
    row = next(line for line in text.splitlines() if line.startswith("| TabPFN family_matched"))
    assert "n/a (no pool rows of this family)" in row
    assert "1 trajectories skipped" in text


def test_main_writes_summary_md(results_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["make_summary.py", "--results-dir", str(results_dir)])
    MS.main()
    assert (results_dir / "summary.md").read_text(encoding="utf-8").startswith("# Results summary")


def test_latency_is_taken_from_tipping_runs_only(results_dir: Path) -> None:
    ctrl = pfn_json("random", "controls", None)
    ctrl["trajectories"] = [{"latency_seconds": 0.2}] * 20  # fast constant-target controls
    (results_dir / "pfn_random_controls.json").write_text(json.dumps(ctrl))
    text = MS.build_summary(MS.Results(results_dir))
    row = next(line for line in text.splitlines() if line.startswith("| TabPFN random |"))
    assert "| 6.0 |" in row  # still the median of the tipping runs: 4, 8, 6


def test_degenerate_control_scores_are_flagged(results_dir: Path) -> None:
    for flag, a in [("val", au(0.9, 0.8, 0.95)), ("controls", None)]:
        run = pfn_json("family_matched", flag, a)
        if flag == "controls":
            run["false_alarms"]["thresholds"]["p_critical"] = 5e-324
        (results_dir / f"pfn_family_matched_{flag}.json").write_text(json.dumps(run))
    text = MS.build_summary(MS.Results(results_dir))
    row = next(line for line in text.splitlines() if line.startswith("| TabPFN family_matched"))
    assert "0.090 ‡" in row
    assert "p_critical was exactly 0" in text
    random_row = next(line for line in text.splitlines() if line.startswith("| TabPFN random |"))
    assert "‡" not in random_row


def test_false_alarm_rate_far_above_target_is_flagged(results_dir: Path) -> None:
    run = json.loads((results_dir / "pfn_random_controls.json").read_text())
    for score in ("p_critical", "neg_median"):
        run["false_alarms"]["scores"][score]["held_out_pooled"]["all_controls"]["window_rate"] = 0.2
    (results_dir / "pfn_random_controls.json").write_text(json.dumps(run))
    text = MS.build_summary(MS.Results(results_dir))
    assert "did not transfer" in text and "0.200" in text


# TabPFN + gate
def gate_block(abstain: float, cov_acc: float, precision: float, recall: float,
               n_bad: int) -> dict[str, Any]:  # fmt: skip
    return {
        "config_status": "tuned", "n_runs": 10, "n_passed": round(10 * (1 - abstain)),
        "abstention_rate": abstain, "coverage_90_mean_all": 0.80,
        "advisories_failed": {"interval_width_cap": 7},
        "coverage_90_mean_accepted": cov_acc, "n_bad": n_bad,
        "blocking": {"precision": precision, "recall": recall},
        "blocking_ci": {"precision": [precision - 0.1, precision], "recall": [0.1, recall]},
        "auroc_accepted": {"tipping_only": {"p_critical": au(0.88, 0.80, 0.93)}},
    }  # fmt: skip


def add_gate(results_dir: Path) -> None:
    for flag, g in [("val", gate_block(0.2, 0.90, 0.9, 0.8, 4)),
                    ("test", gate_block(0.25, 0.88, 0.85, 0.7, 3)),
                    ("ood", gate_block(0.5, 0.70, 0.4, 0.3, 6))]:  # fmt: skip
        path = results_dir / f"pfn_random_{flag}.json"
        run = json.loads(path.read_text())
        run["gate"] = g
        run["trajectories"] = [
            {
                **t,
                "gate_seconds": 0.1 if i else 20.0,
                "context_calibration": {"n_cache_hits": 5 if i else 0},
            }
            for i, t in enumerate(run["trajectories"])
        ]
        path.write_text(json.dumps(run))
    path = results_dir / "pfn_random_controls.json"
    run = json.loads(path.read_text())
    run["gate"] = {"false_alarms": {"held_out_pooled": {
        "n_trajectories": 18, "abstention_rate": 0.1, "window_rate": 0.02,
        "window_rate_accepted": 0.025}}}  # fmt: skip
    path.write_text(json.dumps(run))
    tuning = {
        "status": "tuned", "n_runs": 243, "n_trajectories": 81, "n_bad": 40,
        "n_runs_by_strategy_and_split": {"random/val": 54}, "strategies": ["random"],
        "chosen": {"ood_share_limit": 0.2, "calibration_low": 0.8, "calibration_high": 1.0,
                   "near_tipping_limit": 0.3,
                   "precision": 0.9, "recall": 0.8},
        "out_of_fold": {"folds": 5, "precision": 0.7, "recall": 0.6, "n_infeasible_folds": 0},
        "points": [{}] * 48, "rules": {"objective": "highest recall", "min_precision": 0.6},
        "alert_thresholds": {"random": {"threshold": 0.31, "n_accepted_control_runs": 20,
                                        "n_control_runs": 27}},
        "output_gate_sha256": "a" * 64, "tuned_at_utc": "2026-10-28T10:00:00+00:00",
    }  # fmt: skip
    (results_dir / "gate_tuning.json").write_text(json.dumps(tuning))


def test_gate_rows_come_from_the_files(results_dir: Path) -> None:
    add_gate(results_dir)
    text = MS.build_summary(MS.Results(results_dir))
    rows = [line for line in text.splitlines() if line.startswith("| TabPFN random + gate |")]
    assert len(rows) == 4  # val, test, ood, and the controls table
    ood = next(r for r in rows if "| out-of-prior |" in r)
    assert "0.88 [0.80, 0.93]" in ood and "0.40 [0.30, 0.40]" in ood and "| 0.70 |" in ood
    assert "| 20.0 |" in ood  # gate time per run
    assert "| tuned | 10 | 5 | 0.50 | 7 |" in ood  # status, runs, accepted, abstention, wide
    assert "| tuned, in-sample |" in next(r for r in rows if "| in-prior val |" in r)
    ctrl = rows[-1]
    assert "| 18 |" in ctrl and "0.020" in ctrl and "0.025" in ctrl and "0.090" in ctrl
    knn = [line for line in text.splitlines() if line.startswith("| TabPFN knn + gate |")]
    assert all("not run" in line for line in knn)
    assert "Tuned on 243 in-prior validation runs" in text and "0.3100" in text


def test_gate_findings_are_chosen_by_code(results_dir: Path) -> None:
    add_gate(results_dir)
    text = MS.build_summary(MS.Results(results_dir))
    assert "abstention out of prior (0.50) is higher than in-prior test (0.25)" in text
    assert "recall of BAD runs out of prior is 0.30, lower than the in-sample" in text
    assert "precision out of prior is 0.40, below the tuning minimum 0.6" in text
    assert "out-of-prior: accepting only gated runs lowered mean 90% coverage" in text


def test_untuned_gate_says_so(results_dir: Path) -> None:
    text = MS.build_summary(MS.Results(results_dir))
    assert "The gate has not been tuned" in text and "No gated run to compare yet." in text


def stopped_tuning(results_dir: Path) -> None:
    points = [{"ood_share_limit": o, "calibration_low": lo, "calibration_high": 1.0,
               "near_tipping_limit": 0.5,
               "n_blocked": 10 if o == 0.05 else 8, "n_counted": 20,
               "precision": 0.2 if o == 0.05 else 0.25, "recall": 0.9}
              for o in (0.05, 0.3) for lo in (0.7, 0.85)]  # fmt: skip
    tuning = {
        "status": "stopped", "reason": "no grid point reaches precision 0.6", "n_runs": 243,
        "n_trajectories": 81, "n_bad": 24, "n_runs_by_strategy_and_split": {"random/val": 54},
        "points": points, "rules": {"min_precision": 0.6}, "chosen": None,
    }  # fmt: skip
    (results_dir / "gate_tuning.json").write_text(json.dumps(tuning))


def test_stopped_tuning_is_reported_with_what_it_saw(results_dir: Path) -> None:
    stopped_tuning(results_dir)
    text = MS.build_summary(MS.Results(results_dir))
    assert "Tuning stopped: no grid point reaches precision 0.6" in text
    assert "no held-out run was made with the gate" in text
    assert "Best precision on the grid: 0.25 (recall 0.90, 8 of 20" in text
    assert (
        "Moving ood_share_limit over its grid values, with the other thresholds held, "
        "changes" in text
    )
    assert (
        "Moving calibration_low over its grid values, with the other thresholds held, "
        "never changes" in text
    )
    knn = [line for line in text.splitlines() if line.startswith("| TabPFN knn + gate |")]
    assert knn and all("not run (gate not tuned)" in line for line in knn)

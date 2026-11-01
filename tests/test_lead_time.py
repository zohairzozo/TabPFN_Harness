"""scripts/lead_time.py: the lead-time arithmetic, its summaries, and the frozen-gate rule."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pfngate.gate.freeze import GateNotFrozenError

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lead_time.py"


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location("lead_time_script", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lead_time_script"] = mod
    spec.loader.exec_module(mod)
    return mod


LT = load_module()


def test_the_first_alarm_is_the_first_window_at_or_above_the_threshold() -> None:
    p = np.array([0.0, 0.1, 0.3, 0.9])
    assert LT.first_alarm(p, 0.3) == 2  # at the threshold counts, as in the gate
    assert LT.first_alarm(p, 0.95) is None
    assert LT.first_alarm(np.array([]), 0.1) is None


def test_lead_time_is_the_time_from_the_first_alarm_to_the_tip() -> None:
    times = np.array([10.0, 20.0, 30.0, 40.0])
    p = np.array([0.0, 0.1, 0.3, 0.9])
    assert LT.lead_time(times, p, tip_time=100.0, thr=0.2) == 70.0
    assert LT.lead_time(times, p, tip_time=100.0, thr=0.95) is None  # never alarms
    assert (
        LT.lead_time(times, p, tip_time=25.0, thr=0.2) is None
    )  # the first alarm is after the tip


def rows() -> list[dict[str, Any]]:
    return [
        {"trajectory_id": "a/1", "accepted": True, "lead_time": 10.0},
        {"trajectory_id": "a/2", "accepted": True, "lead_time": 30.0},
        {"trajectory_id": "a/3", "accepted": False, "lead_time": 50.0},
        {"trajectory_id": "a/4", "accepted": True, "lead_time": None},
    ]


def test_the_summary_counts_misses_and_reports_the_gate_accepted_runs_apart() -> None:
    out = LT.summarise(rows())
    all_runs, accepted = out["all_runs"], out["gate_accepted_runs"]
    assert (all_runs["n_runs"], all_runs["n_detected"]) == (4, 3)
    assert all_runs["detection_rate"] == 0.75 and all_runs["lead_median"] == 30.0
    assert (accepted["n_runs"], accepted["n_detected"]) == (3, 2)
    assert accepted["lead_median"] == 20.0  # the blocked run is left out


def test_a_split_with_no_detection_has_no_lead_times() -> None:
    out = LT.summarise([{"trajectory_id": "a/1", "accepted": True, "lead_time": None}])
    assert out["all_runs"]["n_detected"] == 0 and out["all_runs"]["lead_median"] is None
    assert LT.summarise([])["all_runs"]["detection_rate"] is None


def test_the_control_summary_is_the_share_with_an_alarm_in_any_window() -> None:
    controls = [
        {"trajectory_id": "c/1", "accepted": True, "any_alarm": True},
        {"trajectory_id": "c/2", "accepted": True, "any_alarm": False},
        {"trajectory_id": "c/3", "accepted": False, "any_alarm": True},
    ]
    out = LT.summarise_controls(controls)
    assert out["all_runs"] == {"n_runs": 3, "n_with_alarm": 2, "share": 2 / 3}
    assert out["gate_accepted_runs"]["n_with_alarm"] == 1


def test_accepted_ids_come_from_the_gated_result_file(tmp_path: Path) -> None:
    doc = {
        "trajectories": [
            {"trajectory_id": "a/1", "gate": {"failed": []}},
            {"trajectory_id": "a/2", "gate": {"failed": ["ood_share"]}},
        ]
    }
    (tmp_path / "pfn_knn_test.json").write_text(json.dumps(doc), encoding="utf-8")
    assert LT.accepted_ids(tmp_path, "knn", "test") == {"a/1"}
    with pytest.raises(FileNotFoundError, match="run_pfn"):
        LT.accepted_ids(tmp_path, "knn", "ood")


def test_the_gate_must_be_frozen_before_any_data_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse() -> None:
        raise GateNotFrozenError("not frozen")

    monkeypatch.setattr(LT, "assert_frozen", refuse)
    # no result files exist: if the script read anything first, the error would be a missing file
    with pytest.raises(GateNotFrozenError):
        LT.build(tmp_path)


def test_the_thresholds_are_read_from_the_frozen_gate_and_not_tuned_here() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "alert_threshold(strategy)" in text  # the frozen value
    assert "alarm_threshold_for_rate" not in text and "tune" not in text.replace("tuned", "")

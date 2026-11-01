"""scripts/sharpness.py: the climatology baseline, the pinball skill and the width comparison."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pfngate.gate.freeze import GateNotFrozenError

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sharpness.py"


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location("sharpness_script", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sharpness_script"] = mod
    spec.loader.exec_module(mod)
    return mod


SH = load_module()


def test_pinball_loss_of_a_constant_quantile() -> None:
    y = np.array([0.0, 1.0])
    assert SH.pinball(y, 0.5, 0.5) == pytest.approx(0.25)  # half the absolute error
    assert SH.pinball(y, 0.0, 0.9) == pytest.approx(0.45)  # under-prediction costs tau per unit
    assert SH.pinball(y, 1.0, 0.1) == pytest.approx(0.45)  # over-prediction costs (1 - tau)


def test_climatology_is_the_quantiles_of_the_pool_targets() -> None:
    targets = np.linspace(0.0, 1.0, 101)
    clim = SH.climatology(targets, [0.05, 0.5, 0.95])
    assert clim == pytest.approx({0.05: 0.05, 0.5: 0.5, 0.95: 0.95})
    loss = SH.climatology_loss(targets, clim)
    assert loss > 0 and SH.climatology_loss(np.full(10, 0.5), {0.5: 0.5}) == 0.0


def test_skill_is_one_minus_the_loss_ratio() -> None:
    assert SH.skill([0.1, 0.1], [0.2, 0.2]) == pytest.approx(0.5)  # half the climatology loss
    assert SH.skill([0.3], [0.2]) == pytest.approx(-0.5)  # worse than the climatology
    assert SH.skill([], [0.2]) is None and SH.skill([0.1], [0.0]) is None


def traj(tid: str, width: float, loss: float, failed: bool = False) -> dict[str, Any]:
    return {
        "trajectory_id": tid,
        "gate": {
            "failed": ["ood_share"] if failed else [],
            "advisories": [{"name": "interval_width_cap", "value": width}],
        },
        "ground_truth": {"mean_pinball_loss": loss},
    }


def test_the_width_comes_from_the_gate_advisory() -> None:
    assert SH.advisory_width(traj("a/1", 0.7, 0.1)) == 0.7
    assert SH.advisory_width({"gate": {"advisories": []}}) is None


def test_the_block_reports_width_against_climatology_and_the_accepted_runs_apart() -> None:
    runs = [traj("a/1", 0.4, 0.05), traj("a/2", 0.8, 0.10), traj("a/3", 1.2, 0.40, failed=True)]
    clim_loss = {"a/1": 0.2, "a/2": 0.2, "a/3": 0.2}
    out = SH.block(runs, clim_loss, clim_width=0.8)
    assert out["all_runs"]["n_runs"] == 3 and out["gate_accepted_runs"]["n_runs"] == 2
    assert out["all_runs"]["median_width"] == 0.8
    assert out["all_runs"]["median_width_over_climatology"] == pytest.approx(1.0)
    assert out["gate_accepted_runs"]["median_width"] == pytest.approx(0.6)
    assert out["gate_accepted_runs"]["pinball_skill"] == pytest.approx(0.625)  # 1 - 0.075 / 0.2
    assert out["all_runs"]["pinball_skill"] < out["gate_accepted_runs"]["pinball_skill"]


def test_the_gate_must_be_frozen_before_any_target_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse() -> None:
        raise GateNotFrozenError("not frozen")

    monkeypatch.setattr(SH, "assert_frozen", refuse)
    with pytest.raises(GateNotFrozenError):  # no result files exist, so nothing was read first
        SH.build(tmp_path)

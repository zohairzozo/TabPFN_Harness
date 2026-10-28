"""Tests for scripts/plot_reliability.py: pooling, the accepted subset, and file checks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_reliability.py"
LEVELS = ("0.05", "0.25", "0.5", "0.75", "0.95")


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("plot_reliability", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PR = load_module()


def traj(freq: float, n: int, failed: list[str]) -> dict[str, Any]:
    return {"n_windows": n, "gate": {"failed": failed},
            "ground_truth": {"observed_frequency": dict.fromkeys(LEVELS, freq)}}  # fmt: skip


def write(results: Path, strategy: str, flag: str, ts: list[dict[str, Any]]) -> None:
    results.mkdir(parents=True, exist_ok=True)
    run = {"gate": {"config_status": "tuned"}, "trajectories": ts}
    (results / f"pfn_{strategy}_{flag}.json").write_text(json.dumps(run))


def test_frequencies_are_pooled_by_window() -> None:
    pooled = PR.pooled_frequencies([traj(0.2, 100, []), traj(0.8, 300, [])])
    assert pooled == {float(lv): pytest.approx(0.65) for lv in LEVELS}
    assert PR.pooled_frequencies([]) is None


def test_panels_split_all_runs_from_accepted_runs(tmp_path: Path) -> None:
    for s in PR.STRATEGIES:
        write(tmp_path, s, "test", [traj(0.2, 100, []), traj(0.8, 100, ["ood_share"])])
        write(tmp_path, s, "ood", [traj(0.5, 50, ["context_calibration"])])
    data = PR.panel_data(tmp_path)
    test = data[("random", "test")]
    assert test["n_all"] == 2 and test["n_accepted"] == 1
    assert test["all"][0.5] == pytest.approx(0.5) and test["accepted"][0.5] == pytest.approx(0.2)
    assert data[("knn", "ood")]["accepted"] is None  # nothing accepted: no line, a note
    out = PR.draw(data, tmp_path / "figs" / "reliability.png")
    assert out.exists() and out.stat().st_size > 10_000


def test_missing_or_ungated_files_are_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="--gate"):
        PR.load_runs(tmp_path / "pfn_random_test.json")
    (tmp_path / "pfn_random_test.json").write_text(json.dumps({"gate": None,
                                                               "trajectories": []}))  # fmt: skip
    with pytest.raises(ValueError, match="no gate result"):
        PR.load_runs(tmp_path / "pfn_random_test.json")

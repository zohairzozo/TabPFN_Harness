from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "agent_checks.py"


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location("agent_checks", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AC = load_module()


def run(
    role: str, strategy: str | None, verdict: tuple[str, float] | None, final: Any
) -> dict[str, Any]:
    return {
        "ground_truth": {"role": role},
        "record": {
            "final": {"context_strategy": strategy} if strategy else None,
            "verdict": None
            if verdict is None
            else {"decision": verdict[0], "confidence": verdict[1]},
        },
        "decision": final,
    }


def test_counts_strategies_and_confident_alerts_on_controls() -> None:
    runs = [
        run("control", "knn", ("alert", 0.9), "alert"),
        run("control", "knn", ("alert", 0.5), {"decision": "abstain"}),
        run("control", "random", ("no_alert", 0.9), "no_alert"),
        run("control", "knn", None, "no_alert"),
        run("in_prior", "knn", ("alert", 0.99), "alert"),  # not a control: not counted
    ]
    out = AC.check_runs(runs)
    assert out["strategy_counts"] == {"knn": 4, "random": 1}
    assert out["n_control_runs"] == 4
    assert out["n_control_recommended_alert"] == 2
    assert out["n_control_confident_alert"] == 1
    assert out["n_control_without_recommendation"] == 1
    assert out["n_control_final_alert"] == 1


def test_a_run_without_a_final_has_the_strategy_none() -> None:
    assert AC.check_runs([run("control", None, None, "no_alert")])["strategy_counts"] == {"none": 1}


def test_build_reads_only_files_that_exist(tmp_path: Path) -> None:
    data = {"runs": [run("control", "knn", ("no_alert", 0.7), "no_alert")]}
    (tmp_path / "agent_test.json").write_text(json.dumps(data))
    out = AC.build(tmp_path)
    assert list(out["variants"]) == ["agent_test"]
    assert out["variants"]["agent_test"]["n_control_recommended_alert"] == 0

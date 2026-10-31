"""scripts/find_failures.py on small made-up results files: what is counted, ranked and linked."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "find_failures.py"
A = "a_false_alarms_on_controls"
B = "b_bad_runs_the_gate_let_through"
C = "c_good_runs_the_gate_blocked"
D = "d_agent_disagreed_with_gated_decision"


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location("find_failures", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["find_failures"] = mod  # a dataclass looks its module up here
    spec.loader.exec_module(mod)
    return mod


FF = load_module()


def agent_run(
    tid: str,
    *,
    role: str = "in_prior",
    decision: str = "no_alert",
    p_crit: float = 0.1,
    accepted: bool | None = True,
    coverage: float = 0.9,
    verdict: tuple[str, float] | None = None,
) -> dict[str, Any]:
    final = {"decision": decision, "last_window_p_critical": p_crit, "gate_failed": []}
    rec: dict[str, Any] = {"trajectory_id": tid, "final": final, "agent_agrees": None}
    if verdict:
        rec["verdict"] = {"decision": verdict[0], "confidence": verdict[1]}
        rec["agent_agrees"] = verdict[0] == decision
    return {
        "record": rec,
        "ground_truth": {"scenario_id": tid.split("/")[0], "role": role, "end_label": "x"},
        "decision": decision,
        "accepted": accepted,
        "pfn_quality": {"coverage_90": coverage, "bad": coverage < 0.8},
    }


def write(results: Path, name: str, doc: dict[str, Any]) -> None:
    (results / name).write_text(json.dumps(doc), encoding="utf-8")


def write_trace(results: Path, name: str, scenario_ids: list[str]) -> None:
    (results / "traces").mkdir(exist_ok=True)
    lines = [json.dumps({"scenario_id": s, "step": i}) for i, s in enumerate(scenario_ids, 1)]
    (results / "traces" / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def ids(block: dict[str, Any]) -> list[str]:
    return [e["trajectory_id"] for e in block["top"]]


def make_results(tmp_path: Path) -> Path:
    results = tmp_path / "results"
    results.mkdir()
    runs = [
        agent_run("ou_L1/1", role="control", decision="alert", p_crit=0.4),
        agent_run("ou_L1/2", role="control", decision="alert", p_crit=0.9),
        agent_run("ou_L1/3", role="control", decision="no_alert"),
        agent_run("fold_L1/1", role="in_prior", decision="alert"),  # an alert, not a control
        agent_run("fold_L1/2", accepted=True, coverage=0.5),  # bad, accepted
        agent_run("fold_L1/3", accepted=True, coverage=0.7),  # bad, accepted, milder
        agent_run("fold_L1/4", accepted=False, coverage=0.5),  # bad, blocked: fine
        agent_run("fold_L1/5", accepted=False, coverage=0.95),  # good, blocked
        agent_run("fold_L1/6", decision="alert", verdict=("no_alert", 0.9)),
        agent_run("fold_L1/7", decision="alert", verdict=("alert", 0.9)),
    ]
    write(results, "agent_test.json", {"trace": "t.jsonl", "runs": runs})
    # no gate for the naive agent: it is never in b or c
    naive = [agent_run("ou_L1/1", role="control", decision="alert", accepted=None, coverage=0.4)]
    write(results, "naive_agent_test.json", {"trace": "n.jsonl", "runs": naive})
    write_trace(results, "t.jsonl", ["ou_L1/1", "fold_L1/2", "fold_L1/2", "fold_L1/3", "fold_L1/2"])
    return results


def test_ranges_join_consecutive_lines() -> None:
    assert FF.ranges([5, 1, 2, 3, 7]) == [[1, 3], [5, 5], [7, 7]]
    assert FF.ranges([]) == []


def test_false_alarms_are_alerts_on_controls_only_and_ranked_by_strength(tmp_path: Path) -> None:
    block = FF.build(make_results(tmp_path), top=5, include_val=False)["categories"][A]
    assert block["n_failures"] == 3  # agent ou_L1/1, agent ou_L1/2, naive ou_L1/1
    assert block["failures_by_variant"] == {"agent": 2, "naive_agent": 1}
    assert block["n_runs_counted_out_of"] == 4  # every control run
    top = block["top"][0]
    assert top["trajectory_id"] == "ou_L1/2" and top["severity"] == 0.9
    assert top["seed"] == 2 and top["scenario_id"] == "ou_L1"


def test_bad_runs_let_through_are_ranked_by_the_missing_coverage(tmp_path: Path) -> None:
    block = FF.build(make_results(tmp_path), top=5, include_val=False)["categories"][B]
    assert ids(block) == ["fold_L1/2", "fold_L1/3"]
    assert round(block["top"][0]["severity"], 6) == 0.3  # 0.8 minus 0.5
    # the naive agent has no gate, so its bad run is not counted
    assert block["failures_by_variant"] == {"agent": 2}
    assert block["n_runs_counted_out_of"] == 3  # three bad runs with a gate decision


def test_good_runs_blocked_are_counted(tmp_path: Path) -> None:
    block = FF.build(make_results(tmp_path), top=5, include_val=False)["categories"][C]
    assert ids(block) == ["fold_L1/5"]
    assert round(block["top"][0]["severity"], 6) == 0.15  # 0.95 minus 0.8


def test_disagreements_use_the_agents_confidence_and_skip_runs_without_a_verdict(
    tmp_path: Path,
) -> None:
    report = FF.build(make_results(tmp_path), top=5, include_val=False)
    block = report["categories"][D]
    assert block["n_failures"] == 1 and ids(block) == ["fold_L1/6"]
    assert block["top"][0]["severity"] == 0.9
    assert block["n_runs_counted_out_of"] == 2  # only the two runs with a recommendation
    assert "Agent runs with no recommendation: 8" in report["overviews"][D]


def test_each_example_links_to_trace_lines(tmp_path: Path) -> None:
    bad = FF.build(make_results(tmp_path), top=5, include_val=False)["categories"][B]["top"][0]
    assert bad["trace_file"] == "results/traces/t.jsonl"
    assert bad["lines"] == [[2, 3], [5, 5]]  # fold_L1/2 is on lines 2, 3 and 5


def test_top_limits_the_examples_and_ties_break_the_same_way_every_time(tmp_path: Path) -> None:
    results = make_results(tmp_path)
    runs = [
        agent_run(f"ou_L1/{i}", role="control", decision="alert", p_crit=0.5) for i in (3, 1, 2)
    ]
    write(results, "agent_ood.json", {"trace": None, "runs": runs})
    two = FF.build(results, top=2, include_val=False)["categories"][A]
    assert len(two["top"]) == 2
    everything = FF.build(results, top=9, include_val=False)["categories"][A]["top"]
    ties = [e for e in everything if e["severity"] == 0.5]
    assert [(e["split"], e["trajectory_id"]) for e in ties] == [
        ("ood", "ou_L1/1"),
        ("ood", "ou_L1/2"),
        ("ood", "ou_L1/3"),
    ]
    assert FF.build(results, top=9, include_val=False)["categories"][A]["top"] == everything


def test_validation_is_left_out_unless_asked_for(tmp_path: Path) -> None:
    results = make_results(tmp_path)
    val = [agent_run("ou_L1/9", role="control", decision="alert", p_crit=0.99)]
    write(results, "agent_val.json", {"trace": None, "runs": val})
    held_out = FF.build(results, top=5, include_val=False)
    with_val = FF.build(results, top=5, include_val=True)
    assert "ou_L1/9" not in ids(held_out["categories"][A])
    assert ids(with_val["categories"][A])[0] == "ou_L1/9"
    assert held_out["scope"]["includes_validation"] is False


def test_pfn_runs_with_the_gate_count_in_b_and_c(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()

    def traj(tid: str, split: str, passed: bool, coverage: float) -> dict[str, Any]:
        return {
            "trajectory_id": tid,
            "scenario_id": tid.split("/")[0],
            "split": split,
            "n_windows": 10,
            "context_size": 200,
            "gate": {"passed": passed, "failed": [] if passed else ["ood_share"], "reasons": []},
            "ground_truth": {"coverage_90": coverage, "bad": coverage < 0.8},
            "decisions": {"alert": 0, "no_alert": 10, "abstain": 0},
        }

    doc = {
        "trace_file": "results/traces/p.jsonl",
        "trajectories": [
            traj("fold_L1/1", "test_in_prior", True, 0.4),  # bad, accepted
            traj("fold_L1/2", "test_in_prior", False, 0.9),  # good, blocked
            traj("fold_L1/3", "validation", True, 0.1),  # validation: left out
        ],
    }
    write(results, "pfn_knn_test.json", doc)
    write_trace(results, "p.jsonl", ["fold_L1/1", "fold_L1/2"])
    report = FF.build(results, top=5, include_val=False)
    b, c = report["categories"][B], report["categories"][C]
    assert ids(b) == ["fold_L1/1"]
    assert b["top"][0]["variant"] == "pfn_knn+gate" and b["top"][0]["lines"] == [[1, 1]]
    assert ids(c) == ["fold_L1/2"]
    assert c["top"][0]["numbers"]["gate_failed"] == ["ood_share"]


def test_main_writes_the_file_and_prints_four_overviews(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    results = make_results(tmp_path)
    monkeypatch.setattr("sys.argv", ["find_failures", "--results-dir", str(results)])
    assert FF.main() == 0
    out = capsys.readouterr().out
    for title in FF.CATEGORIES.values():
        assert title in out
    written = json.loads((results / "failures.json").read_text(encoding="utf-8"))
    assert list(written["categories"]) == list(FF.CATEGORIES)
    assert all(len(b["top"]) <= 5 for b in written["categories"].values())

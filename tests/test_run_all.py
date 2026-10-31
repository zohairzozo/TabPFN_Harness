"""scripts/run_all.py: how stages are built and how the LLM mode and the daily cap are handled.
The tests replace the subprocess call, so nothing is run."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_all.py"


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_all", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RA = load_module()


def make_run(tmp_path: Path, **flags: Any) -> Any:
    args = argparse.Namespace(
        fast=flags.get("fast", False),
        all=flags.get("all", False),
        variants=flags.get("variants", list(RA.VARIANTS)),
        splits=flags.get("splits", list(RA.SPLITS)),
        skip_checks=True,
    )
    run = RA.Run(args)
    run.llm_cache = tmp_path / "llm"
    run.log_dir = tmp_path / "logs"
    return run


def fake_command(outputs: list[tuple[int, str]], calls: list[list[str]]) -> Any:
    def run_command(run: Any, name: str, cmd: list[str], *, mode: str | None = None) -> Any:
        calls.append(cmd)
        code, text = outputs.pop(0)
        run.stages.append({"stage": name, "exit_code": code, "llm_mode": mode, "seconds": 0.0})
        return subprocess.CompletedProcess(cmd, code, stdout="", stderr=text)

    return run_command


def mode_of(cmd: list[str]) -> str:
    return cmd[cmd.index("--llm-mode") + 1]


def test_the_profiles(tmp_path: Path) -> None:
    default = make_run(tmp_path)
    fast = make_run(tmp_path, fast=True)
    full = make_run(tmp_path, all=True)
    assert (default.limit, default.seeds, default.profile) == (3, 3, "default")
    assert (fast.limit, fast.seeds, fast.profile) == (5, 1, "fast")
    assert (full.limit, full.seeds) == (None, None)
    assert fast.results_dir.name == "fast" and default.results_dir.name == "results"


def test_every_variant_runs_on_every_split() -> None:
    assert set(RA.SPLITS) == {"val", "test", "ood", "real", "real_control"}
    assert set(RA.VARIANTS) == {"classic", "llm_only", "naive", "agent", "oracle"}


def test_an_llm_stage_gets_the_mode_the_cache_and_the_limits(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    cmd = RA.variant_command(run, "agent", "real", "replay")
    assert cmd[cmd.index("--split") + 1] == "real" and mode_of(cmd) == "replay"
    assert cmd[cmd.index("--variant") + 1] == "agent" and "--max-context" in cmd
    assert cmd[cmd.index("--limit") + 1] == "3"
    assert cmd[cmd.index("--seeds-per-scenario") + 1] == "3"
    naive = RA.variant_command(run, "naive", "val", "record")
    assert naive[naive.index("--variant") + 1] == "naive"
    classic = RA.variant_command(run, "classic", "val", None)
    assert "--llm-mode" not in classic and classic[classic.index("--variant") + 1] == "classic"
    unlimited = RA.variant_command(make_run(tmp_path, all=True), "oracle", "ood", None)
    assert "--limit" not in unlimited


def test_replay_is_used_when_the_cache_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = make_run(tmp_path)
    run.llm_cache.mkdir()
    (run.llm_cache / "x.json").write_text("{}")
    calls: list[list[str]] = []
    monkeypatch.setattr(RA, "run_command", fake_command([(0, "")], calls))
    RA.run_variant(run, "agent", "test")
    assert len(calls) == 1 and mode_of(calls[0]) == "replay"


def test_record_is_used_when_there_is_no_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = make_run(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(RA, "run_command", fake_command([(0, "")], calls))
    RA.run_variant(run, "llm_only", "val")
    assert mode_of(calls[0]) == "record"


def test_a_replay_miss_falls_back_to_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = make_run(tmp_path)
    run.llm_cache.mkdir()
    (run.llm_cache / "x.json").write_text("{}")
    calls: list[list[str]] = []
    outputs = [(1, "pfngate.agent.replay.ReplayMissError: no recorded LLM response"), (0, "")]
    monkeypatch.setattr(RA, "run_command", fake_command(outputs, calls))
    RA.run_variant(run, "agent", "val")
    assert [mode_of(c) for c in calls] == ["replay", "record"]


def test_the_daily_cap_is_not_retried_and_skips_what_is_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = make_run(tmp_path)
    calls: list[list[str]] = []
    cap = "ModelRateLimitError: 429 Rate limit exceeded: free-models-per-day"
    monkeypatch.setattr(RA, "run_command", fake_command([(1, cap)], calls))
    RA.run_variant(run, "agent", "val")
    assert len(calls) == 1 and run.quota_out
    RA.run_variant(run, "naive", "val")  # nothing is called: the cap was reached
    assert len(calls) == 1 and run.stages[-1].get("skipped")


def test_a_timeout_is_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = make_run(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(RA, "run_command", fake_command([(1, "504 timeout"), (0, "")], calls))
    monkeypatch.setattr(RA.time, "sleep", lambda s: None)
    RA.run_variant(run, "llm_only", "val")
    assert len(calls) == 2 and not run.quota_out


def test_percentiles_and_pfn_calls(tmp_path: Path) -> None:
    assert RA.percentiles([]) is None
    assert RA.percentiles([1.0, 2.0, 3.0, 4.0, 5.0]) == {"n": 5, "p50": 3.0, "p95": 4.8}
    trace = tmp_path / "t.jsonl"
    steps = [
        ("a/1", "agent_tool:run_pfn"),
        ("a/1", "agent_tool:run_pfn"),
        ("a/1", "pfn_cache_hit"),
        ("a/1", "gate"),
        ("b/1", "pfn_cache_hit"),
    ]
    trace.write_text("\n".join(json.dumps({"scenario_id": s, "tool": t}) for s, t in steps))
    calls = RA.pfn_calls_per_run(trace)
    assert calls["a/1"] == {"agent": 2, "all": 3} and calls["b/1"] == {"agent": 0, "all": 1}
    assert RA.pfn_calls_per_run(tmp_path / "missing.jsonl") == {}


def test_the_gate_stage_reports_a_frozen_gate(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    info = RA.gate_stage(run)
    assert info["frozen"] is True and len(info["sha256"]) == 64
    assert run.stages[-1]["stage"] == "gate_frozen" and run.stages[-1]["exit_code"] == 0


def test_a_replay_miss_that_record_filled_is_not_a_failure() -> None:
    ok = {"stage": "a", "exit_code": 0, "seconds": 2.0}
    miss = {"stage": "a", "exit_code": 1, "seconds": 1.0}
    assert not RA.any_stage_failed([miss, ok])
    assert RA.any_stage_failed([ok, miss])
    assert RA.any_stage_failed([{"stage": "b", "exit_code": 1, "seconds": 0.0}])
    assert RA.seconds_by_stage([miss, ok]) == {"a": 3.0}

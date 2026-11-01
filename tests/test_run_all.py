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
    run.blocker, run.blocker_checked = None, True  # recording is possible unless a test says not
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
    assert (fast.limit, fast.seeds, fast.profile) == (3, 1, "fast")
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


def test_the_summary_stage_runs_the_summary_the_checks_and_the_failure_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    named: list[str] = []

    def record(run: Any, name: str, cmd: list[str], *, mode: str | None = None) -> Any:
        named.append(name)
        run.stages.append({"stage": name, "exit_code": 0, "llm_mode": mode, "seconds": 0.0})
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(RA, "run_command", record)
    monkeypatch.setattr(RA, "check_stage", lambda run: None)
    monkeypatch.setattr(RA, "gate_stage", lambda run: {"frozen": True})
    monkeypatch.setattr(RA, "prepare_results_dir", lambda run: None)
    monkeypatch.setattr(RA, "run_variant", lambda run, variant, split: None)
    monkeypatch.setattr(RA, "pfn_cache_has_entries", lambda: True)
    monkeypatch.setattr(RA, "write_manifest", lambda run, gate, failed: tmp_path / "m.json")
    monkeypatch.setattr("sys.argv", ["run_all", "--skip-checks"])
    assert RA.main() == 0
    assert named == [
        "lead_time",
        "sharpness",
        "make_summary",
        "agent_checks",
        "find_failures",
    ]


# Offline and missing-key behaviour (audit M6): fail at once, with the reason, instead of spinning


def test_without_a_network_a_missing_answer_is_skipped_at_once_and_the_run_is_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = make_run(tmp_path)
    run.blocker = "there is no network"
    calls: list[list[str]] = []
    monkeypatch.setattr(RA, "run_command", fake_command([], calls))
    monkeypatch.setattr(RA.time, "sleep", lambda s: pytest.fail("must not wait"))
    RA.run_variant(run, "agent", "val")  # no cache: it would have to record
    assert calls == [] and run.stages[-1].get("skipped")
    assert "there is no network" in run.stages[-1]["skipped"]
    assert "Record it once online" in run.incomplete[0]


def test_a_replay_miss_offline_does_not_fall_back_to_a_record_that_cannot_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = make_run(tmp_path)
    run.blocker = "there is no network"
    run.llm_cache.mkdir()
    (run.llm_cache / "x.json").write_text("{}")
    calls: list[list[str]] = []
    outputs = [(1, "pfngate.agent.replay.ReplayMissError: no recorded LLM response")]
    monkeypatch.setattr(RA, "run_command", fake_command(outputs, calls))
    RA.run_variant(run, "agent", "val")
    assert [mode_of(c) for c in calls] == ["replay"] and run.incomplete


def test_a_network_error_while_recording_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = make_run(tmp_path)
    calls: list[list[str]] = []
    error = "httpx.ConnectError: [Errno 11001] getaddrinfo failed"
    monkeypatch.setattr(RA, "run_command", fake_command([(1, error)], calls))
    monkeypatch.setattr(RA.time, "sleep", lambda s: pytest.fail("must not wait"))
    RA.run_variant(run, "llm_only", "val")
    assert len(calls) == 1 and "network went away" in run.incomplete[0]
    RA.run_variant(run, "naive", "val")  # the next stage is skipped, not tried
    assert len(calls) == 1 and run.stages[-1].get("skipped")


def test_the_recording_blocker_names_the_missing_network_or_key_and_is_asked_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probes: list[int] = []

    def offline() -> bool:
        probes.append(1)
        return False

    run = RA.Run(
        argparse.Namespace(fast=False, all=False, variants=[], splits=[], skip_checks=True)
    )
    monkeypatch.setattr(RA, "network_available", offline)
    assert RA.recording_blocker(run) == "there is no network"
    assert RA.recording_blocker(run) == "there is no network" and len(probes) == 1

    run = RA.Run(
        argparse.Namespace(fast=False, all=False, variants=[], splits=[], skip_checks=True)
    )
    monkeypatch.setattr(RA, "network_available", lambda: True)
    monkeypatch.setattr(RA, "have_llm_key", lambda: False)
    assert RA.recording_blocker(run) == "OPENROUTER_API_KEY is not set"

    run = RA.Run(
        argparse.Namespace(fast=False, all=False, variants=[], splits=[], skip_checks=True)
    )
    monkeypatch.setattr(RA, "have_llm_key", lambda: True)
    assert RA.recording_blocker(run) is None


def test_the_key_is_looked_for_by_name_without_reading_its_value_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(RA, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert RA.have_llm_key() is False  # no .env, no variable
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=\nOTHER=1\n", encoding="utf-8")
    assert RA.have_llm_key() is False  # named, but empty
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=not-a-real-value\n", encoding="utf-8")
    assert RA.have_llm_key() is True
    (tmp_path / ".env").unlink()
    monkeypatch.setenv("OPENROUTER_API_KEY", "not-a-real-value")
    assert RA.have_llm_key() is True


def test_an_incomplete_run_exits_with_one_and_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def blocked(run: Any, variant: str, split: str) -> None:
        run.incomplete.append(f"{variant}_{split}: there is no network")

    monkeypatch.setattr(RA, "run_command", fake_command([(0, "")] * 20, []))
    monkeypatch.setattr(RA, "check_stage", lambda run: None)
    monkeypatch.setattr(RA, "gate_stage", lambda run: {"frozen": True})
    monkeypatch.setattr(RA, "prepare_results_dir", lambda run: None)
    monkeypatch.setattr(RA, "run_variant", blocked)
    monkeypatch.setattr(RA, "pfn_cache_has_entries", lambda: True)
    monkeypatch.setattr(RA, "write_manifest", lambda run, gate, failed: tmp_path / "m.json")
    monkeypatch.setattr(
        "sys.argv", ["run_all", "--skip-checks", "--variants", "agent", "--splits", "val"]
    )
    assert RA.main() == 1
    assert "incomplete: agent_val: there is no network" in capsys.readouterr().out


def test_the_lead_time_analysis_is_skipped_with_a_reason_when_the_pfn_cache_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = make_run(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(RA, "run_command", fake_command([(0, "")], calls))
    monkeypatch.setattr(RA, "pfn_cache_has_entries", lambda: False)
    RA.analysis_stages(run)
    assert len(calls) == 1 and "sharpness.py" in " ".join(calls[0])  # only sharpness ran
    assert any("would recompute every window" in r for r in run.incomplete)
    monkeypatch.setattr(RA, "pfn_cache_has_entries", lambda: True)
    more: list[list[str]] = []
    monkeypatch.setattr(RA, "run_command", fake_command([(0, ""), (0, "")], more))
    RA.analysis_stages(make_run(tmp_path))
    assert "lead_time.py" in " ".join(more[0]) and "--cache-dir" in more[0]
    assert "sharpness.py" in " ".join(more[1])


def test_the_fast_profile_is_a_subset_of_the_recorded_default_sample() -> None:
    fast, default = RA.PROFILES["fast"], RA.PROFILES["default"]
    # fewer or equal scenarios and seeds, so every fast request was recorded by the default run
    assert fast["limit"] <= default["limit"] and fast["seeds"] <= default["seeds"]

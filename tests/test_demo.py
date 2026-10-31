"""scripts/demo.py: how it picks its two runs, and that it runs offline in replay mode.

The offline test runs the whole demo against the real results and caches that `make full`
leaves behind, with every socket call blocked. It is skipped when they are not on this machine.
"""

from __future__ import annotations

import importlib.util
import socket
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "demo.py"
REAL_RESULTS = ROOT / "results"


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("demo_script", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["demo_script"] = mod
    spec.loader.exec_module(mod)
    return mod


DEMO = load_module()


def run(accepted: bool | None, decision: str, label: str, bad: bool | None) -> dict[str, Any]:
    quality = None if bad is None else {"bad": bad, "coverage_90": 0.5}
    return {
        "record": {"trajectory_id": "x/1"},
        "ground_truth": {"end_label": label},
        "accepted": accepted,
        "decision": decision,
        "pfn_quality": quality,
    }


def write_runs(results: Path, split: str, runs: list[dict[str, Any]]) -> None:
    import json

    (results / f"agent_{split}.json").write_text(json.dumps({"runs": runs}), encoding="utf-8")


# The rules


def test_rule_one_is_an_accepted_correct_alert() -> None:
    rule = DEMO.is_correct_accepted_alert
    assert rule(run(True, "alert", "positive", False))
    assert not rule(run(True, "alert", "negative", False))  # a false alarm
    assert not rule(run(True, "no_alert", "positive", False))
    assert not rule(run(False, "abstain", "positive", False))  # the gate blocked it


def test_rule_two_is_a_bad_run_the_gate_blocked() -> None:
    rule = DEMO.is_blocked_bad_run
    assert rule(run(False, "abstain", "positive", True))
    assert not rule(run(True, "alert", "positive", True))  # bad but let through
    assert not rule(run(False, "abstain", "positive", False))  # blocked but not bad
    assert not rule(run(False, "abstain", "positive", None))  # no PFN output


def test_the_search_takes_the_first_match_and_moves_on_to_the_next_split(tmp_path: Path) -> None:
    write_runs(
        tmp_path,
        "test",
        [run(True, "alert", "positive", True), run(True, "alert", "positive", False)],
    )
    write_runs(tmp_path, "ood", [run(False, "abstain", "positive", True)])
    first = DEMO.pick(tmp_path, DEMO.is_correct_accepted_alert)
    assert first["split"] == "test" and first["recorded"] is not None
    second = DEMO.pick(tmp_path, DEMO.is_blocked_bad_run)
    assert second["split"] == "ood"  # nothing in test satisfies it, so the search went on
    write_runs(tmp_path, "ood", [run(True, "alert", "positive", False)])
    assert DEMO.pick(tmp_path, DEMO.is_blocked_bad_run) is None


def test_the_two_rules_are_the_ones_the_demo_prints() -> None:
    labels = [label for label, _, _ in DEMO.RULES]
    assert labels == ["passed", "blocked"]


# Failing safely


def test_without_caches_the_demo_stops_and_leaves_the_last_transcript(tmp_path: Path) -> None:
    transcript = tmp_path / "demo_transcript.txt"
    transcript.write_text("the last good transcript\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="make full"):
        DEMO.main(["--results-dir", str(tmp_path), "--transcript", str(transcript)])
    assert transcript.read_text(encoding="utf-8") == "the last good transcript\n"


def test_a_run_that_does_not_check_out_is_saved_apart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = tmp_path / "demo_transcript.txt"
    transcript.write_text("the last good transcript\n", encoding="utf-8")
    monkeypatch.setattr(DEMO, "run_demo", lambda results, out, say: say("partial") or False)
    code = DEMO.main(
        [
            "--results-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "o"),
            "--transcript",
            str(transcript),
        ]
    )
    assert code == 1
    assert transcript.read_text(encoding="utf-8") == "the last good transcript\n"
    assert (tmp_path / "demo_transcript.txt.failed").read_text(encoding="utf-8") == "partial\n"


# Offline, in replay mode


LOOPBACK = {"127.0.0.1", "::1", "localhost", ""}


def block_sockets(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Refuse every socket call that leaves this machine and record each attempt.

    Loopback is allowed: on Windows the event loop that runs the agent opens a socket pair to
    itself, which never touches the network.
    """
    attempts: list[str] = []

    def host_of(target: Any) -> Any:
        return target[0] if isinstance(target, tuple) else target

    def guard(name: str, original: Any, host_arg: int) -> Any:
        def guarded(*args: Any, **kwargs: Any) -> Any:
            host = host_of(args[host_arg]) if len(args) > host_arg else None
            if host in LOOPBACK:
                return original(*args, **kwargs)
            attempts.append(f"{name} {host}")
            raise OSError(f"network disabled in this test: {name} {host}")

        return guarded

    monkeypatch.setattr(socket.socket, "connect", guard("connect", socket.socket.connect, 1))
    monkeypatch.setattr(
        socket.socket, "connect_ex", guard("connect_ex", socket.socket.connect_ex, 1)
    )
    monkeypatch.setattr(
        socket, "create_connection", guard("create_connection", socket.create_connection, 0)
    )
    monkeypatch.setattr(socket, "getaddrinfo", guard("getaddrinfo", socket.getaddrinfo, 0))
    monkeypatch.setattr(socket, "gethostbyname", guard("gethostbyname", socket.gethostbyname, 0))
    return attempts


def test_the_demo_runs_in_replay_mode_with_sockets_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    needed = (
        REAL_RESULTS / "agent_test.json",
        REAL_RESULTS / "agent_ood.json",
        REAL_RESULTS / "summary.md",
        REAL_RESULTS / "cache" / "llm",
        REAL_RESULTS / "cache" / "pfn",
    )
    if not all(p.exists() for p in needed):
        pytest.skip("no recorded results and caches on this machine: run make full first")
    attempts = block_sockets(monkeypatch)
    calls: list[dict[str, Any]] = []
    real_evaluate = DEMO.evaluate

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(DEMO, "evaluate", spy)
    transcript = tmp_path / "demo_transcript.txt"
    code = DEMO.main(
        [
            "--results-dir",
            str(REAL_RESULTS),
            "--out-dir",
            str(tmp_path / "demo"),
            "--transcript",
            str(transcript),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert attempts == []  # nothing even tried to reach the network
    assert len(calls) == 2 and all(c["llm_mode"] == "replay" for c in calls)
    assert [c["trajectory_ids"] for c in calls] == [
        [DEMO.pick(REAL_RESULTS, DEMO.RULES[0][2])["recorded"]["record"]["trajectory_id"]],
        [DEMO.pick(REAL_RESULTS, DEMO.RULES[1][2])["recorded"]["record"]["trajectory_id"]],
    ]
    saved = transcript.read_text(encoding="utf-8")
    assert saved.strip() in out.strip()  # the transcript holds what was printed
    for part in ("Rule for the run the gate passed", "Rule for the run the gate blocked"):
        assert part in saved
    for step in ("1. The scenario", "2. What the agent asked", "5. The gate", "6. The final"):
        assert saved.count(step) == 2
    assert "Gate result: PASSED" in saved and "Gate result: BLOCKED" in saved
    assert "Headline row of the results table" in saved and "Full harness" in saved
    assert "index.html" in saved
    assert "as the recorded run: yes" in saved and "NO." not in saved


def test_the_socket_block_really_refuses_and_records_a_network_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = block_sockets(monkeypatch)
    with pytest.raises(OSError, match="network disabled"):
        socket.create_connection(("example.com", 443), timeout=1)
    with pytest.raises(OSError, match="network disabled"):
        socket.getaddrinfo("openrouter.ai", 443)
    assert attempts == ["create_connection example.com", "getaddrinfo openrouter.ai"]

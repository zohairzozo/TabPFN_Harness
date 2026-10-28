"""Tests for the JSONL tracer in src/pfngate/evals/trace.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pfngate.evals.trace import DEFAULT_TRACE_DIR, TRACE_FIELDS, Tracer, new_run_id, read_trace
from pfngate.features import FEATURE_NAMES
from pfngate.pfn import FakeBackend, load_pfn_config, select_context
from pfngate.pfn.context import context_xy, query_x

STEP = dict(scenario_id="fold_L2_slow", tool="pfn_predict", latency_s=0.5)


def test_fields_and_order_match_the_spec() -> None:
    assert TRACE_FIELDS == (
        "run_id", "variant", "scenario_id", "step", "tool", "context_hash", "context_size",
        "strategy", "latency_s", "tokens_in", "tokens_out", "cost_usd", "backend",
        "model_version", "gate_result",
    )  # fmt: skip


def test_one_line_per_step_with_all_fields_and_nulls(tmp_path: Path) -> None:
    with Tracer("random", run_id="run-a", trace_dir=tmp_path) as t:
        t.log_step(**STEP)
        t.log_step(**STEP, context_hash="abc", context_size=10, strategy="knn")
    lines = (tmp_path / "run-a.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first, second = (json.loads(x) for x in lines)
    assert tuple(first) == TRACE_FIELDS  # same fields, same order
    assert (first["run_id"], first["variant"], first["step"]) == ("run-a", "random", 1)
    assert second["step"] == 2 and second["context_size"] == 10 and second["strategy"] == "knn"
    for key in ("context_hash", "tokens_in", "tokens_out", "cost_usd", "backend", "gate_result"):
        assert first[key] is None, key


def test_every_write_is_flushed_before_close(tmp_path: Path) -> None:
    t = Tracer("v", run_id="run-b", trace_dir=tmp_path)
    t.log_step(**STEP)
    assert len(read_trace(t.path)) == 1  # readable while the tracer is still open
    t.log_step(**STEP)
    assert len(read_trace(t.path)) == 2
    t.close()


def test_file_is_new_per_run_and_append_only(tmp_path: Path) -> None:
    Tracer("v", run_id="run-c", trace_dir=tmp_path).close()
    with pytest.raises(FileExistsError):
        Tracer("v", run_id="run-c", trace_dir=tmp_path)
    t = Tracer("v", run_id="run-d", trace_dir=tmp_path)
    t.log_step(**STEP)
    t.close()
    with pytest.raises(ValueError, match="closed"):
        t.log_step(**STEP)
    assert len(read_trace(t.path)) == 1


def test_one_file_per_run_and_generated_ids_are_unique(tmp_path: Path) -> None:
    a, b = Tracer("knn", trace_dir=tmp_path), Tracer("knn", trace_dir=tmp_path)
    assert a.run_id != b.run_id and a.path != b.path
    assert a.path.parent == tmp_path and a.path.suffix == ".jsonl"
    assert a.run_id.startswith("knn-")
    a.close(), b.close()
    assert new_run_id("x") != new_run_id("x")


def test_default_directory_is_results_traces() -> None:
    assert DEFAULT_TRACE_DIR.parts[-2:] == ("results", "traces")


def test_bad_run_id_and_nan_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run_id"):
        Tracer("v", run_id="../escape", trace_dir=tmp_path)
    with Tracer("v", run_id="run-e", trace_dir=tmp_path) as t:
        with pytest.raises(ValueError):
            t.log_step(scenario_id="s", tool="t", latency_s=float("nan"))
        assert read_trace(t.path) == []  # nothing half-written


def test_gate_result_is_stored_when_given(tmp_path: Path) -> None:
    with Tracer("v", run_id="run-f", trace_dir=tmp_path) as t:
        t.log_step(**STEP, gate_result={"decision": "hold"})
    assert read_trace(t.path)[0]["gate_result"] == {"decision": "hold"}


def test_log_pfn_step_records_context_and_backend(tmp_path: Path) -> None:
    cfg = load_pfn_config()
    rng = np.random.default_rng(0)
    pool = pd.DataFrame(rng.standard_normal((200, len(FEATURE_NAMES))), columns=list(FEATURE_NAMES))
    pool["trajectory_id"] = [f"p/{i // 20}" for i in range(200)]
    pool["family"] = "fold"
    pool["window_end"] = 127 + 16 * (np.arange(200) % 20)
    pool["target"] = rng.random(200)
    query = pool[pool["trajectory_id"] == "p/0"].copy()
    query["trajectory_id"] = "q/0"

    sel = select_context("random", pool, query, cfg, max_context=40)
    backend = FakeBackend(cfg)
    x, y = context_xy(sel)
    backend.fit(x, y)
    out = backend.predict(query_x(query))
    with Tracer("random", run_id="run-g", trace_dir=tmp_path) as t:
        t.log_pfn_step("fold_L2_slow", sel, out)
    rec = read_trace(t.path)[0]
    assert (rec["context_hash"], rec["context_size"], rec["strategy"]) == (
        sel.context_hash, 40, "random",
    )  # fmt: skip
    assert (rec["backend"], rec["model_version"]) == ("fake", "fake-1")
    assert rec["latency_s"] >= 0 and rec["tool"] == "pfn_predict"
    assert rec["tokens_in"] is None and rec["gate_result"] is None


def test_oracle_strategy_is_marked_in_the_trace(tmp_path: Path) -> None:
    cfg = load_pfn_config()
    rng = np.random.default_rng(0)
    pool = pd.DataFrame(rng.standard_normal((60, len(FEATURE_NAMES))), columns=list(FEATURE_NAMES))
    pool["trajectory_id"] = [f"p/{i // 20}" for i in range(60)]
    pool["family"] = "fold"
    pool["window_end"] = 127 + 16 * (np.arange(60) % 20)
    pool["target"] = rng.random(60)
    query = pool[pool["trajectory_id"] == "p/0"].copy()
    query["trajectory_id"] = "q/0"
    sel = select_context("family_matched", pool, query, cfg, max_context=20)
    backend = FakeBackend(cfg)
    backend.fit(*context_xy(sel))
    with Tracer("oracle", run_id="run-h", trace_dir=tmp_path) as t:
        t.log_pfn_step("s", sel, backend.predict(query_x(query)))
    assert read_trace(t.path)[0]["strategy"] == "family_matched (ORACLE)"

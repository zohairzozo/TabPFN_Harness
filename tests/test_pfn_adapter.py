"""Tests for the PFN adapter contract, the fake backend and the stubs."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from pydantic import ValidationError

from pfngate.pfn import (
    ClientDisabledError,
    FakeBackend,
    PFNBackend,
    PFNMetadata,
    PFNOutput,
    TabPFNClient,
    TabPFNLocal,
    TipPFNBackend,
    build_output,
    load_pfn_config,
    make_backend,
)
from pfngate.pfn.adapter import interpolate_p_critical, repair_quantiles

CFG = load_pfn_config()
LEVELS = np.array(CFG.quantile_levels)
RNG = np.random.default_rng(0)
CONTEXT_X = RNG.standard_normal((30, 8))
CONTEXT_Y = RNG.random(30)
QUERY_X = RNG.standard_normal((7, 8))


def fitted(**kwargs: object) -> FakeBackend:
    backend = FakeBackend(CFG, **kwargs)  # type: ignore[arg-type]
    backend.fit(CONTEXT_X, CONTEXT_Y)
    return backend


def metadata(**kwargs: object) -> PFNMetadata:
    base = dict(
        backend="t",
        model_version="v",
        n_context=1,
        fit_seconds=0.0,
        predict_seconds=0.0,
        p_critical_method="quantile_interpolation",
        p_critical_threshold=0.2,
    )
    return PFNMetadata(**{**base, **kwargs})  # type: ignore[arg-type]


# Config
def test_config_matches_the_agreed_settings() -> None:
    assert CFG.quantile_levels == (0.05, 0.25, 0.5, 0.75, 0.95)
    assert CFG.critical_threshold == 0.2
    assert CFG.model_version == "v3.5" and CFG.device == "cpu"
    assert CFG.client_enabled is False


def test_config_rejects_bad_quantile_levels(tmp_path: object) -> None:
    from pathlib import Path

    from pfngate.pfn.config import DEFAULT_PFN_CONFIG

    text = DEFAULT_PFN_CONFIG.read_text(encoding="utf-8")
    bad = text.replace("[0.05, 0.25, 0.5, 0.75, 0.95]", "[0.5, 0.25]")
    path = Path(str(tmp_path)) / "pfn.yaml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="increasing"):
        load_pfn_config(path)


# Output shapes
def test_fake_output_shapes_and_metadata() -> None:
    out = fitted().predict(QUERY_X)
    assert isinstance(out, PFNOutput)
    assert out.quantiles.shape == (len(QUERY_X), len(LEVELS))
    assert out.median.shape == out.p_critical.shape == (len(QUERY_X),)
    assert np.array_equal(out.quantile_levels, LEVELS)
    assert out.n_query == len(QUERY_X)
    md = out.metadata
    assert (md.backend, md.model_version, md.n_context) == ("fake", "fake-1", len(CONTEXT_Y))
    assert md.fit_seconds >= 0 and md.predict_seconds >= 0
    assert md.p_critical_threshold == CFG.critical_threshold


def test_fake_backend_is_deterministic_and_matches_hand_values() -> None:
    a, b = fitted().predict(QUERY_X), fitted().predict(QUERY_X)
    assert np.array_equal(a.quantiles, b.quantiles) and np.array_equal(a.p_critical, b.p_critical)
    zero_row = fitted().predict(np.zeros((1, 8)))  # sigmoid(0) = 0.5, so the median is 0.5
    assert zero_row.median[0] == pytest.approx(0.5)
    assert zero_row.quantiles[0, 0] < 0.5 < zero_row.quantiles[0, -1]


def test_output_is_read_only_and_serialises_to_json() -> None:
    out = fitted().predict(QUERY_X)
    with pytest.raises(ValueError):
        out.quantiles[0, 0] = 99.0
    data = json.loads(out.model_dump_json())
    assert np.allclose(data["quantiles"], out.quantiles)
    assert data["metadata"]["backend"] == "fake"


# Quantile ordering and repair
def test_quantiles_are_ordered_and_not_flagged_when_already_ordered() -> None:
    out = fitted().predict(QUERY_X)
    assert np.all(np.diff(out.quantiles, axis=1) >= 0)
    assert out.metadata.quantiles_repaired is False and out.metadata.n_rows_repaired == 0


def test_crossing_quantiles_are_sorted_and_recorded() -> None:
    out = fitted(cross_quantiles=True).predict(QUERY_X)
    assert np.all(np.diff(out.quantiles, axis=1) >= 0)
    assert out.metadata.quantiles_repaired is True
    assert out.metadata.n_rows_repaired == len(QUERY_X)
    assert any("crossing" in n for n in out.metadata.notes)
    # Sorting the reversed fake output gives back the ordered fake output.
    assert np.allclose(out.quantiles, fitted().predict(QUERY_X).quantiles)


def test_repair_quantiles_by_hand() -> None:
    fixed, changed = repair_quantiles(np.array([[0.3, 0.1, 0.2], [0.1, 0.2, 0.3]]))
    assert fixed.tolist() == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    assert changed.tolist() == [True, False]


def test_output_rejects_crossing_quantiles() -> None:
    with pytest.raises(ValidationError, match="cross"):
        PFNOutput(
            quantile_levels=[0.25, 0.5, 0.75],
            quantiles=[[0.3, 0.2, 0.4]],
            median=[0.2],
            p_critical=[0.5],
            metadata=metadata(),
        )


# p_critical
def test_p_critical_is_in_unit_interval() -> None:
    for threshold_shift in (-3.0, 0.0, 3.0):
        out = fitted().predict(QUERY_X + threshold_shift)
        assert np.all((out.p_critical >= 0.0) & (out.p_critical <= 1.0))
        assert out.metadata.p_critical_method == "quantile_interpolation"
        assert any("interpolated" in n for n in out.metadata.notes)


def test_interpolated_p_critical_by_hand() -> None:
    q = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
    assert interpolate_p_critical(LEVELS, q, 0.25)[0] == pytest.approx(0.375)  # halfway 0.25..0.5
    assert interpolate_p_critical(LEVELS, q, 0.3)[0] == pytest.approx(0.5)
    assert interpolate_p_critical(LEVELS, q, 0.05)[0] == 0.0
    assert interpolate_p_critical(LEVELS, q, 0.9)[0] == 1.0


def test_p_critical_goes_down_as_the_target_goes_up() -> None:
    low = fitted().predict(np.full((1, 8), -2.0))  # target centre near 0.12
    high = fitted().predict(np.full((1, 8), 2.0))  # target centre near 0.88
    assert low.p_critical[0] > high.p_critical[0]


def test_given_p_critical_is_kept_and_clipped() -> None:
    out = build_output(
        levels=LEVELS,
        quantiles=np.tile(LEVELS, (2, 1)),
        threshold=0.2,
        backend="x",
        model_version="y",
        n_context=3,
        fit_seconds=0.0,
        predict_seconds=0.0,
        p_critical=np.array([1.0000001, -1e-9]),
        p_critical_method="full_distribution",
    )
    assert out.p_critical.tolist() == [1.0, 0.0]
    assert out.metadata.p_critical_method == "full_distribution"


@pytest.mark.parametrize("bad", [[1.5], [-0.1], [np.nan]])
def test_output_rejects_p_critical_outside_unit_interval(bad: list[float]) -> None:
    with pytest.raises(ValidationError, match="p_critical"):
        PFNOutput(
            quantile_levels=[0.25, 0.5, 0.75],
            quantiles=[[0.1, 0.2, 0.3]],
            median=[0.2],
            p_critical=bad,
            metadata=metadata(),
        )


def test_output_rejects_wrong_shapes() -> None:
    with pytest.raises(ValidationError, match="shape"):
        PFNOutput(
            quantile_levels=[0.25, 0.5, 0.75],
            quantiles=[[0.1, 0.2]],
            median=[0.2],
            p_critical=[0.5],
            metadata=metadata(),
        )


# Input checks
def test_predict_before_fit_and_bad_inputs_raise() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        FakeBackend(CFG).predict(QUERY_X)
    with pytest.raises(ValueError):
        FakeBackend(CFG).fit(CONTEXT_X, CONTEXT_Y[:-1])
    with pytest.raises(ValueError, match="NaN"):
        FakeBackend(CFG).fit(np.full((3, 8), np.nan), np.zeros(3))
    with pytest.raises(ValueError, match="shape"):
        fitted().predict(np.zeros((2, 5)))


# Protocol and factory
def test_backends_follow_the_protocol() -> None:
    for backend in (FakeBackend(CFG), TabPFNLocal(CFG), TabPFNClient(CFG), TipPFNBackend()):
        assert isinstance(backend, PFNBackend), backend.name


def test_make_backend_follows_the_config() -> None:
    assert make_backend().name == "tabpfn_local"
    for name, cls in [("fake", FakeBackend), ("tippfn", TipPFNBackend)]:
        assert isinstance(make_backend(dataclasses.replace(CFG, backend=name)), cls)
    with pytest.raises(ValueError, match="unknown backend"):
        make_backend(dataclasses.replace(CFG, backend="nope"))


# TipPFN stub
def test_tippfn_stub_raises_and_documents_itself() -> None:
    stub = TipPFNBackend()
    with pytest.raises(NotImplementedError):
        stub.fit(CONTEXT_X, CONTEXT_Y)
    with pytest.raises(NotImplementedError):
        stub.predict(QUERY_X)
    doc = TipPFNBackend.__doc__ or ""
    assert "2605.12308" in doc and "99 quantiles" in doc and "context episodes" in doc
    assert "NOT a reimplementation" in doc and "unconfirmed" in doc


# Remote client
def test_remote_client_is_disabled_by_default() -> None:
    client = TabPFNClient(CFG)
    with pytest.raises(ClientDisabledError, match="disabled"):
        client.fit(CONTEXT_X, CONTEXT_Y)
    with pytest.raises(ClientDisabledError):
        client.predict(QUERY_X)
    assert "SENDS YOUR DATA OFF THIS MACHINE" in (TabPFNClient.__doc__ or "")


@pytest.mark.skipif(importlib.util.find_spec("tabpfn_client") is not None, reason="installed")
def test_remote_client_enabled_but_not_installed_says_so() -> None:
    client = TabPFNClient(dataclasses.replace(CFG, client_enabled=True))
    with pytest.raises(RuntimeError, match="not installed"):
        client.fit(CONTEXT_X, CONTEXT_Y)


def test_importing_the_package_does_not_import_tabpfn() -> None:
    code = "import sys, pfngate.pfn; print('tabpfn' in sys.modules, 'tabpfn_client' in sys.modules)"
    run = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert run.stdout.split() == ["False", "False"], run.stderr


# Real TabPFN. Slow, needs the licence and cached weights, so it only runs when asked:
#   PFNGATE_TEST_TABPFN=1 pytest tests/test_pfn_adapter.py -k real_tabpfn
@pytest.mark.skipif(
    os.environ.get("PFNGATE_TEST_TABPFN") != "1", reason="set PFNGATE_TEST_TABPFN=1"
)
def test_real_tabpfn_local_end_to_end() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((60, 8))
    y = np.clip(0.5 + 0.3 * x[:, 0] + 0.05 * rng.standard_normal(60), 0.0, 1.0)
    backend = TabPFNLocal(CFG)
    backend.fit(x, y)
    out = backend.predict(rng.standard_normal((5, 8)))
    assert out.quantiles.shape == (5, len(LEVELS))
    assert np.all(np.diff(out.quantiles, axis=1) >= 0)
    assert np.all((out.p_critical >= 0) & (out.p_critical <= 1))
    md = out.metadata
    assert md.p_critical_method == "full_distribution"
    assert (md.backend, md.model_version, md.n_context) == ("tabpfn_local", "v3.5", 60)
    # With the threshold set to a row median, the CDF there is 0.5: p_critical agrees with
    # the quantiles, so the full-distribution path and the quantile path describe one thing.
    query = rng.standard_normal((1, 8))
    first = backend.predict(query)
    backend.cfg = dataclasses.replace(CFG, critical_threshold=float(first.median[0]))
    assert backend.predict(query).p_critical[0] == pytest.approx(0.5, abs=0.01)

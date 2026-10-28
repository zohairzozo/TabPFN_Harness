"""Tests for the gate: checks, GateResult, the decision mapping, floors and freezing.

The end-to-end tests use FakeBackend in its "neighbours" mode on one seed per scenario of
the real scenario data, so they are fast and deterministic and need no model weights.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from pfngate.evals.gating import FitPredict, gate_one, trajectory_folds
from pfngate.evals.pfn_cache import PFNCache
from pfngate.features import FEATURE_NAMES
from pfngate.features.config import REPO_ROOT
from pfngate.gate import (
    CheckResult,
    GateInput,
    GateNotApplicableError,
    GateNotFrozenError,
    GateResult,
    assert_applicable,
    assert_frozen,
    decide,
    load_gate_config,
)
from pfngate.gate import checks as ck
from pfngate.gate.config import DEFAULT_GATE_CONFIG, DEFAULT_GATE_HASH, FLOORS, sha256_file
from pfngate.gate.freeze import assert_committed, write_frozen_hash
from pfngate.gate.reference import (
    PoolDistance,
    fit_near_tipping_reference,
    fit_pool_reference,
    near_tipping_distances,
)
from pfngate.pfn import FakeBackend, load_pfn_config, select_context
from pfngate.pfn.context import ContextSelection, context_hash, query_x, row_ids

CFG = load_pfn_config()
CAP = 200
# A validation trajectory where the fake backend's neighbour intervals are narrower than
# the width cap, so its advisory passes too. On many other trajectories they are wider.
QUERY = "hopf_L2_fast/2000"

# The hash of a frozen configs/gate.yaml. None while the gate is provisional. Freezing
# the gate means writing the hash here too, so a loosened gate needs three edits.
PINNED_GATE_SHA256: str | None = "6e93455997a265384b3a70eec66e942953cd648b89ce678a31d35e3e4d9473bc"

GateFactory = Callable[..., tuple[Path, Path]]


# Setup
@pytest.fixture(scope="module")
def setup(small_data: dict[str, Any], make_gate: GateFactory, tmp_path_factory: Any) -> Any:
    tmp = tmp_path_factory.mktemp("gate")
    config_path, _ = make_gate(tmp / "frozen", small_data["pool"], small_data["fc"])
    gate = load_gate_config(config_path)
    ref = gate.checks.ood_share
    pool_distance = PoolDistance(small_data["pool"], ref.feature_mean, ref.feature_std, ref.k)
    val = small_data["val"]
    query = val[val["trajectory_id"] == QUERY].reset_index(drop=True)
    assert len(query) > 0 and query["role"].iloc[0] == "in_prior"  # a tipping trajectory
    return {**small_data, "gate": gate, "pool_distance": pool_distance, "query": query,
            "tmp": tmp}  # fmt: skip


def gate_for(setup: Any, selection: ContextSelection, backend: FakeBackend) -> GateResult:
    fp = FitPredict(backend, PFNCache(setup["tmp"] / "c", enabled=False), CFG,
                    backend.model_version, "test")  # fmt: skip
    out, _ = fp(selection.context, setup["query"])
    result, _ = gate_one(setup["gate"], fp, setup["pool_distance"], selection, setup["query"],
                         out, seed=CFG.seed, run_id="test-run")  # fmt: skip
    return result


def neighbours(scale: float = 1.0) -> FakeBackend:
    return FakeBackend(CFG, mode="neighbours", interval_scale=scale)


def selection_from(rows: pd.DataFrame, query_tid: str) -> ContextSelection:
    ctx = rows.copy()
    ctx["row_id"] = row_ids(ctx)
    ctx = ctx.sort_values("row_id").reset_index(drop=True)
    ids = tuple(ctx["row_id"])
    return ContextSelection("random", False, query_tid, CAP, ctx, ids, context_hash(ids))


# The three end-to-end cases
def test_context_of_stationary_controls_fails_for_a_tipping_query(setup: Any) -> None:
    pool = setup["pool"]
    controls = pool[pool["role"] == "control"].sample(n=CAP, random_state=0)
    result = gate_for(setup, selection_from(controls, QUERY), neighbours())
    assert not result.passed
    assert "context_not_degenerate" in result.failed
    assert any(r.startswith("context_not_degenerate: 0 context rows") for r in result.reasons)
    near = next(a for a in result.advisories if a.name == "near_tipping_support")
    assert not near.passed and near.value is None and "only 0 near-tipping" in near.reason


def test_family_matched_context_passes(setup: Any) -> None:
    sel = select_context("family_matched", setup["pool"], setup["query"], CFG, max_context=CAP)
    result = gate_for(setup, sel, neighbours())
    assert result.passed, result.reasons
    assert result.reasons == () and result.failed == ()
    assert result.strategy == "family_matched (ORACLE)"
    assert [c.name for c in result.checks] == [
        "min_context_size", "ood_share", "context_calibration", "context_not_degenerate",
        "output_integrity",
    ]  # fmt: skip
    assert [a.name for a in result.advisories] == ["interval_width_cap", "near_tipping_support"]
    assert result.advisories_failed == ()


def test_a_wide_interval_is_reported_but_does_not_block(setup: Any) -> None:
    sel = select_context("family_matched", setup["pool"], setup["query"], CFG, max_context=CAP)
    result = gate_for(setup, sel, neighbours(scale=4.0))  # intervals 4 times too wide
    width = result.advisories[0]
    assert width.name == "interval_width_cap" and not width.passed
    assert result.advisories_failed == ("interval_width_cap",)
    assert result.passed and result.failed == () and result.reasons == ()
    assert set(decide(result, np.array([0.1, 0.9]), 0.5)) == {"alert", "no_alert"}


def test_overconfident_backend_fails_context_calibration(setup: Any) -> None:
    sel = select_context("family_matched", setup["pool"], setup["query"], CFG, max_context=CAP)
    result = gate_for(setup, sel, neighbours(scale=0.1))  # intervals 10 times too narrow
    assert result.failed == ("context_calibration",)
    cal = next(c for c in result.checks if c.name == "context_calibration")
    assert cal.value is not None and cal.value < setup["gate"].checks.context_calibration.low


def test_gate_is_deterministic(setup: Any) -> None:
    sel = select_context("random", setup["pool"], setup["query"], CFG, max_context=CAP)
    a = gate_for(setup, sel, neighbours()).model_dump_json()
    b = gate_for(setup, sel, neighbours()).model_dump_json()
    assert a == b


def test_calibration_folds_hold_out_whole_trajectories() -> None:
    tids = [f"s/{i}" for i in range(12)]
    folds = trajectory_folds(tids * 3, folds=5, seed=0)  # repeated ids: one fold each
    assert set(folds) == set(tids) and set(folds.values()) == set(range(5))
    assert folds == trajectory_folds(list(reversed(tids)), folds=5, seed=0)  # order-free


# GateResult
def check(name: str, passed: bool) -> CheckResult:
    return CheckResult(name=name, value=1.0, threshold=1.0, comparison=">=", passed=passed,
                       reason="r", source="fixed")  # fmt: skip


def result_with(*passes: bool) -> GateResult:
    cs = tuple(check(f"c{i}", p) for i, p in enumerate(passes))
    return GateResult(
        run_id="r", trajectory_id="t", strategy="random", context_hash="h", checks=cs,
        advisories=(check("width", False),), passed=all(passes),
        reasons=tuple(f"{c.name}: r" for c in cs if not c.passed),
        gate_config_sha256="0" * 64, gate_config_status="tuned", gate_code_version="0",
    )  # fmt: skip


def test_gate_result_cannot_be_mutated() -> None:
    res = result_with(True, False)
    with pytest.raises(ValidationError):
        res.passed = True  # type: ignore[misc]
    with pytest.raises(ValidationError):
        res.checks[1].passed = True  # type: ignore[misc]
    with pytest.raises(AttributeError):
        res.checks.append(check("x", True))  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="cannot be changed"):
        res.model_copy(update={"passed": True})
    assert res.model_copy() == res


def test_gate_result_refuses_an_overall_flag_that_disagrees_with_its_checks() -> None:
    good = result_with(True, False).model_dump()
    with pytest.raises(ValidationError, match="passed must be true exactly"):
        GateResult.model_validate({**good, "passed": True, "reasons": ()})
    with pytest.raises(ValidationError, match="reasons must list"):
        GateResult.model_validate({**good, "reasons": ()})


def test_near_tipping_support_check() -> None:
    d = np.array([0.5, 1.0, 1.5, 2.5, 3.0])  # two of five beyond the cut 2.0
    assert ck.near_tipping_support(d, 10, 5, 2.0, 0.4).passed
    assert not ck.near_tipping_support(d, 10, 5, 2.0, 0.39).passed
    assert ck.near_tipping_support(d, 10, 5, 2.0, 0.4).value == pytest.approx(0.4)
    few = ck.near_tipping_support(d, 4, 5, 2.0, 0.5)
    assert not few.passed and few.value is None  # too few near-tipping rows fails outright


def test_near_tipping_distances_look_only_at_near_tipping_context_rows() -> None:
    cols = list(FEATURE_NAMES)
    ctx = pd.DataFrame(np.zeros((8, len(cols))), columns=cols)
    ctx["target"] = [0.1] * 3 + [0.9] * 5
    ctx.loc[:2, cols[0]] = [1.0, 2.0, 3.0]  # near-tipping rows; the far rows sit at 0
    unit = (tuple([0.0] * len(cols)), tuple([1.0] * len(cols)))
    q = np.zeros((1, len(cols)))
    d = near_tipping_distances(ctx, q, feature_mean=unit[0], feature_std=unit[1], k=2,
                               target_max=0.2)  # fmt: skip
    assert d == pytest.approx([2.0])  # 2nd nearest near-tipping row, not a far row at 0
    d = near_tipping_distances(ctx, q, feature_mean=unit[0], feature_std=unit[1], k=4,
                               target_max=0.2)  # fmt: skip
    assert np.isinf(d).all()


def test_near_tipping_cut_leaves_out_each_rows_own_trajectory(small_data: Any) -> None:
    pool = small_data["pool"]
    ref = fit_pool_reference(pool, k=10, quantile=0.99, n_sample=500, seed=0)
    kw = {"feature_mean": ref.feature_mean, "feature_std": ref.feature_std, "k": 5,
          "target_max": 0.2, "quantile": 0.99, "n_sample": 500, "seed": 0}  # fmt: skip
    a, b = fit_near_tipping_reference(pool, **kw), fit_near_tipping_reference(pool, **kw)
    assert a == b and a.distance_cut > 0.0
    assert a.n_near_rows == int((pool["target"] <= 0.2).sum())
    # Without leaving out the own trajectory, near-tipping rows would find themselves at
    # distance 0, and the cut would come from much closer neighbours.
    one = pool[pool["trajectory_id"] == pool["trajectory_id"].iloc[0]]
    with pytest.raises(ValueError, match="at least two trajectories"):
        fit_near_tipping_reference(one, **kw)


def test_a_check_cannot_be_both_blocking_and_advisory() -> None:
    good = result_with(True, True).model_dump()
    with pytest.raises(ValidationError, match="both blocking and advisory"):
        GateResult.model_validate({**good, "advisories": (check("c0", False),)})


def test_a_config_without_the_advisory_role_is_refused(setup: Any, tmp_path: Path) -> None:
    raw = yaml.safe_load(Path(setup["gate"].source).read_text(encoding="utf-8"))
    for role in (None, "blocking"):
        node = raw["checks"]["interval_width_cap"]
        node.pop("role")
        if role is not None:
            node["role"] = role
        old = tmp_path / "gate.yaml"
        old.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(ValidationError):
            load_gate_config(old)
        node["role"] = "advisory"


def test_gate_input_has_no_field_for_query_ground_truth() -> None:
    for name in GateInput.model_fields:
        assert not any(w in name for w in ("target", "label", "truth", "bad")), name
    assert set(GateInput.model_fields) == {
        "run_id", "trajectory_id", "strategy", "context_hash", "context_size", "context_y",
        "context_n_trajectories", "calibration_coverage", "pool_distances",
        "near_tipping_distances", "output",
    }  # fmt: skip


# Decision mapping
def test_decision_mapping() -> None:
    p = np.array([0.1, 0.5, 0.9])
    assert decide(result_with(True, True), p, 0.5) == ("no_alert", "alert", "alert")
    assert decide(result_with(True, False), p, 0.5) == ("abstain",) * 3  # fail: no alerts
    assert decide(result_with(False, False), np.array([1.0]), 0.0) == ("abstain",)
    assert decide(result_with(True), p, 0.95) == ("no_alert",) * 3
    with pytest.raises(ValueError):
        decide(result_with(True), np.array([[0.5]]), 0.5)


# Each check, just inside and just outside its threshold
def test_checks_at_their_boundaries() -> None:
    assert ck.min_context_size(100, 100).passed and not ck.min_context_size(99, 100).passed
    d = np.array([0.0, 0.0, 0.0, 2.0])  # one of four windows beyond a cut of 1.0
    assert ck.ood_share(d, 1.0, 0.25, "tuned").passed
    assert not ck.ood_share(d, 1.0, 0.24, "tuned").passed
    lo, hi = np.zeros(3), np.array([0.5, 0.6, 0.7])  # median width 0.6
    assert ck.interval_width_cap(lo, hi, 0.6).passed
    assert not ck.interval_width_cap(lo, hi, 0.59).passed
    assert ck.context_calibration(0.7, 5, 5, 0.7, 1.0, "tuned").passed
    assert not ck.context_calibration(0.69, 5, 5, 0.7, 1.0, "tuned").passed
    assert not ck.context_calibration(1.0, 5, 5, 0.7, 0.99, "tuned").passed
    few = ck.context_calibration(0.9, 4, 5, 0.7, 1.0, "tuned")
    assert not few.passed and few.value is None and "too few" in few.reason
    y = np.array([0.1] * 10 + [0.9] * 10)
    assert ck.context_not_degenerate(y, 0.2, 0.6, 10).passed
    assert not ck.context_not_degenerate(y[1:], 0.2, 0.6, 10).passed
    assert not ck.context_not_degenerate(np.ones(50), 0.2, 0.6, 10).passed  # constant target
    ok = ("full_distribution", 0.01, "full_distribution")
    assert ck.output_integrity(1, 100, ok[0], ok[1], ok[2]).passed
    assert not ck.output_integrity(2, 100, ok[0], ok[1], ok[2]).passed
    assert not ck.output_integrity(0, 100, "constant_target", ok[1], ok[2]).passed


# Floors: a loosened config does not load
@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("checks", "min_context_size", "threshold"), 49),
        (("checks", "ood_share", "limit"), 0.31),
        (("checks", "near_tipping_support", "limit"), 0.51),
        (("checks", "interval_width_cap", "threshold"), 0.81),
        (("checks", "context_calibration", "low"), 0.59),
        (("checks", "context_calibration", "folds"), 2),
        (("checks", "context_not_degenerate", "min_rows"), 4),
        (("checks", "output_integrity", "max_repaired_share"), 0.051),
        (("bad_run", "min_coverage"), 0.7),
    ],
)
def test_a_config_looser_than_the_floors_is_refused(
    setup: Any, tmp_path: Path, path: tuple[str, ...], value: float
) -> None:
    raw = yaml.safe_load(Path(setup["gate"].source).read_text(encoding="utf-8"))
    node = raw
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    loose = tmp_path / "gate.yaml"
    loose.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError, match="gate config refused"):
        load_gate_config(loose)


def test_floors_are_pinned() -> None:
    assert FLOORS == {
        "min_context_size_at_least": 50, "ood_share_limit_at_most": 0.3,
        "near_tipping_limit_at_most": 0.5,
        "interval_width_cap_at_most": 0.8, "calibration_low_at_least": 0.6,
        "calibration_folds_at_least": 3, "degenerate_min_rows_at_least": 5,
        "repaired_share_at_most": 0.05,
    }  # fmt: skip


def test_gate_config_is_frozen_all_the_way_down(setup: Any) -> None:
    gate = setup["gate"]
    with pytest.raises(ValidationError):
        gate.checks.ood_share.limit = 1.0  # type: ignore[misc]
    assert isinstance(gate.alert_thresholds, tuple)  # no mutable dict inside


# Freezing
def test_assert_frozen_accepts_the_frozen_file(setup: Any) -> None:
    cfg = assert_frozen(Path(setup["gate"].source), Path(setup["gate"].source).with_name(
        "gate.frozen.sha256"))  # fmt: skip
    assert cfg.sha256 == setup["gate"].sha256 and cfg.status == "tuned"


def test_changing_one_byte_makes_assert_frozen_raise(
    setup: Any, make_gate: GateFactory, tmp_path: Path
) -> None:
    config, digest = make_gate(tmp_path, setup["pool"], setup["fc"])
    assert_frozen(config, digest)
    data = bytearray(config.read_bytes())
    i = data.index(b"limit: 0.3") + len(b"limit: 0.")
    data[i] = ord("2")  # ood_share limit 0.3 -> 0.2: one byte
    config.write_bytes(bytes(data))
    with pytest.raises(GateNotFrozenError, match="changed after it was frozen"):
        assert_frozen(config, digest)


def test_missing_hash_or_config_is_refused(tmp_path: Path) -> None:
    with pytest.raises(GateNotFrozenError, match="not frozen"):
        assert_frozen(tmp_path / "gate.yaml", tmp_path / "gate.frozen.sha256")


def test_only_a_tuned_gate_can_be_frozen_and_never_twice(
    setup: Any, make_gate: GateFactory, tmp_path: Path
) -> None:
    config, digest = make_gate(tmp_path / "p", setup["pool"], setup["fc"], status="provisional",
                               freeze=False)  # fmt: skip
    with pytest.raises(GateNotFrozenError, match="Only a tuned gate"):
        write_frozen_hash(config, digest)
    digest.write_text(sha256_file(config) + "\n")  # frozen by hand
    with pytest.raises(GateNotFrozenError, match="status 'provisional'"):
        assert_frozen(config, digest)
    config, digest = make_gate(tmp_path / "t", setup["pool"], setup["fc"])
    with pytest.raises(FileExistsError):
        write_frozen_hash(config, digest)


def test_assert_committed_refuses_a_file_outside_git(tmp_path: Path) -> None:
    f = tmp_path / "gate.yaml"
    f.write_text("x")
    with pytest.raises(GateNotFrozenError):
        assert_committed(f)


def test_applicability() -> None:
    gate = load_gate_config()
    a = gate.applicability
    ok = {"backend": a.backend, "model_version": a.model_version, "max_context": a.max_context,
          "package_version": a.package_version, "data_version": a.data_version,
          "strategy": a.strategies[0]}  # fmt: skip
    assert_applicable(gate, ok, oracle=False)
    assert_applicable(gate, {**ok, "strategy": "family_matched"}, oracle=True)
    for change in ({"max_context": a.max_context * 10}, {"strategy": "family_matched"},
                   {"model_version": "other"}):  # fmt: skip
        with pytest.raises(GateNotApplicableError):
            assert_applicable(gate, {**ok, **change}, oracle=False)


# The repository's own gate file
def test_repository_gate_config_loads_and_its_frozen_state_is_pinned() -> None:
    gate = load_gate_config(DEFAULT_GATE_CONFIG)
    if DEFAULT_GATE_HASH.exists():
        assert gate.status == "tuned"
        assert DEFAULT_GATE_HASH.read_text().strip() == PINNED_GATE_SHA256 == gate.sha256
    else:
        assert PINNED_GATE_SHA256 is None, "the pinned hash says the gate is frozen"
        assert gate.status == "provisional"
        with pytest.raises(GateNotFrozenError):
            assert_frozen()


def test_gate_files_keep_their_bytes_in_git() -> None:
    attrs = (REPO_ROOT / ".gitattributes").read_text().splitlines()
    assert "configs/gate.yaml -text" in attrs and "configs/gate.frozen.sha256 -text" in attrs


def test_agent_code_cannot_reach_the_gate_writers() -> None:
    forbidden = {"pfngate.gate.freeze", "pfngate.gate.build", "pfngate.gate.config",
                 "pfngate.gate.tune"}  # fmt: skip
    for path in (REPO_ROOT / "src" / "pfngate" / "agent").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module not in forbidden, (path, node.module)
            if isinstance(node, ast.Import):
                assert not {a.name for a in node.names} & forbidden, path


# Pool reference
def test_pool_reference_excludes_the_own_trajectory(small_data: dict[str, Any]) -> None:
    pool = small_data["pool"]
    ref = fit_pool_reference(pool, k=10, quantile=0.99, n_sample=500, seed=0)
    assert ref.distance_cut > 0 and ref.n_pool_rows == len(pool)
    again = fit_pool_reference(pool.sample(frac=1.0, random_state=1), k=10, quantile=0.99,
                               n_sample=len(pool), seed=0)  # fmt: skip
    assert again.feature_mean == pytest.approx(ref.feature_mean)
    pd_ = PoolDistance(pool, ref.feature_mean, ref.feature_std, ref.k)
    tid = str(pool["trajectory_id"].iloc[0])
    with pytest.raises(ValueError, match="is in the pool"):
        pd_.distances(np.zeros((1, 8)), tid)


def test_far_query_windows_are_flagged(setup: Any) -> None:
    pd_ = setup["pool_distance"]
    near = pd_.distances(query_x(setup["query"]), QUERY)
    far = pd_.distances(np.full((5, 8), 50.0), "far/0")
    cut = setup["gate"].checks.ood_share.distance_cut
    assert (near > cut).mean() <= 0.3 and (far > cut).all()

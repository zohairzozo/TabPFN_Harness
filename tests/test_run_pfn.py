"""Tests for the PFN evaluation run, its cache and the shared threshold rule.

The run is exercised with the fake backend on a small slice of the real scenario data
(1 seed per scenario, in a temp cache), so no model weights are needed. Held-out splits
need a frozen gate, so the runs here use a tuned and frozen test gate in a temp folder.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pfngate.evals.baseline_ews import load_baseline_config
from pfngate.evals.metrics import alarm_threshold_for_rate, false_alarm_rate
from pfngate.evals.pfn_cache import PFNCache, cache_key, data_version, query_hash
from pfngate.evals.run_pfn import (
    SCORES,
    SPLIT_FLAGS,
    false_alarm_blocks,
    result_path,
    run,
    window_table,
)
from pfngate.evals.trace import TRACE_FIELDS, read_trace
from pfngate.features import FEATURE_NAMES, build_dataset, load_feature_config
from pfngate.gate import GateNotApplicableError, GateNotFrozenError
from pfngate.pfn import FakeBackend, load_pfn_config
from pfngate.pfn.adapter import PFNOutput

CFG = dataclasses.replace(load_pfn_config(), bootstrap_n=40)
CAP = 60
IN_PRIOR = {"fold", "hopf", "transcritical"}
CONTROLS = {"control_ou", "control_short_ramp", "control_regime_change"}


# Shared threshold rule
def test_alarm_threshold_for_rate_by_hand() -> None:
    scores = np.arange(100.0)  # 0 ... 99
    thr = alarm_threshold_for_rate(scores, 0.05)
    assert (scores >= thr).mean() <= 0.05
    assert (scores >= thr).sum() == 4  # 96, 97, 98, 99: the 95th percentile value does not alarm
    # Ties at the quantile do not alarm, so a block of equal scores gives a rate of 0.
    assert (np.zeros(50) >= alarm_threshold_for_rate(np.zeros(50), 0.05)).sum() == 0


def test_false_alarm_rate_at_the_chosen_threshold_is_at_most_the_target() -> None:
    rng = np.random.default_rng(0)
    scores = rng.standard_normal(2000)
    thr = alarm_threshold_for_rate(scores, 0.05)
    fa = false_alarm_rate(scores, np.arange(2000) // 20, thr)
    assert fa["window_rate"] <= 0.05 + 1e-12


def test_alarm_threshold_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        alarm_threshold_for_rate(np.array([]), 0.05)
    with pytest.raises(ValueError):
        alarm_threshold_for_rate(np.array([1.0]), 1.5)


def test_evaluation_settings_match_the_baseline() -> None:
    base = load_baseline_config()
    pfn = load_pfn_config()
    assert (pfn.bootstrap_n, pfn.bootstrap_seed, pfn.bootstrap_ci) == (
        base.bootstrap_n,
        base.bootstrap_seed,
        base.ci,
    )
    assert pfn.target_false_alarm_rate == base.target_false_alarm_rate


# The threshold uses validation controls only
def synthetic_controls(test_shift: float) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frames = []
    for split, shift in [("validation", 0.0), ("test_in_prior", test_shift)]:
        n = 400
        frames.append(
            pd.DataFrame(
                {
                    "trajectory_id": [f"{split}/{i // 20}" for i in range(n)],
                    "family": "control_ou",
                    "role": "control",
                    "split": split,
                    "neg_median": rng.standard_normal(n) + shift,
                    "p_critical": rng.random(n) * 0.1 + shift,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_false_alarm_threshold_is_set_on_validation_controls_only() -> None:
    a = false_alarm_blocks(synthetic_controls(test_shift=0.0), CFG)
    b = false_alarm_blocks(synthetic_controls(test_shift=50.0), CFG)
    assert a["thresholds"] == b["thresholds"]  # moving the test controls does not move it
    for score in SCORES:
        val = a["scores"][score]["splits"]["validation"]["all_controls"]
        assert val["window_rate"] <= CFG.target_false_alarm_rate
        far = b["scores"][score]["held_out_pooled"]["all_controls"]
        assert far["window_rate"] == 1.0  # shifted held-out scores all alarm: nothing was refit


def test_false_alarms_need_validation_controls() -> None:
    only_test = synthetic_controls(0.0).query("split != 'validation'")
    with pytest.raises(ValueError, match="validation controls"):
        false_alarm_blocks(only_test, CFG)


# Cache
def tiny_output() -> PFNOutput:
    be = FakeBackend(CFG)
    be.fit(np.zeros((5, 8)), np.zeros(5))
    return be.predict(np.ones((3, 8)))


def test_cache_round_trip_is_exact_and_starts_with_the_context_hash(tmp_path: Path) -> None:
    cache = PFNCache(tmp_path)
    key = cache_key(CFG, backend="fake", model_version="fake-1", data_ver="d",
                    context_hash="abcd", query_hash_="q")  # fmt: skip
    assert key.startswith("abcd_") and cache.get(key) is None
    out = tiny_output()
    cache.put(key, out)
    back = cache.get(key)
    assert back is not None
    assert np.array_equal(back.quantiles, out.quantiles)
    assert np.array_equal(back.p_critical, out.p_critical)
    assert back.metadata == out.metadata
    assert [p.suffix for p in tmp_path.iterdir()] == [".json"]  # no temp file left behind


def test_cache_can_be_disabled(tmp_path: Path) -> None:
    cache = PFNCache(tmp_path, enabled=False)
    cache.put("k", tiny_output())
    assert cache.get("k") is None and not list(tmp_path.iterdir())


def test_cache_key_changes_with_everything_that_changes_the_answer() -> None:
    base = dict(backend="fake", model_version="m", data_ver="d", context_hash="c", query_hash_="q")
    k0 = cache_key(CFG, **base)  # type: ignore[arg-type]
    for change in (
        {"backend": "tabpfn_local"}, {"model_version": "m2"}, {"data_ver": "d2"},
        {"context_hash": "c2"}, {"query_hash_": "q2"},
    ):  # fmt: skip
        assert cache_key(CFG, **{**base, **change}) != k0, change  # type: ignore[arg-type]
    for cfg2 in (
        dataclasses.replace(CFG, seed=CFG.seed + 1),
        dataclasses.replace(CFG, critical_threshold=0.3),
        dataclasses.replace(CFG, quantile_levels=(0.1, 0.5, 0.9)),
    ):
        assert cache_key(cfg2, **base) != k0  # type: ignore[arg-type]
    assert cache_key(CFG, **base) == k0  # type: ignore[arg-type]


def test_query_hash_follows_the_features() -> None:
    q = pd.DataFrame(np.random.default_rng(0).standard_normal((4, 8)), columns=list(FEATURE_NAMES))
    q["window_end"] = [127, 143, 159, 175]
    q2 = q.copy()
    q2.loc[0, FEATURE_NAMES[0]] += 1e-9
    assert query_hash(q) == query_hash(q.copy()) and query_hash(q) != query_hash(q2)


def test_data_version_is_stable() -> None:
    assert data_version(load_feature_config()) == data_version(load_feature_config())


# The run itself
@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory, make_gate: Any) -> dict[str, Any]:
    tmp = tmp_path_factory.mktemp("run")
    fc = dataclasses.replace(load_feature_config(), cache_dir=tmp / "features", summary_dir=tmp)
    pool = build_dataset("context", seeds_per_scenario=1, feature_config=fc)
    config, digest = make_gate(tmp / "gate", pool, fc, max_context=CAP,
                               alert={"random": 0.5, "knn": 0.5})  # fmt: skip
    return {"fc": fc, "results": tmp / "results", "gate_config": config, "gate_hash": digest,
            "pool": pool, "tmp": tmp}  # fmt: skip


def do_run(env: dict[str, Any], strategy: str, flag: str, **kw: Any) -> dict[str, Any]:
    kw.setdefault("gate_config_path", env["gate_config"])
    kw.setdefault("gate_hash_path", env["gate_hash"])
    kw.setdefault("results_dir", env["results"])
    return run(strategy, flag, max_context=CAP, backend_name="fake", seeds_per_scenario=1,
               pool_seeds_per_scenario=1, feature_config=env["fc"], cfg=CFG, progress=False,
               **kw)  # fmt: skip


@pytest.fixture(scope="module")
def val_knn(env: dict[str, Any]) -> dict[str, Any]:
    return do_run(env, "knn", "val")


def test_split_flags_cover_the_agreed_splits() -> None:
    assert SPLIT_FLAGS["val"] == ("validation",) and SPLIT_FLAGS["test"] == ("test_in_prior",)
    assert SPLIT_FLAGS["ood"] == ("test_out_of_prior",)
    assert set(SPLIT_FLAGS["controls"]) == {"validation", "test_in_prior", "test_out_of_prior"}


def test_val_run_scores_tipping_trajectories_and_writes_the_json(
    env: dict[str, Any], val_knn: dict[str, Any]
) -> None:
    path = result_path(env["results"], "knn", "val")
    assert path.name == "pfn_knn_val.json" and json.loads(path.read_text()) == json.loads(
        json.dumps(val_knn, default=float)
    )
    assert val_knn["n_trajectories"] == 18  # 18 tipping scenarios, 1 seed each
    assert {t["family"] for t in val_knn["trajectories"]} == IN_PRIOR
    assert (val_knn["max_context"], val_knn["oracle"]) == (CAP, False)
    assert val_knn["max_context_config_default"] == CFG.max_context
    assert all(0 < t["context_size"] <= CAP for t in val_knn["trajectories"])


def test_val_run_reports_both_scores_with_trajectory_bootstrap_cis(val_knn: dict[str, Any]) -> None:
    for subset in ("tipping_only", "family:fold", "family:hopf", "family:transcritical"):
        for score in SCORES:
            a = val_knn["auroc"][subset][score]
            assert 0.0 <= a["ci_low"] <= a["auroc"] <= a["ci_high"] <= 1.0, (subset, score)
            assert a["n_boot"] == CFG.bootstrap_n
    tip = val_knn["auroc"]["tipping_only"]["p_critical"]
    assert tip["n_trajectories"] == 18  # resampled over trajectories, not windows


def test_every_call_is_traced(env: dict[str, Any], val_knn: dict[str, Any]) -> None:
    rows = read_trace(Path(env["results"]) / "traces" / Path(val_knn["trace_file"]).name)
    assert len(rows) == 2 * val_knn["n_trajectories"]  # select_context and pfn_predict each
    assert {tuple(r) for r in rows} == {TRACE_FIELDS}
    assert {r["tool"] for r in rows} == {"select_context", "pfn_predict"}
    calls = [r for r in rows if r["tool"] == "pfn_predict"]
    assert all(r["backend"] == "fake" and r["strategy"] == "knn" for r in calls)
    assert all(0 < r["context_size"] <= CAP and r["gate_result"] is None for r in rows)
    assert [r["step"] for r in rows] == list(range(1, len(rows) + 1))


def test_second_run_hits_the_cache_and_gives_the_same_scores(
    env: dict[str, Any], val_knn: dict[str, Any]
) -> None:
    again = do_run(env, "knn", "val")
    assert val_knn["n_cache_hits"] == 0 and again["n_cache_hits"] == again["n_trajectories"]
    assert again["auroc"] == val_knn["auroc"]
    assert [t["context_hash"] for t in again["trajectories"]] == [
        t["context_hash"] for t in val_knn["trajectories"]
    ]
    # Latency is the latency of the run that computed the result, not of the lookup.
    assert [t["latency_seconds"] for t in again["trajectories"]] == [
        t["latency_seconds"] for t in val_knn["trajectories"]
    ]
    rows = read_trace(Path(env["results"]) / "traces" / Path(again["trace_file"]).name)
    assert {r["tool"] for r in rows} == {"select_context", "pfn_cache_hit"}


def test_family_matched_is_marked_as_an_oracle_everywhere(env: dict[str, Any]) -> None:
    res = do_run(env, "family_matched", "val", use_cache=False)
    assert res["oracle"] is True
    rows = read_trace(Path(env["results"]) / "traces" / Path(res["trace_file"]).name)
    assert {r["strategy"] for r in rows} == {"family_matched (ORACLE)"}


def test_family_matched_cannot_run_on_out_of_prior_families(env: dict[str, Any]) -> None:
    res = do_run(env, "family_matched", "ood")
    assert res["n_trajectories"] == 0 and res["auroc"] is None
    assert len(res["skipped"]) == 12  # 12 out-of-prior scenarios, 1 seed each
    assert all("no pool rows" in s["reason"] for s in res["skipped"])


def test_ood_run_with_a_real_strategy_scores_out_of_prior_families(env: dict[str, Any]) -> None:
    res = do_run(env, "random", "ood")
    assert {t["family"] for t in res["trajectories"]} == {"rate_tipping", "noise_tipping"}
    assert res["auroc"]["tipping_only"]["p_critical"] is not None


def test_controls_run_covers_all_three_splits_and_uses_validation_for_the_threshold(
    env: dict[str, Any],
) -> None:
    res = do_run(env, "random", "controls")
    assert res["n_trajectories"] == 27  # 9 control scenarios in each of 3 splits, 1 seed
    assert {t["family"] for t in res["trajectories"]} == CONTROLS
    assert {t["split"] for t in res["trajectories"]} == set(SPLIT_FLAGS["controls"])
    fa = res["false_alarms"]
    assert fa["target_window_false_alarm_rate"] == CFG.target_false_alarm_rate
    for score in SCORES:
        val = fa["scores"][score]["splits"]["validation"]["all_controls"]
        assert val["threshold"] == fa["thresholds"][score]
        assert val["window_rate"] <= CFG.target_false_alarm_rate
        assert fa["scores"][score]["held_out_pooled"] is not None
    assert "auroc" not in res


def test_run_is_deterministic_without_the_cache(env: dict[str, Any]) -> None:
    a = do_run(env, "random", "test", use_cache=False)
    b = do_run(env, "random", "test", use_cache=False)
    assert a["auroc"] == b["auroc"]
    assert [t["context_hash"] for t in a["trajectories"]] == [
        t["context_hash"] for t in b["trajectories"]
    ]


def test_bad_strategy_and_split_are_refused(env: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        do_run(env, "nope", "val")
    with pytest.raises(ValueError):
        do_run(env, "knn", "nope")


def test_window_table_checks_the_number_of_predictions(val_knn: dict[str, Any]) -> None:
    q = pd.DataFrame(
        {c: ["a"] * 3 for c in ("trajectory_id", "scenario_id", "family", "role", "split")}
    )
    q["label"], q["target"] = [0, 1, 0], [0.9, 0.1, 0.8]
    with pytest.raises(ValueError, match="predictions"):
        window_table(q, tiny_output().model_copy(update={}))  # 3 predictions, fine
        window_table(q.iloc[:2], tiny_output())  # 2 windows, 3 predictions


# Held-out splits need a frozen gate
@pytest.mark.parametrize("flag", ["test", "ood", "controls"])
def test_held_out_splits_refuse_without_a_frozen_gate(env: dict[str, Any], flag: str) -> None:
    missing = env["tmp"] / "nowhere" / "gate.frozen.sha256"
    with pytest.raises(GateNotFrozenError, match="not frozen"):
        do_run(env, "random", flag, gate_hash_path=missing)
    assert not result_path(env["results"] / "refused", "random", flag).exists()


def test_held_out_split_refuses_after_one_byte_of_the_gate_changed(env: dict[str, Any]) -> None:
    changed = env["tmp"] / "changed"
    changed.mkdir()
    config = changed / "gate.yaml"
    data = bytearray(env["gate_config"].read_bytes())
    data[-2] = ord("9") if data[-2] != ord("9") else ord("8")
    config.write_bytes(bytes(data))
    with pytest.raises(GateNotFrozenError, match="changed after it was frozen"):
        do_run(env, "random", "test", gate_config_path=config)


def test_validation_runs_do_not_need_a_frozen_gate(env: dict[str, Any]) -> None:
    missing = env["tmp"] / "nowhere" / "gate.frozen.sha256"
    res = do_run(env, "random", "val", gate_hash_path=missing, use_cache=False)
    assert res["n_trajectories"] == 18 and res["frozen_gate_sha256"] is None


# --gate
@pytest.fixture(scope="module")
def gated_val(env: dict[str, Any]) -> dict[str, Any]:
    backend = FakeBackend(CFG, mode="neighbours")
    return do_run(env, "knn", "val", gate=True, backend=backend,
                  results_dir=env["tmp"] / "gated")  # fmt: skip


def test_gate_result_is_in_the_json_for_every_run(gated_val: dict[str, Any]) -> None:
    names = ["min_context_size", "ood_share", "context_calibration", "context_not_degenerate",
             "output_integrity"]  # fmt: skip
    for t in gated_val["trajectories"]:
        g = t["gate"]
        assert [c["name"] for c in g["checks"]] == names
        assert [a["name"] for a in g["advisories"]] == ["interval_width_cap",
                                                         "near_tipping_support"]  # fmt: skip
        gt = t["ground_truth"]
        freq = gt["observed_frequency"]
        assert list(freq) == ["0.05", "0.25", "0.5", "0.75", "0.95"]
        assert all(0.0 <= v <= 1.0 for v in freq.values())
        assert list(freq.values()) == sorted(freq.values())  # quantiles do not cross
        assert freq["0.95"] >= gt["coverage_90"] - 1e-12  # inside the interval implies below q95
        assert g["passed"] == all(c["passed"] for c in g["checks"])
        assert g["failed"] == [c["name"] for c in g["checks"] if not c["passed"]]
        assert t["ground_truth"]["used_by_gate"] is False
        assert 0.0 <= t["ground_truth"]["coverage_90"] <= 1.0
        n = sum(t["decisions"].values())
        assert n == t["n_windows"]
        if not g["passed"]:
            assert t["decisions"]["abstain"] == n
    summary = gated_val["gate"]
    assert summary["n_runs"] == gated_val["n_trajectories"] == 18
    assert summary["n_passed"] + summary["n_blocked"] == 18
    assert summary["config_status"] == "tuned"


def test_gate_result_is_in_the_trace(env: dict[str, Any], gated_val: dict[str, Any]) -> None:
    rows = read_trace(env["tmp"] / "gated" / "traces" / Path(gated_val["trace_file"]).name)
    assert len(rows) == 4 * gated_val["n_trajectories"]
    gates = [r for r in rows if r["tool"] == "gate"]
    assert len(gates) == 18 and all(r["gate_result"]["checks"] for r in gates)
    by_tid = {t["trajectory_id"]: t["gate"] for t in gated_val["trajectories"]}
    for r in gates:
        g = by_tid[r["scenario_id"]]
        assert r["gate_result"] == {k: v for k, v in g.items() if k != "failed"}
    assert {r["tool"] for r in rows} == {"select_context", "pfn_predict",
                                         "context_calibration", "gate"}  # fmt: skip
    assert all(r["gate_result"] is None for r in rows if r["tool"] != "gate")


def test_gated_held_out_run_works_when_frozen_and_applicable(env: dict[str, Any]) -> None:
    res = do_run(env, "random", "test", gate=True)
    assert res["frozen_gate_sha256"] == res["gate"]["config_sha256"]
    assert res["gate"]["n_runs"] == res["n_trajectories"] > 0


def test_gated_held_out_run_refuses_other_settings(env: dict[str, Any]) -> None:
    with pytest.raises(GateNotApplicableError, match="model_version"):
        do_run(env, "random", "test", gate=True, backend=FakeBackend(CFG, mode="neighbours"))
    with pytest.raises(GateNotApplicableError, match="max_context"):
        run("random", "test", max_context=CAP + 1, backend_name="fake", seeds_per_scenario=1,
            pool_seeds_per_scenario=1, feature_config=env["fc"], cfg=CFG, gate=True,
            gate_config_path=env["gate_config"], gate_hash_path=env["gate_hash"],
            results_dir=env["tmp"] / "other", progress=False)  # fmt: skip


def test_gate_refuses_a_pool_it_was_not_built_on(env: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="gate reference"):
        run("random", "val", max_context=CAP, backend_name="fake", seeds_per_scenario=1,
            pool_seeds_per_scenario=2, feature_config=env["fc"], cfg=CFG, gate=True,
            gate_config_path=env["gate_config"], gate_hash_path=env["gate_hash"],
            results_dir=env["tmp"] / "other", progress=False)  # fmt: skip


# The ledger of PFN compute time
class SlowFake(FakeBackend):
    """A fake backend that says every call took 1.5 s to fit and 2.5 s to predict."""

    def predict(self, query_X: np.ndarray) -> PFNOutput:
        out = super().predict(query_X)
        md = out.metadata.model_copy(update={"fit_seconds": 1.5, "predict_seconds": 2.5})
        return out.model_copy(update={"metadata": md})


def test_the_compute_ledger_counts_a_cached_call_at_its_original_cost(
    tmp_path: Path, small_data: dict[str, Any]
) -> None:
    from pfngate.evals.gating import FitPredict

    pool = small_data["pool"]
    ctx = pool.iloc[:60].copy()
    ctx["row_id"] = ctx["trajectory_id"].astype(str) + "#" + ctx["window_end"].astype(str)
    query = pool.iloc[60:70].copy()
    backend = SlowFake(CFG)
    cache = PFNCache(tmp_path / "c", enabled=True)
    fp = FitPredict(backend, cache, CFG, backend.model_version, "t")
    assert fp.compute_seconds == 0.0
    _, hit = fp(ctx, query)
    assert hit is False and fp.compute_seconds == 4.0  # a miss: what the backend reported
    _, hit = fp(ctx, query)
    assert hit is True and fp.compute_seconds == 8.0  # a hit costs what it cost the first time

"""Tests for the scenario suite in src/pfngate/scenarios, driven by configs/scenarios/."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from pfngate.scenarios import DROPPED, Scenario, Trajectory, binary_labels, load_scenarios
from pfngate.scenarios.labels import target

SCENARIOS = load_scenarios()
TIPPING = [s for s in SCENARIOS.values() if s.role != "control"]
CONTROLS = [s for s in SCENARIOS.values() if s.role == "control"]
COMMON = next(iter(SCENARIOS.values())).common

# How long after the tip we look to confirm the state has really left (time units).
AFTER_TIP_WINDOW = 10.0


def first_seed(sc: Scenario) -> int:
    return sc.seeds()[0]


def mean_between(tr: Trajectory, values: np.ndarray, t0: float, t1: float) -> float:
    mask = (tr.time >= t0) & (tr.time <= t1)
    assert mask.any(), f"no points between {t0} and {t1}"
    return float(values[mask].mean())


def detector_metric(sc: Scenario, tr: Trajectory) -> np.ndarray:
    """The quantity the family's tip detector compares with its threshold."""
    kind = sc.tip_detector["kind"]
    if kind == "state_above":
        return tr.state[:, sc.state_names.index(sc.tip_detector["variable"])]
    if kind == "amplitude_above":
        return np.hypot(tr.state[:, 0], tr.state[:, 1])
    if kind == "moving_frame_above":
        assert tr.control_parameter is not None
        return tr.state[:, 0] + tr.control_parameter
    raise AssertionError(f"no metric for detector kind {kind}")


# (1) Every tipping scenario leaves its starting state around tip_time.
@pytest.mark.parametrize("sc", TIPPING, ids=lambda s: s.scenario_id)
def test_tipping_scenario_leaves_start_state_at_tip(sc: Scenario) -> None:
    tr = sc.generate(first_seed(sc))
    assert tr.tip_time is not None, "expected a tip"
    assert tr.time[0] <= tr.tip_time <= tr.time[-1] + COMMON.obs_dt
    start_end = tr.time[0] + 0.05 * (tr.time[-1] - tr.time[0])

    if sc.tip_detector["kind"] == "escape_below_branch":
        x = tr.state[:, 0]
        p = tr.control_parameter
        assert p is not None
        margin = sc.tip_detector["margin"]
        # Starts at x = 0, far from the escape line.
        assert abs(mean_between(tr, x, 0.0, start_end)) < margin
        # Ends either escaped below the unstable branch or on the new branch x = p.
        tail = tr.time >= tr.time[-1] - 0.1 * (tr.time[-1] - tr.time[0])
        escaped = x[-1] < p[-1] - margin + 0.1  # recorded point is at most one step before stop
        on_branch = x[tail].mean() >= 0.5 * p[tail].mean() > 0
        assert escaped or on_branch
        return

    thr = sc.tip_detector["threshold"]
    metric = detector_metric(sc, tr)
    # Starts on the starting side of the threshold.
    start = mean_between(tr, metric, 0.0, start_end)
    assert start < thr
    # Shortly after the tip the state has really left: on average it is past the
    # midpoint between where it started and the threshold, not just touching it.
    after_end = min(tr.tip_time + AFTER_TIP_WINDOW, tr.time[-1])
    assert mean_between(tr, metric, tr.tip_time, after_end) > (start + thr) / 2
    if sc.tip_detector.get("dwell"):
        # It stayed in the new basin for the whole dwell time.
        i = sc.state_names.index(sc.tip_detector["variable"])
        dwell = (tr.time >= tr.tip_time) & (tr.time <= tr.tip_time + sc.tip_detector["dwell"])
        assert np.all(tr.state[dwell, i] > sc.tip_detector["dwell_level"])
    else:
        # Every recorded point before the tip is still on the starting side, and the
        # run ends past the threshold.
        assert np.all(metric[tr.time < tr.tip_time] <= thr)
        assert metric[-1] > thr

    if sc.label_kind == "bifurcation_distance":
        # The tip is tied to the ramp: not in the first half of the approach.
        assert tr.tip_time > 0.5 * sc.ramp["time_to_critical"]


# (2) Every negative control never tips.
@pytest.mark.parametrize("sc", CONTROLS, ids=lambda s: s.scenario_id)
@pytest.mark.parametrize("seed_set", ["A", "D"])
def test_controls_never_tip(sc: Scenario, seed_set: str) -> None:
    tr = sc.generate(COMMON.seed_sets[seed_set].seeds()[0])
    assert tr.tip_time is None
    assert np.all(tr.target == 1.0)
    x = tr.state[:, 0]
    if sc.sanity_check is not None:
        assert sc.sanity_check["kind"] == "state_below"
        assert np.all(x < sc.sanity_check["threshold"]), "left the lower branch"
    else:
        # OU-type controls stay near their (possibly shifted) level: within 6 sd of the
        # largest noise, plus the mean shift.
        stationary_sd = sc.noise["process_sd"] / np.sqrt(2 * sc.params["theta"])
        shift = sc.params.get("shift_size", 0.0)
        widest = sc.params.get("var_factor", 1.0)
        assert np.max(np.abs(x)) < (shift + 6 * widest) * stationary_sd


# (3) R-tipping: tips above the critical rate and not below (noise-free).
@pytest.mark.parametrize(("factor", "tips"), [(0.5, False), (1.5, True), (3.0, True)])
def test_rate_tipping_critical_rate(factor: float, tips: bool) -> None:
    base = SCENARIOS["rate_L1_slow"]
    r_c = 1.0 / base.params["tau"]
    rate = factor * r_c
    sc = dataclasses.replace(
        base,
        noise={**base.noise, "process_sd": 0.0, "obs_sd": 0.0},
        ramp={**base.ramp, "rate": rate},
    )
    tr = sc.generate(first_seed(sc))
    if tips:
        assert tr.tip_time is not None
        # Tips while lambda is still moving, before it reaches lambda_max.
        assert tr.tip_time < sc.params["ramp_start"] + sc.params["lambda_max"] / rate
    else:
        assert tr.tip_time is None
        # Tracks the moving stable state y = -sqrt(1 - r * tau) once the ramp is under way.
        assert tr.control_parameter is not None
        y = tr.state[:, 0] + tr.control_parameter
        moving = tr.time > sc.params["ramp_start"] + 5 * sc.params["tau"]
        expected = -np.sqrt(1 - rate * sc.params["tau"])
        assert np.allclose(y[moving], expected, atol=0.01)


# (4) Generation is deterministic for a given seed.
@pytest.mark.parametrize("sid", ["hopf_L2_fast", "noise_N2_fast", "regime_L3"])
def test_generation_is_deterministic(sid: str) -> None:
    sc = SCENARIOS[sid]
    seed = first_seed(sc)
    a, b = sc.generate(seed), sc.generate(seed)
    assert np.array_equal(a.observations, b.observations)
    assert np.array_equal(a.target, b.target, equal_nan=True)
    assert a.tip_time == b.tip_time
    other = sc.generate(seed + 1)
    assert not np.array_equal(a.observations[:100], other.observations[:100])


def test_seed_outside_scenario_sets_is_rejected() -> None:
    with pytest.raises(ValueError):
        SCENARIOS["rate_L1_slow"].generate(COMMON.seed_sets["A"].seeds()[0])


# (5) Labels follow docs/LABELS.md on hand-checkable cases.
def test_label_fold_fast_ramp_by_hand() -> None:
    # Fast ramp: p goes from p0 to p_c in 100 time units, so the target is
    # 1 at t = 0, 0.5 at t = 50, 0.25 at t = 75 and 0 at t = 100.
    tr = SCENARIOS["fold_L1_fast"].generate(1000)
    for t, expected in [(0.0, 1.0), (50.0, 0.5), (75.0, 0.25), (100.0, 0.0), (110.0, 0.0)]:
        i = int(np.argmin(np.abs(tr.time - t)))
        assert tr.time[i] == pytest.approx(t)
        assert tr.target[i] == pytest.approx(expected, abs=1e-9)


def test_label_time_to_tip_by_hand() -> None:
    # Tip at t = 100 with horizon H = 100: target = clip((100 - t) / 100, 0, 1).
    assert COMMON.horizon == 100.0
    t = np.array([-50.0, 0.0, 50.0, 90.0, 100.0, 120.0])
    got = target("time_to_tip", t, None, {}, tip_time=100.0, horizon=COMMON.horizon)
    assert np.allclose(got, [1.0, 1.0, 0.5, 0.1, 0.0, 0.0])
    assert np.all(np.isnan(target("time_to_tip", t, None, {}, None, COMMON.horizon)))


def test_label_bifurcation_distance_by_hand() -> None:
    # p0 = -1, p_c = 0: p = -1 -> 1, p = -0.5 -> 0.5, p = 0 -> 0, p = 0.2 -> 0 (clipped).
    p = np.array([-1.0, -0.5, 0.0, 0.2])
    got = target("bifurcation_distance", np.zeros(4), p, {"p0": -1.0, "p_c": 0.0}, None, 100.0)
    assert np.allclose(got, [1.0, 0.5, 0.0, 0.0])


def test_binary_labels_by_hand() -> None:
    # <= 0.2 positive, >= 0.6 negative, in between (and NaN) dropped.
    values = np.array([0.0, 0.2, 0.21, 0.4, 0.59, 0.6, 1.0, np.nan])
    got = binary_labels(values, COMMON.positive_max, COMMON.negative_min)
    assert got.tolist() == [1, 1, DROPPED, DROPPED, DROPPED, 0, 0, DROPPED]


def test_rate_target_matches_tip_time() -> None:
    tr = SCENARIOS["rate_L1_fast"].generate(4000)
    assert tr.tip_time is not None
    i = int(np.argmin(np.abs(tr.time - (tr.tip_time - 50.0))))
    assert tr.target[i] == pytest.approx((tr.tip_time - tr.time[i]) / COMMON.horizon)
    assert np.all(tr.target[tr.time <= tr.tip_time - COMMON.horizon] == 1.0)


def test_suite_size_and_split_tags() -> None:
    assert len(SCENARIOS) == 39
    for sc in SCENARIOS.values():
        if sc.role == "out_of_prior":
            assert set(sc.seed_sets) == {"D"}
        if sc.role == "control":
            assert set(sc.seed_sets) == {"A", "B", "C", "D"}
    assert SCENARIOS["fold_L1_slow"].split_for(2000) == "validation"

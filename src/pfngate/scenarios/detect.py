"""Tip detectors built from the `tip_detector` and `sanity_check` blocks of a family spec."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

# A detector gets the true state and the current drive value and says whether
# the event has happened at this step.
Detector = Callable[[list[float], float], bool]


def make_detector(spec: Mapping[str, Any], state_names: Sequence[str]) -> Detector | None:
    kind = spec["kind"]
    if kind == "none":
        return None
    if kind == "state_above":
        i = list(state_names).index(spec["variable"])
        thr = float(spec["threshold"])
        return lambda s, _: s[i] > thr
    if kind == "state_below":
        i = list(state_names).index(spec["variable"])
        thr = float(spec["threshold"])
        return lambda s, _: s[i] < thr
    if kind == "amplitude_above":
        thr2 = float(spec["threshold"]) ** 2
        return lambda s, _: s[0] * s[0] + s[1] * s[1] > thr2
    if kind == "escape_below_branch":
        margin = float(spec["margin"])
        return lambda s, p: s[0] < p - margin
    if kind == "moving_frame_above":
        thr = float(spec["threshold"])
        return lambda s, lam: s[0] + lam > thr
    raise ValueError(f"unknown detector kind '{kind}'")


def make_stay_condition(spec: Mapping[str, Any], state_names: Sequence[str]) -> Detector | None:
    """Condition the state must keep for `dwell` time units after the detector fires."""
    if not spec.get("dwell"):
        return None
    i = list(state_names).index(spec["variable"])
    level = float(spec["dwell_level"])
    return lambda s, _: s[i] > level

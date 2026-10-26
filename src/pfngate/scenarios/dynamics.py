"""Drift functions for each family. Parameter values come from the YAML specs.

A drift takes the current state and the current value of the driving signal
(p, lambda or mu, depending on the family) and returns dx/dt for each state variable.
Plain Python floats are used because the integrator steps one point at a time.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

Drift = Callable[[list[float], float], list[float]]


def _fold(params: Mapping[str, float]) -> Drift:
    def drift(s: list[float], p: float) -> list[float]:
        x = s[0]
        return [p + x - x**3]

    return drift


def _transcritical(params: Mapping[str, float]) -> Drift:
    def drift(s: list[float], p: float) -> list[float]:
        x = s[0]
        return [p * x - x * x]

    return drift


def _hopf_subcritical(params: Mapping[str, float]) -> Drift:
    omega = params["omega"]

    def drift(s: list[float], p: float) -> list[float]:
        x, y = s
        r2 = x * x + y * y
        radial = p + r2 - r2 * r2
        return [radial * x - omega * y, omega * x + radial * y]

    return drift


def _rate_saddle(params: Mapping[str, float]) -> Drift:
    tau = params["tau"]

    def drift(s: list[float], lam: float) -> list[float]:
        y = s[0] + lam
        return [(y * y - 1.0) / tau]

    return drift


def _double_well(params: Mapping[str, float]) -> Drift:
    def drift(s: list[float], _: float) -> list[float]:
        x = s[0]
        return [x - x**3]

    return drift


def _ou(params: Mapping[str, float]) -> Drift:
    theta = params["theta"]

    def drift(s: list[float], mu: float) -> list[float]:
        return [-theta * (s[0] - mu)]

    return drift


DRIFTS: dict[str, Callable[[Mapping[str, float]], Drift]] = {
    "fold": _fold,
    "transcritical": _transcritical,
    "hopf_subcritical": _hopf_subcritical,
    "rate_saddle": _rate_saddle,
    "double_well": _double_well,
    "ou": _ou,
}


def make_drift(dynamics: str, params: Mapping[str, float]) -> Drift:
    if dynamics not in DRIFTS:
        raise ValueError(f"unknown dynamics '{dynamics}'; known: {sorted(DRIFTS)}")
    return DRIFTS[dynamics](params)


def lower_branch_equilibrium(p0: float) -> float:
    """Lowest real equilibrium of the fold drift p0 + x - x^3, the starting state."""
    roots = np.roots([-1.0, 0.0, 1.0, p0])
    real = roots[np.abs(roots.imag) < 1e-9].real
    return float(real.min())

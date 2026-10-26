"""Euler-Maruyama integration with event detection and early stopping."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pfngate.scenarios.detect import Detector
from pfngate.scenarios.dynamics import Drift


@dataclass(frozen=True)
class RawRun:
    """True states at the observed points, and the first detector event (if any)."""

    states: np.ndarray  # (n_observed, dim)
    event_time: float | None


def _step(x: list[float], f: list[float], dt: float, sd: float, z: list[float]) -> list[float]:
    return [xi + fi * dt + sd * zi for xi, fi, zi in zip(x, f, z, strict=True)]


def euler_maruyama(
    drift: Drift,
    x0: list[float],
    drive: np.ndarray,
    noise_sd: np.ndarray,
    dt: float,
    observe_every: int,
    z: np.ndarray,
    z_burn: np.ndarray,
    detector: Detector | None,
    stop_after: float | None,
    stay: Detector | None = None,
    dwell: float = 0.0,
) -> RawRun:
    """Integrate dx = drift(x, drive) dt + noise_sd dW.

    Burn-in runs first with drive and noise held at their t = 0 values, and is
    discarded. Step k of the main run is at time k * dt. The state is recorded every
    `observe_every` steps.

    The event time is the first step where the detector fires and, if `stay` is given,
    the stay condition then holds for `dwell` time units. If the stay condition breaks
    first, that entry is discarded. When `stop_after` is set, the run stops
    `stop_after` time units after the event is confirmed.
    """
    sqrt_dt = math.sqrt(dt)
    x = list(x0)
    d0, sd0 = float(drive[0]), float(noise_sd[0]) * sqrt_dt
    for zk in z_burn.tolist():
        x = _step(x, drift(x, d0), dt, sd0, zk)

    drive_l = drive.tolist()
    sd_l = (noise_sd * sqrt_dt).tolist()
    z_l = z.tolist()
    n_steps = len(drive_l)
    end = n_steps
    dwell_steps = round(dwell / dt)
    candidate: int | None = None
    event: int | None = None
    recorded: list[list[float]] = []
    for k in range(n_steps):
        if k % observe_every == 0:
            recorded.append(x)
        if detector is not None and event is None:
            if candidate is not None and stay is not None and not stay(x, drive_l[k]):
                candidate = None
            if candidate is None and detector(x, drive_l[k]):
                candidate = k
            if candidate is not None and k - candidate >= dwell_steps:
                event = candidate
                if stop_after is not None:
                    end = min(n_steps, k + round(stop_after / dt) + 1)
        if k >= end - 1:
            break
        x = _step(x, drift(x, drive_l[k]), dt, sd_l[k], z_l[k])
        if not all(math.isfinite(v) and abs(v) < 1e6 for v in x):
            raise FloatingPointError(f"state diverged at t = {k * dt:.2f}: {x}")
    # Round away float noise from k * dt (e.g. 420.21000000000004).
    event_time = None if event is None else round(event * dt, 9)
    return RawRun(states=np.asarray(recorded), event_time=event_time)

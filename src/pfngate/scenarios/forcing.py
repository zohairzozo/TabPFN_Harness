"""Time-varying inputs for each ramp form: the driving signal and the process noise sd.

Every function maps a time grid to two arrays of the same length:
- drive: the value passed to the drift (p, lambda or mu; 0 when unused),
- noise_sd: the sd of the Wiener noise at that time.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

# Name of the control parameter exposed on a Trajectory, per ramp form.
CONTROL_NAMES: dict[str, str | None] = {
    "linear": "p",
    "linear_then_hold": "p",
    "linear_finite": "lambda",
    "linear_noise": "sigma",
    "steps": None,
    "none": None,
}


def drive_and_noise(
    ramp: Mapping[str, Any],
    params: Mapping[str, float],
    noise: Mapping[str, float],
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    form = ramp["form"]
    zeros = np.zeros_like(t)
    if form in ("linear", "linear_then_hold"):
        p0, p_c = params["p0"], params["p_c"]
        p = p0 + (p_c - p0) * t / ramp["time_to_critical"]
        if form == "linear_then_hold":
            p = np.minimum(p, params["p_stop"])
        return p, np.full_like(t, noise["process_sd"])
    if form == "linear_finite":
        lam = np.clip(ramp["rate"] * (t - params["ramp_start"]), 0.0, params["lambda_max"])
        return lam, np.full_like(t, noise["process_sd"])
    if form == "linear_noise":
        frac = np.clip((t - params["ramp_start"]) / ramp["ramp_duration"], 0.0, 1.0)
        sigma = noise["sigma0"] + (params["sigma_max"] - noise["sigma0"]) * frac
        return zeros, sigma
    if form == "steps":
        stationary_sd = noise["process_sd"] / np.sqrt(2.0 * params["theta"])
        mu = np.where(t >= params["t_shift"], params["shift_size"] * stationary_sd, 0.0)
        sd = np.where(
            t >= params["t_var"], params["var_factor"] * noise["process_sd"], noise["process_sd"]
        )
        return mu, sd
    if form == "none":
        return zeros, np.full_like(t, noise["process_sd"])
    raise ValueError(f"unknown ramp form '{form}'")


def control_parameter(
    ramp: Mapping[str, Any], drive: np.ndarray, noise_sd: np.ndarray
) -> tuple[str | None, np.ndarray | None]:
    """The ramped quantity shown on a Trajectory, or None when the family has no ramp."""
    name = CONTROL_NAMES[ramp["form"]]
    if name is None:
        return None, None
    return name, (noise_sd if name == "sigma" else drive)

"""The Trajectory record returned by Scenario.generate()."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Trajectory:
    """One simulated series. All arrays share the observed time grid of length T.

    `state` is the true state before measurement noise. Models must only see
    `observations`; `state` is kept for tests and plots.
    """

    scenario_id: str
    family: str
    role: str  # in_prior, out_of_prior or control
    split: str  # from the seed set: context, validation, test_in_prior, test_out_of_prior
    seed: int
    time: np.ndarray  # (T,)
    observations: np.ndarray  # (T, channels)
    state: np.ndarray  # (T, state_dim)
    control_name: str | None  # "p", "lambda", "sigma" or None
    control_parameter: np.ndarray | None  # (T,) or None
    tip_time: float | None
    target: np.ndarray  # (T,) regression target, NaN if it cannot be defined

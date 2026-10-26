"""Scenario objects built from configs/scenarios/*.yaml, and their generate(seed)."""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pfngate.scenarios import labels
from pfngate.scenarios.config import DEFAULT_CONFIG_DIR, CommonConfig, load_common, read_yaml
from pfngate.scenarios.detect import make_detector, make_stay_condition
from pfngate.scenarios.dynamics import lower_branch_equilibrium, make_drift
from pfngate.scenarios.forcing import control_parameter, drive_and_noise
from pfngate.scenarios.simulate import euler_maruyama
from pfngate.scenarios.trajectory import Trajectory


@dataclass(frozen=True)
class Scenario:
    """One cell of a family grid (noise level x ramp speed), ready to simulate."""

    scenario_id: str
    family: str
    role: str
    dynamics: str
    noise_level: str
    speed: str | None
    noise: dict[str, float]
    ramp: dict[str, Any]
    params: dict[str, float]
    state_names: tuple[str, ...]
    initial_state: dict[str, Any]
    observed: tuple[str, ...]
    tip_detector: dict[str, Any]
    sanity_check: dict[str, Any] | None
    label_kind: str
    n_points: int
    seed_sets: tuple[str, ...]
    common: CommonConfig

    def seeds(self) -> list[int]:
        """All seeds this scenario may be generated with, across its seed sets."""
        return [s for name in self.seed_sets for s in self.common.seed_sets[name].seeds()]

    def split_for(self, seed: int) -> str:
        for name in self.seed_sets:
            if seed in self.common.seed_sets[name]:
                return self.common.seed_sets[name].split
        raise ValueError(
            f"seed {seed} is not in the seed sets {self.seed_sets} of {self.scenario_id}"
        )

    def x0(self) -> list[float]:
        kind = self.initial_state["kind"]
        if kind == "lower_branch_equilibrium":
            return [lower_branch_equilibrium(self.params["p0"])]
        if kind == "value":
            return [float(v) for v in self.initial_state["value"]]
        raise ValueError(f"unknown initial_state kind '{kind}'")

    def generate(self, seed: int) -> Trajectory:
        split = self.split_for(seed)
        c = self.common
        ss = np.random.SeedSequence([zlib.crc32(self.scenario_id.encode()), seed])
        rng = np.random.default_rng(ss)
        dim = len(self.state_names)
        n_steps = self.n_points * c.observe_every
        # Draw all noise up front so early stopping does not change the random stream.
        z_burn = rng.standard_normal((round(c.burn_in / c.dt), dim))
        z = rng.standard_normal((n_steps, dim))
        z_obs = rng.standard_normal((self.n_points, len(self.observed)))

        t_steps = np.arange(n_steps) * c.dt
        drive, noise_sd = drive_and_noise(self.ramp, self.params, self.noise, t_steps)
        run = euler_maruyama(
            drift=make_drift(self.dynamics, self.params),
            x0=self.x0(),
            drive=drive,
            noise_sd=noise_sd,
            dt=c.dt,
            observe_every=c.observe_every,
            z=z,
            z_burn=z_burn,
            detector=make_detector(self.tip_detector, self.state_names),
            stop_after=self.tip_detector.get("stop_after"),
            stay=make_stay_condition(self.tip_detector, self.state_names),
            dwell=float(self.tip_detector.get("dwell") or 0.0),
        )

        n_obs = len(run.states)
        time = t_steps[:: c.observe_every][:n_obs]
        cols = [self.state_names.index(v) for v in self.observed]
        observations = run.states[:, cols] + self.noise["obs_sd"] * z_obs[:n_obs]

        tip_time = run.event_time
        if self.tip_detector.get("tip_at_critical"):
            t_c = float(self.ramp["time_to_critical"])
            tip_time = t_c if tip_time is None else min(tip_time, t_c)

        control_name, control_full = control_parameter(self.ramp, drive, noise_sd)
        control = None if control_full is None else control_full[:: c.observe_every][:n_obs]
        return Trajectory(
            scenario_id=self.scenario_id,
            family=self.family,
            role=self.role,
            split=split,
            seed=seed,
            time=time,
            observations=observations,
            state=run.states,
            control_name=control_name,
            control_parameter=control,
            tip_time=tip_time,
            target=labels.target(self.label_kind, time, control, self.params, tip_time, c.horizon),
        )


def _n_points(
    spec: dict[str, Any], speed: str | None, ramp: dict[str, Any], c: CommonConfig
) -> int:
    lengths = spec["series_length_points"]
    if speed is not None and speed in lengths:
        n = int(lengths[speed])
    elif "fixed" in lengths:
        n = int(lengths["fixed"])
    else:
        n = int(lengths["max"])
    if "run_after_critical" in ramp:
        expected = (1.0 + ramp["run_after_critical"]) * ramp["time_to_critical"] / c.obs_dt
        if not math.isclose(n, expected):
            raise ValueError(
                f"{spec['family']}/{speed}: {n} points, but the ramp implies {expected}"
            )
    if ramp["form"] == "linear_noise":
        p = spec["parameters"]
        end = p["ramp_start"] + ramp["ramp_duration"] + p["hold_after_ramp"]
        n = min(n, round(end / c.obs_dt))
    return n


def _family_scenarios(spec: dict[str, Any], c: CommonConfig) -> list[Scenario]:
    levels = spec["noise_levels"]
    own_levels = isinstance(levels, dict)
    params = {k: float(v) for k, v in spec.get("parameters", {}).items()}
    out = []
    for entry in spec["scenarios"]:
        level, speed = entry["noise_level"], entry["speed"]
        noise = levels[level] if own_levels else c.noise_levels[level]
        ramp: dict[str, Any] = {"form": spec["ramp"]["form"]}
        if speed is not None:
            ramp.update(spec["ramp"]["speeds"][speed])
        if "run_after_critical" in spec["ramp"]:
            ramp["run_after_critical"] = spec["ramp"]["run_after_critical"]
        out.append(
            Scenario(
                scenario_id=entry["id"],
                family=spec["family"],
                role=spec["role"],
                dynamics=spec["dynamics"],
                noise_level=level,
                speed=speed,
                noise={k: float(v) for k, v in noise.items()},
                ramp=ramp,
                params=params,
                state_names=tuple(spec["state"]),
                initial_state=spec["initial_state"],
                observed=tuple(spec["observation"]["variables"]),
                tip_detector=spec["tip_detector"],
                sanity_check=spec.get("sanity_check"),
                label_kind=spec["label"]["kind"],
                n_points=_n_points(spec, speed, ramp, c),
                seed_sets=tuple(spec["seed_sets"]),
                common=c,
            )
        )
    return out


def load_scenarios(config_dir: Path = DEFAULT_CONFIG_DIR) -> dict[str, Scenario]:
    """All scenarios from the family files listed in _common.yaml, keyed by scenario id."""
    c = load_common(config_dir)
    scenarios: dict[str, Scenario] = {}
    for name in c.family_files:
        for sc in _family_scenarios(read_yaml(config_dir / name), c):
            if sc.scenario_id in scenarios:
                raise ValueError(f"duplicate scenario id {sc.scenario_id}")
            unknown = set(sc.seed_sets) - set(c.seed_sets)
            if unknown:
                raise ValueError(f"{sc.scenario_id}: unknown seed sets {unknown}")
            scenarios[sc.scenario_id] = sc
    return scenarios

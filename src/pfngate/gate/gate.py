"""Run the gate on one run, and map its result and p_critical to a decision per window.

evaluate_gate() is a pure function of a frozen GateConfig and a frozen GateInput. GateInput
has no field for the query targets or labels: the gate cannot see the ground truth of the
query trajectory. GateResult is frozen, validates that its overall flag agrees with its
checks, and refuses model_copy with changes. Advisories (the interval width cap) are
computed and reported, but never block a run. See docs/GATE.md.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from pfngate.evals.calibration import level_index
from pfngate.gate import checks
from pfngate.gate.checks import CheckResult
from pfngate.gate.config import GateConfig, GateStatus
from pfngate.pfn.adapter import PFNOutput

Decision = Literal["abstain", "alert", "no_alert"]


def _code_version() -> str:
    try:
        return version("pfngate")
    except PackageNotFoundError:
        return "unknown"


class GateInput(BaseModel):
    """Everything the gate may see about one run. Nothing here comes from the query targets."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    run_id: str
    trajectory_id: str
    strategy: str
    context_hash: str
    context_size: int
    context_y: np.ndarray  # context targets: from the pool (set A), never from the query
    context_n_trajectories: int
    calibration_coverage: float | None  # coverage on held-out context trajectories
    pool_distances: np.ndarray  # per query window, distance to the k-th nearest pool row
    near_tipping_distances: np.ndarray  # per query window, to the k-th near-tipping context row
    output: PFNOutput  # the backend's predictions for the query windows

    @field_validator("context_y", "pool_distances", "near_tipping_distances", mode="before")
    @classmethod
    def _read_only(cls, v: Any) -> np.ndarray:
        arr = np.array(v, dtype=float)
        arr.flags.writeable = False
        return arr

    @model_validator(mode="after")
    def _shapes(self) -> GateInput:
        if len(self.context_y) != self.context_size:
            raise ValueError("context_y must have one value per context row")
        if self.pool_distances.shape != (self.output.n_query,):
            raise ValueError("pool_distances must have one value per query window")
        if self.near_tipping_distances.shape != (self.output.n_query,):
            raise ValueError("near_tipping_distances must have one value per query window")
        return self


class GateResult(BaseModel):
    """The gate's verdict on one run. Frozen: assignment raises, and so does model_copy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    trajectory_id: str
    strategy: str
    context_hash: str
    checks: tuple[CheckResult, ...]  # blocking: the run passes only if all of these pass
    advisories: tuple[CheckResult, ...]  # reported only: they never change `passed`
    passed: bool
    reasons: tuple[str, ...]
    gate_config_sha256: str
    gate_config_status: GateStatus
    gate_code_version: str

    @model_validator(mode="after")
    def _consistent(self) -> GateResult:
        if not self.checks:
            raise ValueError("a gate result needs at least one check")
        if self.passed != all(c.passed for c in self.checks):
            raise ValueError("passed must be true exactly when every check passed")
        if self.reasons != tuple(f"{c.name}: {c.reason}" for c in self.checks if not c.passed):
            raise ValueError("reasons must list the failed checks, in check order")
        if {c.name for c in self.checks} & {a.name for a in self.advisories}:
            raise ValueError("a check cannot be both blocking and advisory")
        return self

    def model_copy(self, *, update: Any = None, deep: bool = False) -> GateResult:
        if update:
            raise TypeError("a GateResult cannot be changed; run the gate again instead")
        return super().model_copy(deep=deep)

    @property
    def failed(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.checks if not c.passed)

    @property
    def advisories_failed(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.advisories if not a.passed)


def evaluate_gate(config: GateConfig, inp: GateInput) -> GateResult:
    """Run every check with the frozen thresholds. The run passes only if all checks pass.

    The interval width cap and near_tipping_support are advisories: reported, never
    blocking.
    """
    c = config.checks
    out = inp.output
    lo_i = level_index(out.quantile_levels, config.bad_run.interval[0])
    hi_i = level_index(out.quantile_levels, config.bad_run.interval[1])
    md = out.metadata
    results = (
        checks.min_context_size(inp.context_size, c.min_context_size.threshold),
        checks.ood_share(
            inp.pool_distances,
            c.ood_share.distance_cut,
            c.ood_share.limit,
            config.source_of("ood_share"),
        ),
        checks.context_calibration(
            inp.calibration_coverage,
            inp.context_n_trajectories,
            c.context_calibration.folds,
            c.context_calibration.low,
            c.context_calibration.high,
            config.source_of("context_calibration"),
        ),
        checks.context_not_degenerate(
            inp.context_y,
            c.context_not_degenerate.positive_max,
            c.context_not_degenerate.negative_min,
            c.context_not_degenerate.min_rows,
        ),
        checks.output_integrity(
            md.n_rows_repaired,
            out.n_query,
            md.p_critical_method,
            c.output_integrity.max_repaired_share,
            c.output_integrity.required_method,
        ),
    )
    advisories = (
        checks.interval_width_cap(
            out.quantiles[:, lo_i], out.quantiles[:, hi_i], c.interval_width_cap.threshold
        ),
        checks.near_tipping_support(
            inp.near_tipping_distances,
            int((inp.context_y <= c.near_tipping_support.target_max).sum()),
            c.near_tipping_support.k,
            c.near_tipping_support.distance_cut,
            c.near_tipping_support.limit,
        ),
    )
    return GateResult(
        run_id=inp.run_id,
        trajectory_id=inp.trajectory_id,
        strategy=inp.strategy,
        context_hash=inp.context_hash,
        checks=results,
        advisories=advisories,
        passed=all(r.passed for r in results),
        reasons=tuple(f"{r.name}: {r.reason}" for r in results if not r.passed),
        gate_config_sha256=config.sha256,
        gate_config_status=config.status,
        gate_code_version=_code_version(),
    )


def decide(
    result: GateResult, p_critical: np.ndarray, alert_threshold: float
) -> tuple[Decision, ...]:
    """One decision per window: abstain if the gate failed, else alert or no_alert."""
    p = np.asarray(p_critical, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all():
        raise ValueError("p_critical must be a 1-D array of finite values")
    if not result.passed:
        return ("abstain",) * len(p)
    return tuple("alert" if v >= alert_threshold else "no_alert" for v in p)

"""The gate thresholds in configs/gate.yaml, loaded into frozen models.

The file is the only source of thresholds. There is no function that sets a threshold at
runtime. Loading refuses a file that is looser than the floors below, so a loosened YAML
fails before any run starts. The floors are part of the code, not of the config: changing
them means changing this file, which shows up in review. See docs/GATE.md.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pfngate.features import FEATURE_NAMES
from pfngate.features.config import REPO_ROOT

DEFAULT_GATE_CONFIG = REPO_ROOT / "configs" / "gate.yaml"
DEFAULT_GATE_HASH = REPO_ROOT / "configs" / "gate.frozen.sha256"

GateStatus = Literal["provisional", "tuned"]

# Hard limits a config may not be looser than (docs/GATE.md, "Tests that fail if someone
# loosens the gate"). A test pins these values.
FLOORS: dict[str, float] = {
    "min_context_size_at_least": 50,
    "ood_share_limit_at_most": 0.3,
    "near_tipping_limit_at_most": 0.5,
    "interval_width_cap_at_most": 0.8,
    "calibration_low_at_least": 0.6,
    "calibration_folds_at_least": 3,
    "degenerate_min_rows_at_least": 5,
    "repaired_share_at_most": 0.05,
}
# The definition of a BAD run is fixed in docs/GATE.md and is not tunable.
BAD_RUN_INTERVAL = (0.05, 0.95)
BAD_RUN_MIN_COVERAGE = 0.80


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BadRunDefinition(_Frozen):
    interval: tuple[float, float]
    min_coverage: float
    tolerance: float = Field(gt=0.0, le=1e-3)


class MinContextSize(_Frozen):
    threshold: int


class OODShare(_Frozen):
    limit: float = Field(gt=0.0)
    k: int = Field(ge=1)
    quantile: float = Field(gt=0.5, lt=1.0)
    n_sample: int = Field(ge=100)
    distance_cut: float = Field(gt=0.0)
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_std: tuple[float, ...]
    n_pool_rows: int = Field(ge=1)
    n_pool_trajectories: int = Field(ge=2)


class NearTippingSupport(_Frozen):
    """Does the context hold near-tipping rows that look like this query? (decision 8)

    Uses the standardisation stored under ood_share. distance_cut comes from the pool.
    Reported in GateResult.advisories and never blocks: its cut is fitted on the density
    of the whole pool, so it blocked almost every random and knn run (docs/GATE.md, round 3).
    """

    role: Literal["advisory"]
    limit: float = Field(gt=0.0)
    k: int = Field(ge=1)
    target_max: float
    quantile: float = Field(gt=0.5, lt=1.0)
    n_sample: int = Field(ge=100)
    distance_cut: float = Field(gt=0.0)
    n_near_rows: int = Field(ge=1)


class IntervalWidthCap(_Frozen):
    """Reported in GateResult.advisories and never blocks a run (docs/GATE.md, decision 7).

    `role` must say so in the file, so a config written before decision 7 does not load.
    """

    threshold: float = Field(gt=0.0)
    role: Literal["advisory"]


class ContextCalibration(_Frozen):
    low: float
    high: float = Field(le=1.0)
    folds: int


class ContextNotDegenerate(_Frozen):
    min_rows: int
    positive_max: float  # binary positive zone, docs/LABELS.md
    negative_min: float  # binary negative zone


class OutputIntegrity(_Frozen):
    max_repaired_share: float = Field(ge=0.0)
    required_method: Literal["full_distribution"]


class Checks(_Frozen):
    min_context_size: MinContextSize
    ood_share: OODShare
    near_tipping_support: NearTippingSupport
    interval_width_cap: IntervalWidthCap
    context_calibration: ContextCalibration
    context_not_degenerate: ContextNotDegenerate
    output_integrity: OutputIntegrity


class Applicability(_Frozen):
    """The run settings the thresholds were set for. Held-out runs must match them."""

    backend: str
    model_version: str
    package_version: str
    strategies: tuple[str, ...]
    max_context: int = Field(ge=1)
    data_version: str


class GateConfig(_Frozen):
    status: GateStatus
    bad_run: BadRunDefinition
    checks: Checks
    # One p_critical alert threshold per strategy, None until tuned. Stored as pairs, not a
    # dict, so the frozen model holds no mutable container.
    alert_thresholds: tuple[tuple[str, float | None], ...]
    applicability: Applicability
    sha256: str  # of the file bytes, filled by load_gate_config
    source: str

    @field_validator("alert_thresholds", mode="before")
    @classmethod
    def _pairs(cls, v: object) -> object:
        return tuple(sorted(v.items())) if isinstance(v, dict) else v

    def alert_threshold(self, strategy: str) -> float | None:
        return dict(self.alert_thresholds).get(strategy)

    @model_validator(mode="after")
    def _respect_floors(self) -> GateConfig:
        c, f = self.checks, FLOORS
        problems = []
        if tuple(self.bad_run.interval) != BAD_RUN_INTERVAL:
            problems.append(f"bad_run.interval must be {BAD_RUN_INTERVAL}")
        if self.bad_run.min_coverage != BAD_RUN_MIN_COVERAGE:
            problems.append(f"bad_run.min_coverage must be {BAD_RUN_MIN_COVERAGE}")
        if c.min_context_size.threshold < f["min_context_size_at_least"]:
            problems.append("min_context_size.threshold is below its floor")
        if c.ood_share.limit > f["ood_share_limit_at_most"]:
            problems.append("ood_share.limit is above its ceiling")
        if c.near_tipping_support.limit > f["near_tipping_limit_at_most"]:
            problems.append("near_tipping_support.limit is above its ceiling")
        if c.near_tipping_support.target_max != c.context_not_degenerate.positive_max:
            problems.append("near_tipping_support.target_max must be the positive zone")
        if c.interval_width_cap.threshold > f["interval_width_cap_at_most"]:
            problems.append("interval_width_cap.threshold is above its ceiling")
        if c.context_calibration.low < f["calibration_low_at_least"]:
            problems.append("context_calibration.low is below its floor")
        if not c.context_calibration.low < c.context_calibration.high:
            problems.append("context_calibration.low must be below high")
        if c.context_calibration.folds < f["calibration_folds_at_least"]:
            problems.append("context_calibration.folds is below its floor")
        if c.context_not_degenerate.min_rows < f["degenerate_min_rows_at_least"]:
            problems.append("context_not_degenerate.min_rows is below its floor")
        if c.output_integrity.max_repaired_share > f["repaired_share_at_most"]:
            problems.append("output_integrity.max_repaired_share is above its ceiling")
        o = c.ood_share
        if o.feature_names != tuple(FEATURE_NAMES):
            problems.append("ood_share.feature_names do not match the feature code")
        if not len(o.feature_mean) == len(o.feature_std) == len(o.feature_names):
            problems.append("ood_share feature mean and std need one value per feature")
        if any(s <= 0.0 for s in o.feature_std):
            problems.append("ood_share.feature_std must be positive")
        if problems:
            raise ValueError("gate config refused: " + "; ".join(problems))
        return self

    def source_of(self, check: str) -> Literal["fixed", "provisional", "tuned"]:
        """Tuned checks carry the file status. The others are fixed by design."""
        return self.status if check in TUNED_CHECKS else "fixed"


TUNED_CHECKS: frozenset[str] = frozenset({"ood_share", "context_calibration"})


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gate_config(path: Path = DEFAULT_GATE_CONFIG) -> GateConfig:
    """Read and validate the gate thresholds. Raises if the file breaks a floor."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or {"sha256", "source"} & set(raw):
        raise ValueError(f"{path} must be a mapping without 'sha256' or 'source' keys")
    return GateConfig.model_validate({**raw, "sha256": sha256_file(path), "source": str(path)})

"""Release criteria: read from configs/release_criteria.yaml, checked on files in results/.

The verdict is plain Python. Nothing is typed by hand: every value comes from a result file
and every threshold from the YAML file. The outcome is also written to
results/report/criteria.json, so the page and the number test can read it.

Each value comes with a 95% interval (exact binomial for shares, bootstrap over runs for the
coverage). The verdict still uses the point estimate, as the criteria file says. An outcome
whose interval spans its threshold is marked undecided, because a few more runs could flip it.
"""

from __future__ import annotations

import json
import operator
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import beta

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CRITERIA = REPO_ROOT / "configs" / "release_criteria.yaml"
CONFIDENCE = 0.95
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0
COMPARISONS: dict[str, Callable[[float, float], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
}


@dataclass(frozen=True)
class Criterion:
    key: str
    label: str
    comparison: str
    threshold: float
    required: bool


@dataclass
class Outcome:
    key: str
    label: str
    comparison: str
    threshold: float
    required: bool
    value: float | None  # None: it could not be computed
    numerator: int | None
    denominator: int | None
    unit: str  # what the numerator and denominator count
    sources: list[str]
    passed: bool
    ci_low: float | None = None  # the interval of the value, at CONFIDENCE
    ci_high: float | None = None
    decided: bool | None = None  # False: the interval spans the threshold


@dataclass
class Verdict:
    verdict: str  # "GO" or "NO-GO"
    stand_in: str
    variant: str
    outcomes: list[Outcome]

    @property
    def required(self) -> list[Outcome]:
        """The outcomes of the criteria that must pass."""
        return [o for o in self.outcomes if o.required]

    @property
    def undecided(self) -> list[Outcome]:
        """The required outcomes whose interval spans their threshold."""
        return [o for o in self.required if o.decided is False]

    @property
    def reported_only(self) -> list[Outcome]:
        """The outcomes of the criteria that are reported but not required."""
        return [o for o in self.outcomes if not o.required]


@dataclass
class Measure:
    value: float | None
    numerator: int | None
    denominator: int | None
    unit: str
    sources: list[str]
    ci: tuple[float, float] | None = None


def clopper_pearson(k: int, n: int, confidence: float = CONFIDENCE) -> tuple[float, float]:
    """The exact binomial interval for k successes in n trials."""
    tail = (1.0 - confidence) / 2.0
    low = 0.0 if k == 0 else float(beta.ppf(tail, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1.0 - tail, k + 1, n - k))
    return low, high


def bootstrap_mean_ci(
    values: list[float], confidence: float = CONFIDENCE
) -> tuple[float, float] | None:
    """A percentile bootstrap interval of the mean over runs, with a fixed seed."""
    if not values:
        return None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(np.asarray(values, dtype=float), size=(BOOTSTRAP_RESAMPLES, len(values)))
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws.mean(axis=1), [tail, 1.0 - tail])
    return float(low), float(high)


def is_decided(comparison: str, threshold: float, ci: tuple[float, float] | None) -> bool | None:
    """True if the whole interval is on one side of the threshold, None if there is none."""
    if ci is None:
        return None
    low, high = ci
    if comparison == ">=":
        return low >= threshold or high < threshold
    return high <= threshold or low > threshold


def load_criteria(path: Path = DEFAULT_CRITERIA) -> tuple[str, str, list[Criterion]]:
    """Read the variant, stand-in and criteria from the criteria file."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    criteria = [
        Criterion(
            key=str(c["key"]),
            label=str(c["label"]),
            comparison=str(c["comparison"]),
            threshold=float(c["threshold"]),
            required=bool(c["required"]),
        )
        for c in raw["criteria"]
    ]
    for c in criteria:
        if c.comparison not in COMPARISONS:
            raise ValueError(f"{c.key}: comparison must be one of {sorted(COMPARISONS)}")
    keys = [c.key for c in criteria]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate criterion key")
    return str(raw["variant"]), str(raw["stand_in"]), criteria


def read_runs(results_dir: Path, variant: str, split: str) -> dict[str, Any] | None:
    """The result file of a variant and split, or None if it is missing."""
    path = results_dir / f"{variant}_{split}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def final_decision(run: dict[str, Any]) -> str | None:
    """The final decision of a run record, as a string."""
    decision = run.get("decision")
    return decision.get("decision") if isinstance(decision, dict) else decision


def coverage_accepted(results_dir: Path, variant: str, split: str) -> Measure:
    """Mean 90% coverage of the accepted runs of a variant and split."""
    name = f"{variant}_{split}.json"
    doc = read_runs(results_dir, variant, split)
    pfn = ((doc or {}).get("summary") or {}).get("pfn") or {}
    n = pfn.get("n_accepted")
    covers = [
        r["pfn_quality"]["coverage_90"]
        for r in (doc or {}).get("runs", [])
        if r.get("accepted") is True and r.get("pfn_quality")
    ]
    return Measure(
        pfn.get("coverage_90_mean_accepted") if n else None,
        None,
        int(n) if n is not None else None,
        "accepted runs",
        [name] if doc else [],
        bootstrap_mean_ci(covers) if n else None,
    )


def gate_recall(results_dir: Path, variant: str, split: str) -> Measure:
    """The share of bad runs that the gate blocked, for a variant and split."""
    name = f"{variant}_{split}.json"
    doc = read_runs(results_dir, variant, split)
    pfn = ((doc or {}).get("summary") or {}).get("pfn") or {}
    n_bad = pfn.get("n_bad")
    blocked = pfn.get("n_bad_blocked")
    if not n_bad or blocked is None:
        return Measure(None, None, int(n_bad) if n_bad is not None else None, "bad runs", [name])
    ci = clopper_pearson(int(blocked), int(n_bad))
    return Measure(blocked / n_bad, int(blocked), int(n_bad), "bad runs blocked", [name], ci)


def false_alarms(results_dir: Path, variant: str, splits: tuple[str, ...]) -> Measure:
    """The share of control runs that alerted, over the given splits."""
    alerts = controls = 0
    sources: list[str] = []
    for split in splits:
        doc = read_runs(results_dir, variant, split)
        if doc is None:
            continue
        sources.append(f"{variant}_{split}.json")
        for run in doc["runs"]:
            if run["ground_truth"]["role"] == "control":
                controls += 1
                alerts += final_decision(run) == "alert"
    if not controls:
        return Measure(None, None, 0, "control runs", sources)
    ci = clopper_pearson(alerts, controls)
    return Measure(alerts / controls, alerts, controls, "control runs that alerted", sources, ci)


def measure(results_dir: Path, variant: str, key: str) -> Measure:
    """The value of one criterion, read from the result files."""
    if key == "coverage_accepted_in_prior":
        return coverage_accepted(results_dir, variant, "test")
    if key == "coverage_accepted_out_of_prior":
        return coverage_accepted(results_dir, variant, "ood")
    if key == "gate_recall_bad_in_prior":
        return gate_recall(results_dir, variant, "test")
    if key == "gate_recall_bad_out_of_prior":
        return gate_recall(results_dir, variant, "ood")
    if key == "false_alarms_simulated_controls":
        return false_alarms(results_dir, variant, ("test", "ood"))
    if key == "false_alarms_real_controls":
        return false_alarms(results_dir, variant, ("real", "real_control"))
    raise ValueError(f"no way to compute the criterion '{key}'")


def evaluate(results_dir: Path, criteria_path: Path = DEFAULT_CRITERIA) -> Verdict:
    """Check every criterion and return the verdict, GO or NO-GO."""
    variant, stand_in, criteria = load_criteria(criteria_path)
    outcomes: list[Outcome] = []
    for c in criteria:
        m = measure(results_dir, variant, c.key)
        passed = m.value is not None and COMPARISONS[c.comparison](m.value, c.threshold)
        outcomes.append(
            Outcome(
                c.key,
                c.label,
                c.comparison,
                c.threshold,
                c.required,
                m.value,
                m.numerator,
                m.denominator,
                m.unit,
                m.sources,
                bool(passed),
                ci_low=m.ci[0] if m.ci else None,
                ci_high=m.ci[1] if m.ci else None,
                decided=is_decided(c.comparison, c.threshold, m.ci),
            )
        )
    go = all(o.passed for o in outcomes if o.required)
    return Verdict("GO" if go else "NO-GO", stand_in, variant, outcomes)


def write_criteria_json(verdict: Verdict, out_dir: Path) -> Path:
    """Write the verdict and its outcomes to criteria.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "criteria.json"
    doc = {
        "verdict": verdict.verdict,
        "stand_in": verdict.stand_in,
        "variant": verdict.variant,
        "required_met": sum(o.passed for o in verdict.required),
        "required_total": len(verdict.required),
        "required_undecided": len(verdict.undecided),
        "confidence": CONFIDENCE,
        "criteria_file": "configs/release_criteria.yaml",
        "outcomes": [asdict(o) for o in verdict.outcomes],
    }
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path

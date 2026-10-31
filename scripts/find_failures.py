"""Write results/failures.json: the top examples of four kinds of failure, ranked and linked.

Reads results/*.json and the trace files they name in results/traces/. Nothing here is typed by
hand. It ranks and links; it does not say why anything failed.

Categories
  a  false alarms on negative controls (simulated or real): a control run whose decision is alert.
     Severity: how strong the alert was, in the variant's own words (see severity_basis).
  b  bad runs the gate let through: a BAD run (90% interval covers less than the minimum share of
     its window targets) that the gate accepted. Severity: minimum coverage minus coverage.
  c  good runs the gate blocked: a run that is not BAD and that the gate blocked. Severity:
     coverage minus minimum coverage.
  d  the agent's recommendation disagreed with the final gated decision (full harness only).
     Severity: the confidence the agent gave its recommendation.

Scope: held-out splits only (test, ood, real, real_control), because the gate thresholds and the
alarm thresholds were set on validation. Use --include-val to add the validation splits. Rows from
different variants are ranked on their own severity scales, which are named in each example.

Usage: python scripts/find_failures.py [--results-dir results] [--top 5] [--include-val]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_CONFIG = REPO_ROOT / "configs" / "gate.yaml"
HELD_OUT = ("test", "ood", "real", "real_control")
VARIANT_FILES = {
    "classic": "classic_ews",
    "llm_only": "llm_only",
    "naive_agent": "naive_agent",
    "agent": "agent",
    "oracle": "oracle",
}
PFN_STRATEGIES = ("random", "knn")
# the split name inside a PFN result file -> the split name used by the variant files
PFN_SPLITS = {"validation": "val", "test_in_prior": "test", "test_out_of_prior": "ood"}
CATEGORIES = {
    "a_false_alarms_on_controls": "False alarms on negative controls",
    "b_bad_runs_the_gate_let_through": "Bad runs the gate let through",
    "c_good_runs_the_gate_blocked": "Good runs the gate blocked",
    "d_agent_disagreed_with_gated_decision": "Agent recommendation differed from final decision",
}


@dataclass
class Run:
    variant: str
    split: str
    trajectory_id: str
    scenario_id: str
    role: str  # "control" or "tipping"
    source_file: str
    trace_name: str | None
    decision: str | None = None
    accepted: bool | None = None  # None: this variant has no gate
    bad: bool | None = None
    coverage_90: float | None = None
    strength: float | None = None
    strength_basis: str = ""
    agent_agrees: bool | None = None
    confidence: float | None = None
    numbers: dict[str, Any] = field(default_factory=dict)

    @property
    def seed(self) -> int | None:
        tail = self.trajectory_id.rsplit("/", 1)[-1]
        return int(tail) if tail.isdigit() else None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def final_decision(run: dict[str, Any]) -> str | None:
    decision = run.get("decision")
    return decision.get("decision") if isinstance(decision, dict) else decision


def min_coverage() -> float | None:
    """The BAD-run line, read from the frozen gate config."""
    if not GATE_CONFIG.exists():
        return None
    value = yaml.safe_load(GATE_CONFIG.read_text(encoding="utf-8")).get("bad_run", {})
    return value.get("min_coverage")


def strength_of(variant: str, rec: dict[str, Any]) -> tuple[float | None, str]:
    """How strongly a variant stated its decision, for ranking alerts. Each in [0, 1]."""
    if variant == "classic":
        return rec.get("score"), "classic baseline score of the last window"
    if variant == "oracle":
        return rec.get("score"), "p_critical of the last window"
    if variant == "llm_only":
        return (rec.get("answer") or {}).get("probability"), "the model's stated probability"
    if variant == "naive_agent":
        return (rec.get("verdict") or {}).get("confidence"), "the agent's stated confidence"
    final = rec.get("final") or {}
    return final.get("last_window_p_critical"), "p_critical of the last window"


def variant_numbers(rec: dict[str, Any], gt: dict[str, Any]) -> dict[str, Any]:
    final = rec.get("final") or {}
    verdict = rec.get("verdict") or {}
    numbers: dict[str, Any] = {
        "end_label": gt.get("end_label"),
        "end_target": gt.get("end_target"),
        "alarm_threshold": rec.get("alarm_threshold") or rec.get("alert_threshold"),
        "n_windows": rec.get("n_windows"),
        "answer_probability": (rec.get("answer") or {}).get("probability"),
        "verdict_decision": verdict.get("decision"),
        "verdict_confidence": verdict.get("confidence"),
        "verdict_context_strategy": verdict.get("context_strategy"),
        "final_decision": final.get("decision"),
        "decided_by": final.get("decided_by"),
        "gate_passed": final.get("gate_passed"),
        "gate_failed": final.get("gate_failed"),
        "last_window_p_critical": final.get("last_window_p_critical"),
        "alert_threshold": final.get("alert_threshold"),
        "window_decisions": final.get("window_decisions"),
        "context_strategy": final.get("context_strategy"),
        "followed_verdict_context": final.get("followed_verdict_context"),
    }
    return {k: v for k, v in numbers.items() if v is not None}


def variant_runs(results_dir: Path, splits: tuple[str, ...]) -> list[Run]:
    out: list[Run] = []
    for variant, prefix in VARIANT_FILES.items():
        for split in splits:
            name = f"{prefix}_{split}.json"
            path = results_dir / name
            if not path.exists():
                continue
            doc = read_json(path)
            for r in doc["runs"]:
                rec, gt = r["record"], r["ground_truth"]
                quality = r.get("pfn_quality") or {}
                strength, basis = strength_of(variant, rec)
                verdict = rec.get("verdict") or {}
                numbers = variant_numbers(rec, gt)
                if quality:
                    numbers["coverage_90"] = quality.get("coverage_90")
                out.append(
                    Run(
                        variant=variant,
                        split=split,
                        trajectory_id=rec["trajectory_id"],
                        scenario_id=gt["scenario_id"],
                        role="control" if gt["role"] == "control" else "tipping",
                        source_file=name,
                        trace_name=doc.get("trace"),
                        decision=final_decision(r),
                        accepted=r.get("accepted"),
                        bad=quality.get("bad"),
                        coverage_90=quality.get("coverage_90"),
                        strength=strength,
                        strength_basis=basis,
                        agent_agrees=rec.get("agent_agrees"),
                        confidence=verdict.get("confidence"),
                        numbers=numbers,
                    )
                )
    return out


def pfn_gate_runs(results_dir: Path, include_val: bool) -> list[Run]:
    """The PFN runs with the gate, per trajectory: random + gate and knn + gate."""
    out: list[Run] = []
    for strategy in PFN_STRATEGIES:
        for suffix in (
            "test",
            "ood",
            "controls",
            *(("val", "val_controls") if include_val else ()),
        ):
            name = f"pfn_{strategy}_{suffix}.json"
            path = results_dir / name
            if not path.exists():
                continue
            doc = read_json(path)
            role = "control" if suffix.endswith("controls") else "tipping"
            for t in doc["trajectories"]:
                split = PFN_SPLITS.get(t["split"], t["split"])
                if split == "val" and not include_val:
                    continue
                gate, gt = t.get("gate") or {}, t.get("ground_truth") or {}
                out.append(
                    Run(
                        variant=f"pfn_{strategy}+gate",
                        split=split,
                        trajectory_id=t["trajectory_id"],
                        scenario_id=t["scenario_id"],
                        role=role,
                        source_file=name,
                        trace_name=doc.get("trace_file"),
                        accepted=gate.get("passed"),
                        bad=gt.get("bad"),
                        coverage_90=gt.get("coverage_90"),
                        numbers={
                            "coverage_90": gt.get("coverage_90"),
                            "gate_failed": gate.get("failed"),
                            "gate_reasons": gate.get("reasons"),
                            "context_strategy": gate.get("strategy"),
                            "context_size": t.get("context_size"),
                            "n_windows": t.get("n_windows"),
                            "window_decisions": t.get("decisions"),
                            "calibration_error": gt.get("calibration_error"),
                        },
                    )
                )
    return out


def ranges(numbers: list[int]) -> list[list[int]]:
    """Consecutive line numbers as [first, last] pairs."""
    out: list[list[int]] = []
    for n in sorted(numbers):
        if out and n == out[-1][1] + 1:
            out[-1][1] = n
        else:
            out.append([n, n])
    return out


class Traces:
    """Line numbers (1-based) of a trajectory in a trace file, parsed once per file."""

    def __init__(self, results_dir: Path) -> None:
        self.results_dir = results_dir
        self.cache: dict[str, dict[str, list[int]]] = {}

    def lines(self, trace_name: str | None, trajectory_id: str) -> dict[str, Any]:
        if not trace_name:
            return {"trace_file": None, "lines": []}
        name = Path(trace_name).name
        if name not in self.cache:
            index: dict[str, list[int]] = {}
            path = self.results_dir / "traces" / name
            if path.exists():
                with path.open(encoding="utf-8") as f:
                    for n, line in enumerate(f, start=1):
                        if line.strip():
                            sid = json.loads(line).get("scenario_id")
                            index.setdefault(str(sid), []).append(n)
            self.cache[name] = index
        found = self.cache[name].get(trajectory_id, [])
        return {"trace_file": f"{self.results_dir.name}/traces/{name}", "lines": ranges(found)}


def is_false_alarm(r: Run) -> bool:
    return r.role == "control" and r.decision == "alert"


def severity(category: str, r: Run, floor: float | None) -> tuple[float | None, str]:
    if category.startswith("a_"):
        return r.strength, r.strength_basis
    if category.startswith("b_") and r.coverage_90 is not None and floor is not None:
        return floor - r.coverage_90, "minimum coverage minus the run's 90% coverage"
    if category.startswith("c_") and r.coverage_90 is not None and floor is not None:
        return r.coverage_90 - floor, "the run's 90% coverage minus the minimum coverage"
    if category.startswith("d_"):
        return r.confidence, "the confidence the agent gave its recommendation"
    return None, ""


def select(category: str, runs: list[Run]) -> list[Run]:
    if category.startswith("a_"):
        return [r for r in runs if r.variant in VARIANT_FILES and is_false_alarm(r)]
    if category.startswith("b_"):
        return [r for r in runs if r.accepted is True and r.bad is True]
    if category.startswith("c_"):
        return [r for r in runs if r.accepted is False and r.bad is False]
    return [r for r in runs if r.variant == "agent" and r.agent_agrees is False]


def population(category: str, runs: list[Run]) -> list[Run]:
    """The runs the category is counted out of."""
    if category.startswith("a_"):
        return [r for r in runs if r.variant in VARIANT_FILES and r.role == "control"]
    if category.startswith("b_"):
        return [r for r in runs if r.accepted is not None and r.bad is True]
    if category.startswith("c_"):
        return [r for r in runs if r.accepted is not None and r.bad is False]
    return [r for r in runs if r.variant == "agent" and r.agent_agrees is not None]


def example(category: str, r: Run, floor: float | None, traces: Traces) -> dict[str, Any]:
    sev, basis = severity(category, r, floor)
    return {
        "severity": sev,
        "severity_basis": basis,
        "variant": r.variant,
        "split": r.split,
        "scenario_id": r.scenario_id,
        "seed": r.seed,
        "trajectory_id": r.trajectory_id,
        "role": r.role,
        "result_file": r.source_file,
        **traces.lines(r.trace_name, r.trajectory_id),
        "numbers": r.numbers,
    }


def rank_key(item: tuple[float | None, Run]) -> tuple[Any, ...]:
    sev, r = item
    return (-(sev if sev is not None else float("-inf")), r.variant, r.split, r.trajectory_id)


def count_by(runs: list[Run], key: str) -> dict[str, int]:
    return dict(sorted(Counter(getattr(r, key) for r in runs).items()))


def category_block(
    category: str, runs: list[Run], floor: float | None, top: int, traces: Traces
) -> dict[str, Any]:
    chosen = select(category, runs)
    pool = population(category, runs)
    ranked = sorted(((severity(category, r, floor)[0], r) for r in chosen), key=rank_key)
    return {
        "title": CATEGORIES[category],
        "n_failures": len(chosen),
        "n_runs_counted_out_of": len(pool),
        "failures_by_variant": count_by(chosen, "variant"),
        "runs_by_variant": count_by(pool, "variant"),
        "failures_by_split": count_by(chosen, "split"),
        "top": [example(category, r, floor, traces) for _, r in ranked[:top]],
    }


def share(n: int, total: int) -> str:
    return f"{n} of {total}"


def by_variant_text(block: dict[str, Any]) -> str:
    runs, fails = block["runs_by_variant"], block["failures_by_variant"]
    parts = [f"{v} {share(fails.get(v, 0), n)}" for v, n in runs.items() if fails.get(v)]
    return ", ".join(parts) if parts else "none"


def overview(category: str, block: dict[str, Any], runs: list[Run]) -> str:
    n, total = block["n_failures"], block["n_runs_counted_out_of"]
    split_text = ", ".join(f"{s} {c}" for s, c in block["failures_by_split"].items()) or "none"
    head = f"{block['title']}: {share(n, total)} runs."
    if category.startswith("a_"):
        text = f"{head} Control runs that ended in an alert, by variant: {by_variant_text(block)}."
    elif category.startswith("b_"):
        text = (
            f"{head} Runs the gate accepted although their 90% interval covered too little, "
            f"counted out of all BAD runs that have a gate decision, by variant: "
            f"{by_variant_text(block)}."
        )
    elif category.startswith("c_"):
        text = (
            f"{head} Runs the gate blocked although they were not BAD, counted out of all runs "
            f"with a gate decision that were not BAD, by variant: {by_variant_text(block)}."
        )
    else:
        text = (
            f"{head} Runs of the full harness where the agent's recommended decision differs "
            f"from the final decision, counted out of the agent runs that gave a recommendation: "
            f"{by_variant_text(block)}. "
            f"Agent runs with no recommendation: "
            f"{sum(r.variant == 'agent' and r.agent_agrees is None for r in runs)}."
        )
    if block["top"]:
        top = block["top"][0]
        text += (
            f" By splits: {split_text}. The listed examples are ranked by {top['severity_basis']};"
            f" the first is {top['variant']} on {top['trajectory_id']} ({top['split']})."
        )
    else:
        text += f" By splits: {split_text}."
    return text


def build(results_dir: Path, top: int, include_val: bool) -> dict[str, Any]:
    splits = ("val", *HELD_OUT) if include_val else HELD_OUT
    runs = variant_runs(results_dir, splits) + pfn_gate_runs(results_dir, include_val)
    floor = min_coverage()
    traces = Traces(results_dir)
    categories = {c: category_block(c, runs, floor, top, traces) for c in CATEGORIES}
    return {
        "scope": {
            "splits": list(splits),
            "includes_validation": include_val,
            "top_per_category": top,
            "min_coverage_of_a_bad_run": floor,
            "note": "Held-out splits only unless validation is included: the gate and alarm "
            "thresholds were set on validation. Severity scales differ by variant and are named "
            "in each example. Ties are broken by variant, split and trajectory id, in that "
            "order. This file ranks and links; it gives no causes.",
        },
        "n_runs_read": len(runs),
        "categories": categories,
        "overviews": {c: overview(c, categories[c], runs) for c in CATEGORIES},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"), type=Path)
    parser.add_argument("--top", default=5, type=int)
    parser.add_argument("--include-val", action="store_true", help="add the validation splits")
    args = parser.parse_args()
    report = build(args.results_dir, args.top, args.include_val)
    out = args.results_dir / "failures.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for text in report["overviews"].values():
        print(text + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

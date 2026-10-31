"""The failure gallery: the top examples of results/failures.json, each with its evidence.

For one example the report needs the scenario plot, the numbers, the gate result, the agent's
own rationale and a short excerpt of the trace. All of it is read from files in results/; the
scenario plot is drawn again from the simulator (or the raw real data, if it is on disk).
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pfngate.scenarios import Trajectory  # noqa: E402

INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OBSERVED, TARGET, TIP = "#2a78d6", "#eb6834", "#b3261e"
EXCERPT_LINES = 8  # first and last lines shown when a run has more


@dataclass
class GateCheck:
    name: str
    value: Any
    threshold: Any
    passed: bool
    reason: str


@dataclass
class TraceLine:
    line: int
    tool: str
    latency_s: float | None
    strategy: str | None
    context_size: int | None


@dataclass
class GalleryEntry:
    category: str
    example: dict[str, Any]
    rationale: str | None = None
    verdict_decision: str | None = None
    verdict_confidence: float | None = None
    final_decision: str | None = None
    gate_passed: bool | None = None
    gate_checks: list[GateCheck] = field(default_factory=list)
    trace: list[TraceLine] = field(default_factory=list)
    trace_omitted: int = 0
    plot_png: bytes | None = None
    plot_note: str = ""


def find_record(results_dir: Path, example: dict[str, Any]) -> dict[str, Any] | None:
    """The run (variant files) or trajectory (PFN files) behind an example."""
    path = results_dir / example["result_file"]
    if not path.exists():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    tid = example["trajectory_id"]
    for run in doc.get("runs", []):
        if run["record"]["trajectory_id"] == tid:
            return {"kind": "run", **run}
    for t in doc.get("trajectories", []):
        if t["trajectory_id"] == tid:
            return {"kind": "trajectory", **t}
    return None


def gate_checks(gate: dict[str, Any] | None) -> list[GateCheck]:
    if not gate:
        return []
    return [
        GateCheck(c["name"], c["value"], c["threshold"], bool(c["passed"]), c.get("reason", ""))
        for c in gate.get("checks", [])
    ]


def fill_from_record(entry: GalleryEntry, rec: dict[str, Any]) -> None:
    if rec["kind"] == "run":
        record = rec["record"]
        verdict = record.get("verdict") or {}
        answer = record.get("answer") or {}
        final = record.get("final") or {}
        gate = record.get("gate_result")
        entry.rationale = verdict.get("rationale") or answer.get("rationale")
        entry.verdict_decision = verdict.get("decision") or answer.get("decision")
        entry.verdict_confidence = verdict.get("confidence", answer.get("probability"))
        entry.final_decision = final.get("decision") or rec.get("decision")
        entry.gate_passed = None if gate is None else bool(gate.get("passed"))
        entry.gate_checks = gate_checks(gate)
    else:
        gate = rec.get("gate")
        entry.gate_passed = None if gate is None else bool(gate.get("passed"))
        entry.gate_checks = gate_checks(gate)


def trace_excerpt(results_dir: Path, example: dict[str, Any]) -> tuple[list[TraceLine], int]:
    """A short excerpt of the example's own trace lines, and how many lines were left out."""
    trace_file = example.get("trace_file")
    if not trace_file:
        return [], 0
    path = results_dir / "traces" / Path(trace_file).name
    wanted = [n for first, last in example["lines"] for n in range(first, last + 1)]
    if not path.exists() or not wanted:
        return [], 0
    keep = set(wanted)
    found: list[TraceLine] = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if n in keep and line.strip():
                row = json.loads(line)
                found.append(
                    TraceLine(
                        n,
                        str(row.get("tool")),
                        row.get("latency_s"),
                        row.get("strategy"),
                        row.get("context_size"),
                    )
                )
    if len(found) <= EXCERPT_LINES:
        return found, 0
    half = EXCERPT_LINES // 2
    return found[:half] + found[-half:], len(found) - EXCERPT_LINES


def find_trajectory(example: dict[str, Any]) -> Trajectory | None:
    """The trajectory of an example, from the simulator or from the real data on disk."""
    from pfngate.data import REAL_CONTROL_SPLIT, REAL_SPLIT
    from pfngate.gate import assert_frozen
    from pfngate.scenarios import load_scenarios

    sid, seed, split = example["scenario_id"], example["seed"], example["split"]
    if split in (REAL_SPLIT, REAL_CONTROL_SPLIT):
        from pfngate.data.load import real_trajectories

        assert_frozen()  # real data is test only: the gate must be frozen before it is read
        trajectories, _ = real_trajectories(split)
        return next((t for t in trajectories if t.scenario_id == sid and t.seed == seed), None)
    scenario = load_scenarios().get(sid)
    return scenario.generate(seed) if scenario and seed is not None else None


def draw(tr: Trajectory, example: dict[str, Any]) -> bytes:
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(6.4, 3.6), sharex=True, constrained_layout=True, dpi=110
    )
    top.plot(tr.time, tr.observations[:, 0], lw=0.7, color=OBSERVED)
    bottom.plot(tr.time, tr.target, lw=1.0, color=TARGET)
    if tr.tip_time is not None:
        for ax in (top, bottom):
            ax.axvline(tr.tip_time, color=TIP, ls=":", lw=1)
    top.set_ylabel("observed", fontsize=8, color=INK_MUTED)
    bottom.set_ylabel("target", fontsize=8, color=INK_MUTED)
    bottom.set_xlabel("time", fontsize=8, color=INK_MUTED)
    for ax in (top, bottom):
        ax.tick_params(labelsize=7, colors=INK_MUTED)
        ax.grid(color=GRID, lw=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    top.set_title(f"{example['trajectory_id']} ({example['split']})", fontsize=9, color=INK)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def build_entry(results_dir: Path, category: str, example: dict[str, Any]) -> GalleryEntry:
    entry = GalleryEntry(category=category, example=example)
    rec = find_record(results_dir, example)
    if rec:
        fill_from_record(entry, rec)
    entry.trace, entry.trace_omitted = trace_excerpt(results_dir, example)
    try:
        tr = find_trajectory(example)
        if tr is None:
            entry.plot_note = "The scenario could not be found, so there is no plot."
        else:
            entry.plot_png = draw(tr, example)
    except Exception as err:  # raw data not on this machine, gate not frozen, and so on
        entry.plot_note = f"No plot: {type(err).__name__}: {err}"
    return entry


def build_gallery(results_dir: Path, per_category: int) -> list[GalleryEntry]:
    failures = json.loads((results_dir / "failures.json").read_text(encoding="utf-8"))
    out: list[GalleryEntry] = []
    for category, block in failures["categories"].items():
        out.extend(build_entry(results_dir, category, ex) for ex in block["top"][:per_category])
    return out


def data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")

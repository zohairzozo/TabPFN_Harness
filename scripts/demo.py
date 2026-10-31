"""A short, offline walkthrough of two held-out runs of the full harness (make demo).

It picks two runs from the recorded results by rule, runs the agent on them again with
--llm-mode replay (no network, no key; the PFN output comes from the cache), and prints each
step in plain language: the scenario, what the agent asked, the context it chose, the PFN's
interval and p_critical, every gate check, and the final decision. It ends with the path to the
report and the headline row of the results table. Everything printed is also saved to
results/demo_transcript.txt, which is only replaced when a run finishes, so the last good
transcript stays as a fallback if a live run fails.

It needs what `make full` leaves behind: results/agent_{test,ood}.json, results/summary.md and
the caches in results/cache/llm and results/cache/pfn.

Usage: python scripts/demo.py [--results-dir results] [--out-dir results/demo]
                              [--transcript results/demo_transcript.txt]
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from pfngate.agent.store import RunData, RunStore
from pfngate.evals.llm_eval import DEFAULT_GATE_CONFIG
from pfngate.evals.run_agent import evaluate
from pfngate.gate.freeze import assert_committed
from pfngate.report.text_sources import plain, read_text, section_table

REPO_ROOT = Path(__file__).resolve().parents[1]
TIME_BUDGET_SECONDS = 180.0
SEARCH_SPLITS = (("test", "the in-prior test split"), ("ood", "the out-of-prior test split"))
HEADLINE_VARIANT = "Full harness"
BAD_COVERAGE_LINE = 0.80  # docs/GATE.md: a run is BAD if its 90% interval covers less than this
FAMILIES = {
    "fold": "a fold bifurcation: the stable state disappears as a parameter drifts",
    "hopf": "a subcritical Hopf bifurcation: oscillations take over as a parameter drifts",
    "transcritical": "a transcritical bifurcation: two states exchange stability",
    "noise_tipping": "noise-induced tipping: noise alone kicks the system over (out of prior)",
    "rate_tipping": "rate-induced tipping: the parameter moves too fast to follow (out of prior)",
    "control_ou": "a control: a steady Ornstein-Uhlenbeck process that never tips",
    "control_short_ramp": "a control: a short ramp that never reaches a tipping point",
    "control_regime_change": "a control: a regime change that is not a tip",
}
TOOL_SENTENCES = {
    "describe_query": "asked for a description of the query series",
    "describe_context_pool": "asked what the context pool holds",
    "choose_context": "asked for a context to be built",
    "run_pfn": "asked the PFN to predict from a context",
}
SEPARATOR = "-" * 72


class Printer:
    """Prints a line and keeps it for the transcript."""

    def __init__(self, sink: Callable[[str], None] = print) -> None:
        self.lines: list[str] = []
        self.sink = sink

    def __call__(self, text: str = "") -> None:
        self.lines.append(text)
        self.sink(text)

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


class KeepingStore(RunStore):
    """A run store that keeps the data of each run after it ends, so the demo can read it."""

    def __init__(self) -> None:
        super().__init__()
        self.kept: dict[str, RunData] = {}

    def drop(self, run_id: str) -> None:
        self.kept[run_id] = self.get(run_id)
        super().drop(run_id)


# Picking the runs, by rule


def is_correct_accepted_alert(run: dict[str, Any]) -> bool:
    return (
        run["accepted"] is True
        and run["decision"] == "alert"
        and run["ground_truth"]["end_label"] == "positive"
    )


def is_blocked_bad_run(run: dict[str, Any]) -> bool:
    quality = run.get("pfn_quality") or {}
    return run["accepted"] is False and bool(quality.get("bad"))


RULES: tuple[tuple[str, str, Callable[[dict[str, Any]], bool]], ...] = (
    (
        "passed",
        "the first run, in the order of results/agent_<split>.json, that the gate accepted "
        "and that ended in an alert while the series really was near tipping (a correct alert)",
        is_correct_accepted_alert,
    ),
    (
        "blocked",
        "the first run whose PFN interval was bad (its 90% coverage was below the line of "
        "docs/GATE.md) and that the gate blocked",
        is_blocked_bad_run,
    ),
)


def pick(results_dir: Path, rule: Callable[[dict[str, Any]], bool]) -> dict[str, Any] | None:
    """The first run that satisfies the rule, searching the test splits in order."""
    for split, label in SEARCH_SPLITS:
        path = results_dir / f"agent_{split}.json"
        if not path.exists():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        for run in doc["runs"]:
            if rule(run):
                return {"split": split, "split_label": label, "doc": doc, "recorded": run}
    return None


# Printing a run


def fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() and abs(value) >= 1 else f"{value:.{digits}f}"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(fmt(v, digits) for v in value) + "]"
    return str(value)


def trace_rows(trace_path: Path, trajectory_id: str) -> list[dict[str, Any]]:
    rows = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("scenario_id") == trajectory_id:
                rows.append(row)
    return rows


def describe_asks(rows: list[dict[str, Any]], say: Printer) -> None:
    asks = [r for r in rows if str(r["tool"]).startswith("agent_tool")]
    model_calls = sum(str(r["tool"]).startswith("llm") for r in rows)
    say(f"   The agent made {model_calls} model calls and, in this order:")
    for n, row in enumerate(asks, start=1):
        tool = str(row["tool"])
        if tool == "agent_tool_refused":
            say(f"     {n}. it asked for one more thing, and the harness refused (tool cap)")
            continue
        name = tool.split(":", 1)[1]
        sentence = TOOL_SENTENCES.get(name, f"called {name}")
        extra = ""
        if name == "choose_context":
            extra = f" ({row.get('strategy')}, {fmt(row.get('context_size'))} rows)"
        elif name == "run_pfn":
            extra = f" (context {str(row.get('context_hash'))[:8]})"
        say(f"     {n}. {sentence}{extra}")


def last_window(data: RunData | None, context_hash: str | None) -> dict[str, Any] | None:
    """The PFN's output for the last window of the run's final context."""
    if data is None or context_hash is None or context_hash not in data.outputs:
        return None
    out = data.outputs[context_hash]
    levels = [float(v) for v in out.quantile_levels]
    quantiles = [float(v) for v in out.quantiles[-1]]
    return {
        "levels": levels,
        "quantiles": quantiles,
        "median": float(out.median[-1]),
        "p_critical": float(out.p_critical[-1]),
        "n_windows": int(len(out.median)),
    }


def describe_pfn(window: dict[str, Any] | None, truth: dict[str, Any], say: Printer) -> None:
    if window is None:
        say("   There is no PFN output for the final context (no context was built).")
        return
    q = dict(zip(window["levels"], window["quantiles"], strict=True))
    low, high = min(q), max(q)
    say("   The PFN predicts how far the system is from tipping: 0 means at the tipping")
    say("   point, 1 means far from it. For the last window of the series it says:")
    say(f"     median prediction        {fmt(window['median'])}")
    say(f"     {fmt(high - low, 2)} central interval  [{fmt(q[low])}, {fmt(q[high])}]")
    say(f"     p_critical               {fmt(window['p_critical'])}   (chance the target is at or")
    say("                              below the critical level)")
    say(f"   The true target at the last window is {fmt(truth.get('end_target'))} (read by the")
    say("   evaluation only; neither the agent nor the gate sees it).")


def describe_gate(final: dict[str, Any], gate: dict[str, Any] | None, say: Printer) -> None:
    if not gate:
        say("   There was no gate result: the agent built no context.")
        return
    say("   The gate checks the context and the output, without seeing the truth:")
    for check in gate["checks"]:
        mark = "passed" if check["passed"] else "FAILED"
        say(
            f"     {check['name']:<24} value {fmt(check['value'])}  "
            f"{check['comparison']}  threshold {fmt(check['threshold'])}  -> {mark}"
        )
        say(f"         {check['reason']}")
    for adv in gate.get("advisories", []):
        mark = "ok" if adv["passed"] else "over the advisory line (reported, never blocks)"
        say(
            f"     {adv['name']:<24} value {fmt(adv['value'])}  "
            f"{adv['comparison']}  threshold {fmt(adv['threshold'])}  -> {mark}"
        )
    verdict = "PASSED" if gate["passed"] else "BLOCKED"
    say(f"   Gate result: {verdict}.")


def describe_run(
    number: int,
    why: str,
    choice: dict[str, Any],
    result: dict[str, Any],
    data: RunData | None,
    out_dir: Path,
    say: Printer,
) -> bool:
    """Print one run. Returns True if the rerun matches the recorded run."""
    recorded = choice["recorded"]
    tid = recorded["record"]["trajectory_id"]
    run = next(r for r in result["runs"] if r["record"]["trajectory_id"] == tid)
    rec, truth = run["record"], run["ground_truth"]
    verdict, final = rec.get("verdict"), rec["final"]
    say(SEPARATOR)
    say(f"RUN {number}: {why}")
    say(SEPARATOR)
    seed = tid.rsplit("/", 1)[-1]
    say(f"1. The scenario: {truth['scenario_id']}, seed {seed}, from {choice['split_label']}.")
    say(f"   Family: {truth['family']}, {FAMILIES.get(truth['family'], 'no description here')}.")
    ending = "near tipping" if truth["end_label"] == "positive" else "far from tipping"
    say(f"   Truth, for scoring only: the series ends {ending} (label {truth['end_label']}).")
    say()
    say("2. What the agent asked:")
    trace = out_dir / "traces" / result["trace"]
    describe_asks(trace_rows(trace, tid), say)
    if rec.get("stop_detail"):
        say(f"   It stopped because: {rec['stop_detail']}.")
    say()
    say("3. The context strategy it picked, and why:")
    if verdict:
        say(
            f"   It picked {verdict['context_strategy']} with {fmt(verdict['context_size'])} rows "
            f"and recommended {verdict['decision']} (confidence {fmt(verdict['confidence'])})."
        )
        say(f"   Its own reason: {verdict['rationale']}")
    else:
        say("   The agent gave no recommendation, so there is no reason to quote.")
        why_not = rec.get("stop_detail") or "it stopped early"
        say(f"   Why: {why_not}. The harness gates the last context the agent built.")
    matches_text = (
        f" (it matches the recommendation: {fmt(final['followed_verdict_context'])})"
        if verdict
        else ""
    )
    say(
        f"   The context that was gated: {final['context_strategy']}, "
        f"{fmt(final['context_size'])} rows{matches_text}."
    )
    say()
    say("4. The PFN's output for that context:")
    describe_pfn(last_window(data, final["context_hash"]), truth, say)
    say()
    say("5. The gate:")
    describe_gate(final, rec.get("gate_result"), say)
    say()
    say("6. The final decision:")
    agree = "the same as" if rec.get("agent_agrees") else "different from"
    say(f"   {final['decision'].upper()}: {final['reason']}.")
    if verdict:
        say(f"   The agent recommended {verdict['decision']}: {agree} the final decision.")
    quality = run.get("pfn_quality") or {}
    if quality:
        label = "BAD" if quality["bad"] else "not bad"
        say(
            f"   Afterwards, with the truth: the 90% interval covered "
            f"{fmt(quality['coverage_90'])} of the window targets ({label}, line "
            f"{fmt(BAD_COVERAGE_LINE, 2)})."
        )
    same = (
        final["decision"] == recorded["record"]["final"]["decision"]
        and final["context_hash"] == recorded["record"]["final"]["context_hash"]
        and run["accepted"] == recorded["accepted"]
    )
    say(
        f"   Same decision, context and gate result as the recorded run: {'yes' if same else 'NO'}."
    )
    say()
    return bool(same)


# The end


def headline(results_dir: Path, say: Printer) -> None:
    table = section_table(read_text(results_dir / "summary.md"), "Variants, run level")
    row = None
    if table:
        row = next((r for r in table.rows if plain(r[0]).startswith(HEADLINE_VARIANT)), None)
    say("Headline row of the results table (results/summary.md, 'Variants, run level'):")
    if table is None or row is None:
        say("   not found: run `make full` first.")
        return
    for header, cell in zip(table.headers, row, strict=True):
        say(f"   {plain(header)}: {plain(cell)}")


def report_lines(results_dir: Path, say: Printer) -> None:
    index = results_dir / "report" / "index.html"
    if index.exists():
        say(f"The report: {index}")
        criteria = results_dir / "report" / "criteria.json"
        if criteria.exists():
            doc = json.loads(criteria.read_text(encoding="utf-8"))
            say(
                f"   Verdict in the report: {doc['verdict']} "
                f"({doc['required_met']} of {doc['required_total']} required criteria met)."
            )
    else:
        say(f"The report is not built yet ({index}): run `make report`.")


def check_caches(results_dir: Path) -> None:
    for name in ("llm", "pfn"):
        folder = results_dir / "cache" / name
        if not folder.exists() or not any(folder.iterdir()):
            raise SystemExit(
                f"{folder} is empty. The demo replays recorded answers: run `make full` once first."
            )


def clear_previous(out_dir: Path) -> None:
    """Remove the demo's own files from an earlier run (nothing else lives in this folder)."""
    for pattern in ("agent_*.json", "traces/agent-*.jsonl"):
        for path in out_dir.glob(pattern):
            path.unlink()


def run_demo(results_dir: Path, out_dir: Path, say: Printer) -> bool:
    start = time.perf_counter()
    say("pfngate demo: two held-out runs of the full harness (agent + gate)")
    say("Mode: --llm-mode replay. No network, no API key. The PFN output comes from the cache.")
    say()
    check_caches(results_dir)
    assert_committed(DEFAULT_GATE_CONFIG)  # held-out data needs the gate committed (SPLITS.md)
    chosen: list[tuple[str, str, dict[str, Any]]] = []
    for label, text, rule in RULES:
        found = pick(results_dir, rule)
        say(f"Rule for the run the gate {label}: {text}.")
        if found is None:
            say("   No recorded run satisfies this rule, so there is nothing to show.")
            continue
        tid = found["recorded"]["record"]["trajectory_id"]
        say(f"   Picked: {tid}, from {found['split_label']} (results/agent_{found['split']}.json).")
        if found["split"] != SEARCH_SPLITS[0][0]:
            say(f"   No run in {SEARCH_SPLITS[0][1]} satisfies the rule, so the search went on to")
            say(f"   {found['split_label']}.")
        chosen.append((label, text, found))
    say()
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_previous(out_dir)
    matches: list[bool] = []
    for number, (label, _, found) in enumerate(chosen, start=1):
        doc, tid = found["doc"], found["recorded"]["record"]["trajectory_id"]
        store = KeepingStore()
        result = evaluate(
            "agent",
            found["split"],
            max_context=doc["settings"]["max_context"],
            seeds_per_scenario=doc["seeds_per_scenario"],
            limit=doc["limit"],
            trajectory_ids=[tid],
            store=store,
            results_dir=out_dir,
            cache_dir=results_dir / "cache" / "pfn",
            llm_cache_dir=results_dir / "cache" / "llm",
            llm_mode="replay",
            progress=False,
        )
        run_id = result["runs"][0]["record"]["run_id"]
        why = f"the first run the gate {label} ({found['split']} split)"
        matches.append(
            describe_run(number, why, found, result, store.kept.get(run_id), out_dir, say)
        )
    say(SEPARATOR)
    report_lines(results_dir, say)
    say()
    headline(results_dir, say)
    elapsed = time.perf_counter() - start
    say()
    note = "" if elapsed < TIME_BUDGET_SECONDS else "  (over the time budget)"
    say(f"Finished in {elapsed:.1f} s{note}.")
    return len(chosen) == len(RULES) and all(matches) and elapsed < TIME_BUDGET_SECONDS


def save_transcript(path: Path, say: Printer) -> None:
    """Replace the transcript only now, so a failed run leaves the last good one in place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(say.text(), encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--transcript", type=Path, default=None)
    args = parser.parse_args(argv)
    out_dir = args.out_dir or args.results_dir / "demo"
    transcript = args.transcript or args.results_dir / "demo_transcript.txt"
    np.random.seed(0)
    say = Printer()
    ok = run_demo(args.results_dir, out_dir, say)
    if not ok:  # keep the last good transcript where it is
        transcript = transcript.with_name(transcript.name + ".failed")
    save_transcript(transcript, say)
    say.sink(f"Saved this transcript to {transcript}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

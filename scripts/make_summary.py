"""Write results/summary.md from the result files in results/.

Every number in the summary is read from results/baseline_ews.json, results/pfn_*.json,
results/gate_tuning.json, results/timing_tabpfn.json or the results of the variants
(results/{classic_ews,llm_only,naive_agent,agent,oracle}_{test,ood}.json). The text around
the numbers does not state any result: sentences that depend on a number are chosen by code
from that number.

Usage: python scripts/make_summary.py [--results-dir results]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ("random", "knn", "knn_balanced", "family_matched")
METHODS: list[tuple[str, str]] = [
    ("Classic EWS baseline", "baseline"),
    ("TabPFN random", "random"),
    ("TabPFN knn", "knn"),
    ("TabPFN knn_balanced", "knn_balanced"),
    ("TabPFN family_matched (ORACLE)", "family_matched"),
]
# Columns of the main table: (header, dataset split in the baseline file, run flag for PFN).
AUROC_COLUMNS = [
    ("in-prior val", "validation", "val"),
    ("in-prior test", "test_in_prior", "test"),
    ("out-of-prior", "test_out_of_prior", "ood"),
]
SCORE_TITLES = {
    "p_critical": "p_critical (probability that the target is at or below the threshold)",
    "neg_median": "-median (minus the median of the predicted target)",
}
LEAKAGE_AUROC = 0.98  # an AUROC above this usually means leakage
DEGENERATE_THRESHOLD = 1e-12  # a p_critical alarm threshold below this means scores of 0
RATE_TOLERANCE = 1.5  # held-out false-alarm rate above this times the target is flagged
GATE_STRATEGIES = ("random", "knn")
# The variants of the comparison table: (row title, result file prefix, metrics source in the file).
VARIANTS: list[tuple[str, str, str]] = [
    ("Classic EWS baseline", "classic_ews", "classic_ews"),
    ("LLM only", "llm_only", "llm_only"),
    ("Naive agent (random context, no gate)", "naive_agent", "naive_agent"),
    ("Full harness (agent + gate)", "agent", "agent_final"),
    ("Family-matched oracle (ORACLE)", "oracle", "oracle"),
]
VARIANT_SPLITS = ("test", "ood")  # in-prior test, out-of-prior
REAL_VARIANT_SPLITS = (
    "real",
    "real_control",
)  # real data, test only (docs/DAPHNIA.md, docs/GRID.md)
GATE_SPLITS = [
    ("in-prior val", "val"),
    ("in-prior test", "test"),
    ("out-of-prior", "ood"),
]


def load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


class Results:
    """All result files of one results/ directory."""

    def __init__(self, results_dir: Path) -> None:
        self.dir = results_dir
        self.baseline = load(results_dir / "baseline_ews.json")
        self.timing = load(results_dir / "timing_tabpfn.json")
        self.tuning = load(results_dir / "gate_tuning.json")
        self.variants: dict[tuple[str, str], dict[str, Any]] = {}
        for _, method, _ in VARIANTS:
            for split in (*VARIANT_SPLITS, *REAL_VARIANT_SPLITS):
                data = load(results_dir / f"{method}_{split}.json")
                if data is not None:
                    self.variants[(method, split)] = data
        self.pfn: dict[tuple[str, str], dict[str, Any]] = {}
        for strategy in STRATEGIES:
            for flag in ("val", "test", "ood", "controls"):
                data = load(results_dir / f"pfn_{strategy}_{flag}.json")
                if data is not None:
                    self.pfn[(strategy, flag)] = data

    def auroc(
        self, method: str, split: str, flag: str, score: str, subset: str = "tipping_only"
    ) -> dict[str, Any] | None:
        """AUROC entry (with CI) or None. The reason it is None is given by `missing`."""
        if method == "baseline":
            if self.baseline is None:
                return None
            return self.baseline["splits"][split]["auroc"].get(subset)
        run = self.pfn.get((method, flag))
        if run is None or run.get("auroc") is None:
            return None
        return (run["auroc"].get(subset) or {}).get(score)

    def missing(self, method: str, flag: str) -> str:
        if method == "baseline":
            return "not run"
        run = self.pfn.get((method, flag))
        if run is None:
            return "not run"
        if run["skipped"] and run["n_trajectories"] == 0:
            return "n/a (no pool rows of this family)"
        return "n/a"

    def false_alarm(self, method: str, score: str) -> dict[str, Any] | None:
        if method == "baseline":
            if self.baseline is None:
                return None
            return self.baseline["false_alarms_at_validation_threshold"]
        run = self.pfn.get((method, "controls"))
        if run is None or run.get("false_alarms") is None:
            return None
        fa = run["false_alarms"]
        return {
            "target_window_false_alarm_rate": fa["target_window_false_alarm_rate"],
            "threshold": fa["thresholds"][score],
            "splits": fa["scores"][score]["splits"],
            "held_out_pooled": fa["scores"][score]["held_out_pooled"],
        }

    def degenerate_controls(self, method: str) -> bool:
        """True if p_critical was exactly 0 on (at least) 95% of the validation controls.

        The alarm threshold then sits at the smallest positive float. A constant context
        target makes TabPFN return a constant, which gives this. The false-alarm rate of
        such a run says nothing about the method.
        """
        run = self.pfn.get((method, "controls"))
        if run is None or run.get("false_alarms") is None:
            return False
        return run["false_alarms"]["thresholds"]["p_critical"] < DEGENERATE_THRESHOLD

    def latencies(self, method: str) -> list[float]:
        return [
            t["latency_seconds"]
            for (m, flag), run in self.pfn.items()
            if m == method and flag != "controls"
            for t in run["trajectories"]
        ]


def cell(entry: dict[str, Any] | None, fallback: str) -> str:
    if entry is None:
        return fallback
    text = f"{entry['auroc']:.2f} [{entry['ci_low']:.2f}, {entry['ci_high']:.2f}]"
    if entry["ci_low"] <= 0.5 <= entry["ci_high"]:
        text += " †"
    if entry["auroc"] > LEAKAGE_AUROC:
        text += " ⚠"
    return text


def rate(block: dict[str, Any] | None, key: str = "window_rate") -> str:
    return "not run" if block is None else f"{block['all_controls'][key]:.3f}"


def latency_cell(res: Results, method: str) -> str:
    if method == "baseline":
        return "not measured"
    lat = res.latencies(method)
    return f"{np.median(lat):.1f}" if lat else "not run"


def main_table(res: Results, score: str) -> list[str]:
    head = ["Method"] + [f"AUROC {c[0]}" for c in AUROC_COLUMNS]
    head += ["False alarms on held-out controls (window rate)", "Median latency per trajectory (s)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for label, method in METHODS:
        row = [label]
        for _, split, flag in AUROC_COLUMNS:
            row.append(cell(res.auroc(method, split, flag, score), res.missing(method, flag)))
        fa = res.false_alarm(method, score)
        far = rate(fa["held_out_pooled"]) if fa and fa["held_out_pooled"] else "not run"
        row.append(far + (" ‡" if res.degenerate_controls(method) else ""))
        row.append(latency_cell(res, method))
        lines.append("| " + " | ".join(row) + " |")
    return lines


def family_tables(res: Results, score: str) -> list[str]:
    out: list[str] = []
    for title, split, flag in AUROC_COLUMNS:
        families: set[str] = set()
        for _, method in METHODS:
            a: dict[str, Any] = {}
            a = (
                res.baseline["splits"][split]["auroc"]
                if method == "baseline" and res.baseline
                else {}
            )
            if method != "baseline" and (method, flag) in res.pfn:
                a = res.pfn[(method, flag)].get("auroc") or {}
            families |= {k for k in a if k.startswith("family:")}
        if not families:
            continue
        out += ["", f"**{title}**", "", "| Family | " + " | ".join(m[0] for m in METHODS) + " |"]
        out.append("|" + "---|" * (len(METHODS) + 1))
        for fam in sorted(families):
            cells = [
                cell(res.auroc(m, split, flag, score, fam), res.missing(m, flag))
                for _, m in METHODS
            ]
            out.append(f"| {fam.removeprefix('family:')} | " + " | ".join(cells) + " |")
    return out


def false_alarm_table(res: Results) -> list[str]:
    head = (
        "| Method (score) | Threshold | Target | Validation controls | "
        "Held-out controls: windows | Held-out controls: trajectories with any alarm |"
    )
    lines = [head, "|" + "---|" * 6]
    for label, method in METHODS:
        for score in ("p_critical", "neg_median") if method != "baseline" else (None,):
            fa = res.false_alarm(method, score or "p_critical")
            name = label if score is None else f"{label} ({score})"
            if fa is None or fa["held_out_pooled"] is None:
                lines.append(f"| {name} | not run | | | | |")
                continue
            val = fa["splits"].get("validation")
            held = fa["held_out_pooled"]["all_controls"]
            lines.append(
                f"| {name} | {fa['threshold']:.4g} | {fa['target_window_false_alarm_rate']:.3f} | "
                f"{val['all_controls']['window_rate']:.3f} | {held['window_rate']:.3f} | "
                f"{held['trajectory_rate']:.3f} |"
            )
    return lines


def settings_table(res: Results) -> list[str]:
    head = [
        "Run",
        "max_context (config default)",
        "Seeds per scenario",
        "Trajectories",
        "Windows",
        "Cache hits",
        "Skipped",
        "Wall time (min)",
        "Model",
        "Backend package",
    ]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for (strategy, flag), run in sorted(res.pfn.items()):
        seeds = run["seeds_per_scenario"]
        lines.append(
            f"| {strategy} {flag} | {run['max_context']} ({run['max_context_config_default']}) | "
            f"{'all' if seeds is None else seeds} | {run['n_trajectories']} | {run['n_windows']} | "
            f"{run['n_cache_hits']} | {len(run['skipped'])} | "
            f"{run['timing']['total_wall_seconds'] / 60:.1f} | {run['model_version']} | "
            f"{run['package_version']} |"
        )
    return lines


def reduction_notes(res: Results) -> list[str]:
    """Say, from the result files, if the runs were reduced to fit the CPU."""
    runs = list(res.pfn.values())
    if not runs:
        return []
    notes = []
    caps = {r["max_context"] for r in runs}
    default = runs[0]["max_context_config_default"]
    if caps != {default}:
        used_caps = ", ".join(map(str, sorted(caps)))
        text = f"**max_context was reduced** from the default {default} to {used_caps}"
        t = (res.timing or {}).get("timings", {}).get(str(default))
        if t:
            text += (
                f" because the CPU is slow: results/timing_tabpfn.json measured "
                f"{t['predict_seconds_median']:.0f} s to predict {res.timing['n_test']} rows "
                f"with {default} context rows"
            )
        notes.append(text + ".")
    seeds = {r["seeds_per_scenario"] for r in runs}
    if seeds != {None}:
        used = ", ".join(str(x) for x in sorted(s for s in seeds if s is not None))
        notes.append(
            f"**Only the first {used} "
            "seeds of each scenario "
            "were run** (seeds_per_scenario). All scenarios are included, but each has fewer "
            "trajectories than the design, so the confidence intervals are wider. This was done "
            "after max_context had been reduced, because a full run was still too slow on CPU."
        )
    return notes


def comparison_lines(res: Results, score: str) -> list[str]:
    """Where the TabPFN CI is clearly above, clearly below or overlapping the baseline CI."""
    out = []
    for title, split, flag in AUROC_COLUMNS:
        base = res.auroc("baseline", split, flag, score)
        for label, method in METHODS[1:]:
            a = res.auroc(method, split, flag, score)
            if a is None or base is None:
                continue
            if a["ci_low"] > base["ci_high"]:
                word = "above the baseline (CIs do not overlap)"
            elif a["ci_high"] < base["ci_low"]:
                word = "below the baseline (CIs do not overlap)"
            else:
                word = "not clearly different from the baseline (CIs overlap)"
            out.append(f"- {title}, {label}: {word}.")
    return out


def check_lines(res: Results) -> list[str]:
    out = []
    for score in SCORE_TITLES:
        for title, split, flag in AUROC_COLUMNS:
            for _, method in METHODS:
                a = res.auroc(method, split, flag, score)
                if a is not None and a["auroc"] > LEAKAGE_AUROC:
                    out.append(
                        f"- ⚠ {method} {score} {title}: AUROC {a['auroc']:.3f} is above "
                        f"{LEAKAGE_AUROC}. This usually means leakage. Check before trusting it."
                    )
    for _, method in METHODS[1:]:
        if res.degenerate_controls(method):
            out.append(
                f"- ‡ {method} on controls: the p_critical alarm threshold is at the smallest "
                "positive float, so p_critical was exactly 0 on at least 95% of validation "
                "control windows. The score is degenerate on controls (a constant context "
                "target gives a constant prediction), so its false-alarm rate says nothing "
                "about the method."
            )
        for score in SCORE_TITLES:
            fa = res.false_alarm(method, score)
            if fa is None or fa["held_out_pooled"] is None or res.degenerate_controls(method):
                continue
            held = fa["held_out_pooled"]["all_controls"]["window_rate"]
            target = fa["target_window_false_alarm_rate"]
            if held > RATE_TOLERANCE * target:
                out.append(
                    f"- {method} {score}: held-out window false-alarm rate {held:.3f} is above "
                    f"{RATE_TOLERANCE} times the target {target:.3f}. The validation threshold "
                    "did not transfer."
                )
    for (strategy, flag), run in sorted(res.pfn.items()):
        if run["skipped"]:
            reasons = {s["reason"].split(" for ")[0] for s in run["skipped"]}
            why = "; ".join(sorted(reasons))
            out.append(f"- {strategy} {flag}: {len(run['skipped'])} trajectories skipped ({why}).")
    return out or ["- No check was triggered."]


def num(v: float | None, fallback: str = "n/a", digits: int = 2) -> str:
    return fallback if v is None else f"{v:.{digits}f}"


def with_ci(v: float | None, ci: list[float] | None, fallback: str) -> str:
    if v is None:
        return fallback
    return f"{v:.2f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")


def gate_of(res: Results, strategy: str, flag: str) -> dict[str, Any] | None:
    run = res.pfn.get((strategy, flag))
    return None if run is None else run.get("gate")


def gate_table(res: Results) -> list[str]:
    head = [
        "Method", "Split", "Gate", "Runs", "Accepted runs", "Abstention rate",
        "Runs over the width cap (advisory)", "AUROC, accepted runs", "AUROC, all runs",
        "90% coverage, accepted runs", "90% coverage, all runs", "BAD runs",
        "Precision (blocking BAD)", "Recall (blocking BAD)",
        "Gate time per run, uncached refits (s)",
    ]  # fmt: skip
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for strategy in GATE_STRATEGIES:
        for title, flag in GATE_SPLITS:
            g = gate_of(res, strategy, flag)
            name = f"TabPFN {strategy} + gate"
            if g is None:
                why = "not run" if tuned(res) else "not run (gate not tuned)"
                lines.append(f"| {name} | {title} | {why} |" + " - |" * (len(head) - 3))
                continue
            run = res.pfn[(strategy, flag)]
            status = g.get("config_status", "?")
            kind = f"{status}, in-sample" if status == "tuned" and flag == "val" else status
            acc = ((g.get("auroc_accepted") or {}).get("tipping_only") or {}).get("p_critical")
            b, ci = g["blocking"] or {}, g["blocking_ci"] or {}
            gate_s = [t["gate_seconds"] for t in run["trajectories"] if "gate_seconds" in t
                      and t["context_calibration"]["n_cache_hits"] == 0]  # fmt: skip
            lines.append(
                f"| {name} | {title} | {kind} | {g['n_runs']} | {g['n_passed']} | "
                f"{num(g['abstention_rate'])} | "
                f"{(g.get('advisories_failed') or {}).get('interval_width_cap', 0)} | "
                f"{cell(acc, 'n/a (no accepted run)')} | "
                f"{cell(res.auroc(strategy, '', flag, 'p_critical'), 'n/a')} | "
                f"{num(g['coverage_90_mean_accepted'], 'n/a (no accepted run)')} | "
                f"{num(g['coverage_90_mean_all'])} | {g['n_bad']} | "
                f"{with_ci(b.get('precision'), ci.get('precision'), 'n/a (nothing blocked)')} | "
                f"{with_ci(b.get('recall'), ci.get('recall'), 'n/a (no BAD runs)')} | "
                f"{num(float(np.median(gate_s)) if gate_s else None, 'n/a (refits cached)', 1)} |"
            )
    return lines


def gate_false_alarm_table(res: Results) -> list[str]:
    head = [
        "Method", "Held-out control runs", "Abstention rate",
        "Window false alarms (abstain = no alarm)", "Window false alarms, accepted runs only",
        "Window false alarms without the gate",
    ]  # fmt: skip
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for strategy in GATE_STRATEGIES:
        g = gate_of(res, strategy, "controls")
        fa = None if g is None else (g.get("false_alarms") or {}).get("held_out_pooled")
        name = f"TabPFN {strategy} + gate"
        if fa is None:
            why = "not run" if tuned(res) else "not run (gate not tuned)"
            lines.append(f"| {name} | {why} |" + " - |" * (len(head) - 2))
            continue
        ungated = res.false_alarm(strategy, "p_critical")
        lines.append(
            f"| {name} | {fa['n_trajectories']} | {num(fa['abstention_rate'])} | "
            f"{num(fa['window_rate'], digits=3)} | "
            f"{num(fa['window_rate_accepted'], 'n/a (no accepted run)', 3)} | "
            f"{rate(ungated['held_out_pooled'] if ungated else None)} |"
        )
    return lines


def tuned(res: Results) -> bool:
    return (res.tuning or {}).get("status") == "tuned"


def stopped_lines(t: dict[str, Any]) -> list[str]:
    """What a stopped tuning saw: the best grid point and what each check could do."""
    by = ", ".join(f"{k} {v}" for k, v in t["n_runs_by_strategy_and_split"].items())
    out = [
        f"- Tuning stopped: {t['reason']}. configs/gate.yaml stays provisional and no "
        "held-out run was made with the gate (the evaluation refuses a gate that is not "
        "frozen).",
        f"- Input: {t['n_runs']} in-prior validation runs ({t['n_trajectories']} "
        f"trajectories; {by}), of which {t['n_bad']} were BAD.",
    ]
    points = t.get("points") or []
    if points:
        best = max(points, key=lambda r: (r["precision"] or 0.0, r["recall"] or 0.0))
        out.append(
            f"- Best precision on the grid: {num(best['precision'])} (recall "
            f"{num(best['recall'])}, {best['n_blocked']} of {best['n_counted']} counted runs "
            f"blocked) at ood_share limit {best['ood_share_limit']}, context_calibration "
            f"[{best['calibration_low']}, {best['calibration_high']}]. "
            + (
                "There is no minimum precision (decision 12)."
                if t["rules"].get("min_precision") is None
                else f"The minimum is {t['rules']['min_precision']}."
            )
        )
        keys = ("ood_share_limit", "calibration_low", "calibration_high")
        for key in keys:
            groups: dict[tuple[float, ...], set[int]] = {}
            for r in points:
                rest = tuple(r[k] for k in keys if k != key)
                groups.setdefault(rest, set()).add(r["n_blocked"])
            moves = any(len(v) > 1 for v in groups.values())
            out.append(
                f"- Moving {key} over its grid values, with the other thresholds held, "
                + ("changes" if moves else "never changes")
                + " the number of blocked runs."
            )
    return out


def current_gate_sha(res: Results) -> str | None:
    """The configs/gate.yaml hash the gated validation runs were made with, if any."""
    for strategy in GATE_STRATEGIES:
        g = gate_of(res, strategy, "val")
        if g is not None:
            return g.get("config_sha256")
    return None


def tuning_lines(res: Results) -> list[str]:
    lines = tuning_record_lines(res)
    t, sha = res.tuning, current_gate_sha(res)
    known = {t.get("input_gate_sha256"), t.get("output_gate_sha256")} if t else set()
    if t is not None and sha is not None and sha not in known:
        lines.append(
            f"- This tuning record was made with an earlier provisional gate "
            f"({str(t.get('input_gate_sha256'))[:16]}...). The validation rows below use the "
            f"current one ({sha[:16]}...), which has not been tuned yet."
        )
    return lines


def tuning_record_lines(res: Results) -> list[str]:
    t = res.tuning
    if t is None:
        return ["- The gate has not been tuned: results/gate_tuning.json does not exist."]
    if t["status"] != "tuned":
        return stopped_lines(t)
    c, o = t["chosen"], t["out_of_fold"]
    by = ", ".join(f"{k} {v}" for k, v in t["n_runs_by_strategy_and_split"].items())
    alerts = "; ".join(
        f"{s} {num(a['threshold'], 'none', 4)} ({a['n_accepted_control_runs']} of "
        f"{a['n_control_runs']} control runs accepted)"
        for s, a in t["alert_thresholds"].items()
    )
    return [
        f"- Tuned on {t['n_runs']} in-prior validation runs ({t['n_trajectories']} "
        f"trajectories; {by}), of which {t['n_bad']} were BAD. Strategies: "
        f"{', '.join(t['strategies'])}.",
        f"- Chosen point: ood_share limit {c['ood_share_limit']}, context_calibration "
        f"[{c['calibration_low']}, {c['calibration_high']}], out of {len(t['points'])} grid "
        f"points. Rule: {t['rules']['objective']}.",
        f"- In-sample on validation: precision {num(c['precision'])}, recall "
        f"{num(c['recall'])}, F1 {num(c.get('f1'))}. Out-of-fold ({o['folds']} folds by "
        f"trajectory): precision "
        f"{num(o['precision'])}, recall {num(o['recall'])}"
        + (
            f", {o['n_infeasible_folds']} folds had no feasible point"
            if o["n_infeasible_folds"]
            else ""
        )  # fmt: skip
        + ". The in-sample numbers are optimistic by construction.",
        f"- Alert thresholds on p_critical, from accepted validation control runs: {alerts}.",
        f"- Gate frozen with SHA-256 {t['output_gate_sha256'][:16]}... at {t['tuned_at_utc']}.",
    ]


def compare(
    a: float | None, b: float | None, higher: str, lower: str, same: str = "the same as"
) -> str | None:
    """Compares the numbers as printed (2 decimals), so the word matches the table."""
    if a is None or b is None:
        return None
    a, b = round(a, 2), round(b, 2)
    return higher if a > b else lower if a < b else same


def gate_finding_lines(res: Results) -> list[str]:
    """Sentences chosen by code from the gate numbers: does the gate carry over?"""
    out = []
    min_p = (res.tuning or {}).get("rules", {}).get("min_precision")
    for strategy in GATE_STRATEGIES:
        val, test, ood = (gate_of(res, strategy, f) for f in ("val", "test", "ood"))
        if val is None or ood is None:
            continue
        name = f"TabPFN {strategy}"
        vb, ob = val["blocking"] or {}, ood["blocking"] or {}
        word = compare(ood["abstention_rate"], (test or {}).get("abstention_rate"),
                       "higher than", "lower than")  # fmt: skip
        if word:
            out.append(
                f"- {name}: abstention out of prior ({num(ood['abstention_rate'])}) is {word} "
                f"in-prior test ({num(test['abstention_rate'])})."
            )
        if ood["n_bad"] == 0:
            out.append(f"- {name}: no out-of-prior run was BAD, so recall is undefined there.")
        else:
            word = compare(ob.get("recall"), vb.get("recall"), "higher than", "lower than")
            out.append(
                f"- {name}: recall of BAD runs out of prior is {num(ob.get('recall'))}, "
                f"{word or 'not comparable with'} the in-sample validation recall "
                f"({num(vb.get('recall'))})."
            )
        if min_p is not None and ob.get("precision") is not None:
            word = "at or above" if ob["precision"] >= min_p else "below"
            out.append(
                f"- {name}: precision out of prior is {num(ob['precision'])}, {word} the "
                f"tuning minimum {min_p}."
            )
        for title, flag in GATE_SPLITS:
            g = gate_of(res, strategy, flag)
            if g is None or g["coverage_90_mean_accepted"] is None:
                continue
            word = compare(g["coverage_90_mean_accepted"], g["coverage_90_mean_all"],
                           "raised", "lowered", "did not change")  # fmt: skip
            out.append(
                f"- {name}, {title}: accepting only gated runs {word} mean 90% coverage "
                f"({num(g['coverage_90_mean_all'])} on all runs, "
                f"{num(g['coverage_90_mean_accepted'])} on accepted runs)."
            )
    return out or ["- No gated run to compare yet."]


def per_split(res: Results, method: str, fn: Any) -> str:
    """One cell with a value for in-prior test and one for out-of-prior, "a / b".

    If both are the same "n/a (reason)" or "not run", the cell says it once.
    """
    parts = []
    for split in VARIANT_SPLITS:
        d = res.variants.get((method, split))
        parts.append("not run" if d is None else fn(d))
    if len(set(parts)) == 1 and parts[0].startswith(("n/a", "not run")):
        return parts[0]
    return " / ".join(parts)


def has_result(res: Results, method: str) -> bool:
    return any((method, split) in res.variants for split in VARIANT_SPLITS)


def pfn_of(res: Results, method: str, split: str) -> dict[str, Any] | None:
    """The PFN block of a variant's summary for one split, or None."""
    d = res.variants.get((method, split))
    return None if d is None else (d["summary"] or {}).get("pfn")


def variant_auroc(res: Results, method: str, source: str, split: str) -> str:
    d = res.variants.get((method, split))
    if d is None:
        return "not run"
    m = d["metrics"].get(source) or {}
    a = m.get("auroc_end")
    if a:
        text = f"{a['auroc']:.2f} [{a['ci_low']:.2f}, {a['ci_high']:.2f}]"
        scorable = m["n_positive_end"] + m["n_negative_end"]
        return text + (f" (n={m['n_scored']})" if m["n_scored"] < scorable else "")
    if m.get("n_positive_end", 0) == 0:
        skipped = "tipping runs skipped: family not in the pool"
        return f"n/a ({skipped if d.get('skipped') else 'no tipping run'})"
    return "n/a (no run had a score)" if m.get("n_scored", 0) == 0 else "n/a (one class only)"


def variant_coverage(res: Results, method: str) -> str:
    """Coverage of all runs, in-prior / out-of-prior, and of the accepted runs if a gate ruled."""
    blocks = [pfn_of(res, method, split) for split in VARIANT_SPLITS]
    present = [b for b in blocks if b is not None]
    if not present:
        return "n/a (no intervals)" if has_result(res, method) else "not run"
    text = " / ".join("not run" if b is None else num(b["coverage_90_mean_all"]) for b in blocks)
    if not any(b["gated"] for b in present):
        return text
    accepted = " / ".join(
        "not run" if b is None else num(b["coverage_90_mean_accepted"], "none accepted")
        for b in blocks
    )
    return f"{text} (accepted runs: {accepted})"


def variant_calibration(res: Results, method: str) -> str:
    def one(d: dict[str, Any]) -> str:
        p = (d["summary"] or {}).get("pfn")
        return "n/a (no intervals)" if p is None else num(p["calibration_error_mean"], digits=3)

    return per_split(res, method, one)


def variant_false_alarms(res: Results, method: str, source: str) -> str:
    def one(d: dict[str, Any]) -> str:
        m = (d["metrics"].get(source) or {}).get("control_alert") or {}
        if m.get("value") is None:
            return "n/a (no control run)"
        return f"{m['value']:.2f} (n={m['n']})"

    return per_split(res, method, one)


def variant_bad_blocked(res: Results, method: str) -> str:
    def one(d: dict[str, Any]) -> str:
        p = (d["summary"] or {}).get("pfn")
        if p is None or not p["gated"]:
            return "n/a (no gate)"
        if p["n_bad"] == 0:
            return "n/a (no BAD run)"
        return f"{p['bad_blocked_share']:.2f} ({p['n_bad_blocked']}/{p['n_bad']})"

    return per_split(res, method, one)


def variant_latency(res: Results, method: str) -> str:
    """The p95 of the run latencies of both splits together."""
    seconds = [
        r["timing"]["total_seconds"]
        for split in VARIANT_SPLITS
        for r in (res.variants.get((method, split)) or {}).get("runs", [])
    ]
    return f"{float(np.percentile(seconds, 95)):.3g} s" if seconds else "not run"


def variant_tool_calls(res: Results, method: str) -> str:
    n = total = 0.0
    for split in VARIANT_SPLITS:
        t = ((res.variants.get((method, split)) or {}).get("summary") or {}).get("tool_calls")
        if t:
            n += t["n"]
            total += t["mean"] * t["n"]
    if n == 0:
        return "n/a (no tools)" if has_result(res, method) else "not run"
    return f"{total / n:.2f}"


def variant_pass_k(res: Results, method: str) -> str:
    """Scenarios where the 3 seeds agree, over both splits together."""
    agree = scenarios = 0
    for split in VARIANT_SPLITS:
        pk = ((res.variants.get((method, split)) or {}).get("summary") or {}).get("pass_k")
        if pk:
            agree, scenarios = agree + pk["n_agree"], scenarios + pk["n_scenarios"]
    if scenarios == 0:
        return "n/a" if has_result(res, method) else "not run"
    return f"{agree / scenarios:.2f} ({agree}/{scenarios})"


def variant_table(res: Results) -> list[str]:
    head = [
        "Variant", "In-prior AUROC", "Out-of-prior AUROC",
        "90% coverage (in-prior / out-of-prior)", "Calibration error (in-prior / out-of-prior)",
        "False alarms on controls (in-prior / out-of-prior)",
        "Share of BAD runs blocked (in-prior / out-of-prior)", "p95 latency",
        "Tool calls per verdict", "pass^3: seeds that agree",
    ]  # fmt: skip
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for title, method, source in VARIANTS:
        cells = [
            title,
            variant_auroc(res, method, source, "test"),
            variant_auroc(res, method, source, "ood"),
            variant_coverage(res, method),
            variant_calibration(res, method),
            variant_false_alarms(res, method, source),
            variant_bad_blocked(res, method),
            variant_latency(res, method),
            variant_tool_calls(res, method),
            variant_pass_k(res, method),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def sample_line(limit: int | None, seeds: int | None) -> str:
    if limit is None:
        kinds = "all scenarios"
    else:
        kinds = f"{limit} scenarios of each kind (tipping and control)"
    trial = " (a trial run)" if limit is not None else ""
    return f"- Sample: {kinds}, {'all' if seeds is None else seeds} seeds per scenario{trial}."


def variant_notes(res: Results) -> list[str]:
    """What was run, read from the result files."""
    if not res.variants:
        return ["- No variant has been run yet (results/<variant>_{test,ood}.json)."]
    samples = {(d.get("limit"), d.get("seeds_per_scenario")) for d in res.variants.values()}
    lines = [sample_line(limit, seeds) for limit, seeds in sorted(samples, key=str)]
    counts = []
    for title, method, _ in VARIANTS:
        found = [res.variants.get((method, split)) for split in VARIANT_SPLITS]
        ran = " / ".join("not run" if d is None else str(d["n_runs"]) for d in found)
        skipped = sum(len((d or {}).get("skipped") or []) for d in found)
        note = f" ({skipped} skipped: family not in the pool)" if skipped else ""
        counts.append(f"{title.split(' (')[0]}: {ran}{note}")
    lines.append("- Runs per variant, in-prior test / out-of-prior: " + "; ".join(counts) + ".")
    llm = [d for d in res.variants.values() if d.get("model")]
    if llm:
        models = ", ".join(sorted({d["model"] for d in llm}))
        modes = ", ".join(sorted({str(d["llm_mode"]) for d in llm}))
        runs = sum(d["n_runs"] for d in llm)
        applied = sum(d["usage"]["temperature_applied_runs"] for d in llm)
        lines.append(
            f"- LLM: {models}; --llm-mode {modes}. Temperature 0 was applied in {applied} of "
            f"{runs} runs."
        )
    if res.degenerate_controls("family_matched"):
        lines.append(
            "- The oracle's context for a control run holds a constant target, so its output "
            "there is a constant (see ‡ above): its control rows say nothing about the method."
        )
    return lines


VARIANT_DEFINITIONS = [
    "A run is one trajectory. Every number in this table is run level: one verdict at the end of "
    "the series, scored against the label of the last window (near tipping, far from it, or "
    "the middle band, which is left out). The per-window AUROC and false-alarm rates above are "
    "different quantities and must not be compared with these.",
    "**AUROC**: of the run's score for the end of the series, over the tipping runs (positive) and "
    "the control runs (negative) of the split. The score is p_critical at the last window "
    "(oracle, and the full harness), the baseline score of the last window (classic), the "
    "language model's own probability (LLM only) or its confidence (naive agent). The full harness "
    "is scored on the runs the gate accepted, and (n=) says how many. CIs are a bootstrap over "
    "runs.",
    "**90% coverage** and **calibration error**: for variants that use the PFN, the mean over runs "
    "of the share of window targets inside the 90% interval of the PFN's output on the run's "
    "final context, and of the mean absolute gap between the five quantile levels and the "
    "frequencies observed. The truth is read by the evaluation only. The classic baseline and the "
    "LLM-only baseline give no intervals.",
    "**False alarms on controls**: the share of control runs whose decision is alert. An "
    "abstention is not an alert.",
    "**Share of BAD runs blocked**: a run is BAD if its 90% interval covers less than 0.80 of its "
    "windows (docs/GATE.md). Blocked means the gate failed. Only the full harness has a gate.",
    "**p95 latency**: the 95th percentile of the run latency over both splits. A run's latency is "
    "its model calls (as recorded, so a replay reports the live latency) plus the PFN fit and "
    "predict time (as first measured, gate refits included).",
    "**Tool calls per verdict**: the mean number of tool calls of a run (cap 6).",
    "**pass^3**: the share of scenarios, with all 3 seeds present, where the 3 seeds give the same "
    "final decision (abstain counts as a decision). It measures consistency, not correctness.",
]


def real_rate(res: Results, method: str, source: str, split: str, key: str) -> str:
    """One rate from a real-split result file, "value (n=...)"."""
    d = res.variants.get((method, split))
    if d is None:
        return "not run"
    m = (d["metrics"].get(source) or {}).get(key) or {}
    if m.get("value") is None:
        return "n/a (no such run)"
    return f"{m['value']:.2f} (n={m['n']})"


def real_table(res: Results) -> list[str]:
    head = [
        "Variant",
        "Daphnia (surrogate labels): AUROC",
        "Daphnia: alert on extinct populations",
        "Daphnia: alert on persisting populations",
        "Grid (normal operation): false-alarm rate",
    ]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for title, method, source in VARIANTS:
        cells = [
            variant_auroc(res, method, source, "real"),
            real_rate(res, method, source, "real", "recall_end"),
            real_rate(res, method, source, "real", "control_alert"),
            real_rate(res, method, source, "real_control", "control_alert"),
        ]
        lines.append("| " + " | ".join([title, *cells]) + " |")
    return lines


def real_section(res: Results) -> list[str]:
    if not any(split in REAL_VARIANT_SPLITS for (_, split) in res.variants):
        return []
    return [
        "## Real data (test only)",
        "",
        "Two real datasets, never in the context pool and never used to tune the gate. "
        "The Daphnia labels are surrogates (time to extinction over a fixed horizon): they are "
        "not the distance to a bifurcation (docs/DAPHNIA.md). The grid data is normal operation, "
        "not a pre-blackout run, and every window is negative, so it measures false alarms only "
        "and has no AUROC (docs/GRID.md).",
        "",
        *real_table(res),
        "",
        *real_sample_lines(res),
        "",
    ]


def real_sample_lines(res: Results) -> list[str]:
    lines = []
    for split, label in (("real", "Daphnia"), ("real_control", "grid")):
        parts = []
        for title, method, _ in VARIANTS:
            d = res.variants.get((method, split))
            if d is not None:
                parts.append(f"{title.split(' (')[0]}: {d['n_runs']}")
        if parts:
            lines.append(f"- Runs per variant on the {label} split: " + "; ".join(parts) + ".")
    return lines


def variants_section(res: Results) -> list[str]:
    return [
        "## Variants, run level",
        "",
        "One row per variant, on the same trajectories. The test and out-of-prior splits are "
        "held out: the gate was tuned and frozen before they were run.",
        "",
        *variant_table(res),
        "",
        *variant_notes(res),
        "",
        *[f"- {text}" for text in VARIANT_DEFINITIONS],
        "",
    ]


def build_summary(res: Results) -> str:
    lines = [
        "# Results summary",
        "",
        "Written by `scripts/make_summary.py` from the files in `results/`.",
        "Nothing here is typed by hand.",
        "",
    ]
    notes = reduction_notes(res)
    if notes:
        lines += ["## Reduced runs", "", *[f"- {n}" for n in notes], ""]
    for score, title in SCORE_TITLES.items():
        lines += [f"## AUROC, score = {title}", ""]
        lines += main_table(res, score)
        legend = (
            "Higher is better and 0.5 is chance. Cells show AUROC with the bootstrap CI "
            "resampled over trajectories. † = the CI includes 0.5. "
            f"⚠ = AUROC above {LEAKAGE_AUROC}, check for leakage."
        )
        lines += ["", legend, ""]
        lines += [
            "Compared with the baseline:",
            "",
            *(comparison_lines(res, score) or ["- no data"]),
            "",
        ]
        lines += [
            "<details><summary>AUROC per family</summary>",
            *family_tables(res, score),
            "",
            "</details>",
            "",
        ]
    lines += [
        "## False alarms on controls",
        "",
        *false_alarm_table(res),
        "",
        "The alarm threshold is set on the validation controls only, so that at most the "
        "target share of their windows alarm. It is then applied unchanged to the held-out "
        "controls (test_in_prior and test_out_of_prior). A trajectory counts as an alarm if "
        "any of its windows alarms, so that rate is high for any threshold. The window "
        "rate is the informative one. The main table shows the window rate. "
        "‡ = degenerate score on controls, see the checks below.",
        "",
    ]
    lines += [
        "## TabPFN + gate",
        "",
        "The gate (docs/GATE.md) decides per run whether to trust it. A blocked run "
        "abstains on all its windows. A run is BAD if its 90% interval covers less than "
        "0.80 of its window targets. Thresholds are tuned on in-prior validation runs only "
        "and frozen before any held-out run. Scores are p_critical. CIs come from a "
        "bootstrap over runs (trajectories).",
        "",
        "### Tuning",
        "",
        *tuning_lines(res),
        "",
        "### Gate on tipping runs",
        "",
        *gate_table(res),
        "",
        "AUROC on accepted runs and on all runs are computed on different sets of "
        "trajectories, so a higher AUROC on accepted runs can partly come from abstaining "
        "on hard trajectories. The interval width cap is an advisory: runs over it are "
        "counted and reported, but not blocked (docs/GATE.md, decision 7). The Gate column "
        "says which "
        "configs/gate.yaml produced the row: a provisional row shows the loosest grid "
        "point, not a tuned gate. AUROC on accepted runs from one or two accepted runs "
        "only ranks windows inside those trajectories.",
        "",
        "### Gate on control runs",
        "",
        *gate_false_alarm_table(res),
        "",
        "### Does the gate carry over? (computed from the numbers above)",
        "",
        *gate_finding_lines(res),
        "",
    ]
    lines += variants_section(res)
    lines += real_section(res)
    lines += ["## Run settings", "", *settings_table(res), ""]
    lines += ["## Checks computed from the results", "", *check_lines(res), ""]
    lines += [
        "## Definitions",
        "",
        "- **AUROC**: balanced AUROC on windows of tipping trajectories, positives and "
        "negatives as defined in `docs/LABELS.md` (the middle band is dropped). Controls "
        "are not in the AUROC. They are scored by the false-alarm rate.",
        "- **Baseline**: mean Kendall tau of rolling variance and lag-1 autocorrelation over "
        "the previous K windows (`configs/baseline_ews.yaml`). Its settings were fixed "
        "before the first run.",
        "- **TabPFN**: regression on the scale-free window features, with a context chosen "
        "from the in-prior context pool. The context never contains the query trajectory.",
        "- **In-prior splits**: the context pool holds the same scenarios as the in-prior "
        "val and test splits, with other seeds (docs/SPLITS.md). In-prior AUROC is "
        "therefore easier than for a new system. The out-of-prior column is the "
        "test of generalisation.",
        "- **knn_balanced**: knn, except that a fixed share of the context is kept for pool "
        "rows near tipping (target at or below the critical threshold), the ones nearest to "
        "the query. It reads pool targets only, never the query's family, so it is "
        "deployable (docs/GATE.md, decision 6).",
        "- **family_matched (ORACLE)**: the context is sampled from the family of the query, "
        "which a real run would not know. It is an upper bound for context selection and not "
        "a method. It cannot run on out-of-prior families, which are not in the pool.",
        "- **Latency per trajectory**: fit plus predict for all windows of one trajectory, "
        "median over the tipping trajectories of the strategy (val, test and ood runs), on "
        "this CPU. Control runs are left out. A cached result keeps "
        "the latency of the run that computed it. The baseline latency was not measured.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    args = p.parse_args()
    text = build_summary(Results(args.results_dir))
    out = args.results_dir / "summary.md"
    out.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

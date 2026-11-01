"""The go/no-go report as one static HTML page.

Plain Python templating. Everything comes from files: the verdict from the release criteria and
the result files, the tables from results/summary.md, the figures from results/figs/, the
gallery from results/failures.json, the provenance from results/run_manifest.json and
data/SOURCES.md. The prose in the method section holds no numbers, so every number on the page
is a number from a file. See check.py.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from pfngate.report.check import normalise, numbers_in_text
from pfngate.report.criteria import CONFIDENCE, Outcome, Verdict
from pfngate.report.gallery import GalleryEntry, build_gallery, data_uri
from pfngate.report.style import CSS
from pfngate.report.text_sources import (
    Table,
    parse_sources,
    plain,
    read_text,
    section_bullets,
    section_table,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
GALLERY_PER_CATEGORY = 2
FAMILY_FIGURES = (
    ("fold", "Fold (in prior)"),
    ("hopf", "Subcritical Hopf (in prior)"),
    ("transcritical", "Transcritical (in prior)"),
    ("noise_tipping", "Noise-induced tipping (out of prior)"),
    ("rate_tipping", "Rate-induced tipping (out of prior)"),
    ("control_ou", "Control: Ornstein-Uhlenbeck"),
    ("control_short_ramp", "Control: short ramp"),
    ("control_regime_change", "Control: regime change"),
)
CATEGORY_TITLES = {
    "a_false_alarms_on_controls": "False alarms on negative controls",
    "b_bad_runs_the_gate_let_through": "Bad runs the gate let through",
    "c_good_runs_the_gate_blocked": "Good runs the gate blocked",
    "d_agent_disagreed_with_gated_decision": "Agent recommendation differed from final decision",
}
VARIANT_GROUPS = (
    ("Before: no agent and no gate", ("Classic EWS baseline", "LLM only")),
    ("An agent, but no gate", ("Naive agent",)),
    ("After: the full harness, agent and gate", ("Full harness",)),
    ("Upper bound, not a method", ("Family-matched oracle",)),
)
NUMBER_LABELS = {
    "end_label": "label at the end of the series",
    "end_target": "target at the end (low means near tipping)",
    "alarm_threshold": "alarm threshold",
    "alert_threshold": "alert threshold on p_critical",
    "n_windows": "windows in the run",
    "answer_probability": "probability the model stated",
    "verdict_decision": "agent recommendation",
    "verdict_confidence": "agent confidence",
    "verdict_context_strategy": "context strategy the agent named",
    "final_decision": "final decision",
    "decided_by": "decided by",
    "gate_passed": "gate passed",
    "gate_failed": "gate checks that failed",
    "gate_reasons": "reasons the gate gave",
    "last_window_p_critical": "p_critical at the last window",
    "window_decisions": "decisions over the windows",
    "context_strategy": "context strategy used",
    "context_size": "context rows",
    "followed_verdict_context": "gated context is the one the agent named",
    "coverage_90": "90% interval coverage of the window targets",
    "calibration_error": "calibration error",
}
DECISION_FIELDS = ("verdict_decision", "final_decision", "decided_by")
MODEL_LABELS = {"llm_agent": "LLM agent model", "llm_debug": "LLM debug model"}


def esc(value: Any) -> str:
    """Escape a value for HTML."""
    return html.escape(str(value), quote=True)


LEDGER: set[str] = set()


def reset_ledger() -> None:
    """Forget the numbers formatted so far. build_page calls this first."""
    LEDGER.clear()


def rendered_numbers() -> set[str]:
    """The numbers the page formatted from data. The number check accepts these."""
    return set(LEDGER)


def note_number(token: str) -> str:
    """Record a number that is about to be shown, and return it unchanged."""
    LEDGER.add(normalise(token))
    return token


def quoted(text: str) -> str:
    """Text quoted word for word from a result file, such as an agent rationale.

    The numbers inside it were written by the agent into the file, so they are recorded as
    quoted, not as formatted by the page.
    """
    LEDGER.update(numbers_in_text(text))
    return text


def num(value: Any, digits: int = 3) -> str:
    """A number for the page: fixed decimals, never scientific notation. It is recorded."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return note_number(str(value))
    if isinstance(value, float):
        return note_number(f"{value:.{digits}f}")
    return str(value)


def render_value(value: Any) -> str:
    """Render a value as page text: dicts and lists as text, numbers as num()."""
    if isinstance(value, dict):
        return ", ".join(f"{k} {render_value(v)}" for k, v in value.items())
    if isinstance(value, list):
        return ", ".join(render_value(v) for v in value)
    return num(value)


def wrap_table(head: str, body: str) -> str:
    """Wrap a table head and body in the scrolling table element."""
    return (
        f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def table_html(headers: list[str], rows: list[list[str]], classes: list[str] | None = None) -> str:
    """A table with the given headers and rows of cell HTML."""
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = []
    for row in rows:
        cells = "".join(f"<td>{c}</td>" for c in row)
        body.append(f"<tr>{cells}</tr>")
    return wrap_table(head, "".join(body))


def variant_group(name: str) -> str:
    """The row group of a variant in the before-and-after table."""
    for title, prefixes in VARIANT_GROUPS:
        if name.startswith(prefixes):
            return title
    return "Other"


def degenerate_note(summary: str) -> str:
    """The legend of the double dagger mark, taken from the checks in summary.md."""
    for bullet in section_bullets(summary, "Checks computed"):
        if bullet.startswith("‡"):
            return f'<p class="note">{esc(plain(bullet))}</p>'
    return ""


def summary_table_html(table: Table | None, group_variants: bool = False) -> str:
    """A table from a parsed summary table, with a group row per kind of variant."""
    if table is None:
        return '<p class="note">This table is not in results/summary.md.</p>'
    head = "".join(f"<th>{esc(plain(h))}</th>" for h in table.headers)
    rows = []
    last_group = None
    for row in table.rows:
        if group_variants:
            group = variant_group(plain(row[0]))
            if group != last_group:
                rows.append(f'<tr class="group"><td colspan="{len(row)}">{esc(group)}</td></tr>')
                last_group = group
        rows.append("<tr>" + "".join(f"<td>{esc(plain(c))}</td>" for c in row) + "</tr>")
    return wrap_table(head, "".join(rows))


def figure_html(path: Path, caption: str) -> str:
    """A figure with the PNG embedded, or a note if the file is missing."""
    if not path.exists():
        return f'<p class="note">Missing figure {esc(path.name)}: {esc(caption)}.</p>'
    png = base64_png(path)
    return (
        f'<figure><img src="{png}" alt="{esc(caption)}" loading="lazy">'
        f"<figcaption>{esc(caption)}</figcaption></figure>"
    )


def base64_png(path: Path) -> str:
    """A PNG file as a data URI."""
    return data_uri(path.read_bytes())


def section(number: str, anchor: str, title: str, body: str) -> str:
    """A page section with its number, title and body."""
    return (
        f'<section id="{anchor}"><h2><span class="secno">{esc(number)}</span>{esc(title)}</h2>'
        f"{body}</section>"
    )


# 1. Verdict


def outcome_row(o: Outcome) -> list[str]:
    """One table row for a criterion outcome, with its interval and whether it is decided."""
    value = "not computed" if o.value is None else num(o.value, 2)
    interval = ""
    if o.ci_low is not None and o.ci_high is not None:
        interval = f' <span class="mono">[{num(o.ci_low, 2)}, {num(o.ci_high, 2)}]</span>'
    if o.numerator is not None and o.denominator is not None:
        detail = f" ({num(o.numerator)} of {num(o.denominator)} {esc(o.unit)})"
    elif o.denominator is not None:
        detail = f" ({num(o.denominator)} {esc(o.unit)})"
    else:
        detail = ""
    mark = '<span class="pass">PASS</span>' if o.passed else '<span class="fail">FAIL</span>'
    if o.required and o.decided is False:
        mark += ' <span class="muted">(undecided: the interval spans the threshold)</span>'
    if not o.required:
        mark = '<span class="muted">reported only</span>'
    return [
        esc(o.label),
        f'<span class="mono">{esc(value)}</span>{interval}{detail}',
        f'<span class="mono">{esc(o.comparison)} {esc(num(o.threshold, 2))}</span>',
        mark,
    ]


def tree_notice(results_dir: Path) -> str:
    """A warning when the numbers come from a working tree with uncommitted changes."""
    path = results_dir / "run_manifest.json"
    if not path.exists():
        return ""
    git = json.loads(path.read_text(encoding="utf-8")).get("git") or {}
    n = git.get("uncommitted_files")
    if not n:
        return ""
    commit = esc(str(git.get("commit", ""))[:12])
    return (
        '<p class="warn"><strong>Not reproducible from git.</strong> These numbers were produced '
        f"from commit {commit} with {num(n)} uncommitted files, so no commit reproduces them. "
        "Commit, then run the suite again on the clean tree.</p>"
    )


def verdict_section(
    verdict: Verdict, criteria_file: str, notice: str = "", confidence: float = CONFIDENCE
) -> str:
    """Section 1: the verdict banner, the criteria and the reported criteria."""
    required = verdict.required
    met = sum(o.passed for o in required)
    css = "go" if verdict.verdict == "GO" else "nogo"
    failed = [o.label for o in required if not o.passed]
    reason = (
        "Every required criterion is met." if not failed else "Not met: " + "; ".join(failed) + "."
    )
    undecided = verdict.undecided
    sure = (
        f" {num(len(undecided))} of {num(len(required))} required results are undecided at "
        f"{num(confidence * 100, 0)}% confidence: the interval spans the threshold, so a few "
        "more runs could change them."
        if undecided
        else ""
    )
    banner = (
        f'<div class="banner {css}"><div class="word">{esc(verdict.verdict)}</div><div>'
        f"<strong>{esc(verdict.stand_in)}</strong><br>"
        f"{num(met)} of {num(len(required))} required criteria met. {esc(reason)}{sure}</div></div>"
    )
    headers = ["Criterion", "Value [interval]", "Threshold", "Result"]
    rows = [outcome_row(o) for o in required]
    out_rows = [outcome_row(o) for o in verdict.reported_only]
    note = (
        '<p class="note">The thresholds are illustrative. They were written after the '
        f"experiments, for demonstration, in <code>{esc(criteria_file)}</code>. Out-of-prior "
        "performance is reported and is never a pass condition. A criterion that cannot be "
        "computed counts as not met. The verdict uses the point estimate; the interval is exact "
        "binomial for shares and a bootstrap over runs for coverage. The gate recall rests on "
        "very few bad runs.</p>"
    )
    return (
        banner
        + notice
        + table_html(headers, rows)
        + "<h3>Reported, not a pass condition</h3>"
        + table_html(headers, out_rows)
        + note
    )


# 2. Before and after


def comparison_section(summary: str) -> str:
    """Section 2: the before-and-after table, real data and reduced runs."""
    variants = section_table(summary, "Variants, run level")
    real = section_table(summary, "Real data (test only)")
    body = [
        "<p>One row per variant on the same trajectories. The first two rows are the "
        "before: a classic early-warning statistic and a language model alone. The naive "
        "agent adds an agent but no gate. The full harness is the after. The oracle is an "
        "upper bound, not a method.</p>",
        summary_table_html(variants, group_variants=True),
        degenerate_note(summary),
    ]
    if variants and variants.notes:
        body.append(
            '<ul class="note">'
            + "".join(f"<li>{esc(plain(n))}</li>" for n in variants.notes)
            + "</ul>"
        )
    body.append("<h3>Real data (test only)</h3>")
    body.append(summary_table_html(real))
    if real and real.notes:
        body.append(
            '<ul class="note">' + "".join(f"<li>{esc(plain(n))}</li>" for n in real.notes) + "</ul>"
        )
    reduced = section_bullets(summary, "Reduced runs")
    if reduced:
        body.append("<h3>How big these runs are</h3><ul>")
        body.extend(f"<li>{esc(plain(b))}</li>" for b in reduced)
        body.append("</ul>")
    return "".join(body)


# 3. Figures


def figures_section(figs_dir: Path, summary: str) -> str:
    """Section 3: the reliability diagram, the strategy table and the figures."""
    strategies = section_table(summary, "AUROC, score = p_critical")
    legend = (
        f'<p class="note">{esc(plain(strategies.after))}</p>'
        if strategies and strategies.after
        else ""
    )
    parts = [
        "<h3>Reliability diagram</h3>",
        "<p>How often the true value falls at or below each predicted quantile, for all runs "
        "and for the runs the gate accepted. A well calibrated model lies on the diagonal.</p>",
        figure_html(figs_dir / "reliability.png", "Reliability diagram of the predicted quantiles"),
        "<h3>Context strategies</h3>",
        "<p>No separate context ablation has been run. The table below compares the ways of "
        "choosing the context for the same model, from the same results file.</p>",
        summary_table_html(strategies),
        legend,
        degenerate_note(summary),
        figure_html(figs_dir / "baseline_auroc.png", "AUROC of the classic baseline"),
        "<h3>Scenario figures</h3>",
        '<div class="grid">',
    ]
    parts.extend(figure_html(figs_dir / f"{name}.png", title) for name, title in FAMILY_FIGURES)
    parts.append("</div>")
    return "".join(parts)


# 4. Failure gallery


def check_value(value: Any) -> str:
    """A gate value for the page: whole numbers above one without decimals, others as num()."""
    if isinstance(value, float) and value.is_integer() and abs(value) >= 2:
        return note_number(str(int(value)))
    if isinstance(value, list):
        return ", ".join(check_value(v) for v in value)
    return render_value(value)


def field_value(key: str, value: Any) -> str:
    """A number-list value for the gallery: yes and no for flags, none for empty, words for ids."""
    if value is None or value == [] or value == "":
        return "none"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if key in DECISION_FIELDS and isinstance(value, str):
        return value.replace("_", " ")
    if isinstance(value, dict):
        return ", ".join(f"{str(k).replace('_', ' ')} {render_value(v)}" for k, v in value.items())
    return render_value(value)


def gate_table(entry: GalleryEntry) -> str:
    """The gate checks of a gallery entry, as a table."""
    if not entry.gate_checks:
        return '<p class="note">This variant has no gate result for this run.</p>'
    rows = [
        [
            esc(c.name),
            f'<span class="mono">{esc(check_value(c.value))}</span>',
            f'<span class="mono">{esc(check_value(c.threshold))}</span>',
            '<span class="pass">passed</span>' if c.passed else '<span class="fail">failed</span>',
        ]
        for c in entry.gate_checks
    ]
    state = {True: "passed", False: "blocked", None: "none"}[entry.gate_passed]
    return f"<p>Gate: <strong>{state}</strong>.</p>" + table_html(
        ["Check", "Value", "Threshold", "Result"], rows
    )


def trace_table(entry: GalleryEntry) -> str:
    """The trace excerpt of a gallery entry, as a table."""
    if not entry.trace:
        return '<p class="note">No trace lines found.</p>'
    rows = [
        [
            f'<span class="mono">{num(t.line)}</span>',
            esc(t.tool),
            esc(num(t.latency_s)),
            esc(t.strategy or "-"),
            esc(num(t.context_size)),
        ]
        for t in entry.trace
    ]
    omitted = (
        f'<p class="note">{num(entry.trace_omitted)} more lines are left out of this excerpt.</p>'
        if entry.trace_omitted
        else ""
    )
    return table_html(["Line", "Tool", "Latency (s)", "Strategy", "Context size"], rows) + omitted


def gallery_card(entry: GalleryEntry, position: str) -> str:
    """One gallery card: the plot, numbers, gate, rationale and trace excerpt."""
    ex = entry.example
    title = f"{position}: {ex['variant']} on {ex['trajectory_id']} ({ex['split']})"
    facts = {
        "scenario": ex["scenario_id"],
        "seed": ex["seed"],
        "variant": ex["variant"],
        "split": ex["split"],
        "severity": ex["severity"],
        "severity measures": ex["severity_basis"],
        "result file": ex["result_file"],
    }
    dl = "".join(f"<dt>{esc(k)}</dt><dd>{esc(render_value(v))}</dd>" for k, v in facts.items())
    numbers = "".join(
        f"<dt>{esc(NUMBER_LABELS.get(k, k.replace('_', ' ')))}</dt>"
        f"<dd>{esc(field_value(k, v))}</dd>"
        for k, v in ex["numbers"].items()
    )
    plot = (
        f'<img src="{data_uri(entry.plot_png)}" alt="Scenario plot of {esc(ex["trajectory_id"])}">'
        if entry.plot_png
        else f'<p class="note">{esc(entry.plot_note)}</p>'
    )
    if entry.rationale:
        conf = (
            f" (confidence {esc(num(entry.verdict_confidence))})"
            if entry.verdict_confidence is not None
            else ""
        )
        rationale = (
            "<h5>Agent rationale</h5><p>Recommended "
            f"<strong>{esc(field_value('verdict_decision', entry.verdict_decision))}</strong>"
            f"{conf}; final decision "
            f"<strong>{esc(field_value('final_decision', entry.final_decision))}</strong>.</p>"
            f"<blockquote>{esc(quoted(entry.rationale))}</blockquote>"
        )
    else:
        rationale = (
            '<h5>Agent rationale</h5><p class="note">This variant has no agent rationale.</p>'
        )
    lines = ", ".join(num(a) if a == b else f"{num(a)} to {num(b)}" for a, b in ex["lines"])
    return (
        f'<div class="card"><h4>{esc(title)}</h4>'
        f'<div class="cols"><div>{plot}<dl>{dl}</dl></div>'
        f"<div><h5>Numbers involved</h5><dl>{numbers}</dl></div></div>"
        f"<h5>Gate result</h5>{gate_table(entry)}{rationale}"
        f"<h5>Trace excerpt</h5>"
        f'<p class="note"><code>{esc(ex["trace_file"])}</code>, lines {esc(lines)}</p>'
        f"{trace_table(entry)}</div>"
    )


def gallery_section(results_dir: Path, per_category: int) -> str:
    """Section 4: the failure gallery, with the top examples per category."""
    path = results_dir / "failures.json"
    if not path.exists():
        return '<p class="note">results/failures.json is missing: run scripts/find_failures.py.</p>'
    failures = json.loads(path.read_text(encoding="utf-8"))
    scope = failures["scope"]
    entries = build_gallery(results_dir, per_category)
    ordinal = ("First", "Second", "Third")
    parts = [
        "<p>The two highest-ranked examples of each kind of failure, from "
        "<code>results/failures.json</code>. They are ranked and linked, not explained. "
        f"{esc(scope['note'])}</p>"
    ]
    for category in failures["categories"]:
        parts.append(f"<h3>{esc(CATEGORY_TITLES[category])}</h3>")
        parts.append(f"<p>{esc(failures['overviews'][category])}</p>")
        shown = [e for e in entries if e.category == category]
        for i, entry in enumerate(shown):
            parts.append(gallery_card(entry, ordinal[i] if i < len(ordinal) else "Next"))
        if not shown:
            parts.append('<p class="note">No example in this category.</p>')
    return "".join(parts)


# 5. Method and limits

METHOD_PARAGRAPHS = (
    "<h3>What is simulated</h3><p>Most of the evidence comes from simulators. In-prior systems "
    "are the fold, the subcritical Hopf and the transcritical bifurcation. Out-of-prior systems "
    "are rate-induced and noise-induced tipping. Controls are series that never tip, including "
    "one that changes regime without tipping. Data are split by system family, never by window. "
    "The context pool uses its own seeds. The gate thresholds were tuned on in-prior "
    "validation runs only. Held-out simulated results had already been computed and read "
    "before the gate was designed, so they are not blind (docs/AUDIT.md has the timeline).</p>",
    "<h3>What is surrogate labelled</h3><p>The two real datasets are test only. They never "
    "enter the context pool and never touch the gate tuning. The Daphnia populations have no "
    "known distance to a bifurcation. Their label is a surrogate: how long until extinction, "
    "over a fixed horizon, so its scores are not comparable to the simulated ones. The power "
    "grid frequency data is normal operation, not a run up to a blackout. Every window is "
    "negative, so it can only measure false alarms and has no AUROC.</p>",
    "<h3>What the gate cannot know</h3><p>The gate is plain Python with frozen thresholds. It "
    "sees the context and the model output, never the truth. It checks the context size, how "
    "far the query is from the pool, how well the context predicts held-out context rows, "
    "whether the context holds both near-tipping and far-from-tipping rows, and the integrity "
    "of the output. It cannot tell that a series comes from a system unlike anything in the "
    "pool when the output still looks plausible. A run is called bad only afterwards, when "
    "its prediction interval is compared with the truth. The recall of the gate is measured "
    "on few bad runs, so it is uncertain. An LLM agent may recommend but can never override or "
    "edit a gate decision. Its tools do show it some of the gate's own inputs, such as how far "
    "the query is from the pool, so it can pick a context that will pass.</p>",
    "<h3>The model is a stand-in</h3><p>The model behind every number here is TabPFN, run "
    "locally. TipPFN is not public. The backend is swappable, and the TipPFN backend in this "
    "repository is a stub that raises an error and documents the input and output it would "
    "need. Nothing on this page reproduces TipPFN or says how TipPFN would score.</p>",
    "<h3>Size of the runs</h3><p>These are trial-size runs on a slow CPU: a reduced context "
    "and a few seeds per scenario, with a free language model. The intervals are wide. "
    "Release criteria are illustrative and were set after the experiments.</p>",
)


def method_section() -> str:
    """Section 5: the method and limits text."""
    return "".join(METHOD_PARAGRAPHS)


# 6. Provenance


def dl_rows(items: dict[str, Any]) -> str:
    """Definition-list rows for a dictionary of labels and values."""
    rows = []
    for k, v in items.items():
        text = esc(render_value(v)).replace("\n", "<br>")  # nested source bullets keep their lines
        rows.append(f"<dt>{esc(k)}</dt><dd>{text}</dd>")
    return "".join(rows)


def pfn_model_info(results_dir: Path) -> dict[str, Any]:
    """The PFN backend, model version, package version and context cap, from a result file."""
    path = results_dir / "agent_test.json"
    if not path.exists():
        return {}
    settings = json.loads(path.read_text(encoding="utf-8")).get("settings", {})
    info = {
        "PFN backend": settings.get("backend"),
        "PFN model version": settings.get("model_version"),
        "PFN package version": settings.get("package_version"),
        "context rows (cap)": settings.get("max_context"),
    }
    return {k: v for k, v in info.items() if v is not None}


def provenance_section(results_dir: Path, sources_text: str) -> str:
    """Section 6: commit, packages, seeds, models and data sources."""
    path = results_dir / "run_manifest.json"
    if not path.exists():
        return '<p class="note">results/run_manifest.json is missing: run scripts/run_all.py.</p>'
    m = json.loads(path.read_text(encoding="utf-8"))
    git = m["git"]
    run = {
        "git commit at the run": git["commit"],
        "branch": git["branch"],
        "uncommitted files": git["uncommitted_files"],
        "profile": m["profile"],
        "run started (UTC)": m["started_utc"],
        "wall clock (s)": m["wall_clock_seconds_total"],
        "gate SHA-256": m["gate"]["sha256"],
        "gate status": m["gate"]["status"],
        "python": m["python"],
        "platform": m["platform"],
    }
    models = {MODEL_LABELS.get(k, k): v for k, v in m["models"].items() if k != "pfn_backend"}
    models.update(pfn_model_info(results_dir))
    seeds = m["seeds"]
    seed_info = {
        "PFN seed": seeds["pfn_seed"],
        "LLM temperature": seeds["llm_temperature"],
        "seeds per scenario": seeds["seeds_per_scenario"],
        "scenarios per kind (limit)": seeds["limit"],
        "seed sets": seeds["scenario_seed_sets"],
    }
    totals = m["totals"]
    usage = {
        "tokens in": totals["tokens_in"],
        "tokens out": totals["tokens_out"],
        "cost, USD": totals["cost_usd_approx"],
        "cost note": totals["cost_note"],
    }
    parts = [
        "<h3>Run</h3><dl>" + dl_rows(run) + "</dl>",
        "<h3>Models</h3><dl>" + dl_rows(models) + "</dl>",
        "<h3>Seeds</h3><dl>" + dl_rows(seed_info) + "</dl>",
        "<h3>Packages</h3><dl>" + dl_rows(m["packages"]) + "</dl>",
        "<h3>LLM usage</h3><dl>" + dl_rows(usage) + "</dl>",
        "<h3>Datasets, sources and licences</h3>",
    ]
    for entry in parse_sources(sources_text):
        items = {label: text for label, text in entry.items}
        parts.append(f"<h4>{esc(entry.title)}</h4><dl>{dl_rows(items)}</dl>")
    return "".join(parts)


NAV = (
    ("verdict", "Verdict"),
    ("before-after", "Before and after"),
    ("figures", "Figures"),
    ("gallery", "Failure gallery"),
    ("method", "Method and limits"),
    ("provenance", "Provenance"),
)


def build_page(
    results_dir: Path,
    verdict: Verdict,
    sources_path: Path,
    criteria_file: str = "configs/release_criteria.yaml",
    per_category: int = GALLERY_PER_CATEGORY,
) -> str:
    """Build the whole report page as one HTML string."""
    reset_ledger()
    summary = read_text(results_dir / "summary.md")
    sources = read_text(sources_path)
    nav = "".join(f'<a href="#{a}">{esc(t)}</a>' for a, t in NAV)
    sections = [
        section(
            "1",
            "verdict",
            "Verdict",
            verdict_section(verdict, criteria_file, tree_notice(results_dir)),
        ),
        section("2", "before-after", "Before and after", comparison_section(summary)),
        section("3", "figures", "Figures", figures_section(results_dir / "figs", summary)),
        section("4", "gallery", "Failure gallery", gallery_section(results_dir, per_category)),
        section("5", "method", "Method and limits", method_section()),
        section("6", "provenance", "Provenance", provenance_section(results_dir, sources)),
    ]
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>pfngate release report</title>"
        f"<style>{CSS}</style></head><body><main>"
        "<h1>pfngate release report</h1>"
        '<p class="muted">A release gate for early-warning models: go or no-go for the '
        "stand-in model, with the evidence behind it. Every number comes from a file in "
        "<code>results/</code>.</p>"
        f"<nav>{nav}</nav>{''.join(sections)}</main></body></html>\n"
    )

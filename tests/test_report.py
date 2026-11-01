"""The go/no-go report: the verdict, the number check and the page, on small made-up results.

One test builds the page from the real results/ when they are on disk, and fails if the page
shows a number that no file in results/ (or data/SOURCES.md) holds.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from pfngate.report import page as report_page
from pfngate.report.build import UnsourcedNumbersError, build_report
from pfngate.report.check import (
    allowed_numbers,
    numbers_in_json,
    numbers_on_page,
    unsourced_numbers,
)
from pfngate.report.criteria import (
    DEFAULT_CRITERIA,
    evaluate,
    load_criteria,
    measure,
    write_criteria_json,
)
from pfngate.report.text_sources import parse_sources, section_table

REAL_RESULTS = Path(__file__).resolve().parents[1] / "results"
SECTION_IDS = ["verdict", "before-after", "figures", "gallery", "method", "provenance"]


# Made-up results


def run(
    tid: str,
    role: str,
    decision: str,
    accepted: bool | None,
    coverage: float,
    verdict: tuple[str, float] | None = None,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "trajectory_id": tid,
        "final": {"decision": decision},
        "gate_result": {
            "passed": accepted,
            "checks": [
                {"name": "ood_share", "value": 0.1, "threshold": 0.3, "passed": True, "reason": ""}
            ],
        },
    }
    if verdict:
        rec["verdict"] = {"decision": verdict[0], "confidence": verdict[1], "rationale": "why"}
    return {
        "record": rec,
        "ground_truth": {"scenario_id": tid.split("/")[0], "role": role},
        "decision": decision,
        "accepted": accepted,
        "pfn_quality": {"coverage_90": coverage, "bad": coverage < 0.8},
    }


def write_variant(
    results: Path, split: str, runs: list[dict[str, Any]], pfn: dict[str, Any] | None
) -> None:
    doc: dict[str, Any] = {"trace": "t.jsonl", "runs": runs, "summary": {"pfn": pfn}}
    (results / f"agent_{split}.json").write_text(json.dumps(doc), encoding="utf-8")


def make_results(tmp_path: Path, *, recall_blocked: int = 1) -> Path:
    results = tmp_path / "results"
    (results / "traces").mkdir(parents=True)
    in_prior = {
        "n_accepted": 5,
        "coverage_90_mean_accepted": 0.91,
        "n_bad": 2,
        "n_bad_blocked": recall_blocked,
    }
    write_variant(
        results,
        "test",
        [
            run("ou_L1/3000", "control", "no_alert", True, 0.9),
            run("fold_L1_slow/3000", "in_prior", "alert", True, 0.5, ("no_alert", 0.8)),
        ],
        in_prior,
    )
    write_variant(
        results,
        "ood",
        [run("ou_L1/4000", "control", "no_alert", False, 0.95)],
        {"n_accepted": 3, "coverage_90_mean_accepted": 0.7, "n_bad": 1, "n_bad_blocked": 0},
    )
    write_variant(
        results, "real", [run("daphnia_constant/2", "control", "no_alert", True, 0.9)], None
    )
    write_variant(
        results, "real_control", [run("grid_fingrid/0", "control", "no_alert", True, 0.9)], None
    )
    trace = [json.dumps({"scenario_id": "ou_L1/3000", "tool": "llm", "latency_s": 1.5})] * 3
    (results / "traces" / "t.jsonl").write_text("\n".join(trace) + "\n", encoding="utf-8")
    example = {
        "severity": 0.7,
        "severity_basis": "a made-up basis",
        "variant": "agent",
        "split": "test",
        "scenario_id": "ou_L1",
        "seed": 3000,
        "trajectory_id": "ou_L1/3000",
        "role": "control",
        "result_file": "agent_test.json",
        "trace_file": "results/traces/t.jsonl",
        "lines": [[1, 3]],
        "numbers": {"coverage_90": 0.9, "window_decisions": {"alert": 2, "no_alert": 9}},
    }
    cats = ("a_false_alarms_on_controls", "b_bad_runs_the_gate_let_through")
    cats += ("c_good_runs_the_gate_blocked", "d_agent_disagreed_with_gated_decision")
    failures = {
        "scope": {"note": "A made-up scope note."},
        "categories": {c: {"top": [example, example, example]} for c in cats},
        "overviews": {c: f"Overview of {c}: 4 of 9 runs." for c in cats},
    }
    (results / "failures.json").write_text(json.dumps(failures), encoding="utf-8")
    (results / "summary.md").write_text(
        "## Reduced runs\n\n- Only 3 seeds were run.\n\n"
        "## Variants, run level\n\n| Variant | AUROC |\n|---|---|\n"
        "| Classic EWS baseline | 0.79 |\n| Full harness (agent + gate) | 0.98 |\n\n"
        "- Sample: 3 scenarios.\n\n## Real data (test only)\n\n| Variant | Rate |\n|---|---|\n"
        "| Classic EWS baseline | 0.67 |\n",
        encoding="utf-8",
    )
    manifest = {
        "started_utc": "2026-10-31T10:00:00+00:00",
        "profile": "default",
        "git": {"commit": "abc1234def", "branch": "main", "uncommitted_files": 5},
        "python": "3.12.0",
        "platform": "Windows-11",
        "packages": {"tabpfn": "9.2.0"},
        "models": {"llm_agent": "openrouter:some/model:free"},
        "seeds": {
            "pfn_seed": 0,
            "llm_temperature": 0.0,
            "seeds_per_scenario": 3,
            "limit": 3,
            "scenario_seed_sets": {"A": {"start": 1000, "count": 20}},
        },
        "gate": {"sha256": "6e93455997a265", "status": "tuned"},
        "wall_clock_seconds_total": 12.5,
        "totals": {"tokens_in": 10, "tokens_out": 20, "cost_usd_approx": 0.0, "cost_note": "free"},
    }
    (results / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return results


@pytest.fixture
def sources(tmp_path: Path) -> Path:
    path = tmp_path / "SOURCES.md"
    path.write_text(
        "# Data sources\n\n## some_file.csv\n\n- **What:** a slice of 5 days.\n"
        "- **Licence:** CC BY 4.0 (see [the text](https://example.org/x)).\n"
        "- **SHA-256:** `473d8b29`\n",
        encoding="utf-8",
    )
    return path


# Release criteria


def test_the_criteria_file_is_complete_and_says_it_is_illustrative() -> None:
    variant, stand_in, criteria = load_criteria(DEFAULT_CRITERIA)
    assert variant == "agent" and stand_in
    required = {c.key for c in criteria if c.required}
    assert required == {
        "coverage_accepted_in_prior",
        "false_alarms_simulated_controls",
        "false_alarms_real_controls",
        "gate_recall_bad_in_prior",
    }
    # every criterion in the file can be computed by the code
    for c in criteria:
        measure(Path("no_such_dir"), variant, c.key)
    text = DEFAULT_CRITERIA.read_text(encoding="utf-8")
    assert "ILLUSTRATIVE" in text and "AFTER the experiments" in text
    # out of prior is reported but never a pass condition
    assert all(not c.required for c in criteria if c.key.endswith("out_of_prior"))


def test_go_when_every_required_criterion_passes(tmp_path: Path) -> None:
    verdict = evaluate(make_results(tmp_path, recall_blocked=2), DEFAULT_CRITERIA)
    assert verdict.verdict == "GO"
    by_key = {o.key: o for o in verdict.outcomes}
    assert by_key["coverage_accepted_in_prior"].value == 0.91
    assert by_key["false_alarms_simulated_controls"].denominator == 2  # two controls in test, ood
    assert by_key["false_alarms_simulated_controls"].numerator == 0
    # the out-of-prior criterion fails and still does not change the verdict
    assert by_key["coverage_accepted_out_of_prior"].passed is False
    assert not by_key["coverage_accepted_out_of_prior"].required


def test_no_go_when_one_required_criterion_fails(tmp_path: Path) -> None:
    verdict = evaluate(make_results(tmp_path, recall_blocked=0), DEFAULT_CRITERIA)
    assert verdict.verdict == "NO-GO"
    failed = [o.key for o in verdict.required if not o.passed]
    assert failed == ["gate_recall_bad_in_prior"]


def test_no_go_when_a_criterion_cannot_be_computed(tmp_path: Path) -> None:
    results = make_results(tmp_path)
    (results / "agent_real_control.json").unlink()
    (results / "agent_real.json").unlink()
    verdict = evaluate(results, DEFAULT_CRITERIA)
    real = next(o for o in verdict.outcomes if o.key == "false_alarms_real_controls")
    assert real.value is None and real.passed is False
    assert verdict.verdict == "NO-GO"


def test_the_criteria_are_written_for_the_number_check(tmp_path: Path) -> None:
    verdict = evaluate(make_results(tmp_path), DEFAULT_CRITERIA)
    path = write_criteria_json(verdict, tmp_path / "out")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["verdict"] == verdict.verdict
    assert doc["required_total"] == 4 and "required_met" in doc


# Text sources


def test_section_table_reads_the_table_and_the_bullets_under_it() -> None:
    text = "## A\n\n| x | y |\n|---|---|\n| 1 | 2 |\n\n- note one\n- note two\n\n## B\n"
    table = section_table(text, "A")
    assert table is not None
    assert table.headers == ["x", "y"] and table.rows == [["1", "2"]]
    assert table.notes == ["note one", "note two"]
    assert section_table(text, "Missing") is None


def test_sources_keep_labels_and_drop_links(sources: Path) -> None:
    entries = parse_sources(sources.read_text(encoding="utf-8"))
    assert [e.title for e in entries] == ["some_file.csv"]
    items = dict(entries[0].items)
    assert "https://" not in items["Licence"] and "CC BY 4.0" in items["Licence"]


# The number check


def test_the_check_catches_a_number_that_is_in_no_file() -> None:
    allowed = numbers_in_json({"rate": 0.9412, "n": 18, "inside": {"x": [0.5]}})
    assert unsourced_numbers("<p>The rate is 0.94 in 18 runs, half of 0.5.</p>", allowed) == []
    assert unsourced_numbers("<p>We reach 0.77 accuracy.</p>", allowed) == ["0.77"]
    assert unsourced_numbers("<p>That is 94 percent.</p>", allowed) == []  # a percentage of 0.94


def test_ids_dates_versions_section_numbers_and_years_are_allowed() -> None:
    page = (
        '<h2><span class="secno">3</span>Title</h2>'
        "<p>fold_L1_slow/3000 and v3.5 and abc123def at 2026-10-31T10:00:00+00:00, "
        "version 9.2.0, run 20261031T163524Z, doi 10.5281/zenodo.3744121, year 2019, "
        "n=9.</p><style>.x { margin: 77px }</style>"
    )
    assert numbers_on_page(page) == ["2019", "9"]  # the year is found and then allowed
    assert unsourced_numbers(page, {"9"}) == []
    assert unsourced_numbers(page, set()) == ["9"]


def test_a_typed_number_in_the_template_fails_the_build(
    tmp_path: Path, sources: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = make_results(tmp_path)
    monkeypatch.setattr(
        report_page, "METHOD_PARAGRAPHS", ("<p>Our model reaches 0.8123 accuracy.</p>",)
    )
    with pytest.raises(UnsourcedNumbersError, match=r"0\.8123"):
        build_report(results, criteria_path=DEFAULT_CRITERIA, sources_path=sources)


# The page


def test_the_page_has_the_six_sections_in_order_and_is_self_contained(
    tmp_path: Path, sources: Path
) -> None:
    results = make_results(tmp_path, recall_blocked=0)
    path = build_report(results, criteria_path=DEFAULT_CRITERIA, sources_path=sources)
    page = path.read_text(encoding="utf-8")
    assert re.findall(r'<section id="([^"]+)"', page) == SECTION_IDS
    assert re.findall(r'<span class="secno">([^<]+)</span>', page)[:6] == list("123456")
    assert not re.findall(r'(?:src|href)="(?:https?:)?//', page)  # no network assets
    assert "<link" not in page and "<script" not in page
    assert "NO-GO" in page  # the made-up gate recall is too low
    assert (path.parent / "criteria.json").exists()


def test_the_gallery_shows_two_examples_per_category_with_their_evidence(
    tmp_path: Path, sources: Path
) -> None:
    results = make_results(tmp_path)
    page = build_report(results, criteria_path=DEFAULT_CRITERIA, sources_path=sources).read_text(
        encoding="utf-8"
    )
    assert page.count('<div class="card">') == 8  # two in each of four categories
    assert page.count("Agent rationale") >= 8 and "Gate result" in page
    assert "Trace excerpt" in page and "results/traces/t.jsonl" in page
    assert "data:image/png;base64," in page  # the scenario plot is embedded


def test_the_method_text_holds_no_numbers() -> None:
    text = re.sub(r"<[^>]+>", " ", " ".join(report_page.METHOD_PARAGRAPHS))
    assert not re.search(r"\d", text)


# The real results


def test_the_real_report_shows_only_numbers_from_the_result_files(tmp_path: Path) -> None:
    needed = ("run_manifest.json", "failures.json", "summary.md", "agent_test.json")
    if not all((REAL_RESULTS / name).exists() for name in needed):
        pytest.skip("no real results on this machine: run make full first")
    out = tmp_path / "report"
    path = build_report(REAL_RESULTS, out_dir=out)  # the strict check raises on a stray number
    page = path.read_text(encoding="utf-8")
    assert "GO" in re.findall(r'<div class="word">([^<]+)</div>', page)[0]


def test_no_go_when_a_control_alerts_too_often(tmp_path: Path) -> None:
    results = make_results(tmp_path, recall_blocked=2)
    doc = json.loads((results / "agent_test.json").read_text(encoding="utf-8"))
    doc["runs"][0]["decision"] = "alert"  # the one simulated control in test now alerts
    (results / "agent_test.json").write_text(json.dumps(doc), encoding="utf-8")
    verdict = evaluate(results, DEFAULT_CRITERIA)
    failed = [o.key for o in verdict.required if not o.passed]
    assert verdict.verdict == "NO-GO" and failed == ["false_alarms_simulated_controls"]


# The fixes after the first review of the page


def test_the_result_split_names_map_to_the_real_loader_splits() -> None:
    from pfngate.data import REAL_CONTROL_SPLIT, REAL_SPLIT
    from pfngate.report.gallery import real_split_of

    # the result files say "real"; the loader says "real_tipping". The plot lookup used to miss it
    assert real_split_of("real") == REAL_SPLIT
    assert real_split_of("real_control") == REAL_CONTROL_SPLIT
    assert real_split_of(REAL_SPLIT) == REAL_SPLIT
    assert real_split_of("test") is None and real_split_of("ood") is None


def test_variants_are_grouped_by_what_they_have() -> None:
    group = report_page.variant_group
    assert group("Classic EWS baseline") == group("LLM only")
    assert group("Naive agent (random context, no gate)") == "An agent, but no gate"
    assert group("Full harness (agent + gate)").startswith("After")
    assert group("Family-matched oracle (ORACLE)").startswith("Upper bound")
    names = ["Classic EWS baseline", "Naive agent (x)", "Full harness (x)", "Family-matched oracle"]
    assert len({group(n) for n in names}) == 4
    assert group("Something else") == "Other"


def test_the_before_and_after_table_no_longer_files_the_naive_agent_under_after(
    tmp_path: Path, sources: Path
) -> None:
    results = make_results(tmp_path)
    summary = (results / "summary.md").read_text(encoding="utf-8")
    summary = summary.replace(
        "| Full harness (agent + gate) | 0.98 |",
        "| Naive agent (random context, no gate) | 0.99 |\n| Full harness (agent + gate) | 0.98 |",
    )
    (results / "summary.md").write_text(summary, encoding="utf-8")
    page = build_report(results, criteria_path=DEFAULT_CRITERIA, sources_path=sources).read_text(
        encoding="utf-8"
    )
    assert "After and bounds" not in page
    for title in (
        "Before: no agent and no gate",
        "An agent, but no gate",
        "After: the full harness",
    ):
        assert title in page
    assert page.index("An agent, but no gate") < page.index("After: the full harness")


def test_the_dagger_legends_come_from_summary_md_and_sit_by_the_table() -> None:
    summary = (
        "## AUROC, score = p_critical (x)\n\n| Method | AUROC |\n|---|---|\n| a | 0.5 † |\n\n"
        "Higher is better. † = the CI includes half.\n\n"
        "## Checks computed from the results\n\n- ‡ oracle on controls: degenerate score.\n"
    )
    from pfngate.report.text_sources import section_table as read_table

    table = read_table(summary, "AUROC, score = p_critical")
    assert table is not None and table.after == "Higher is better. † = the CI includes half."
    out = report_page.figures_section(Path("no_such_figs"), summary)
    assert "† = the CI includes half" in out
    assert "‡ oracle on controls: degenerate score" in out
    assert out.index("the CI includes half") > out.index("<table")  # the legend is under the table


def test_nested_source_bullets_keep_their_own_lines() -> None:
    text = (
        "## f.csv\n\n- **Files used:**\n  - `a.csv`:\n    `abc123`\n  - `b.csv`:\n    `def456`\n"
        "- **Licence:** CC0\n"
    )
    entry = parse_sources(text)[0]
    items = dict(entry.items)
    assert items["Files used"] == "- a.csv: abc123\n- b.csv: def456"
    assert items["Licence"] == "CC0"
    assert report_page.dl_rows({"a": "x\ny"}) == "<dt>a</dt><dd>x<br>y</dd>"


def test_gallery_values_read_in_plain_words() -> None:
    value, check = report_page.field_value, report_page.check_value
    assert value("gate_failed", []) == "none" and value("gate_failed", None) == "none"
    assert value("gate_passed", True) == "yes" and value("gate_passed", False) == "no"
    assert value("final_decision", "no_alert") == "no alert"
    assert value("decided_by", "gate_and_decision_mapping") == "gate and decision mapping"
    assert value("window_decisions", {"no_alert": 9, "alert": 2}) == "no alert 9, alert 2"
    assert check(200.0) == "200" and check(0.012) == "0.012" and check(1.0) == "1.000"
    assert check([0.7, 1.0]) == "0.700, 1.000"


def test_the_gallery_uses_labels_not_field_names_and_provenance_names_the_pfn_model(
    tmp_path: Path, sources: Path
) -> None:
    results = make_results(tmp_path)
    doc = json.loads((results / "agent_test.json").read_text(encoding="utf-8"))
    doc["settings"] = {
        "backend": "tabpfn_local",
        "model_version": "v3.5",
        "package_version": "9.2.0",
        "max_context": 200,
    }
    (results / "agent_test.json").write_text(json.dumps(doc), encoding="utf-8")
    page = build_report(results, criteria_path=DEFAULT_CRITERIA, sources_path=sources).read_text(
        encoding="utf-8"
    )
    assert "90% interval coverage of the window targets" in page
    assert "<dt>coverage_90</dt>" not in page and "<dt>window_decisions</dt>" not in page
    assert "PFN model version" in page and "v3.5" in page
    assert "see variants" not in page  # the old placeholder for the PFN backend is gone


# The strict number check, the intervals and the notice (audit M1, M5, H2)


def test_a_typed_number_that_some_result_file_holds_is_still_refused(
    tmp_path: Path, sources: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = make_results(tmp_path)
    (results / "other_result.json").write_text(json.dumps({"x": 0.618}), encoding="utf-8")
    monkeypatch.setattr(report_page, "METHOD_PARAGRAPHS", ("<p>We reach 0.618 here.</p>",))
    with pytest.raises(UnsourcedNumbersError, match=r"0\.618"):
        build_report(results, criteria_path=DEFAULT_CRITERIA, sources_path=sources)
    # the coarse check, which accepts any value found anywhere in results/, let it through
    page = (results / "report" / "index.html").read_text(encoding="utf-8")
    assert unsourced_numbers(page, allowed_numbers(results, sources)) == []


def test_numbers_quoted_from_an_agent_rationale_are_accepted(tmp_path: Path, sources: Path) -> None:
    results = make_results(tmp_path)
    doc = json.loads((results / "agent_test.json").read_text(encoding="utf-8"))
    doc["runs"][1]["record"]["verdict"]["rationale"] = "The last window is 0.4321 with p 0.0987."
    (results / "agent_test.json").write_text(json.dumps(doc), encoding="utf-8")
    failures = json.loads((results / "failures.json").read_text(encoding="utf-8"))
    for block in failures["categories"].values():
        for ex in block["top"]:
            ex["trajectory_id"], ex["scenario_id"] = "fold_L1_slow/3000", "fold_L1_slow"
    (results / "failures.json").write_text(json.dumps(failures), encoding="utf-8")
    page = build_report(results, criteria_path=DEFAULT_CRITERIA, sources_path=sources).read_text(
        encoding="utf-8"
    )
    assert "0.4321" in page  # shown, and the strict check let it pass as a quotation


def test_a_number_with_thousands_commas_is_one_number_and_a_list_is_not() -> None:
    from pfngate.report.check import numbers_on_page

    assert numbers_on_page("<p>432,000 rows</p>") == ["432000"]
    assert numbers_on_page("<p>[0.91, 1.00]</p>") == ["0.91", "1.00"]


def test_no_number_literal_is_passed_to_the_page_formatters() -> None:
    import ast

    tree = ast.parse(Path(report_page.__file__).read_text(encoding="utf-8"))
    formatters = {"num", "note_number", "check_value", "quoted"}
    typed = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") in formatters and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, int | float):
                typed.append(first.lineno)
    assert typed == []  # a literal there would be recorded as if it came from data


def test_shares_get_an_exact_interval_and_coverage_a_bootstrap_interval() -> None:
    from pfngate.report.criteria import bootstrap_mean_ci, clopper_pearson, is_decided

    low, high = clopper_pearson(0, 2)
    assert low == 0.0 and high == pytest.approx(0.8419, abs=1e-4)
    assert clopper_pearson(1, 18)[1] == pytest.approx(0.273, abs=1e-3)  # about 0.27, not 0.06
    assert clopper_pearson(5, 5)[1] == 1.0
    ci = bootstrap_mean_ci([0.9, 0.95, 1.0, 0.85])
    assert ci == bootstrap_mean_ci([0.9, 0.95, 1.0, 0.85])  # fixed seed: the same every time
    assert ci is not None and 0.85 <= ci[0] <= ci[1] <= 1.0
    assert bootstrap_mean_ci([]) is None
    # decided means the whole interval is on one side of the threshold
    assert is_decided(">=", 0.85, (0.90, 0.97)) is True
    assert is_decided(">=", 0.85, (0.10, 0.60)) is True  # clearly below
    assert is_decided(">=", 0.85, (0.50, 0.95)) is False
    assert is_decided("<=", 0.10, (0.00, 0.27)) is False
    assert is_decided("<=", 0.10, (0.00, 0.05)) is True
    assert is_decided("<=", 0.10, None) is None


def test_an_outcome_whose_interval_spans_the_threshold_is_marked_undecided(
    tmp_path: Path, sources: Path
) -> None:
    results = make_results(tmp_path, recall_blocked=2)
    verdict = evaluate(results, DEFAULT_CRITERIA)
    sim = next(o for o in verdict.outcomes if o.key == "false_alarms_simulated_controls")
    assert sim.value == 0.0 and sim.decided is False  # 0 of 2 controls: the interval is wide
    assert sim.ci_high == pytest.approx(0.8419, abs=1e-4)
    assert verdict.verdict == "GO" and sim in verdict.undecided  # the verdict rule is unchanged
    page = build_report(results, criteria_path=DEFAULT_CRITERIA, sources_path=sources).read_text(
        encoding="utf-8"
    )
    assert "undecided: the interval spans the threshold" in page
    assert "required results are undecided at 95% confidence" in page
    doc = json.loads((results / "report" / "criteria.json").read_text(encoding="utf-8"))
    assert doc["confidence"] == 0.95 and doc["required_undecided"] == len(verdict.undecided)


def test_the_page_warns_when_the_numbers_come_from_a_dirty_tree(
    tmp_path: Path, sources: Path
) -> None:
    results = make_results(tmp_path)  # the made-up manifest has uncommitted files
    page = build_report(results, criteria_path=DEFAULT_CRITERIA, sources_path=sources).read_text(
        encoding="utf-8"
    )
    assert "Not reproducible from git" in page and "abc1234def" in page
    manifest = json.loads((results / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["git"]["uncommitted_files"] = 0
    (results / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    clean = build_report(results, criteria_path=DEFAULT_CRITERIA, sources_path=sources).read_text(
        encoding="utf-8"
    )
    assert "Not reproducible from git" not in clean


def test_the_name_number_in_ninety_percent_interval_matches_the_gate_config() -> None:
    import yaml

    from pfngate.report.check import NAME_NUMBERS

    interval = yaml.safe_load(
        (REAL_RESULTS.parent / "configs" / "gate.yaml").read_text(encoding="utf-8")
    )["bad_run"]["interval"]
    assert round((interval[1] - interval[0]) * 100) == 90 and NAME_NUMBERS == {"90"}

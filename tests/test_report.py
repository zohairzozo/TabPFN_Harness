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
from pfngate.report.build import DEFAULT_SOURCES, UnsourcedNumbersError, build_report
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
    path = build_report(REAL_RESULTS, out_dir=out, check=False)
    page = path.read_text(encoding="utf-8")
    allowed = allowed_numbers(REAL_RESULTS, DEFAULT_SOURCES, extra=[out / "criteria.json"])
    assert unsourced_numbers(page, allowed) == []
    assert "GO" in re.findall(r'<div class="word">([^<]+)</div>', page)[0]


def test_no_go_when_a_control_alerts_too_often(tmp_path: Path) -> None:
    results = make_results(tmp_path, recall_blocked=2)
    doc = json.loads((results / "agent_test.json").read_text(encoding="utf-8"))
    doc["runs"][0]["decision"] = "alert"  # the one simulated control in test now alerts
    (results / "agent_test.json").write_text(json.dumps(doc), encoding="utf-8")
    verdict = evaluate(results, DEFAULT_CRITERIA)
    failed = [o.key for o in verdict.required if not o.passed]
    assert verdict.verdict == "NO-GO" and failed == ["false_alarms_simulated_controls"]

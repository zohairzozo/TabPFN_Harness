"""scripts/render_readme.py and README.md: the prose has no number that a result file does not hold.

The rule is enforced in two ways. The prose (everything outside the generated blocks) has no
digits at all, so any number on the page comes from a generated block. Every number in the
whole README, generated or not, must appear in a file of results/ (or data/SOURCES.md).
The renderer is also tested on made-up results, so the tests do not depend on a full run.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from pfngate.report.check import allowed_numbers, numbers_in_text, unsourced_numbers

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_readme.py"
README = ROOT / "README.md"
RESULTS = ROOT / "results"


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("render_readme", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["render_readme"] = mod
    spec.loader.exec_module(mod)
    return mod


RR = load_module()


def prose(text: str) -> str:
    """The README with each generated block emptied: what a reader reads as the author's words."""
    for start, end in RR.MARKERS.values():
        text = re.sub(re.escape(start) + r".*?" + re.escape(end), start + end, text, flags=re.S)
    return text


def digits_in_prose(text: str) -> list[str]:
    return re.findall(r"\d", prose(text))


def have_results() -> bool:
    needed = ("summary.md", "failures.json", "report/criteria.json")
    return README.exists() and all((RESULTS / name).exists() for name in needed)


# The prose rule


def test_the_prose_has_no_digits() -> None:
    assert digits_in_prose(README.read_text(encoding="utf-8")) == []


def test_the_prose_rule_catches_a_typed_number() -> None:
    typed = "The gate blocks 17 per cent.\n<!-- results:start -->\n0.5\n<!-- results:end -->\n"
    assert digits_in_prose(typed) == ["1", "7"]  # digits outside the block are caught
    inside_only = "<!-- results:start -->0.5<!-- results:end -->"
    assert digits_in_prose(inside_only) == []  # a number inside a generated block is allowed


def test_every_number_in_the_readme_is_quoted_from_summary_md_or_formatted_by_the_script() -> None:
    if not have_results():
        pytest.skip("no results on this machine: run make full and make report first")
    current = README.read_text(encoding="utf-8")
    RR.render(current, RESULTS)  # fills RR.LEDGER with the numbers the script formats
    allowed = numbers_in_text((RESULTS / "summary.md").read_text(encoding="utf-8")) | RR.LEDGER
    assert unsourced_numbers(current.replace("`", " "), allowed) == []  # identifiers keep digits


def test_the_readme_is_up_to_date_with_the_results() -> None:
    if not have_results():
        pytest.skip("no results on this machine: run make full and make report first")
    current = README.read_text(encoding="utf-8")
    assert RR.render(current, RESULTS) == current, "run python scripts/render_readme.py"


def test_rendering_twice_changes_nothing() -> None:
    if not have_results():
        pytest.skip("no results on this machine: run make full and make report first")
    once = RR.render(README.read_text(encoding="utf-8"), RESULTS)
    assert RR.render(once, RESULTS) == once


# The renderer, on made-up results


SUMMARY = """# Results summary

## Variants, run level

| Variant | In-prior AUROC | Out-of-prior AUROC |
|---|---|---|
| Classic EWS baseline | 0.58 | 0.52 |
| Full harness (agent + gate) | 0.98 [0.91, 1.00] | 0.91 |

- Sample: a made-up sample.

## Real data (test only)

| Variant | Rate |
|---|---|
| Classic EWS baseline | 0.67 |

- Decisions per variant on the grid split: Classic EWS baseline: alert 2, no alert 1.

## Early warning: detection and lead time (window level)

| Split | Strategy | Tipping runs |
|---|---|---|
| In-prior test | knn | 54 |

- The first alarm can be a false one: 28 of 54 control trajectories.

## Sharpness: is the PFN better than a prior that ignores the query?

| Split | Strategy | Median width |
|---|---|---|
| Out-of-prior | knn | 0.840 |
"""


def example(
    variant: str, tid: str, split: str, severity: float, basis: str, numbers: dict[str, Any]
) -> dict[str, Any]:
    return {
        "variant": variant,
        "trajectory_id": tid,
        "split": split,
        "severity": severity,
        "severity_basis": basis,
        "numbers": numbers,
    }


def make_results(tmp_path: Path, floor: float | None = 0.8) -> Path:
    results = tmp_path / "results"
    (results / "report").mkdir(parents=True)
    (results / "summary.md").write_text(SUMMARY, encoding="utf-8")
    criteria = {
        "verdict": "NO-GO",
        "required_met": 1,
        "required_total": 2,
        "required_undecided": 2,
        "confidence": 0.95,
    }
    (results / "report" / "criteria.json").write_text(json.dumps(criteria), encoding="utf-8")
    manifest = {"git": {"commit": "abc1234def5678", "uncommitted_files": 5}}
    (results / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    settings = {
        "backend": "tabpfn_local",
        "model_version": "v3.5",
        "package_version": "9.2.0",
        "max_context": 200,
    }
    (results / "agent_test.json").write_text(json.dumps({"settings": settings}), encoding="utf-8")
    scope: dict[str, Any] = {}
    if floor is not None:
        scope["min_coverage_of_a_bad_run"] = floor
    categories = {
        "a_false_alarms_on_controls": {
            "top": [
                example(
                    "classic",
                    "grid_fingrid/0",
                    "real_control",
                    0.925,
                    "classic baseline score of the last window",
                    {},
                )
            ]
        },
        "b_bad_runs_the_gate_let_through": {
            "top": [
                example(
                    "agent",
                    "rate_L2_slow/4001",
                    "ood",
                    0.3,
                    "minimum coverage minus the run's coverage",
                    {"coverage_90": 0.5},
                )
            ]
        },
        "c_good_runs_the_gate_blocked": {"top": []},
        "d_agent_disagreed_with_gated_decision": {
            "top": [
                example(
                    "agent",
                    "regime_L3/4000",
                    "ood",
                    0.85,
                    "the confidence the agent gave its recommendation",
                    {
                        "verdict_decision": "no_alert",
                        "verdict_confidence": 0.85,
                        "final_decision": "abstain",
                    },
                )
            ]
        },
    }
    (results / "failures.json").write_text(
        json.dumps({"scope": scope, "categories": categories}), encoding="utf-8"
    )
    return results


def template() -> str:
    blocks = ("\n\n").join(f"<!-- {n}:start -->\n<!-- {n}:end -->" for n in RR.MARKERS)
    return f"Intro with no numbers.\n\n{blocks}\n"


def test_the_blocks_come_from_the_files_and_the_prose_is_kept(tmp_path: Path) -> None:
    out = RR.render(template(), make_results(tmp_path))
    assert out.startswith("Intro with no numbers.")
    assert "| Classic EWS baseline | 0.58 | 0.52 |" in out
    assert "| Full harness (agent + gate) | 0.98 [0.91, 1.00] | 0.91 |" in out
    assert "**Release verdict for the stand-in model: NO-GO**, with 1 of 2" in out
    assert "a made-up sample" not in out  # only the table is copied, not the notes


def test_the_failure_sentences_use_the_numbers_in_the_files(tmp_path: Path) -> None:
    out = RR.render(template(), make_results(tmp_path))
    assert "with a score of 0.925 (classic baseline score of the last window)" in out
    assert "covered 0.500 of the window targets, below the line of 0.800" in out
    assert "recommended no alert with confidence 0.850" in out  # words, not no_alert
    assert "the gate's decision was abstain" in out
    assert "no_alert" not in out
    assert out.count("[failure gallery](results/report/index.html#gallery)") == 3


def test_the_rendered_numbers_are_all_in_the_made_up_files(tmp_path: Path) -> None:
    results = make_results(tmp_path)
    out = RR.render(template(), results)
    assert unsourced_numbers(out.replace("`", " "), allowed_numbers(results, None)) == []


def test_a_missing_result_file_is_a_clear_error(tmp_path: Path) -> None:
    results = make_results(tmp_path)
    (results / "failures.json").unlink()
    with pytest.raises(RR.RenderError, match="failures.json is missing"):
        RR.render(template(), results)


def test_a_missing_marker_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(RR.RenderError, match="results:start"):
        RR.render("no markers here", make_results(tmp_path))


def test_the_gate_floor_is_required_for_the_missed_block(tmp_path: Path) -> None:
    with pytest.raises(RR.RenderError, match="minimum coverage"):
        RR.render(template(), make_results(tmp_path, floor=None))


def test_an_empty_category_says_so(tmp_path: Path) -> None:
    results = make_results(tmp_path)
    doc = json.loads((results / "failures.json").read_text(encoding="utf-8"))
    doc["categories"]["d_agent_disagreed_with_gated_decision"]["top"] = []
    (results / "failures.json").write_text(json.dumps(doc), encoding="utf-8")
    out = RR.render(template(), results)
    assert "- No example of this kind is in the results." in out


# The blocks added after the first review: early warning, sharpness, real data, and the notices


def test_the_early_warning_sharpness_and_real_data_blocks_are_copied_from_summary_md(
    tmp_path: Path,
) -> None:
    out = RR.render(template(), make_results(tmp_path))
    assert "| In-prior test | knn | 54 |" in out
    assert "- The first alarm can be a false one: 28 of 54 control trajectories." in out
    assert "| Out-of-prior | knn | 0.840 |" in out
    assert (
        "- Decisions per variant on the grid split: Classic EWS baseline: alert 2, no alert 1."
        in out
    )


def test_the_verdict_line_says_how_undecided_it_is_and_when_the_tree_was_dirty(
    tmp_path: Path,
) -> None:
    results = make_results(tmp_path)
    line = RR.verdict_line(results)
    assert "2 of 2 required results are undecided at 95% confidence" in line
    assert "commit abc1234def56 with 5 uncommitted files, so no commit reproduces them" in line
    manifest = json.loads((results / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["git"]["uncommitted_files"] = 0
    (results / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert "reproduces" not in RR.verdict_line(results)
    doc = json.loads((results / "report" / "criteria.json").read_text(encoding="utf-8"))
    doc["required_undecided"] = 0
    (results / "report" / "criteria.json").write_text(json.dumps(doc), encoding="utf-8")
    assert "undecided" not in RR.verdict_line(results)


def test_the_model_behind_the_numbers_is_named_from_the_result_file(tmp_path: Path) -> None:
    results = make_results(tmp_path)
    assert RR.model_line(results) == (
        "Model behind every number: tabpfn_local, model version v3.5, "
        "package version 9.2.0, context capped at 200 rows."
    )
    (results / "agent_test.json").unlink()
    assert RR.model_line(results) == ""


def test_the_strict_check_refuses_a_typed_number_that_some_result_file_holds(
    tmp_path: Path,
) -> None:
    results = make_results(tmp_path)
    (results / "other_result.json").write_text(json.dumps({"x": 0.618}), encoding="utf-8")
    out = RR.render(template(), results)
    allowed = numbers_in_text(SUMMARY) | RR.LEDGER
    assert unsourced_numbers(out.replace("`", " "), allowed) == []
    typed = out.replace("`", " ") + " We reach 0.618 here."
    assert unsourced_numbers(typed, allowed) == ["0.618"]  # not in the summary, not formatted
    assert unsourced_numbers(typed, allowed_numbers(results, None)) == []  # the coarse check agreed


def test_a_summary_without_one_of_the_new_sections_is_a_clear_error(tmp_path: Path) -> None:
    results = make_results(tmp_path)
    text = (results / "summary.md").read_text(encoding="utf-8")
    (results / "summary.md").write_text(text.replace("## Sharpness", "## Something else"))
    with pytest.raises(RR.RenderError, match="Sharpness"):
        RR.render(template(), results)


def test_the_licence_wording_and_the_tippfn_citation_stay_in_place() -> None:
    readme = README.read_text(encoding="utf-8")
    assert "gated weights" in readme and "their own terms" in readme and "Apache licence" in readme
    stub = (ROOT / "src" / "pfngate" / "pfn" / "tippfn.py").read_text(encoding="utf-8")
    assert "In-context learning to predict critical" in stub and "Only the abstract" in stub
    labels = (ROOT / "docs" / "LABELS.md").read_text(encoding="utf-8")
    assert "only its abstract was checked" in labels


def test_the_gate_freeze_claim_says_what_happened_and_the_old_claim_is_gone() -> None:
    readme = README.read_text(encoding="utf-8")
    assert "not blind" in readme and "docs/AUDIT.md" in readme
    assert "frozen before the held-out runs" not in readme
    page = (ROOT / "src" / "pfngate" / "report" / "page.py").read_text(encoding="utf-8")
    assert "not blind" in page and "frozen before any held-out run" not in page
    summary_script = (ROOT / "scripts" / "make_summary.py").read_text(encoding="utf-8")
    assert "frozen before any held-out run" not in summary_script
    assert "not blind" in summary_script


def test_the_agent_seeing_the_gate_inputs_is_disclosed_in_the_readme_and_the_report() -> None:
    readme = README.read_text(encoding="utf-8")
    assert "gate's own inputs" in readme and "still cannot change a gate result" in readme
    page = (ROOT / "src" / "pfngate" / "report" / "page.py").read_text(encoding="utf-8")
    assert "show it some of the gate" in page


def test_the_repository_has_an_mit_licence_that_leaves_the_weights_alone() -> None:
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("MIT License") and "Zohair Ahmed Shehzad" in text
    readme = README.read_text(encoding="utf-8")
    assert "MIT licence" in readme and "does not cover the weights" in readme

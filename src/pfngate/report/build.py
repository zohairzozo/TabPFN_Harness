"""Build results/report/index.html and check that every number on it comes from a file.

The check is strict: a number must be text quoted from summary.md, SOURCES.md or failures.json,
or a number the page formatted from data (see page.num). A number typed into the template is
refused even if some result file holds the same value.
"""

from __future__ import annotations

from pathlib import Path

from pfngate.report.check import exact_allowed, unsourced_numbers
from pfngate.report.criteria import DEFAULT_CRITERIA, REPO_ROOT, evaluate, write_criteria_json
from pfngate.report.page import build_page, rendered_numbers

DEFAULT_SOURCES = REPO_ROOT / "data" / "SOURCES.md"


class UnsourcedNumbersError(RuntimeError):
    """The page shows a number that no source file holds."""


def build_report(
    results_dir: Path,
    out_dir: Path | None = None,
    criteria_path: Path = DEFAULT_CRITERIA,
    sources_path: Path = DEFAULT_SOURCES,
    check: bool = True,
) -> Path:
    """Build the report page, its criteria file and its number check."""
    out = out_dir or results_dir / "report"
    out.mkdir(parents=True, exist_ok=True)
    verdict = evaluate(results_dir, criteria_path)
    write_criteria_json(verdict, out)
    try:
        criteria_file = criteria_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        criteria_file = criteria_path.name
    page = build_page(results_dir, verdict, sources_path, criteria_file)
    path = out / "index.html"
    path.write_text(page, encoding="utf-8")
    if check:
        bad = unsourced_numbers(page, exact_allowed(results_dir, sources_path, rendered_numbers()))
        if bad:
            raise UnsourcedNumbersError(
                f"{len(bad)} numbers on the page were neither quoted from a source nor "
                f"formatted from data: {', '.join(bad)}"
            )
    return path

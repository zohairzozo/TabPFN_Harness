"""Build results/report/index.html and check that every number on it comes from a file."""

from __future__ import annotations

from pathlib import Path

from pfngate.report.check import allowed_numbers, unsourced_numbers
from pfngate.report.criteria import DEFAULT_CRITERIA, REPO_ROOT, evaluate, write_criteria_json
from pfngate.report.page import build_page

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
    out = out_dir or results_dir / "report"
    out.mkdir(parents=True, exist_ok=True)
    verdict = evaluate(results_dir, criteria_path)
    criteria_json = write_criteria_json(verdict, out)
    try:
        criteria_file = criteria_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        criteria_file = criteria_path.name
    page = build_page(results_dir, verdict, sources_path, criteria_file)
    path = out / "index.html"
    path.write_text(page, encoding="utf-8")
    if check:
        bad = unsourced_numbers(
            page, allowed_numbers(results_dir, sources_path, extra=[criteria_json])
        )
        if bad:
            raise UnsourcedNumbersError(
                f"{len(bad)} numbers on the page are in no source file: {', '.join(bad)}"
            )
    return path

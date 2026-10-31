"""python -m pfngate.report: write results/report/index.html (make report)."""

from __future__ import annotations

import argparse
from pathlib import Path

from pfngate.report.build import DEFAULT_SOURCES, UnsourcedNumbersError, build_report
from pfngate.report.criteria import DEFAULT_CRITERIA, REPO_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=REPO_ROOT / "results", type=Path)
    parser.add_argument("--criteria", default=DEFAULT_CRITERIA, type=Path)
    parser.add_argument("--sources", default=DEFAULT_SOURCES, type=Path)
    parser.add_argument("--no-check", action="store_true", help="skip the number check")
    args = parser.parse_args()
    try:
        path = build_report(
            args.results_dir,
            criteria_path=args.criteria,
            sources_path=args.sources,
            check=not args.no_check,
        )
    except UnsourcedNumbersError as err:
        print(f"report written, but the number check failed: {err}")
        return 1
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

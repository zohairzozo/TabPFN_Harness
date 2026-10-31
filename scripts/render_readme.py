"""Fill the generated blocks of README.md from the result files in results/.

The README has two generated blocks. Each sits between a pair of HTML comment markers:

  results   the variants table (results/summary.md, "Variants, run level") and the release
            verdict line (results/report/criteria.json)
  failures  one sentence for each of three kinds of failure, from results/failures.json

The rest of the README is prose with no digits, so every number on the page comes from a file.
The prose is written by hand. Nothing in the generated blocks is typed.

The failures are picked by rule: for each of three kinds, the top-ranked example in
results/failures.json. The kinds are the false alarm on a control (category a), the missed block
by the gate (category b) and the disagreement between the agent and the gated decision
(category d). Over-blocking (category c) is in the file and in the report gallery, not here.

Usage:
  python scripts/render_readme.py            write README.md
  python scripts/render_readme.py --check    exit 1 if README.md is out of date
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))  # run from a checkout without installing

from pfngate.report.text_sources import plain, read_text, section_table  # noqa: E402

DEFAULT_README = REPO_ROOT / "README.md"
DEFAULT_RESULTS = REPO_ROOT / "results"
REPORT_LINK = "results/report/index.html"
MARKERS = {
    "results": ("<!-- results:start -->", "<!-- results:end -->"),
    "failures": ("<!-- failures:start -->", "<!-- failures:end -->"),
}
FAILURE_KINDS: tuple[tuple[str, str], ...] = (
    ("a_false_alarms_on_controls", "false_alarm"),
    ("b_bad_runs_the_gate_let_through", "missed_block"),
    ("d_agent_disagreed_with_gated_decision", "disagreement"),
)
VARIANT_NAMES = {
    "classic": "the classic baseline",
    "llm_only": "the LLM-only model",
    "naive_agent": "the naive agent",
    "agent": "the full harness",
    "oracle": "the oracle",
    "pfn_knn+gate": "TabPFN with knn context and the gate",
    "pfn_random+gate": "TabPFN with random context and the gate",
}


class RenderError(RuntimeError):
    """A result file or a marker is missing, so the README cannot be filled."""


def fmt(value: Any) -> str:
    """A number as the README shows it: three decimals, or the text unchanged."""
    if value is None or isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def load_json(path: Path, hint: str) -> dict[str, Any]:
    if not path.exists():
        raise RenderError(f"{path.name} is missing in {path.parent}: {hint}")
    return json.loads(path.read_text(encoding="utf-8"))


def variants_table(results_dir: Path) -> str:
    table = section_table(read_text(results_dir / "summary.md"), "Variants, run level")
    if table is None:
        raise RenderError("results/summary.md has no 'Variants, run level' table: run make full")
    head = "| " + " | ".join(plain(h) for h in table.headers) + " |"
    sep = "|" + "---|" * len(table.headers)
    rows = ["| " + " | ".join(plain(c) for c in row) + " |" for row in table.rows]
    return "\n".join([head, sep, *rows])


def verdict_line(results_dir: Path) -> str:
    doc = load_json(results_dir / "report" / "criteria.json", "run make report first")
    return (
        f"**Release verdict for the stand-in model: {doc['verdict']}**, with "
        f"{doc['required_met']} of {doc['required_total']} required criteria met. "
        "The criteria are illustrative (configs/release_criteria.yaml)."
    )


def results_block(results_dir: Path) -> str:
    return f"{verdict_line(results_dir)}\n\n{variants_table(results_dir)}"


def failure_sentence(kind: str, example: dict[str, Any], floor: float) -> str:
    name = VARIANT_NAMES.get(example["variant"], example["variant"])
    numbers = example.get("numbers", {})
    where = f"`{example['trajectory_id']}` ({example['split']} split)"
    gallery = f"see the [failure gallery]({REPORT_LINK}#gallery)"
    if kind == "false_alarm":
        return (
            f"**False alarm:** {name} alerted on the negative control {where}, where its "
            f"{example['severity_basis']} was {fmt(example['severity'])}; {gallery}."
        )
    if kind == "missed_block":
        covered = numbers.get("coverage_90")
        if covered is None:  # the severity is the shortfall below the line
            covered = floor - example["severity"]
        return (
            f"**Missed block:** the gate accepted {name} on {where}, where its prediction "
            f"interval covered {fmt(covered)} of the window targets, below the line of "
            f"{fmt(floor)}; {gallery}."
        )
    if kind == "disagreement":
        confidence = numbers.get("verdict_confidence", example["severity"])
        return (
            f"**Disagreement:** {name} recommended {numbers.get('verdict_decision')} with "
            f"confidence {fmt(confidence)} on {where}, but the gate's decision was "
            f"{numbers.get('final_decision')}; {gallery}."
        )
    raise RenderError(f"unknown failure kind {kind}")


def failures_block(results_dir: Path) -> str:
    doc = load_json(results_dir / "failures.json", "run make full, which writes it")
    floor = doc["scope"].get("min_coverage_of_a_bad_run")
    if floor is None:
        raise RenderError("results/failures.json has no minimum coverage of a bad run")
    lines = []
    for category, kind in FAILURE_KINDS:
        top = doc["categories"][category]["top"]
        if not top:
            lines.append("- No example of this kind is in the results.")
        else:
            lines.append("- " + failure_sentence(kind, top[0], floor))
    return "\n".join(lines)


def render(text: str, results_dir: Path) -> str:
    """The README with both generated blocks replaced. Text outside the markers is kept."""
    bodies = {"results": results_block(results_dir), "failures": failures_block(results_dir)}
    for name, body in bodies.items():
        start, end = MARKERS[name]
        if text.count(start) != 1 or text.count(end) != 1:
            raise RenderError(f"README.md needs exactly one {start} and one {end}")
        head, rest = text.split(start, 1)
        _, tail = rest.split(end, 1)
        text = f"{head}{start}\n{body}\n{end}{tail}"
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--check", action="store_true", help="do not write; fail if out of date")
    args = parser.parse_args(argv)
    try:
        current = args.readme.read_text(encoding="utf-8")
        rendered = render(current, args.results_dir)
    except RenderError as err:
        print(f"render_readme: {err}")
        return 1
    if args.check:
        if rendered != current:
            print("README.md is out of date: run python scripts/render_readme.py")
            return 1
        print("README.md is up to date")
        return 0
    if rendered != current:
        args.readme.write_text(rendered, encoding="utf-8")
        print(f"wrote {args.readme}")
    else:
        print("README.md already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

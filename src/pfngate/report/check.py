"""The number check: every number on the report page must come from a file.

The page is parsed back to text. Identifiers (scenario ids, hashes, versions, dates), section
numbers and years are set aside. Every other number must appear in the files the report reads:
the JSON and JSONL files in results/, results/summary.md and data/SOURCES.md. A number is
matched at any rounding from zero to six decimals, and as a percentage.
"""

from __future__ import annotations

import html as html_lib
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
TOKEN_SPLIT = re.compile(r"[\s(),;:=\[\]<>|–—]+")
ID_CHARS = re.compile(r"[A-Za-z0-9_./+\-^#@~%]+")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ][\d:.+Z-]*)?")
COMPACT_TIME = re.compile(r"\b\d{8}T\d{6}Z\b")
DOI = re.compile(r"\b10\.\d{4,}/\S+")
VERSION = re.compile(r"\b\d+(?:\.\d+){2,}\b")
YEAR = re.compile(r"(?:19|20)\d{2}")
SECTION_NO = re.compile(r'<span class="secno">.*?</span>', re.S)
SKIP_BLOCKS = re.compile(r"<(script|style)\b.*?</\1>", re.S | re.I)
TAG = re.compile(r"<[^>]+>")


def normalise(token: str) -> str:
    """A number as a plain string: no thousands commas, no trailing zeros after the point."""
    token = token.replace(",", "")
    if "." in token:
        token = token.rstrip("0").rstrip(".")
    return token or "0"


def variants(value: float) -> Iterator[str]:
    """The strings a value can be shown as: rounded to 0..6 decimals, or as a percentage."""
    value = abs(float(value))
    for scale in (1.0, 100.0):
        for digits in range(0, 7):
            yield normalise(f"{value * scale:.{digits}f}")


def numbers_in_text(text: str) -> set[str]:
    return {normalise(m) for m in NUMBER.findall(text)}


def numbers_in_json(node: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(node, bool) or node is None:
        return out
    if isinstance(node, int | float):
        if node == node and abs(node) != float("inf"):
            out.update(variants(node))
    elif isinstance(node, str):
        out |= numbers_in_text(node)
    elif isinstance(node, list):
        for item in node:
            out |= numbers_in_json(item)
    elif isinstance(node, dict):
        for key, item in node.items():
            out |= numbers_in_text(str(key))
            out |= numbers_in_json(item)
    return out


def source_files(
    results_dir: Path, data_sources: Path | None, extra: list[Path] | None = None
) -> list[Path]:
    files = sorted(results_dir.glob("*.json")) + sorted((results_dir / "datasets").glob("*.json"))
    files += sorted((results_dir / "report").glob("*.json"))
    files += sorted((results_dir / "traces").glob("*.jsonl"))
    files += [results_dir / "summary.md"]
    if data_sources is not None:
        files.append(data_sources)
    files += extra or []
    return [f for f in dict.fromkeys(files) if f.exists()]


def allowed_numbers(
    results_dir: Path, data_sources: Path | None, extra: list[Path] | None = None
) -> set[str]:
    allowed: set[str] = set()
    for path in source_files(results_dir, data_sources, extra):
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            allowed |= numbers_in_json(json.loads(text))
        elif path.suffix == ".jsonl":
            for line in text.splitlines():
                if line.strip():
                    allowed |= numbers_in_json(json.loads(line))
        else:
            allowed |= numbers_in_text(text)
    return allowed


def visible_text(page: str) -> str:
    """The text a reader sees: no scripts, styles, tags or section numbers."""
    page = SKIP_BLOCKS.sub(" ", page)
    page = SECTION_NO.sub(" ", page)
    return html_lib.unescape(TAG.sub(" ", page))


def is_identifier(token: str) -> bool:
    """A name with letters and digits in it, such as fold_L1_slow/3000 or v3.5."""
    return (
        bool(ID_CHARS.fullmatch(token))
        and any(c.isalpha() for c in token)
        and any(c.isdigit() for c in token)
    )


def numbers_on_page(page: str) -> list[str]:
    text = visible_text(page)
    for pattern in (DATE, COMPACT_TIME, DOI, VERSION):
        text = pattern.sub(" ", text)
    found: list[str] = []
    for token in TOKEN_SPLIT.split(text):
        token = token.strip(".,'\"’“”*")
        if not token or is_identifier(token):
            continue
        found.extend(NUMBER.findall(token))
    return found


def unsourced_numbers(page: str, allowed: set[str]) -> list[str]:
    """The numbers on the page that no source file holds (years are always allowed)."""
    bad: list[str] = []
    for token in numbers_on_page(page):
        if YEAR.fullmatch(token) or normalise(token) in allowed:
            continue
        bad.append(token)
    return sorted(set(bad))

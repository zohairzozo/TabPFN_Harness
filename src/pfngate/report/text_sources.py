"""Read the two text files the report quotes: results/summary.md and data/SOURCES.md.

summary.md is written by scripts/make_summary.py, so its tables are results, not prose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")


@dataclass
class Table:
    headers: list[str]
    rows: list[list[str]]
    notes: list[str] = field(default_factory=list)  # the bullet block right after the table


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def is_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?", line.strip()))


def section_table(text: str, heading_start: str, index: int = 0) -> Table | None:
    """The index-th pipe table under the first heading that starts with heading_start."""
    lines = text.splitlines()
    start = next(
        (i for i, ln in enumerate(lines) if ln.startswith("## " + heading_start)),
        None,
    )
    if start is None:
        return None
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    block = lines[start + 1 : end]
    seen = -1
    i = 0
    while i < len(block):
        if block[i].lstrip().startswith("|"):
            j = i
            while j < len(block) and block[j].lstrip().startswith("|"):
                j += 1
            seen += 1
            if seen == index:
                rows = [split_row(ln) for ln in block[i:j] if not is_separator(ln)]
                return Table(rows[0], rows[1:], bullet_block(block, j))
            i = j
        else:
            i += 1
    return None


def bullet_block(block: list[str], start: int) -> list[str]:
    """The bullets after a table (skipping one blank line), up to the next blank line."""
    i = start
    while i < len(block) and not block[i].strip():
        i += 1
    out: list[str] = []
    while i < len(block) and block[i].startswith("- "):
        out.append(block[i][2:].strip())
        i += 1
    return out


def section_bullets(text: str, heading_start: str) -> list[str]:
    """The top-level bullets directly under a heading."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("## " + heading_start)), None)
    if start is None:
        return []
    out: list[str] = []
    for ln in lines[start + 1 :]:
        if ln.startswith("## "):
            break
        if ln.startswith("- "):
            out.append(ln[2:].strip())
    return out


def plain(text: str) -> str:
    """Markdown inline marks removed: links keep their text, bold and code lose their marks."""
    text = LINK.sub(r"\1", text)
    return text.replace("**", "").replace("`", "")


@dataclass
class SourceEntry:
    title: str
    items: list[tuple[str, str]]  # (label, text) for the bullets that start with a bold label


def parse_sources(text: str) -> list[SourceEntry]:
    """data/SOURCES.md: one entry per '## ' heading, with its '- **Label:** text' bullets."""
    entries: list[SourceEntry] = []
    for line in text.splitlines():
        if line.startswith("## "):
            entries.append(SourceEntry(plain(line[3:].strip()), []))
        elif entries and line.startswith("- "):
            m = re.match(r"- \*\*([^*]+?):\*\*\s*(.*)", line)
            if m:
                entries[-1].items.append((m.group(1), plain(m.group(2))))
        elif entries and line.startswith("  ") and entries[-1].items and line.strip():
            label, body = entries[-1].items[-1]
            entries[-1].items[-1] = (label, f"{body} {plain(line.strip())}")
    return [e for e in entries if e.items]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""

"""List and fetch a bounded slice of the Zenodo power-grid frequency record 3744121.

Kruse, Schafer and Witthaut, "Pre-Processed Power Grid Frequency Time Series", Zenodo record
3744121 (https://doi.org/10.5281/zenodo.3744121).

The record has two big zip files (Data_converted.zip, Data_cleansed.zip). Each holds yearly
files for three areas: Continental Europe (TransnetBW, NOT under an open licence), Great
Britain (National Grid) and Nordic (Fingrid). Per the record text, the National Grid and
Fingrid files are released under CC BY 4.0; the TransnetBW files are not.

This script never downloads a TransnetBW file. It reads the zip directory with HTTP range
requests (a few kilobytes), keeps only members of the allowed areas, and then streams just the
first part of ONE yearly file of an allowed area, until it has found a stretch of days that
looks stationary. Only that slice is written, to data/raw/grid/.

    python scripts/fetch_grid.py list                   # files, sizes and licences
    python scripts/fetch_grid.py members                # members of the two big zips
    python scripts/fetch_grid.py licences               # the LICENSE.txt of each area
    python scripts/fetch_grid.py fetch                  # download the slice (Fingrid, 5 days)

Every fetch is recorded in data/SOURCES.md: source, access date, licence and SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import sys
import urllib.request
import zipfile
import zlib
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORD = "3744121"
API = f"https://zenodo.org/api/records/{RECORD}"
DOI = "10.5281/zenodo.3744121"
DATA_ZIPS = ("Data_cleansed.zip", "Data_converted.zip")
RAW_DIR = REPO_ROOT / "data" / "raw" / "grid"
SOURCES = REPO_ROOT / "data" / "SOURCES.md"

# Licence per area, as stated in the record text (DESCRIPTION.md, section "License").
# Only areas with an open licence are allowed. Anything else is refused.
AREA_LICENCE: dict[str, str] = {
    "fingrid": "CC BY 4.0 (Fingrid, Nordic; pre-processed version released by the record authors)",
    "nationalgrid": "CC BY 4.0 (National Grid ESO, GB; pre-processed version, same authors)",
}
BLOCKED_WORDS = ("transnet", "continental")
NOMINAL_HZ = 50.0
SECONDS_PER_DAY = 86_400
# Selection rule for a stationary stretch, fixed before any window or result is looked at:
# consecutive days that are complete (one row per second, at most MAX_NAN_PER_DAY NaN) with no
# large event, that is, every value within MAX_DEVIATION_HZ of the nominal 50 Hz.
MAX_DEVIATION_HZ = 0.25
MAX_NAN_PER_DAY = 100  # the cleansed file marks a few seconds as NaN; the loader interpolates them
MAX_BYTES = 400 * 1024 * 1024  # stop reading the remote file after this many bytes
USER_AGENT = "pfngate-fetch-grid/1.0 (research demo)"


class RemoteFile(io.RawIOBase):
    """A read-only seekable file over HTTP range requests, so a zip directory can be read."""

    def __init__(self, url: str, size: int) -> None:
        self.url, self.size, self.pos = url, size, 0
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n < 0 or self.pos + n > self.size:
            n = self.size - self.pos
        if n <= 0:
            return b""
        req = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}", "User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        self.pos += len(data)
        self.bytes_read += len(data)
        return data

    def readinto(self, b: bytearray) -> int:  # type: ignore[override]
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def get_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def record_files() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rec = get_json(API)
    return rec, rec["files"]


def allowed_area(member: str) -> str | None:
    """The allowed area name that a member path belongs to, or None. TransnetBW is refused."""
    low = member.lower().replace("\\", "/")
    if any(w in low for w in BLOCKED_WORDS):
        return None
    for key in AREA_LICENCE:
        if f"/{key}/" in low:
            return key
    return None


def cmd_list(_: argparse.Namespace) -> None:
    rec, files = record_files()
    meta = rec["metadata"]
    print(
        f"{meta['title']}  (doi:{DOI}, version {meta.get('version')}, {meta['publication_date']})"
    )
    print(f"record licence field: {meta.get('license')}  (a mixed record: see the text below)")
    print("\nlicence statement in the record text:")
    print("  scripts (Scripts.zip): MIT")
    print(f"  Fingrid (Nordic), pre-processed: {AREA_LICENCE['fingrid']}")
    print(f"  Nationalgrid (GB), pre-processed: {AREA_LICENCE['nationalgrid']}")
    print("  Continental Europe (TransnetBW): NOT open, no licence for the pre-processed files")
    print("\nfiles published in the record:")
    for f in files:
        print(f"  {f['key']:<22} {f['size']:>14,} bytes  {f['checksum']}")


def open_remote_zip(name: str) -> tuple[zipfile.ZipFile, RemoteFile]:
    _, files = record_files()
    info = next(f for f in files if f["key"] == name)
    remote = RemoteFile(info["links"]["self"], int(info["size"]))
    return zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 20)), remote


def cmd_members(_: argparse.Namespace) -> None:
    for name in DATA_ZIPS:
        zf, remote = open_remote_zip(name)
        print(f"\n{name}: {len(zf.infolist())} members (read {remote.bytes_read:,} bytes to list)")
        for m in zf.infolist():
            if m.is_dir():
                continue
            area = allowed_area(m.filename)
            use = f"allowed: {area}" if area else "NOT USED (licence not open or unknown)"
            print(f"  {m.filename:<60} {m.file_size:>13,} bytes  {use}")


def cmd_licences(_: argparse.Namespace) -> None:
    """Print the LICENSE.txt that sits next to the yearly files of each area."""
    zf, _ = open_remote_zip("Data_cleansed.zip")
    for m in zf.infolist():
        if m.filename.endswith("LICENSE.txt"):
            print(f"{m.filename}: {zf.read(m).decode('utf-8', 'replace').strip()}")


def stream_rows(stream: Any) -> Iterator[tuple[str, float]]:
    """(timestamp, frequency) rows of the first file in an inner zip, read as a stream."""
    header = stream.read(30)
    sig, _v, _flags, method, _t, _d, _crc, _cs, _us, name_len, extra_len = struct.unpack(
        "<4sHHHHHLLLHH", header
    )
    if sig != b"PK\x03\x04" or method not in (0, 8):
        raise ValueError("unexpected inner zip header")
    stream.read(name_len + extra_len)
    inflater = zlib.decompressobj(-15) if method == 8 else None
    tail = ""
    while chunk := stream.read(1 << 20):
        text = tail + (inflater.decompress(chunk) if inflater else chunk).decode("ascii")
        *lines, tail = text.split("\n")
        for line in lines:
            ts, _, value = line.strip().partition(",")
            if ts and ts[0].isdigit():
                yield ts, float(value) if value not in ("", "NaN", "nan") else float("nan")


def find_stationary_days(
    rows: Iterator[tuple[str, float]], n_days: int, read_bytes: Callable[[], int]
) -> tuple[list[str], list[float], list[str]]:
    """The first n_days consecutive complete days with no large event, and their rows."""
    times: list[str] = []
    values: list[float] = []
    run_days: list[str] = []
    day = ""
    day_t: list[str] = []
    day_v: list[float] = []

    def day_is_ok() -> bool:
        if len(day_v) != SECONDS_PER_DAY:
            return False
        valid = [v for v in day_v if v == v]  # drops NaN
        if len(day_v) - len(valid) > MAX_NAN_PER_DAY:
            return False
        return all(abs(v - NOMINAL_HZ) <= MAX_DEVIATION_HZ for v in valid)

    def close_day() -> bool:
        if not day_is_ok():
            times.clear()
            values.clear()
            run_days.clear()
            return False
        times.extend(day_t)
        values.extend(day_v)
        run_days.append(day)
        return len(run_days) == n_days

    for ts, v in rows:
        if ts[:10] != day:
            if day and close_day():
                return times, values, run_days
            day, day_t, day_v = ts[:10], [], []
        day_t.append(ts)
        day_v.append(v)
        if read_bytes() > MAX_BYTES:
            raise RuntimeError(f"no stationary stretch of {n_days} days within the byte budget")
    if day and close_day():
        return times, values, run_days
    raise RuntimeError(f"no stationary stretch of {n_days} days in this file")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cmd_fetch(args: argparse.Namespace) -> None:
    member = f"Data_cleansed/{args.area}/{args.year}.zip"
    area = allowed_area(member)
    if area is None:
        raise SystemExit(f"'{args.area}' is not an allowed area. Allowed: {list(AREA_LICENCE)}")
    zf, remote = open_remote_zip("Data_cleansed.zip")
    info = zf.getinfo(member)  # KeyError if the year does not exist
    print(f"reading the start of {member} ({info.file_size:,} bytes) with range requests ...")
    with zf.open(info) as inner:
        times, values, days = find_stationary_days(
            stream_rows(inner), args.days, lambda: remote.bytes_read
        )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"{area}_{days[0]}_{len(days)}d.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        f.write("time,frequency_hz\n")
        f.writelines(f"{t},{v}\n" for t, v in zip(times, values, strict=True))
    digest = sha256_file(out)
    _, files = record_files()
    outer = next(x for x in files if x["key"] == "Data_cleansed.zip")
    record_source(
        out, digest, member, outer["checksum"], len(values), days, remote.bytes_read, area
    )
    print(f"wrote {out} ({len(values):,} rows, days {days[0]}..{days[-1]}), sha256 {digest}")
    print(f"downloaded {remote.bytes_read:,} bytes from Zenodo. Recorded in {SOURCES}.")


def record_source(
    path: Path,
    digest: str,
    member: str,
    outer_checksum: str,
    n_rows: int,
    days: list[str],
    bytes_read: int,
    area: str,
) -> None:
    """Append one entry to data/SOURCES.md."""
    SOURCES.parent.mkdir(parents=True, exist_ok=True)
    if not SOURCES.exists():
        SOURCES.write_text(
            "# Data sources\n\nEvery external dataset: where it came from, when, under which "
            "licence, and the SHA-256 of the file as used. The files themselves are not in git "
            "(`data/raw/` is ignored).\n",
            encoding="utf-8",
        )
    entry = f"""
## {path.name}

- **What:** a slice of {len(days)} consecutive days of 1-second grid frequency, {n_rows:,} rows,
  days {days[0]} to {days[-1]} (naive local time). Used as the split `real_control`.
- **Source:** Kruse, Schafer and Witthaut, "Pre-Processed Power Grid Frequency Time Series",
  Zenodo record {RECORD}, doi:{DOI}. File `Data_cleansed.zip`, member `{member}`
  (Zenodo checksum of the whole zip: {outer_checksum}).
- **Access date (UTC):** {datetime.now(UTC).date().isoformat()}
- **Licence:** {AREA_LICENCE[area]}. The LICENSE.txt in the same folder of the zip says CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/). Attribution is required.
- **Not used:** every TransnetBW file (no open licence). None was downloaded.
- **How fetched:** `scripts/fetch_grid.py fetch`, with HTTP range requests, {bytes_read:,} bytes
  read from Zenodo. The slice is the first stretch that is complete and has no large event
  (every value within {MAX_DEVIATION_HZ} Hz of {NOMINAL_HZ:.0f} Hz, at most {MAX_NAN_PER_DAY} NaN
  seconds a day). The rule is in the script.
- **File:** `{path.relative_to(REPO_ROOT).as_posix()}`
- **SHA-256:** `{digest}`
"""
    with SOURCES.open("a", encoding="utf-8") as f:
        f.write(entry)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(func=cmd_list)
    sub.add_parser("members").set_defaults(func=cmd_members)
    sub.add_parser("licences").set_defaults(func=cmd_licences)
    fetch = sub.add_parser("fetch", help="download the stationary slice of one allowed area")
    fetch.add_argument(
        "--area", default="Fingrid", help="Fingrid or Nationalgrid (default Fingrid)"
    )
    fetch.add_argument("--year", default="2019")
    fetch.add_argument("--days", type=int, default=5, help="consecutive days (default 5)")
    fetch.set_defaults(func=cmd_fetch)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())

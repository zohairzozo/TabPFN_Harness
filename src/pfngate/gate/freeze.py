"""Freeze the gate thresholds, and refuse held-out evaluation unless they are frozen.

After tuning on validation runs, write_frozen_hash() writes the SHA-256 of
configs/gate.yaml to configs/gate.frozen.sha256. Every evaluation of test, out-of-prior or
real data calls assert_frozen() first. The hash guards against accidental changes, not
deliberate ones: the git history is the audit trail. See docs/GATE.md.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pfngate.gate.config import (
    DEFAULT_GATE_CONFIG,
    DEFAULT_GATE_HASH,
    GateConfig,
    load_gate_config,
    sha256_file,
)


class GateNotFrozenError(RuntimeError):
    """The gate thresholds are missing, untuned, changed, or uncommitted."""


class GateNotApplicableError(RuntimeError):
    """The run settings differ from the settings the gate was tuned for."""


def write_frozen_hash(
    config_path: Path = DEFAULT_GATE_CONFIG, hash_path: Path = DEFAULT_GATE_HASH
) -> str:
    """Write the hash of a tuned gate config. Refuses to overwrite an existing hash."""
    if hash_path.exists():
        raise FileExistsError(
            f"{hash_path} exists. Re-tuning voids every held-out result made under the old "
            "hash: delete the file by hand and write a PROGRESS.md entry saying why."
        )
    cfg = load_gate_config(config_path)
    if cfg.status != "tuned":
        raise GateNotFrozenError(
            f"{config_path} has status '{cfg.status}'. Only a tuned gate may be frozen."
        )
    digest = sha256_file(config_path)
    hash_path.write_bytes((digest + "\n").encode("ascii"))
    return digest


def assert_frozen(
    config_path: Path = DEFAULT_GATE_CONFIG, hash_path: Path = DEFAULT_GATE_HASH
) -> GateConfig:
    """Return the gate config if it is tuned and matches the frozen hash, else raise."""
    if not config_path.exists() or not hash_path.exists():
        missing = [str(p) for p in (config_path, hash_path) if not p.exists()]
        raise GateNotFrozenError(
            f"gate is not frozen: missing {', '.join(missing)}. Tune the gate on validation "
            "runs and freeze it before evaluating test, out-of-prior or real data."
        )
    expected = hash_path.read_text(encoding="ascii").strip()
    actual = sha256_file(config_path)
    if actual != expected:
        raise GateNotFrozenError(
            f"{config_path} changed after it was frozen: its SHA-256 is {actual}, "
            f"{hash_path} says {expected}. Held-out evaluation refused."
        )
    cfg = load_gate_config(config_path)
    if cfg.status != "tuned":
        raise GateNotFrozenError(f"{config_path} is frozen but has status '{cfg.status}'")
    return cfg


def assert_committed(path: Path = DEFAULT_GATE_CONFIG) -> None:
    """Raise unless the file is tracked by git and has no uncommitted changes."""
    path = path.resolve()
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", path.name],
            cwd=path.parent, capture_output=True, text=True, check=False,
        )  # fmt: skip
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", path.name],
            cwd=path.parent, capture_output=True, text=True, check=True,
        )  # fmt: skip
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GateNotFrozenError(f"cannot ask git about {path}: {exc}") from exc
    if tracked.returncode != 0:
        raise GateNotFrozenError(f"{path} is not committed to git")
    if status.stdout.strip():
        raise GateNotFrozenError(f"{path} has uncommitted changes")


def assert_applicable(cfg: GateConfig, settings: Mapping[str, Any], *, oracle: bool) -> None:
    """Raise if a held-out run uses settings the gate was not tuned for.

    Oracle strategies are gated and reported with the label ORACLE, so they need not be
    in the tuned strategy list. Everything else must match exactly.
    """
    a = cfg.applicability
    problems = [
        f"{name}: run has {settings[name]!r}, gate was tuned for {getattr(a, name)!r}"
        for name in ("backend", "model_version", "package_version", "max_context", "data_version")
        if settings[name] != getattr(a, name)
    ]
    if not oracle and settings["strategy"] not in a.strategies:
        problems.append(f"strategy {settings['strategy']!r} is not in {list(a.strategies)}")
    if problems:
        raise GateNotApplicableError("gate not applicable to this run: " + "; ".join(problems))

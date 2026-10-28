"""An append-only JSONL trace: one line per step of a run, written and flushed at once.

A run has one file, results/traces/<run_id>.jsonl. Every line has the same fields, in the
same order. Fields that do not apply to a step (a PFN call has no tokens) are null.
gate_result is null except on the "gate" step that run_pfn --gate writes for every run.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self

from pfngate.features.config import REPO_ROOT

if TYPE_CHECKING:
    from pfngate.pfn import PFNOutput
    from pfngate.pfn.context import ContextSelection

DEFAULT_TRACE_DIR = REPO_ROOT / "results" / "traces"

TRACE_FIELDS: tuple[str, ...] = (
    "run_id",
    "variant",
    "scenario_id",
    "step",
    "tool",
    "context_hash",
    "context_size",
    "strategy",
    "latency_s",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "backend",
    "model_version",
    "gate_result",
)

_RUN_ID_OK = re.compile(r"^[A-Za-z0-9._-]+$")


def new_run_id(variant: str) -> str:
    """A readable and unique id: variant, UTC time, and 8 random hex characters."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{variant}-{stamp}-{uuid.uuid4().hex[:8]}"


class Tracer:
    """Writes one JSON object per line. The file is created new: a run id is never reused."""

    def __init__(
        self, variant: str, run_id: str | None = None, trace_dir: Path = DEFAULT_TRACE_DIR
    ) -> None:
        self.variant = variant
        self.run_id = run_id or new_run_id(re.sub(r"[^A-Za-z0-9._-]", "_", variant))
        if not _RUN_ID_OK.match(self.run_id):
            raise ValueError(f"run_id may only contain letters, digits, '.', '_' and '-': {run_id}")
        trace_dir.mkdir(parents=True, exist_ok=True)
        self.path = trace_dir / f"{self.run_id}.jsonl"
        self._file = self.path.open("x", encoding="utf-8")  # FileExistsError if it exists
        self._step = 0

    def log_step(
        self,
        *,
        scenario_id: str,
        tool: str,
        latency_s: float,
        context_hash: str | None = None,
        context_size: int | None = None,
        strategy: str | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        cost_usd: float | None = None,
        backend: str | None = None,
        model_version: str | None = None,
        gate_result: Any = None,
    ) -> dict[str, Any]:
        """Append one line and flush it. Returns the record that was written."""
        if self._file.closed:
            raise ValueError("tracer is closed")
        self._step += 1
        record = {
            "run_id": self.run_id,
            "variant": self.variant,
            "scenario_id": scenario_id,
            "step": self._step,
            "tool": tool,
            "context_hash": context_hash,
            "context_size": context_size,
            "strategy": strategy,
            "latency_s": latency_s,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": cost_usd,
            "backend": backend,
            "model_version": model_version,
            "gate_result": gate_result,
        }
        assert tuple(record) == TRACE_FIELDS
        line = json.dumps(record, allow_nan=False)  # NaN would be invalid JSON: fail loudly
        self._file.write(line + "\n")
        self._file.flush()
        return record

    def log_pfn_step(
        self,
        scenario_id: str,
        selection: ContextSelection,
        output: PFNOutput,
        tool: str = "pfn_predict",
    ) -> dict[str, Any]:
        """Log one PFN call: the context that was used and what the backend reported."""
        md = output.metadata
        return self.log_step(
            scenario_id=scenario_id,
            tool=tool,
            latency_s=md.fit_seconds + md.predict_seconds,
            context_hash=selection.context_hash,
            context_size=selection.size,
            strategy=selection.label,
            backend=md.backend,
            model_version=md.model_version,
        )

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def read_trace(path: Path) -> list[dict[str, Any]]:
    """Read a trace file back, one dict per line."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

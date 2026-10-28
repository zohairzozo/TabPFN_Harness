"""Deterministic release gate written in plain Python.

The gate reads frozen thresholds from configs/gate.yaml and never sees the ground truth of
the query trajectory. Nothing here lets an agent change a GateResult or a threshold.
"""

from pfngate.gate.checks import CheckResult
from pfngate.gate.config import GateConfig, load_gate_config
from pfngate.gate.freeze import (
    GateNotApplicableError,
    GateNotFrozenError,
    assert_applicable,
    assert_frozen,
)
from pfngate.gate.gate import Decision, GateInput, GateResult, decide, evaluate_gate

__all__ = [
    "CheckResult",
    "Decision",
    "GateConfig",
    "GateInput",
    "GateNotApplicableError",
    "GateNotFrozenError",
    "GateResult",
    "assert_applicable",
    "assert_frozen",
    "decide",
    "evaluate_gate",
    "load_gate_config",
]

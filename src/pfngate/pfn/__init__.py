"""Swappable PFN backends (TabPFN now, TipPFN stub).

Importing this package does not import tabpfn or tabpfn-client. Each backend imports
its own package when it is fitted.
"""

from pfngate.pfn.adapter import PFNBackend, PFNMetadata, PFNOutput, build_output
from pfngate.pfn.config import PFNConfig, load_pfn_config
from pfngate.pfn.context import ContextSelection, select_context
from pfngate.pfn.fake import FakeBackend
from pfngate.pfn.tabpfn_local import TabPFNLocal
from pfngate.pfn.tabpfn_remote import ClientDisabledError, TabPFNClient
from pfngate.pfn.tippfn import TipPFNBackend


def make_backend(config: PFNConfig | None = None) -> PFNBackend:
    """The backend named by `backend` in configs/pfn.yaml."""
    cfg = config or load_pfn_config()
    backends: dict[str, type] = {
        "tabpfn_local": TabPFNLocal,
        "tabpfn_client": TabPFNClient,
        "tippfn": TipPFNBackend,
        "fake": FakeBackend,
    }
    if cfg.backend not in backends:
        raise ValueError(f"unknown backend '{cfg.backend}', expected one of {sorted(backends)}")
    cls = backends[cfg.backend]
    return cls() if cls is TipPFNBackend else cls(cfg)


__all__ = [
    "ClientDisabledError",
    "ContextSelection",
    "FakeBackend",
    "PFNBackend",
    "PFNConfig",
    "PFNMetadata",
    "PFNOutput",
    "TabPFNClient",
    "TabPFNLocal",
    "TipPFNBackend",
    "build_output",
    "load_pfn_config",
    "make_backend",
    "select_context",
]

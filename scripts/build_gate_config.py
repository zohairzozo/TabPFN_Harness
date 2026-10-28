"""Write a provisional configs/gate.yaml from docs/GATE.md and the context pool (set A).

    python scripts/build_gate_config.py --max-context 200

Reads only set A features. Refuses to run if the gate is already frozen. The thresholds
that are tuned later start at the loosest value of their grid, and the file is marked
provisional, so it cannot be frozen and cannot gate held-out evaluation.
"""

from __future__ import annotations

import argparse

from pfngate.evals.pfn_cache import data_version, package_version
from pfngate.features import build_dataset, load_feature_config
from pfngate.gate.build import build_gate_config, write_gate_config
from pfngate.gate.config import DEFAULT_GATE_CONFIG, DEFAULT_GATE_HASH, load_gate_config
from pfngate.pfn import load_pfn_config


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-context", type=int, required=True,
                   help="the max_context the validation runs will use")  # fmt: skip
    p.add_argument("--strategies", nargs="+", default=["random", "knn"])
    args = p.parse_args()
    if DEFAULT_GATE_HASH.exists():
        raise SystemExit(f"{DEFAULT_GATE_HASH} exists: the gate is frozen. Not overwriting.")

    cfg, fc = load_pfn_config(), load_feature_config()
    pool = build_dataset("context", feature_config=fc)
    applicability = {
        "backend": cfg.backend,
        "model_version": cfg.model_version,
        "package_version": package_version(cfg.backend),
        "max_context": args.max_context,
        "data_version": data_version(fc),
    }
    data = build_gate_config(
        pool, applicability=applicability, strategies=tuple(args.strategies), seed=cfg.seed
    )
    path = write_gate_config(data, DEFAULT_GATE_CONFIG)
    gate = load_gate_config(path)  # validates floors and shapes
    print(f"wrote {path} (status {gate.status}, sha256 {gate.sha256[:16]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

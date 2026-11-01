"""Write results/sharpness.json: is the PFN sharper than a prior that knows nothing about the query?

Coverage near the nominal level is cheap if the intervals are wide. This puts the PFN next to a
climatology baseline: constant quantiles of the context pool's own target values (the pool is
data a context may use). It reports, for each held-out tipping split and strategy:

- the median 90% interval width of the PFN runs, and the width of the climatology interval,
- the pinball skill: 1 minus the PFN mean pinball loss over the climatology mean pinball loss,
  on the same windows. Zero means no better than the climatology, one means perfect.

The PFN numbers are read from results/pfn_<strategy>_<split>.json (the gate advisory
interval_width_cap, and ground_truth.mean_pinball_loss). The gate must be frozen first.

Usage: python scripts/sharpness.py [--results-dir results]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))  # run from a checkout without installing

from pfngate.evals.metrics import write_json  # noqa: E402
from pfngate.evals.run_pfn import load_queries  # noqa: E402
from pfngate.features import build_dataset, load_feature_config  # noqa: E402
from pfngate.gate import assert_frozen  # noqa: E402

STRATEGIES = ("random", "knn")
FLAGS = ("test", "ood")
WIDTH_CHECK = "interval_width_cap"


def pinball(y: np.ndarray, pred: float, tau: float) -> float:
    """Mean pinball loss of one constant quantile prediction at level tau."""
    diff = np.asarray(y, dtype=float) - pred
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))


def climatology(targets: np.ndarray, levels: list[float]) -> dict[float, float]:
    """The quantiles of the target values at each level."""
    return {tau: float(np.quantile(targets, tau)) for tau in levels}


def climatology_loss(y: np.ndarray, clim: dict[float, float]) -> float:
    """Mean over levels of the pinball loss of the constant climatology quantiles."""
    return float(np.mean([pinball(y, q, tau) for tau, q in clim.items()]))


def skill(pfn_losses: list[float], clim_losses: list[float]) -> float | None:
    """1 minus the ratio of the mean PFN loss to the mean climatology loss, or None."""
    if not pfn_losses or not clim_losses or float(np.mean(clim_losses)) == 0.0:
        return None
    return 1.0 - float(np.mean(pfn_losses)) / float(np.mean(clim_losses))


def advisory_width(trajectory: dict[str, Any]) -> float | None:
    """The median 90% interval width the gate recorded for a run."""
    for adv in trajectory["gate"].get("advisories", []):
        if adv["name"] == WIDTH_CHECK:
            return float(adv["value"])
    return None


def block(
    trajectories: list[dict[str, Any]], clim_by_tid: dict[str, float], clim_width: float
) -> dict[str, Any]:
    """The sharpness numbers for one strategy and split."""

    def part(sub: list[dict[str, Any]]) -> dict[str, Any]:
        widths = [w for t in sub if (w := advisory_width(t)) is not None]
        median = float(np.median(widths)) if widths else None
        return {
            "n_runs": len(sub),
            "median_width": median,
            "median_width_over_climatology": median / clim_width if median is not None else None,
            "pinball_skill": skill(
                [t["ground_truth"]["mean_pinball_loss"] for t in sub],
                [clim_by_tid[t["trajectory_id"]] for t in sub],
            ),
        }

    accepted = [t for t in trajectories if not t["gate"]["failed"]]
    return {"all_runs": part(trajectories), "gate_accepted_runs": part(accepted)}


def build(results_dir: Path) -> dict[str, Any]:
    """Compute the sharpness results for every strategy and held-out tipping split."""
    assert_frozen()  # held-out targets are read: the gate must be frozen first
    fc = load_feature_config()
    ref = json.loads((results_dir / "pfn_random_test.json").read_text(encoding="utf-8"))
    levels, seeds = [float(v) for v in ref["quantile_levels"]], ref["seeds_per_scenario"]
    pool = build_dataset("context", feature_config=fc)
    clim = climatology(pool["target"].to_numpy(float), levels)
    clim_width = clim[max(levels)] - clim[min(levels)]
    out: dict[str, Any] = {
        "reference": "constant quantiles of the context pool target values",
        "levels": levels,
        "climatology_quantiles": {str(k): v for k, v in clim.items()},
        "climatology_width": clim_width,
        "splits": {},
    }
    for flag in FLAGS:
        clim_by_tid = {
            str(q["trajectory_id"].iloc[0]): climatology_loss(q["target"].to_numpy(float), clim)
            for q in load_queries(flag, fc, seeds)
        }
        for strategy in STRATEGIES:
            path = results_dir / f"pfn_{strategy}_{flag}.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            out["splits"].setdefault(flag, {})[strategy] = block(
                doc["trajectories"], clim_by_tid, clim_width
            )
    return out


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: write results/sharpness.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args(argv)
    path = args.results_dir / "sharpness.json"
    write_json(path, build(args.results_dir))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

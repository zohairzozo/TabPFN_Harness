"""Reliability diagram of the predicted quantiles: results/figs/reliability.png.

    python scripts/plot_reliability.py

For each of the five quantile levels, it plots the share of true values at or below the
predicted quantile (observed) against the level itself (nominal). It does this for all runs
and for the runs the gate accepted, with in-prior test and out-of-prior side by side, one
row per gated strategy.

It reads results/pfn_<strategy>_{test,ood}.json only. Each trajectory there carries its
observed frequency per level (ground_truth.observed_frequency) and its gate result. Windows
are pooled over runs: a run with more windows counts more.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ("random", "knn")  # the strategies the gate applies to
SPLITS = (("test", "In-prior test"), ("ood", "Out-of-prior test"))
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ALL_RUNS, ACCEPTED = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 (validated pair)


def load_runs(path: Path) -> list[dict[str, Any]]:
    """The trajectories of one gated result file. Refuses a file without the data."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run run_pfn with --gate first")
    run = json.loads(path.read_text(encoding="utf-8"))
    trajectories = run["trajectories"]
    if run.get("gate") is None or any(
        "observed_frequency" not in t["ground_truth"] for t in trajectories
    ):
        raise ValueError(f"{path.name} has no gate result or no observed frequencies")
    return trajectories


def accepted(t: dict[str, Any]) -> bool:
    return not t["gate"]["failed"]


def pooled_frequencies(trajectories: list[dict[str, Any]]) -> dict[float, float] | None:
    """Observed frequency per level, pooled over windows. None if there is no run."""
    if not trajectories:
        return None
    n = sum(t["n_windows"] for t in trajectories)
    levels = trajectories[0]["ground_truth"]["observed_frequency"]
    return {
        float(lv): sum(t["ground_truth"]["observed_frequency"][lv] * t["n_windows"]
                       for t in trajectories) / n
        for lv in levels
    }  # fmt: skip


def panel_data(results_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for strategy in STRATEGIES:
        for flag, _ in SPLITS:
            ts = load_runs(results_dir / f"pfn_{strategy}_{flag}.json")
            acc = [t for t in ts if accepted(t)]
            out[(strategy, flag)] = {
                "all": pooled_frequencies(ts),
                "accepted": pooled_frequencies(acc),
                "n_all": len(ts),
                "n_accepted": len(acc),
            }
    return out


def draw(data: dict[tuple[str, str], dict[str, Any]], out_path: Path) -> Path:
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK})
    fig, axes = plt.subplots(len(STRATEGIES), len(SPLITS), figsize=(9.5, 4.6 * len(STRATEGIES)),
                             facecolor=SURFACE, squeeze=False)  # fmt: skip
    for i, strategy in enumerate(STRATEGIES):
        for j, (flag, title) in enumerate(SPLITS):
            ax, d = axes[i][j], data[(strategy, flag)]
            ax.set_facecolor(SURFACE)
            ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)),
                    label="perfectly calibrated")  # fmt: skip
            levels = list(d["all"])
            ax.plot(levels, list(d["all"].values()), "-o", color=ALL_RUNS, linewidth=2,
                    markersize=8, markeredgecolor=SURFACE, mew=2,
                    label=f"all runs ({d['n_all']})")  # fmt: skip
            if d["accepted"] is not None:
                ax.plot(list(d["accepted"]), list(d["accepted"].values()), "--s",
                        color=ACCEPTED, linewidth=2, markersize=8, markeredgecolor=SURFACE,
                        mew=2, label=f"runs the gate accepted ({d['n_accepted']})")  # fmt: skip
            else:
                ax.text(0.5, 0.08, "the gate accepted no run", ha="center", color=INK_MUTED)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks(levels)
            ax.set_aspect("equal")
            ax.tick_params(colors=INK_MUTED, length=0)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.grid(color=GRID, linewidth=0.8)
            ax.set_title(f"TabPFN, {strategy} context: {title}", loc="left", color=INK)
            ax.set_xlabel("Quantile level the model predicted")
            ax.set_ylabel("Share of true values at or below\nthe predicted quantile")
            ax.legend(loc="upper left", frameon=False, labelcolor=INK)
    fig.text(0.01, 0.005, "Above the dashed line: the predicted quantile is too high. Below "
             "it: too low. Runs pooled by window. Data: results/pfn_*_{test,ood}.json.",
             color=INK_MUTED, fontsize=8)  # fmt: skip
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "figs" / "reliability.png")
    args = p.parse_args(argv)
    path = draw(panel_data(args.results_dir), args.out)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

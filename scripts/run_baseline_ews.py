"""Run the classic early-warning baseline on the evaluation splits.

Writes results/baseline_ews.json and results/figs/baseline_auroc.png (paths from
configs/baseline_ews.yaml). The figure is drawn from the JSON, not from memory.

Usage: python scripts/run_baseline_ews.py
"""

from __future__ import annotations

import json
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from pfngate.evals.baseline_ews import (  # noqa: E402
    BaselineConfig,
    load_baseline_config,
    score_split,
)
from pfngate.evals.metrics import bootstrap_auroc, false_alarm_rate, write_json  # noqa: E402
from pfngate.features import load_feature_config  # noqa: E402
from pfngate.scenarios import load_scenarios  # noqa: E402

# Figure colours: reference palette from the dataviz skill (light surface).
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ROLE_COLOURS = {"in_prior": "#2a78d6", "out_of_prior": "#eb6834"}


def auroc_block(labelled: pd.DataFrame, cfg: BaselineConfig) -> dict[str, Any]:
    """AUROC on all labelled windows, on tipping trajectories only, and per tipping family."""
    tipping = labelled[labelled["role"] != "control"]
    subsets = {"all_windows": labelled, "tipping_only": tipping}
    for fam in sorted(tipping["family"].unique()):
        subsets[f"family:{fam}"] = tipping[tipping["family"] == fam]
    out = {}
    for name, s in subsets.items():
        out[name] = bootstrap_auroc(
            s["label"].to_numpy(),
            s["score"].to_numpy(),
            s["trajectory_id"].to_numpy(),
            n_boot=cfg.bootstrap_n,
            seed=cfg.bootstrap_seed,
            ci=cfg.ci,
        )
        out[name]["role"] = str(s["role"].iloc[0]) if name.startswith("family:") else None
    return out


def false_alarm_block(controls: pd.DataFrame, cfg: BaselineConfig) -> dict[str, Any]:
    out = {"all_controls": controls}
    out.update({fam: g for fam, g in controls.groupby("family")})
    return {
        name: false_alarm_rate(
            g["score"].to_numpy(), g["trajectory_id"].to_numpy(), cfg.alarm_threshold
        )
        for name, g in out.items()
    }


def run(cfg: BaselineConfig) -> dict[str, Any]:
    fc = load_feature_config()
    scenarios = load_scenarios()
    result: dict[str, Any] = {
        "method": "classic_ews",
        "description": (
            "score = mean of Kendall tau (against time) of detrended window variance and of "
            "window lag-1 autocorrelation, over the current window and the k-1 before it"
        ),
        "config": {
            "k_windows": cfg.k_windows,
            "combine": cfg.combine,
            "alarm_threshold": cfg.alarm_threshold,
            "bootstrap": {"n": cfg.bootstrap_n, "seed": cfg.bootstrap_seed, "ci": cfg.ci},
            "window_length": fc.window_length,
            "stride": fc.stride,
        },
        "versions": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy")},
        "auroc_note": (
            "balanced sample weights are constant within a class, so the balanced AUROC "
            "equals the plain AUROC; the CI resamples trajectories, not windows"
        ),
        "splits": {},
    }
    all_controls = []
    for split in cfg.splits:
        t0 = time.perf_counter()
        df, dropped = score_split(split, scenarios, fc, cfg)
        scored = df[df["score"].notna()]
        controls = scored[scored["role"] == "control"]
        all_controls.append(controls)
        result["splits"][split] = {
            "n_trajectories": int(df["trajectory_id"].nunique()),
            "n_windows": len(df),
            "n_windows_unscored_first_k_minus_1": int(df["score"].isna().sum()),
            "dropped_trajectories_no_tip": dropped,
            "auroc": auroc_block(scored[scored["label"] >= 0], cfg),
            "false_alarms": false_alarm_block(controls, cfg),
        }
        print(f"{split}: {len(df)} windows in {time.perf_counter() - t0:.1f} s")
    result["false_alarms_pooled"] = false_alarm_block(pd.concat(all_controls), cfg)
    return result


def plot(json_path: Path, fig_path: Path) -> Path:
    """AUROC with CI per split and subset, and false-alarm rates on controls."""
    res = json.loads(json_path.read_text(encoding="utf-8"))
    rows = []
    for split, block in res["splits"].items():
        for name, a in block["auroc"].items():
            role = a["role"] or ("out_of_prior" if split == "test_out_of_prior" else "in_prior")
            rows.append((f"{split} | {name.replace('family:', '')}", a, role))

    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK})
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 5.6), gridspec_kw={"width_ratios": [3, 2]}, facecolor=SURFACE
    )
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=INK_MUTED, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    y = list(range(len(rows)))[::-1]
    for yi, (_, a, role) in zip(y, rows, strict=True):
        c = ROLE_COLOURS[role]
        ax1.plot(
            [a["ci_low"], a["ci_high"]], [yi, yi], color=c, linewidth=2, solid_capstyle="round"
        )
        ax1.plot(a["auroc"], yi, "o", color=c, markersize=8, markeredgecolor=SURFACE, mew=2)
        ax1.annotate(
            f"{a['auroc']:.2f}",
            (a["ci_high"], yi),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=INK_MUTED,
        )
    ax1.axvline(0.5, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax1.set_yticks(y, [r[0] for r in rows])
    ax1.set_xlim(0, 1.08)
    ax1.set_xlabel(
        f"AUROC with {int(100 * res['config']['bootstrap']['ci'])}% CI (trajectory bootstrap)"
    )
    ax1.set_title("Classic EWS baseline: window AUROC", loc="left", color=INK)
    for role, c in ROLE_COLOURS.items():
        ax1.plot([], [], "o", color=c, label=role.replace("_", "-") + " tipping")
    ax1.legend(loc="lower left", frameon=False, labelcolor=INK)

    fa = res["false_alarms_pooled"]
    names = [n for n in fa if n != "all_controls"] + ["all_controls"]
    yy = list(range(len(names)))[::-1]
    for yi, n in zip(yy, names, strict=True):
        ax2.plot(fa[n]["window_rate"], yi, "o", color=INK, markersize=8)
        ax2.plot(
            fa[n]["trajectory_rate"],
            yi,
            "o",
            color=INK,
            markersize=8,
            markerfacecolor=SURFACE,
            mew=2,
        )
    ax2.set_yticks(yy, [n.replace("control_", "") for n in names])
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_xlabel(f"false-alarm rate at score >= {res['config']['alarm_threshold']}")
    ax2.set_title("Controls, pooled over evaluated splits", loc="left", color=INK)
    ax2.plot([], [], "o", color=INK, label="share of windows")
    ax2.plot(
        [],
        [],
        "o",
        color=INK,
        markerfacecolor=SURFACE,
        mew=2,
        label="share of trajectories with any alarm",
    )
    ax2.legend(loc="lower right", frameon=False, labelcolor=INK)

    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return fig_path


def main() -> None:
    cfg = load_baseline_config()
    write_json(cfg.json_path, run(cfg))
    print(f"wrote {cfg.json_path}")
    print(f"wrote {plot(cfg.json_path, cfg.figure_path)}")


if __name__ == "__main__":
    main()

"""Save one figure per scenario family to results/figs/.

Each figure shows one trajectory (observations and true state), the control
parameter, the realised tip time and the regression target. Run with:
    uv run python scripts/plot_scenarios.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pfngate.scenarios import Scenario, load_scenarios  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "figs"
# Prefer the middle noise level and the slow ramp when choosing one scenario per family.
PREFERRED_LEVEL_SUFFIX = "2"
PREFERRED_SPEED = "slow"


def pick_scenario(scenarios: list[Scenario]) -> Scenario:
    def score(sc: Scenario) -> tuple[bool, bool]:
        return (sc.noise_level.endswith(PREFERRED_LEVEL_SUFFIX), sc.speed == PREFERRED_SPEED)

    return max(scenarios, key=score)


def plot_family(sc: Scenario, out_path: Path) -> None:
    seed = sc.seeds()[0]
    tr = sc.generate(seed)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, constrained_layout=True)

    ax = axes[0]
    for c, name in enumerate(sc.observed):
        ax.plot(tr.time, tr.observations[:, c], lw=0.6, label=f"observed {name}")
        ax.plot(
            tr.time,
            tr.state[:, sc.state_names.index(name)],
            lw=0.8,
            alpha=0.7,
            label=f"true {name}",
        )
    det = sc.tip_detector
    if det["kind"] == "moving_frame_above" and tr.control_parameter is not None:
        # Rate tipping happens in the moving frame y = x + lambda.
        ax.plot(
            tr.time,
            tr.state[:, 0] + tr.control_parameter,
            color="black",
            lw=0.9,
            label="x + lambda (moving frame)",
        )
    if det["kind"] in ("state_above", "moving_frame_above"):
        ax.axhline(det["threshold"], color="tab:red", ls=":", lw=1, label="tip threshold")
    ax.set_ylabel("value")
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    ax = axes[1]
    if tr.control_parameter is not None:
        ax.plot(tr.time, tr.control_parameter, color="tab:purple", lw=1.2)
        ax.set_ylabel(f"control: {tr.control_name}")
        if "p_c" in sc.params and tr.control_name == "p":
            ax.axhline(sc.params["p_c"], color="tab:red", ls=":", lw=1, label="p_c")
            ax.legend(loc="upper left", fontsize=8)
    else:
        ax.text(
            0.5,
            0.5,
            "no ramped control parameter",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_ylabel("control")

    ax = axes[2]
    ax.plot(tr.time, tr.target, color="tab:green", lw=1.2)
    ax.axhline(sc.common.positive_max, color="grey", ls=":", lw=1)
    ax.axhline(sc.common.negative_min, color="grey", ls=":", lw=1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(f"target ({sc.label_kind})")
    ax.set_xlabel("time")

    for ax in axes:
        if tr.tip_time is not None:
            ax.axvline(tr.tip_time, color="tab:red", ls="--", lw=1)
    tip = "none" if tr.tip_time is None else f"{tr.tip_time:.1f}"
    fig.suptitle(
        f"{sc.family}: {sc.scenario_id}, seed {seed} ({tr.split}), tip time {tip}", fontsize=11
    )
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_family: dict[str, list[Scenario]] = {}
    for sc in load_scenarios().values():
        by_family.setdefault(sc.family, []).append(sc)
    for family, scenarios in by_family.items():
        out = OUT_DIR / f"{family}.png"
        plot_family(pick_scenario(scenarios), out)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

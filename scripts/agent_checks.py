"""Write results/agent_checks.json: two failure checks on the agent variants, read from results/.

1. Does the agent always pick the same context strategy? (runs per strategy)
2. Does the agent confidently recommend "alert" on control runs, where nothing is tipping?
   (its own recommendation, and the final decision after the gate, on the control runs)

The numbers come only from results/{agent,naive_agent}_{test,ood}.json. The naive agent's
context is chosen by the harness, so check 1 is only meaningful for the full harness.

Usage: python scripts/agent_checks.py [--results-dir results]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("agent", "naive_agent")
SPLITS = ("test", "ood")
CONFIDENT = 0.8  # a recommendation at or above this confidence counts as confident


def final_decision(run: dict[str, Any]) -> str | None:
    decision = run.get("decision")
    return decision.get("decision") if isinstance(decision, dict) else decision


def strategy_of(run: dict[str, Any]) -> str | None:
    final = run["record"].get("final") or {}
    return final.get("context_strategy")


def check_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    strategies = Counter(strategy_of(r) or "none" for r in runs)
    controls = [r for r in runs if r["ground_truth"]["role"] == "control"]
    verdicts = [r["record"].get("verdict") for r in controls]
    alerts = [v for v in verdicts if v and v["decision"] == "alert"]
    return {
        "n_runs": len(runs),
        "strategy_counts": dict(strategies),
        "n_control_runs": len(controls),
        "n_control_without_recommendation": sum(v is None for v in verdicts),
        "n_control_recommended_alert": len(alerts),
        "n_control_confident_alert": sum(v["confidence"] >= CONFIDENT for v in alerts),
        "n_control_final_alert": sum(final_decision(r) == "alert" for r in controls),
    }


def build(results_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"confident_at": CONFIDENT, "variants": {}}
    for variant in VARIANTS:
        for split in SPLITS:
            path = results_dir / f"{variant}_{split}.json"
            if path.exists():
                runs = json.loads(path.read_text(encoding="utf-8"))["runs"]
                out["variants"][f"{variant}_{split}"] = check_runs(runs)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    args = p.parse_args()
    target = args.results_dir / "agent_checks.json"
    target.write_text(json.dumps(build(args.results_dir), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()

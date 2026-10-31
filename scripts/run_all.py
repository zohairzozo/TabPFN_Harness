"""Run the whole suite with one command: `make full` (or `python scripts/run_all.py`).

Stages, in order:
  1. ruff (check and format check) and pytest,
  2. the gate: configs/gate.frozen.sha256 must match configs/gate.yaml (assert_frozen),
  3. every variant on every split: classic, LLM-only, naive agent, full harness (agent + gate)
     and the family-matched oracle, on val, test, ood, real (the Daphnia populations) and
     real_control (the grid frequency). The controls are inside each split. LLM stages use
     --llm-mode replay when the cache answers every request, and record otherwise,
  4. scripts/make_summary.py (and scripts/agent_checks.py),
  5. results/run_manifest.json.

Profiles (how much of each split is run):
  default  --limit 3 --seeds-per-scenario 3   the trial size used for all results so far
  --fast   --limit 5 --seeds-per-scenario 1   5 scenarios of each kind, 1 seed each; writes to
                                              results/fast/ so it never overwrites results/
  --all    no limit, all seeds                the full split sizes of docs/SPLITS.md (hours)

The gate is plain Python and is only read here: this script never edits or tunes it.
A daily-cap error from the LLM provider (429) is not retried: the LLM stages that are left are
marked skipped, the summary and manifest are still written, and a later run continues from the
recorded responses.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pfngate.gate import assert_frozen  # noqa: E402

PYTHON = sys.executable
SPLITS = ("val", "test", "ood", "real", "real_control")
VARIANTS = ("classic", "llm_only", "naive", "agent", "oracle")
# variant -> (module, result file prefix)
MODULES: dict[str, tuple[str, str]] = {
    "classic": ("pfngate.evals.run_reference", "classic_ews"),
    "oracle": ("pfngate.evals.run_reference", "oracle"),
    "llm_only": ("pfngate.evals.baseline_llm", "llm_only"),
    "naive": ("pfngate.evals.run_agent", "naive_agent"),
    "agent": ("pfngate.evals.run_agent", "agent"),
}
LLM_VARIANTS = ("llm_only", "naive", "agent")
PROFILES: dict[str, dict[str, int | None]] = {
    "default": {"limit": 3, "seeds": 3},
    "fast": {"limit": 5, "seeds": 1},
    "all": {"limit": None, "seeds": None},
}
SHARED_FILES = ("baseline_ews.json", "gate_tuning.json", "timing_tabpfn.json")
PACKAGES = (
    "tabpfn",
    "pydantic-ai-slim",
    "pydantic",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "pyyaml",
    "ruff",
    "pytest",
)
MAX_TRIES = 5


class Run:
    """Settings and state of one run_all call."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.profile = "fast" if args.fast else "all" if args.all else "default"
        self.limit, self.seeds = (PROFILES[self.profile][k] for k in ("limit", "seeds"))
        self.results_dir = REPO_ROOT / "results" / ("fast" if args.fast else "")
        self.llm_cache = REPO_ROOT / "results" / "cache" / "llm"
        self.log_dir = REPO_ROOT / "results" / "cache" / "run_logs"
        self.variants = [v for v in VARIANTS if v in args.variants]
        self.splits = [s for s in SPLITS if s in args.splits]
        self.skip_checks = args.skip_checks
        self.stages: list[dict[str, Any]] = []
        self.quota_out = False
        self.started = time.perf_counter()
        self.started_utc = datetime.now(UTC).isoformat(timespec="seconds")


def run_command(
    run: Run, name: str, cmd: list[str], *, mode: str | None = None
) -> subprocess.CompletedProcess[str]:
    """Run one command, log its output, and record the stage."""
    run.log_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8")
    seconds = time.perf_counter() - start
    (run.log_dir / f"{name}.log").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    print(f"  [{name}] exit {proc.returncode} in {seconds:.1f} s" + (f" ({mode})" if mode else ""))
    run.stages.append(
        {
            "stage": name,
            "command": " ".join(cmd[1:]) if cmd[0] == PYTHON else " ".join(cmd),
            "llm_mode": mode,
            "exit_code": proc.returncode,
            "seconds": round(seconds, 2),
        }
    )
    return proc


def skip(run: Run, name: str, reason: str) -> None:
    print(f"  [{name}] skipped: {reason}")
    run.stages.append(
        {"stage": name, "skipped": reason, "exit_code": None, "seconds": 0.0, "llm_mode": None}
    )


def variant_command(run: Run, variant: str, split: str, mode: str | None) -> list[str]:
    module, _ = MODULES[variant]
    cmd = [PYTHON, "-W", "ignore::DeprecationWarning", "-m", module, "--split", split]
    if module.endswith("run_reference"):
        cmd += ["--variant", variant]
    if module.endswith("run_agent"):
        cmd += ["--variant", "naive" if variant == "naive" else "agent", "--max-context", "200"]
    if module.endswith("run_agent") or module.endswith("baseline_llm"):
        cmd += ["--llm-mode", mode or "replay", "--llm-cache-dir", str(run.llm_cache)]
    if run.limit is not None:
        cmd += ["--limit", str(run.limit)]
    if run.seeds is not None:
        cmd += ["--seeds-per-scenario", str(run.seeds)]
    return cmd + ["--results-dir", str(run.results_dir)]


def cache_has_entries(run: Run) -> bool:
    return run.llm_cache.exists() and any(run.llm_cache.glob("*.json"))


def run_variant(run: Run, variant: str, split: str) -> None:
    name = f"{variant}_{split}"
    if variant not in LLM_VARIANTS:
        run_command(run, name, variant_command(run, variant, split, None) + [])
        return
    if run.quota_out and not cache_has_entries(run):
        skip(run, name, "provider daily cap reached earlier in this run")
        return
    # replay when the cache answers every request, otherwise record (which also replays)
    if cache_has_entries(run):
        proc = run_command(run, name, variant_command(run, variant, split, "replay"), mode="replay")
        if proc.returncode == 0:
            return
        if "ReplayMissError" not in proc.stderr + proc.stdout:
            return
    if run.quota_out:
        skip(run, name, "provider daily cap reached earlier in this run")
        return
    for _attempt in range(MAX_TRIES):
        proc = run_command(run, name, variant_command(run, variant, split, "record"), mode="record")
        text = proc.stderr + proc.stdout
        if proc.returncode == 0:
            return
        if "free-models-per-day" in text:
            run.quota_out = True  # not retried: a later run continues from the saved responses
            return
        time.sleep(20)  # a timeout (504) from the provider: the finished responses are saved


def check_stage(run: Run) -> None:
    print("1. ruff and pytest")
    if run.skip_checks:
        for name in ("ruff_check", "ruff_format", "pytest"):
            skip(run, name, "--skip-checks")
        return
    run_command(run, "ruff_check", [PYTHON, "-m", "ruff", "check", "."])
    run_command(run, "ruff_format", [PYTHON, "-m", "ruff", "format", "--check", "."])
    run_command(run, "pytest", [PYTHON, "-m", "pytest", "-q"])


def gate_stage(run: Run) -> dict[str, Any]:
    print("2. gate")
    start = time.perf_counter()
    info: dict[str, Any] = {}
    try:
        cfg = assert_frozen()
        info = {"frozen": True, "status": cfg.status}
        digest = (REPO_ROOT / "configs" / "gate.frozen.sha256").read_text(encoding="ascii")
        info["sha256"] = digest.strip()
        code = 0
    except Exception as err:  # noqa: BLE001 - any failure here stops the run
        info = {"frozen": False, "error": str(err)}
        code = 1
    seconds = time.perf_counter() - start
    print(f"  [gate_frozen] exit {code}")
    run.stages.append(
        {"stage": "gate_frozen", "exit_code": code, "seconds": round(seconds, 3), "llm_mode": None}
    )
    return info


def prepare_results_dir(run: Run) -> None:
    run.results_dir.mkdir(parents=True, exist_ok=True)
    if run.results_dir.name != "fast":
        return
    for name in SHARED_FILES:  # a quick run reads the same reference files, in its own folder
        shutil.copy(REPO_ROOT / "results" / name, run.results_dir / name)
    for path in (REPO_ROOT / "results").glob("pfn_*.json"):
        shutil.copy(path, run.results_dir / path.name)


def percentiles(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    return {
        "n": len(a),
        "p50": round(float(np.percentile(a, 50)), 3),
        "p95": round(float(np.percentile(a, 95)), 3),
    }


def pfn_calls_per_run(trace: Path) -> dict[str, dict[str, int]]:
    """Per trajectory in a trace: the agent's run_pfn calls and every PFN step."""
    out: dict[str, dict[str, int]] = {}
    if not trace.exists():
        return out
    for line in trace.read_text(encoding="utf-8").splitlines():
        step = json.loads(line)
        tool = str(step["tool"])
        entry = out.setdefault(step["scenario_id"], {"agent": 0, "all": 0})
        if tool == "agent_tool:run_pfn":
            entry["agent"] += 1
        if tool == "agent_tool:run_pfn" or tool.startswith("pfn_"):
            entry["all"] += 1
    return out


def variant_manifest(run: Run, variant: str) -> dict[str, Any]:
    """Latency, tokens, cost and PFN calls of one variant over the splits it produced."""
    prefix = MODULES[variant][1]
    latency_by_scenario: dict[str, list[float]] = {}
    all_latency: list[float] = []
    usage = {"model_requests": 0, "replayed_requests": 0, "tokens_in": 0, "tokens_out": 0}
    cost = 0.0
    models: set[str] = set()
    modes: set[str] = set()
    splits: dict[str, Any] = {}
    agent_calls: list[int] = []
    all_calls: list[int] = []
    for split in run.splits:
        path = run.results_dir / f"{prefix}_{split}.json"
        if not path.exists() or path.stat().st_mtime < run_start_time(run):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        splits[split] = {"n_runs": data["n_runs"], "wall_seconds": round(data["wall_seconds"], 1)}
        if data.get("model"):
            models.add(data["model"])
        if data.get("llm_mode"):
            modes.add(data["llm_mode"])
        u = data.get("usage") or {}
        for key in usage:
            usage[key] += int(u.get(key) or 0)
        cost += float(u.get("cost_usd_approx") or 0.0)
        calls = pfn_calls_per_run(run.results_dir / "traces" / data["trace"])
        for r in data["runs"]:
            sid = r["ground_truth"]["scenario_id"]
            total = float(r["timing"]["total_seconds"])
            latency_by_scenario.setdefault(sid, []).append(total)
            all_latency.append(total)
            tid = r["record"].get("trajectory_id") if "record" in r else None
            if variant in ("naive", "agent") and tid in calls:
                agent_calls.append(calls[tid]["agent"])
                all_calls.append(calls[tid]["all"])
            elif variant == "oracle":
                all_calls.append(1)  # one PFN fit and predict per run (a cache hit counts)
                agent_calls.append(0)
            else:
                all_calls.append(0)
                agent_calls.append(0)
    return {
        "splits_run": splits,
        "models": sorted(models),
        "llm_modes": sorted(modes),
        "latency_seconds_overall": percentiles(all_latency),
        "latency_seconds_by_scenario": {
            k: percentiles(v) for k, v in sorted(latency_by_scenario.items())
        },
        "tokens": {k: usage[k] for k in ("tokens_in", "tokens_out")},
        "llm_requests": {k: usage[k] for k in ("model_requests", "replayed_requests")},
        "cost_usd_approx": round(cost, 6),
        "pfn_calls_per_verdict": {
            "agent_run_pfn_tool_calls": round(float(np.mean(agent_calls)), 3)
            if agent_calls
            else None,
            "all_pfn_steps": round(float(np.mean(all_calls)), 3) if all_calls else None,
            "n_verdicts": len(all_calls),
        },
    }


def run_start_time(run: Run) -> float:
    return datetime.fromisoformat(run.started_utc).timestamp() - 1.0


def git_info() -> dict[str, Any]:
    def git(*a: str) -> str:
        return subprocess.run(
            ["git", *a], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "uncommitted_files": len([line for line in git("status", "--short").splitlines() if line]),
    }


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def seeds_info(run: Run) -> dict[str, Any]:
    pfn = yaml.safe_load((REPO_ROOT / "configs" / "pfn.yaml").read_text(encoding="utf-8"))
    common = yaml.safe_load(
        (REPO_ROOT / "configs" / "scenarios" / "_common.yaml").read_text(encoding="utf-8")
    )
    agent = yaml.safe_load((REPO_ROOT / "configs" / "agent.yaml").read_text(encoding="utf-8"))
    return {
        "pfn_seed": pfn.get("seed"),
        "bootstrap": pfn.get("metrics", {}).get("bootstrap")
        if isinstance(pfn.get("metrics"), dict)
        else None,
        "scenario_seed_sets": common.get("seed_sets"),
        "seeds_per_scenario": run.seeds,
        "limit": run.limit,
        "llm_temperature": agent.get("temperature"),
    }


def any_stage_failed(stages: list[dict[str, Any]]) -> bool:
    """A stage fails when its LAST attempt failed: a replay miss that record then filled is fine."""
    last = {s["stage"]: s["exit_code"] for s in stages}
    return any(code not in (0, None) for code in last.values())


def seconds_by_stage(stages: list[dict[str, Any]]) -> dict[str, float]:
    """Wall-clock per stage, adding the attempts of a stage that was tried more than once."""
    out: dict[str, float] = {}
    for s in stages:
        out[s["stage"]] = round(out.get(s["stage"], 0.0) + s["seconds"], 3)
    return out


def write_manifest(run: Run, gate: dict[str, Any], failed: bool) -> Path:
    agent_cfg = yaml.safe_load((REPO_ROOT / "configs" / "agent.yaml").read_text(encoding="utf-8"))
    variants = {v: variant_manifest(run, v) for v in run.variants}
    manifest = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "started_utc": run.started_utc,
        "profile": run.profile,
        "complete": not failed and not run.quota_out,
        "llm_daily_cap_reached": run.quota_out,
        "git": git_info(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "models": {
            "llm_agent": agent_cfg.get("model"),
            "llm_debug": agent_cfg.get("debug_model"),
            "pfn_backend": "see variants[*].models and results settings",
        },
        "seeds": seeds_info(run),
        "gate": gate,
        "wall_clock_seconds_total": round(time.perf_counter() - run.started, 1),
        "wall_clock_seconds_by_stage": seconds_by_stage(run.stages),
        "stages": run.stages,
        "totals": {
            "tokens_in": sum(v["tokens"]["tokens_in"] for v in variants.values()),
            "tokens_out": sum(v["tokens"]["tokens_out"] for v in variants.values()),
            "cost_usd_approx": round(sum(v["cost_usd_approx"] for v in variants.values()), 6),
            "cost_note": "approximate: the price table in configs/agent.yaml, where the models "
            "used cost 0 (free tier)",
        },
        "variants": variants,
    }
    path = run.results_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--fast", action="store_true", help="5 scenarios per kind, 1 seed each")
    profile.add_argument("--all", action="store_true", help="no limit: the full split sizes")
    parser.add_argument("--variants", default=",".join(VARIANTS), type=lambda s: s.split(","))
    parser.add_argument("--splits", default=",".join(SPLITS), type=lambda s: s.split(","))
    parser.add_argument("--skip-checks", action="store_true", help="skip ruff and pytest")
    args = parser.parse_args()
    run = Run(args)
    print(f"run_all, profile '{run.profile}', results in {run.results_dir}")

    check_stage(run)
    gate = gate_stage(run)
    failed = any_stage_failed(run.stages)
    if not gate.get("frozen"):
        print("the gate is not frozen: no variant is run")
        write_manifest(run, gate, failed=True)
        return 1

    print("3. variants")
    prepare_results_dir(run)
    for variant in run.variants:
        for split in run.splits:
            run_variant(run, variant, split)

    print("4. summary")
    summary_cmd = [PYTHON, "scripts/make_summary.py", "--results-dir", str(run.results_dir)]
    run_command(run, "make_summary", summary_cmd)
    run_command(
        run,
        "agent_checks",
        [PYTHON, "scripts/agent_checks.py", "--results-dir", str(run.results_dir)],
    )

    print("5. manifest")
    failed = any_stage_failed(run.stages)
    path = write_manifest(run, gate, failed)
    print(f"wrote {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

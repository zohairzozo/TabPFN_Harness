"""Smoke test for TabPFNRegressor on CPU: timings, quantile ordering, full output keys.

Writes results/timing_tabpfn.json. Run with:
    uv run python scripts/smoke_tabpfn.py --model-version v3.5
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import warnings
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

SEED = 0
N_FEATURES = 8
N_TEST = 200
N_TRAIN_SIZES = [500, 2000, 5000]
REPEATS = 3
QUANTILES = [0.05, 0.5, 0.95]
OUT_PATH = Path(__file__).resolve().parents[1] / "results" / "timing_tabpfn.json"

MODEL_VERSIONS = ["v3", "v3.5", "v3.5-fast"]
DEFAULT_MODEL_VERSION = "v3.5"


def cpu_model() -> str:
    """Return a readable CPU name for the current machine."""
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except ImportError:
        pass
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def package_versions() -> dict[str, str]:
    """Return installed versions of the packages that matter for timings."""
    names = ["tabpfn", "torch", "numpy", "scikit-learn", "scipy", "pandas"]
    return {name: metadata.version(name) for name in names}


def make_problem(n_train: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """Noisy nonlinear regression problem with 8 numeric features."""
    n = n_train + N_TEST
    x = rng.normal(size=(n, N_FEATURES))
    signal = (
        np.sin(x[:, 0] * x[:, 1]) + 0.5 * x[:, 2] ** 2 - np.abs(x[:, 3]) + 0.3 * x[:, 4] * x[:, 5]
    )
    y = signal + 0.3 * rng.normal(size=n)
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


def licence_help(version: str) -> str:
    """Plain-language instructions for gated weights. We do not work around the licence."""
    return (
        f"\nCould not load TabPFN model version '{version}'. These weights are gated and\n"
        "are licensed for non-commercial use. This script does not work around the licence.\n"
        "To use them, you need to:\n"
        "  1. Open https://ux.priorlabs.ai, log in (or register).\n"
        "  2. Accept the licence on the Licenses tab.\n"
        "  3. Copy your API key from https://ux.priorlabs.ai/account.\n"
        "  4. Put it in .env as TABPFN_TOKEN=<your-api-key> and run again.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-version",
        default=DEFAULT_MODEL_VERSION,
        choices=MODEL_VERSIONS,
        help="TabPFN model version (default: %(default)s).",
    )
    parser.add_argument(
        "--allow-cpu-large",
        action="store_true",
        help="Set TABPFN_ALLOW_CPU_LARGE_DATASET=true so sizes above the CPU limit run. "
        "Off by default; the result is recorded in the JSON.",
    )
    args = parser.parse_args()

    load_dotenv()
    # tabpfn reads this setting at import time, so set it before importing tabpfn.
    if args.allow_cpu_large:
        os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "true"
        print("NOTE: --allow-cpu-large set TABPFN_ALLOW_CPU_LARGE_DATASET=true")

    from tabpfn import TabPFNRegressor
    from tabpfn.browser_auth import get_cached_token
    from tabpfn.constants import ModelVersion

    version = ModelVersion(args.model_version)
    result: dict[str, Any] = {
        "python": sys.version,
        "os": platform.platform(),
        "cpu_model": cpu_model(),
        "package_versions": package_versions(),
        "model_version_requested": version.value,
        "cpu_large_override_flag_used": args.allow_cpu_large,
        "n_test": N_TEST,
        "n_features": N_FEATURES,
        "repeats": REPEATS,
        "seed": SEED,
        "timings": {},
        "warnings": [],
        "errors": [],
    }

    def finish(code: int) -> int:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(result, indent=2, default=str) + "\n")
        print(f"Wrote {OUT_PATH}")
        return code

    # Do not start a browser login flow for gated weights; ask for a token instead.
    if not get_cached_token():
        print(licence_help(version.value))
        result["errors"].append(f"no TABPFN_TOKEN available for gated version {version.value}")
        return finish(2)

    def new_regressor() -> TabPFNRegressor:
        return TabPFNRegressor.create_default_for_version(version, device="cpu", random_state=SEED)

    rng = np.random.default_rng(SEED)
    quantile_ok = True
    for n_train in N_TRAIN_SIZES:
        x_tr, y_tr, x_te, _ = make_problem(n_train, rng)
        fit_times: list[float] = []
        predict_times: list[float] = []
        entry: dict[str, Any] = {}
        try:
            for _ in range(REPEATS):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    reg = new_regressor()
                    t0 = time.perf_counter()
                    reg.fit(x_tr, y_tr)
                    fit_times.append(time.perf_counter() - t0)
                    t0 = time.perf_counter()
                    reg.predict(x_te)
                    predict_times.append(time.perf_counter() - t0)
                for w in caught:
                    msg = f"n_train={n_train}: {w.category.__name__}: {w.message}"
                    if msg not in result["warnings"]:
                        result["warnings"].append(msg)
        except Exception as exc:  # noqa: BLE001
            text = str(exc)
            if "CPU" in text and "not allowed by default" in text:
                entry["skipped"] = "over the default CPU sample limit; rerun with --allow-cpu-large"
                print(f"n_train={n_train}: skipped, over the CPU limit (use --allow-cpu-large)")
            elif any(k in text.lower() for k in ("licen", "gated", "token", "authenticat")):
                print(licence_help(version.value))
                result["errors"].append(text)
                return finish(2)
            else:
                entry["error"] = f"{type(exc).__name__}: {text}"
                result["errors"].append(entry["error"])
                print(f"n_train={n_train}: failed: {entry['error']}")
            result["timings"][str(n_train)] = entry
            continue

        cpu_limit = reg.inference_config_.MAX_CPU_SAMPLES
        entry.update(
            fit_seconds_median=statistics.median(fit_times),
            predict_seconds_median=statistics.median(predict_times),
            fit_seconds_all=fit_times,
            predict_seconds_all=predict_times,
            default_cpu_sample_limit=cpu_limit,
            needed_cpu_large_override=n_train > cpu_limit,
        )
        result["timings"][str(n_train)] = entry
        result["model_path_used"] = str(reg.model_path)
        print(
            f"n_train={n_train:>5}  fit {entry['fit_seconds_median']:7.2f}s  "
            f"predict {entry['predict_seconds_median']:6.2f}s  "
            f"(cpu limit {cpu_limit}, override needed: {entry['needed_cpu_large_override']})"
        )

        quantiles = np.stack(reg.predict(x_te, output_type="quantiles", quantiles=QUANTILES))
        ordered = bool(
            np.all(quantiles[0] <= quantiles[1]) and np.all(quantiles[1] <= quantiles[2])
        )
        entry["quantile_order_ok"] = ordered
        quantile_ok &= ordered
        if n_train == N_TRAIN_SIZES[0]:
            full = reg.predict(x_te, output_type="full")
            result["full_output_keys"] = sorted(full.keys())
            print(f"output_type='full' keys: {result['full_output_keys']}")

    result["quantile_order_ok_all_sizes"] = quantile_ok
    print(f"quantile ordering 0.05 <= 0.5 <= 0.95 holds at all sizes run: {quantile_ok}")
    return finish(0 if not result["errors"] else 1)


if __name__ == "__main__":
    raise SystemExit(main())

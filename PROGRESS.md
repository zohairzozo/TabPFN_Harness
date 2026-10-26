# Progress

Append to this file at the end of every session.

## Log

### 2026-10-25

- Created the project skeleton: uv project, src layout, package tree, CLAUDE.md, Makefile, one smoke test.
- Installed uv 0.13.5 with winget.
- On this machine `uv` cannot inspect the Store Python or install a managed Python. Workaround: `python -m pip install --user uv`, then `python -m uv sync --python <path to python.exe>`.
- GNU Make is not installed. Run the Makefile commands directly until it is.


### 2026-10-26 (TabPFN smoke test)

- Added `tabpfn` 9.2.0 (pulls in torch 2.14.1). Wrote `scripts/smoke_tabpfn.py` with `--model-version` (v3, v3.5, v3.5-fast; default v3.5) and `--allow-cpu-large`.
- Model versions come from `tabpfn.constants.ModelVersion`. A version is selected with `TabPFNRegressor.create_default_for_version(ModelVersion.V3_5, ...)`. These weights are gated: accept the licence and set `TABPFN_TOKEN` in `.env`. Without a token the script stops and prints what to do. It does not work around the licence.
- v3.5 runs on CPU at n_train 500, 2000 and 5000 without the CPU override (the v3.x CPU limit is 5000 rows). Quantile ordering held at every size. `output_type="full"` returns the keys criterion, logits, mean, median, mode, quantiles.
- Timings are in `results/timing_tabpfn.json`, not copied here. Fit is lazy (a fraction of a second); predict is the cost and grows quickly with n_train. The first fit includes a one-off weight download.
- The package warns "more than 1000 samples" on CPU even for v3.5. It is recorded in the JSON warnings and did not block the run.

### 2026-10-26 (scenario suite design)

- Drafted the scenario design for approval: `docs/LABELS.md`, `docs/SPLITS.md` and `configs/scenarios/*.yaml` (a common file plus 8 family files). No simulator code yet.
- 39 scenarios: fold, subcritical Hopf and transcritical (in-prior, 6 each); rate-induced and noise-induced tipping (out-of-prior, 6 each); OU, short-ramp and regime-change controls (3 each).
- Splits use seed sets A (context), B (validation), C (in-prior test) and D (out-of-prior test). Controls are in every split.
- Status: approved as drafted. Open questions are listed at the end of both docs and are still open.

### 2026-10-26 (scenario simulator)

- Implemented the approved design in `src/pfngate/scenarios/` (config, dynamics, forcing, detect, simulate, labels, trajectory, scenario). `load_scenarios()` reads the YAML files; `Scenario.generate(seed)` returns a `Trajectory`. All parameter values come from the YAML.
- The YAML files gained machine-readable fields (dynamics, observation variables, structured tip detectors, a scenario list, split names). The design itself did not change, except as noted below.
- Change from the approved design: the noise-induced tip detector now requires the state to stay in the right basin (x > 0) for 10 time units after entering x > 0.5. Without this, at high noise the state touched x > 0.5 and fell back, which is not a tip. LABELS.md updated.
- `tests/test_scenarios.py` covers: tipping scenarios leave their start state at the tip, controls never tip, R-tipping tips at 1.5 and 3 times r_c but not at 0.5 times r_c (noise-free), determinism, and hand-checked labels.
- `scripts/plot_scenarios.py` writes one figure per family to `results/figs/`.
- A one-off scratch check over every planned seed found: every tipping trajectory tips, no control tips, and no noise-induced trajectory had to be dropped. Transcritical escapes before p_c in most L2 and L3 slow-ramp runs. This confirms the open question in LABELS.md. The check is not yet a script in `scripts/` with output in `results/`.

### 2026-10-26 (features)

- Implemented `src/pfngate/features/` (config, windows, compute, dataset). Settings are in `configs/features.yaml`: window 128 points, stride 16, linear detrend, sub-windows of 32 with stride 8, low-frequency share of the spectrum up to 10% of Nyquist.
- Window settings moved from `configs/scenarios/_common.yaml` (which said 400 / 50 and was unused by code) to `configs/features.yaml`. LABELS.md updated.
- Features per window, computed on the linearly detrended window only: `variance_log_ratio`, `ac1`, `skewness`, `kurtosis`, `trend_slope`, `tau_variance`, `tau_ac1`, `low_freq_power`. All are scale-free (unchanged by `a * x + b` with `a > 0`). Plain variance is replaced by the log ratio of late to early variance, because raw variance depends on units and noise level. Hopf's two channels are averaged.
- `build_dataset(split)` returns one row per window (ids, family, split, seed, window end, target, binary label, features), caches it as a pickle under `results/cache/` (gitignored), and for full builds writes counts to `results/datasets/<split>_summary.json`.
- `tests/test_features.py`: shapes, no NaNs, features unchanged when later data is altered, no trajectory in two splits (plan level and dataset level), scale-free features, Kendall tau equals scipy.
- Built all four splits. Each takes well under a minute on this CPU and loads from the cache almost instantly afterwards. Window and label counts are in `results/datasets/`.

### 2026-10-26 (classic EWS baseline)

- Implemented `src/pfngate/evals/baseline_ews.py`. For each window it takes the detrended variance and the lag-1 autocorrelation (averaged over channels). The score is the mean of their Kendall taus against time over the current window and the K - 1 before it. Only past windows are used. The first K - 1 windows of each trajectory have no score and are left out.
- Implemented `src/pfngate/evals/metrics.py`: class-balanced AUROC (sklearn, balanced sample weights), a percentile bootstrap CI that resamples whole trajectories, false-alarm rates on controls (share of windows and share of trajectories with any alarm), and `write_json`. With weights that are constant within a class, the balanced AUROC equals the plain AUROC. The JSON says so.
- Settings are in `configs/baseline_ews.yaml`: K = 16, alarm threshold 0.5, 1000 bootstrap resamples. They were fixed before the first run and have not been tuned.
- `scripts/run_baseline_ews.py` scores the validation, in-prior test and out-of-prior test splits. It writes `results/baseline_ews.json` and draws `results/figs/baseline_auroc.png` from that JSON. A full rerun gave an identical JSON. Window counts match `results/datasets/`.
- `tests/test_baseline_ews.py`: the rolling tau matches scipy, the score does not change when later data is altered, the score is the mean of the two taus, the bootstrap resamples trajectories rather than windows, the false-alarm rate is checked by hand, and the JSON writer handles numpy types and NaN.
- **Result: the baseline is weak, and it is committed as it is.** Numbers are in `results/baseline_ews.json` and are not copied here.
  - Sanity check 1 only partly passes. In-prior AUROC is above 0.5 with a CI that excludes 0.5, but not by much. Transcritical is at chance.
  - Sanity check 2 passes. Out-of-prior AUROC is near 0.5.
  - Sanity check 3 fails. Controls alarm on a sizeable share of windows, and every control trajectory alarms at least once.
- Suspected cause: the choice of K = 16, not a bug. The windows are 128 points long with stride 16, so 16 windows span 368 points, only about three independent windows. The tau is therefore close to noise, even on controls.
  - A scratch check on the validation split found that the window-level lag-1 autocorrelation and the variance relative to the first window separate positives from negatives far better than the trend score. The signal is in the data, and the short-span tau loses it.
  - The classic method (Dakos et al. 2012) takes the tau from the start of the series. Switching to it would change the baseline's definition after test results have been seen. That change was not made.
- The transcritical result matches open question 1 in LABELS.md: at higher noise, trajectories escape before p_c, so the labels and the actual tip disagree.
- A trajectory-level false-alarm rate of 100% is built into the metric: any per-window threshold is crossed at least once in about 300 windows. Treat the window rate as the informative number.

## Decisions

- Model version: v3.5. Keep the model version in the results JSON for every run. Never redistribute the weights.
- Features are scale-free only. No raw variance or raw level, so the model cannot tell systems apart by their units or noise level.
- The classic EWS baseline stays at K = 16 as it was configured before the first run. It is reported as weak and is not retuned after seeing test results.

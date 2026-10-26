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

## Decisions

- Model version: v3.5. Keep the model version in the results JSON for every run. Never redistribute the weights.

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

## Decisions

- Model version: v3.5. Keep the model version in the results JSON for every run. Never redistribute the weights.

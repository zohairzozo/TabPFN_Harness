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

### 2026-10-27 (PFN adapter)

- Implemented the backend contract in `src/pfngate/pfn/adapter.py`: a `PFNBackend` protocol (`fit(context_X, context_y)`, `predict(query_X) -> PFNOutput`) and a pydantic `PFNOutput` (quantile levels, quantiles, median, p_critical, metadata). The output validates itself: levels increasing, quantiles non-decreasing per row, p_critical inside [0, 1], shapes consistent. Crossing quantiles are sorted per row and the metadata records that a repair happened and for how many rows.
- Settings are in `configs/pfn.yaml`: backend `tabpfn_local`, model version v3.5 (verified on 2026-10-26), CPU, seed 0, quantile levels 0.05, 0.25, 0.5, 0.75, 0.95, p_critical threshold 0.2.
- Backends are in sibling modules and importable from `pfngate.pfn`; `make_backend()` picks one from the config.
  - `TabPFNLocal`: wraps `tabpfn.TabPFNRegressor`. The API was read from the installed tabpfn 9.2.0 source, not from memory. One `predict(output_type="full", quantiles=levels)` call returns the quantiles and the predictive distribution (`criterion`, `logits`). p_critical is the CDF of that distribution at the threshold. If the distribution is missing, it falls back to interpolating the quantiles and says so in the metadata. If the context target is constant, tabpfn returns no usable distribution, so p_critical is a step function and the metadata says so.
  - `TabPFNClient`: off by default (`tabpfn_client.enabled: false`), with a loud docstring that data leaves the machine. tabpfn-client is not installed here, so this class has not been run against the real package. It checks the predict signature at fit time and refuses otherwise.
  - `TipPFNBackend`: stub that raises NotImplementedError. Its docstring gives the input and output as we understand them (time-indexed feature table plus 0 to 3 context episodes in, 99 quantiles of relative distance to criticality out) and lists what is uncertain. The paper (arXiv 2605.12308) was not read for this.
  - `FakeBackend`: deterministic quantiles, with an option to emit crossing quantiles to test the repair.
- `tests/test_pfn_adapter.py`: output shapes, quantile ordering and repair, p_critical in [0, 1] and by hand, FakeBackend determinism, the TipPFN stub raising, the remote client being disabled, and that importing `pfngate.pfn` does not import tabpfn. One extra test runs the real TabPFN and only runs with `PFNGATE_TEST_TABPFN=1`: it checks shapes, ordering, the method, and that p_critical is 0.5 when the threshold equals the row median. It passed when run once on 2026-10-27.
- Not done yet: running the backend on the scenario windows with context strategies, and the trace file for every call. The adapter returns the metadata a trace needs and `PFNOutput.model_dump_json()` works, but nothing writes traces yet.

### 2026-10-27 (context selection and trace)

- Implemented `src/pfngate/pfn/context.py`. The pool is the context split (seed set A) from `build_dataset("context")`. The query is a batch of windows from one trajectory. Every strategy drops the query trajectory from the pool first, caps the context at `max_context` rows (2000 by default, from `configs/pfn.yaml`), sorts rows by row id so the pool order does not matter, and returns the chosen row ids and a context hash (sha256 of the sorted row ids, 16 hex characters). A row id is the trajectory id plus the window end index.
  - `random`: uniform sample, seeded by the config seed and the query trajectory id, so each query trajectory gets its own draw and the same draw when repeated.
  - `knn`: standardised features (mean and sd of the eligible pool), `knn_neighbors` nearest rows per query window (50 by default). The union is filled by neighbour rank, so every query window gets its nearest row before any gets its second. It uses no random numbers and can return fewer rows than the cap if the neighbours overlap.
  - `family_matched`: ORACLE upper bound. It reads the family of the query, which a real run would not know. The selection has `oracle=True` and its label is `family_matched (ORACLE)`, which is also what the trace records. It raises `NoMatchingFamilyError` for a family that is not in the pool, such as the out-of-prior families.
- Context targets are the regression `target` of each window, including windows the binary task drops.
- Implemented `src/pfngate/evals/trace.py`: a JSONL tracer with the 15 requested fields in a fixed order. Each run writes one new file `results/traces/<run_id>.jsonl`, created in exclusive mode so a run id is never reused. Every line is flushed when written. `gate_result` is null until the gate exists. NaN is refused instead of being written as invalid JSON. `log_pfn_step` fills the context hash, size, strategy, backend and model version from a selection and a `PFNOutput`.
- Tests: `tests/test_context.py` and `tests/test_trace.py`. They cover caps, determinism, independence from pool order, that `family_matched` uses only its family, that a query trajectory never contributes to its own context (synthetic pool and real scenario data), and the trace format and flushing. As a check that the tests can fail, I removed the self-exclusion, the family filter and the cap one at a time. Each removal broke a test. The code was then restored.
- Not done yet: the runner that ties the pool, a strategy, a backend and the tracer together on the scenario windows. No trace has been written by a real run, and nothing under `results/traces/` is committed.

### 2026-10-27 (PFN runs and summary)

- Implemented `src/pfngate/evals/run_pfn.py`, run as `python -m pfngate.evals.run_pfn --strategy {random,knn,family_matched} --split {val,test,ood,controls} --max-context N`. For every trajectory it picks a context, fits once, predicts all windows in one batch, and writes two trace lines (context selection, PFN call). Each run writes `results/pfn_<strategy>_<split>.json` and one file under `results/traces/`.
  - Two scores per window: minus the median and `p_critical`. Both are scored with the same balanced AUROC and trajectory bootstrap as the baseline (`bootstrap_auroc`).
  - Splits: `val`, `test`, `ood` hold the tipping trajectories of validation, test_in_prior and test_out_of_prior. `controls` holds the control trajectories of all three. Controls exist in every split but are run once, so the alarm threshold can be set from the validation controls only and then applied unchanged to the others. The threshold rule is `alarm_threshold_for_rate` in `metrics.py` (at most 5% of validation control windows alarm; the target is in `configs/pfn.yaml`).
  - The baseline got the same rule: `results/baseline_ews.json` has a new block `false_alarms_at_validation_threshold`. Its AUROC numbers did not change (checked against the committed file).
- Cache: `results/cache/pfn/`, keyed by context hash. TabPFN fit is lazy, so the cache stores the whole fit-and-predict result for a (context, query) pair, plus the backend, model and package versions, seed, levels, threshold and a hash of the feature and scenario configs. A cache hit keeps the latency of the run that computed it.
- `scripts/make_summary.py` writes `results/summary.md` from the result files. Wording that depends on a number is chosen by code: CI includes 0.5, AUROC above 0.98 (leakage check), threshold not transferring, degenerate scores.
- Tests: `tests/test_run_pfn.py` (fake backend on real scenario data: caps, trace, cache hits, controls use validation for the threshold, oracle marking) and `tests/test_make_summary.py`. Removing the validation-only threshold rule, or swapping the control and tipping selection, each broke a test.
- **Reduced runs.** The full design is 750 trajectories per strategy. On this CPU, predict time per trajectory grows with the context size, and the default `max_context` of 2000 was too slow (see `results/timing_tabpfn.json`). A short scratch timing (not saved) showed predict time falling only slowly below a few hundred rows, because the query windows and fixed overhead dominate. So I ran `--max-context 200` and then also `--seeds-per-scenario 3`. All 39 scenarios are in, with the first 3 of 10 seeds each. Per-trajectory timings are in the `timing` block of each `results/pfn_*.json` and in `results/summary.md`. Wider confidence intervals follow from the smaller sample. A full run needs `--max-context` raised and `--seeds-per-scenario` dropped; the cache lets an interrupted run resume.
- **Results** are in `results/summary.md`; no number is copied here.
  - Sanity check: TabPFN in-prior AUROC is clearly above the baseline and not suspiciously perfect. Out-of-prior AUROC is lower and its CI overlaps the baseline CI. The rate-tipping family includes 0.5 for p_critical.
  - Hopf with knn and with the oracle context is flagged as above 0.98. The raw lag-1 autocorrelation of Hopf windows separates the classes strongly (checked in a scratch diagnostic for the baseline, not saved), so a strong signal is plausible. It was not investigated further. Leakage guards exist: features use only in-window data, the query trajectory is never in its own context, and the context seeds are disjoint from the query seeds.
  - The in-prior context pool holds the same scenarios as the in-prior val and test splits, with other seeds. In-prior AUROC is therefore optimistic. The out-of-prior column is the generalisation test.
  - **The oracle's false-alarm result on controls is degenerate.** I checked the cache: for all 81 control trajectories the family_matched context had a constant target (all 1), so TabPFN took its constant-target path and returned a constant. The threshold sat at the smallest float and nothing alarmed. Its 0.000 rate and its very low control latency say nothing about the method. The summary marks this with a double dagger, and median latency is computed from the tipping runs only.
  - knn p_critical: the held-out false-alarm rate is above 1.5 times the target, so the validation threshold did not transfer. The summary flags it.
  - family_matched cannot run on the out-of-prior families, which are not in the pool: those 36 trajectories are skipped and listed.
- Squashed the PFN adapter, the context and trace commits and this work into one commit, as asked.

### 2026-10-27 (gate)

- `docs/GATE.md` approved with the five recommended decisions and implemented. Nothing is tuned and nothing is frozen yet.
- `src/pfngate/gate/`:
  - `config.py`: frozen pydantic models for `configs/gate.yaml`. Loading refuses a file looser than the floors written in code. The BAD definition (90% interval, coverage below 0.80) is fixed.
  - `checks.py`: the six checks as pure functions.
  - `gate.py`: `GateInput` (no field for query targets or labels), the frozen `GateResult` (its overall flag must agree with its checks, and `model_copy` with changes raises), `evaluate_gate()` and `decide()` (fail → abstain; pass and p_critical at or above the alert threshold → alert; else no_alert).
  - `freeze.py`: `write_frozen_hash()` (tuned files only, never overwrites), `assert_frozen()`, `assert_committed()` and `assert_applicable()`.
  - `reference.py`: the distance of a query window to its 10th nearest pool row, and the cut (99th percentile of the same distance for pool rows, leaving out each row's own trajectory).
  - `build.py` plus `scripts/build_gate_config.py`: write `configs/gate.yaml` from the design constants and set A features.
- `src/pfngate/evals/calibration.py`: interval coverage, pinball loss per level, mean pinball loss (documented as a rough CRPS proxy, not CRPS), calibration error over the five levels, and reliability diagram data.
- `src/pfngate/evals/gating.py`: cached fit and predict, the context calibration refits (5 folds of whole context trajectories), the per-run ground truth, and gate summary counts.
- `run_pfn.py`:
  - `--gate` gates every run. The GateResult goes into the result JSON and into a "gate" trace line, next to a "context_calibration" line. The ground truth is recorded next to it and marked as not used by the gate.
  - The test, ood and controls splits now call `assert_frozen()` before reading any data, with or without `--gate`. From the command line the file must also be committed.
  - So the committed test and out-of-prior results cannot be re-run until the gate is tuned and frozen. This is deliberate (docs/SPLITS.md).
- `configs/gate.yaml` was written from the full set A at max_context 200. Its status is **provisional**: the tuned thresholds hold the loosest grid values and the alert thresholds are empty. It can gate validation runs but cannot be frozen. `.gitattributes` keeps its bytes (and hash) stable under `core.autocrlf`.
- `FakeBackend` has a second mode, "neighbours", that learns from its context (empirical quantiles of the nearest context targets). An `interval_scale` below 1 makes it overconfident.
- Tests:
  - `tests/test_gate.py` and `tests/test_calibration.py`.
  - New gate cases in `tests/test_run_pfn.py`, with a shared temp gate in `tests/conftest.py`.
  - The three requested end-to-end cases use real scenario windows (one seed per scenario) and the neighbours fake:
    - A context of controls fails `context_not_degenerate` for a tipping query.
    - A family-matched context passes.
    - An overconfident fake fails only `context_calibration`.
  - As a check that the tests can fail, I removed each of these one at a time, and each removal broke a test: the frozen check in the runner, the hash comparison, the degenerate-context rule, the calibration lower bound, the abstain rule, one floor, and the applicability check.
- Seen while writing the tests, with the fake backend only (not TabPFN): the fake's neighbour intervals are often wider than the 0.6 width cap, so many family-matched runs failed that check. The passing test uses a Hopf trajectory where they are narrower. Whether the cap blocks many real TabPFN runs is unknown until the validation runs exist.
- Not done yet:
  - the tuner;
  - gated validation runs with TabPFN;
  - alert thresholds;
  - the report that recomputes every GateResult;
  - the baseline script, which does not call `assert_frozen()`.

### 2026-10-28 (gate tuning)

- Tuner: `src/pfngate/gate/tune.py` and `scripts/tune_gate.py`.
  - It reads only `results/pfn_{random,knn,family_matched}_{val,val_controls}.json`. It refuses a run gated with another `configs/gate.yaml`, a run with any trajectory outside the validation split, and a run whose backend, model version or max_context differ from the gate's applicability block.
  - It evaluates the 48 grid points of docs/GATE.md, picks the highest recall with precision at least 0.6, and estimates the result out of fold (5 folds grouped by trajectory id).
  - Alert thresholds per strategy come from the accepted validation control runs.
  - Fewer than 30 runs: it says so loudly, prints how to add seeds, and writes nothing.
  - A stop rule (too few or too many BAD runs, or no feasible point) writes `results/gate_tuning.json` with the reason and the whole grid, prints the trade-off table, and leaves `configs/gate.yaml` unchanged.
  - Only a successful tuning writes the tuned `configs/gate.yaml` and calls `write_frozen_hash()`.
- `run_pfn.py --split val_controls` runs the validation control trajectories, so the tuner sees controls (and the alert thresholds can be set from them).
- `scripts/make_summary.py` has a "TabPFN + gate" section, with rows for each strategy and split:
  - AUROC on accepted runs and on all runs;
  - 90% coverage on accepted runs and on all runs;
  - precision and recall of blocking BAD runs, with bootstrap CIs;
  - the abstention rate, and false alarms on held-out controls.

  Every row says which gate produced it (provisional or tuned). A stopped tuning is reported with its best grid point and with which thresholds actually move the blocking. All sentences in "Does the gate carry over?" are chosen by code.
- Runs: random, knn and family_matched on `val` and `val_controls`, with `--gate` under the provisional gate (the loosest grid point). This used 3 seeds per scenario, not the 10 of decision 4. The five calibration refits per run made 10 seeds cost several hours on this CPU. The oracle's runs were included in tuning at your request.
- **Result: tuning stopped.** No grid point reaches the minimum precision. `configs/gate.yaml` stays provisional, nothing is frozen, and **no held-out run was made with the gate**. `run_pfn` refuses an unfrozen gate on test, ood and controls, as designed. So there is no in-prior versus out-of-prior comparison of the gate yet. The gate failed one step earlier, on validation. The numbers are in `results/gate_tuning.json` and `results/summary.md`. What the runs show:
  - **The width cap blocks almost every random and knn run.** TabPFN's 90% intervals at max_context 200 are wider than the fixed 0.6 cap on most runs. By design these runs count as "not useful" and are left out of precision and recall, so the tuner sees few runs. Taken alone, this check says the model is not usable at this context size.
  - **context_calibration does not separate BAD from good runs on validation.** The context refits of BAD runs cover as well as, or better than, those of good runs. Every BAD run's context coverage lies inside every grid interval. A BAD run looks fine on its context and fails on the query, so a check on context coverage alone cannot catch it. The per-run values are in the "runs" list of `results/gate_tuning.json`.
  - **ood_share has no signal in-prior.** Docs/GATE.md predicted this.
  - Most of the BAD runs that do get blocked are knn runs caught by `context_not_degenerate`. That fixed check also blocks every oracle control run and many knn control runs, which are good runs, and that is what holds precision down.
- What would unblock the gate. Each option is a design change and must be decided before any held-out gated run:
  - revisit the width cap (decision 2);
  - replace or add a check that compares context and query (the BAD runs fail exactly there);
  - remove the oracle from tuning, which takes out the degenerate control runs;
  - add validation seeds.
- Combined the commits from 2026-10-27 and 2026-10-28 into one commit, at your request. The proof that thresholds are frozen before held-out runs now lives in `results/gate_tuning.json` (its time stamp and hashes) and in the `frozen_gate_sha256` of future held-out results, not in commit order.

### 2026-10-28 (gate revision 2: decisions 7 and 9)

- docs/GATE.md has a revision 2, written after the stopped tuning. You approved two of its decisions, and both are implemented. Decisions 6 (a `knn_balanced` context), 8 (a context-vs-query check), 10 (all 10 seeds) and 11 (one more round only) are still open.
- **Decision 7: the width cap only reports.**
  - `GateResult` has a new `advisories` field, apart from the blocking `checks`. `interval_width_cap` moved there. It is computed for every run and never changes `passed` or the decision.
  - A check cannot be both blocking and advisory.
  - `configs/gate.yaml` must say `role: advisory`, so a file written before this change does not load.
  - Precision and recall now count every run, and the "blocked only as not useful" exclusion is gone.
  - The summary reports how many runs are over the cap.
- **Decision 9: the oracle leaves the tuning set.** The tuner reads only random and knn, and a test shows it never opens the oracle's files. Oracle runs are still gated and reported, labelled as oracle.
- `configs/gate.yaml` was rebuilt as provisional. Only the width line changed: the pool reference came out the same.
- The validation runs were gated again under it. Every prediction and refit came from the cache, and the context calibration values match the first gating.
- The summary's gate time column now uses only runs whose refits were computed, not read from the cache. Cached refits made it near zero, which hid the real cost. The first gating's times are in the results committed in 8ccab02.
- The summary flags a tuning record made with an earlier provisional gate.
- **Tuning, round 2: stopped again.** No grid point reaches the minimum precision. `results/gate_tuning.json` now holds this round, and round 1's record is in git (8ccab02). Before running the real tuner I ran the same selection in memory. It found no feasible point, so the real run could not freeze anything. What the runs show (numbers in `results/summary.md`):
  - **random:** at the loosest point the gate blocks nothing. Its BAD runs pass every check, so no current check sees them.
  - **knn, tipping runs:** `context_not_degenerate` blocks almost only BAD runs, and most of them. On tipping runs alone, the gate works.
  - **knn, control runs:** the same check blocks most of them, because a knn context for a control has few near-tipping rows. Those runs cover well, so they are good runs, and they are what holds precision below the minimum.
  - So two gaps remain:
    - random BAD runs that no check sees. Decision 8 targets this.
    - knn controls blocked for having no near-tipping context. Decision 6 changes the context. A "cannot warn" block could also be counted as usefulness rather than trust, but that would be a new decision.

### 2026-10-28 (gate revision 2: decisions 6 and 8)

- **Decision 6, `knn_balanced`** (`src/pfngate/pfn/context.py`):
  - It is knn, except that a fixed share of the context (`context.balanced_near_share` in `configs/pfn.yaml`, 25%) is kept for pool rows near tipping (target at or below `critical_threshold`). Those rows are the ones nearest to the query by the knn rank rule. The rest is filled by knn.
  - It reads pool targets only, never the query's family, so it is deployable.
  - knn's own code moved into shared helpers without changing what it selects: the knn re-gating hit the cache for every run.
- **Decision 8, `near_tipping_support`** (check 7):
  - The value is the share of query windows whose 5th nearest near-tipping context row is beyond a cut, in the `ood_share` features. A context with fewer than 5 near-tipping rows fails outright.
  - The cut is the 99th percentile, over pool rows, of the same distance to near-tipping pool rows of other trajectories. It is stored in `configs/gate.yaml`.
  - The limit is tuned on {0.1, 0.2, 0.3, 0.5}, with a floor of 0.5 in code. The grid grew from 48 to 192 points.
- `configs/gate.yaml` was rebuilt as provisional for random, knn and knn_balanced. The tuner pools those three. Tests cover the new strategy, the check, its distances and cut, and the tuner's new limit.
- Ran: validation runs with `--gate` for all four strategies. Only knn_balanced needed new TabPFN fits and refits; the others came from the cache.
- **Tuning, round 3: stopped again** (`results/gate_tuning.json`). No grid point reaches the minimum precision. Nothing is frozen and no held-out gated run was made. What the runs show (numbers in `results/summary.md`; per-run values in the run files):
  - **knn_balanced removes the BAD runs on validation.** None of its tipping runs is BAD, its mean coverage rises, and so does its coverage on windows near tipping. The cause found in round 1, a context without rows near tipping, was the right one.
  - **But it ranks windows much worse.** Its p_critical AUROC on validation is well below knn's and random's, and its p_critical on validation controls is much higher (`results/pfn_knn_balanced_val_controls.json`). The reserved rows make the model see tipping almost everywhere. Calibration got better and discrimination got worse. This is a real trade-off, not a free fix, and it must not be reported as one.
  - **`near_tipping_support` as specified separates strategies, not runs.** Its cut was fitted on the density of all near-tipping rows in the pool. A context without a quota holds far fewer, so at the loosest limit the check blocks every random run and almost every knn run, BAD or not. Within knn its value does rank BAD runs, but only in-sample, on the seeds the design was built on. The design error: the cut should be fitted on contexts of the size the gate actually sees, not on the whole pool.
  - knn_balanced has no BAD runs, so recall on it is undefined. The tuning set's BAD runs all come from random and knn.
- **Three design rounds have now used seeds 2000 to 2002.** Any further change (a smaller share or a minimum count for knn_balanced, a cut fitted on context-sized samples) would be fitted to those seeds. It should be checked on seeds 2003 to 2009 (decision 10), under a one-more-round rule (decision 11). Both decisions are open.

### 2026-10-28 (round 4: frozen gate, held-out evaluation, reliability plot)

- **Back to the best configuration.** On validation, round 2's checks and tuning set gave the highest precision and recall of the three rounds, by F1. The near-tipping check blocking, or knn_balanced in the tuning set, scored lower or the same. So:
  - `near_tipping_support` is now an advisory, like the width cap: reported for every run, never blocking.
  - `knn_balanced` stays as a context strategy, outside the gate.
  - The tuning set is random and knn.
- **Decision 12, at your request: no minimum precision.** The 0.6 floor was the reason rounds 1 to 3 stopped. The tuner now picks the highest F1 of blocking BAD runs. Ties go to higher precision, then lower abstention on good runs, then the looser point. This loosening was made after seeing three stopped rounds, and docs/GATE.md says so.
- **Tuned and frozen** (`results/gate_tuning.json`, `configs/gate.frozen.sha256`, pinned in `tests/test_gate.py`).
  - The chosen point is the loosest on the grid, so in practice only the fixed `context_not_degenerate` check blocks runs.
  - Out-of-fold precision and recall equal the in-sample ones.
  - The frozen gate was committed before the held-out runs, as `run_pfn` requires.
- **Held-out runs with `--gate`:** random and knn on in-prior test, out-of-prior and controls. Predictions came from the cache; the context calibration refits were new. Validation was gated again under the frozen gate, so its rows read "tuned, in-sample".
- **What the held-out runs show** (numbers in `results/summary.md`):
  - **knn, in-prior test: the gate carries over from validation.** It blocks mostly BAD runs and catches a good share of them. Coverage on accepted runs is higher than on all runs.
  - **knn, out of prior: it does not carry over.** About half the runs are BAD, the gate blocks very few of them, and coverage stays low. The one working check looks for contexts without near-tipping rows. Out of prior the failure is different: the model meets a family it has never seen. `ood_share` at its loosest limit does not fire. This is the finding the gate design predicted (risk 2 in docs/GATE.md), not a failure of the harness.
  - **random:** the gate blocks almost nothing, in or out of prior. Its BAD runs pass every check.
  - **knn, held-out controls:** the gate abstains on most of them (knn contexts for controls hold few near-tipping rows). That lowers the window false-alarm rate, but mostly by abstaining. On accepted runs the rate stays close to knn without the gate.
- **`scripts/plot_reliability.py`** writes `results/figs/reliability.png`.
  - It plots the observed share of true values at or below each predicted quantile against the level, for all runs and for gate-accepted runs, with in-prior test and out-of-prior side by side.
  - It reads `results/pfn_*_{test,ood}.json` only. `run_pfn` now records the observed frequency per level for every run (`ground_truth.observed_frequency`).
  - The figure shows quantiles that are too high in prior and too low out of prior, for both strategies. The gate moves knn's in-prior curve toward the diagonal and leaves the out-of-prior curves where they are.
- **Combined into one commit**, at your request: everything since a873b25. The proof that the gate was frozen before the held-out runs now lives in the files, not in commit order:
  - the tuning time and hashes in `results/gate_tuning.json`;
  - the `frozen_gate_sha256` and status "tuned" in every held-out result;
  - the time stamps in the trace run ids.

### 2026-10-28 (LLM agent)

- **Added `pydantic-ai-slim[anthropic,openai]`** with uv (pyproject and uv.lock changed, nothing committed). It is 2.63.0, a major version I did not know, so I read the installed package before writing code:
  - Model strings are `provider:model` (`pydantic_ai.models.infer_model`). `anthropic:claude-sonnet-5-5` and `anthropic:claude-haiku-5-5` are both known names. A real model needs its API key when it is built, so the agent takes the model at run time and tests never need a key.
  - `Hooks.on.model_request` wraps every model call (used for tokens, cost and latency). An exception raised inside a tool that is not a retry propagates out of `run_sync` unchanged.
  - `RunResult.usage` is a property, not a method. The banner is switched off with `pydantic_ai.BANNER_ENABLED = False`.
  - I used the slim package, not the full `pydantic-ai`: same code, without every other provider's SDK. The `openai` extra covers free OpenAI-compatible endpoints (OpenRouter, Ollama).
- **`src/pfngate/agent/`** (small modules): `config.py` with `configs/agent.yaml`, `prompts.py`, `verdict.py`, `store.py`, `tools.py`, `final.py`, `agent.py`.
  - Model names, temperature, limits and the price table are in `configs/agent.yaml`. A test fails if a model id appears in agent code. Loading refuses a temperature other than 0, more than 6 tool calls or more than 3 contexts, and prices that are not marked approximate. The prices are copied from the snapshot bundled with the installed genai-prices, not checked against the provider's page.
  - `Verdict`: decision (alert, no_alert, abstain), context_strategy (random or knn only, so the oracle is not an option), context_size, rationale (at most 80 words), confidence in [0, 1].
  - The four tools (`describe_query`, `describe_context_pool`, `choose_context`, `run_pfn`) read from a server-side store keyed by run id and return small dicts, never arrays. The query is cut down to the trajectory id, the window position and the features when it enters the store, so no target, family or scenario name can be returned. The prompts and tool output never contain the trajectory id (it names the family).
  - **Cap of 6 tool calls**, counted in code: the 7th call raises, the run stops and is recorded as `tool_cap`. At most 3 contexts; a 4th `choose_context` returns an error. Both are also in the system prompt, built from the same numbers.
  - **The gate runs after the agent**, in code. `run_pfn.gate_run` was split: `gate_and_trace` runs the gate, writes its trace steps and returns the per-window decisions, and `gate_run` calls it and adds the ground truth as before. The agent calls `gate_and_trace` with a query that has no labels, so both paths use the same gate code.
  - **Run-level decision (my design choice):** the gate and the mapping give one decision per window. A run needs one answer, so it is the answer for the end of the series: abstain if the gate failed, otherwise the decision of the last window. The agent's verdict is stored next to it, never instead of it.
  - A run that hits a cap, or answers with something unusable, is stopped, recorded and still gated on the last context it built. A run with no context abstains. Network and provider errors are not caught: they must not look like an agent decision.
  - Every model call (tokens, approximate cost, latency) and every tool call goes into the JSONL trace. The agent's verdict and the final decision are in `<run_id>.agent.json` next to the trace, because the trace has no field for free text.
- Tests (`tests/test_agent.py`, 38), with a scripted fake model: the tool cap, the context budget, a stopped run still being gated, the agent unable to change a gate result (and tools not exposing one), a gate failure always giving abstain for every verdict, a passing gate taking its decision from the mapping, the model seeing no raw data, ids or labels, traces with tokens and cost, and the config and code rules.
  - I broke the code on purpose in eight ways (no cap, the verdict decides, no tracing, a leaked field, an unsanitised store, a wrong context fallback, a skipped gate, and a plain "decision = verdict"), and a test failed each time. The first version of the "verdict decides" break was caught only through the reason text, so I added the plain version.
  - Four of them run the real `AnthropicModel` built from the YAML string, against a mock transport that answers in the Anthropic streaming format, so no network or key is used. They check the request (model name, tools, schema, no leaked label) and the parsing, tokens and cost.
- **Finding: temperature 0 is not applied for the models I was asked to use.** For `claude-sonnet-5-5` and `claude-haiku-5-5`, PydanticAI warns that sampling parameters are not supported and drops `temperature` (no `temperature` in the request). For the 4.5 models it is sent. CLAUDE.md says LLM calls use temperature 0, so this is a gap, not a pass. The run record has `temperature_requested` and `temperature_applied`, set from that warning. Runs with these models are therefore not deterministic by themselves.
- **Not done:** the record/replay cache, the LLM-only baseline and an evaluation script for the agent, which are in the goal but not in this step. No real model call was made: there is no API key here (only `TABPFN_TOKEN` is in `.env`) and no local model. The model's real behaviour (tool use, how often the gate blocks it) is untested.

### 2026-10-28 (LLM replay cache, LLM-only baseline, naive agent)

You chose to keep claude-sonnet-5-5 and claude-haiku-5-5 even though temperature 0 is not applied for them, and to rely on a replay cache for reproducibility. This step builds that cache and two comparison variants.

- **Replay cache, `src/pfngate/agent/replay.py`.** `RecordReplayModel` wraps a PydanticAI model and has three modes:
  - `live`: every call goes to the model, and the cache is not touched.
  - `record`: a call with a stored key is answered from the cache, any other call goes to the model and is stored. This needs the API key.
  - `replay`: every call is answered from the cache. A missing key raises `ReplayMissError` with the key, the model and what to do. The wrapped model is never called. It is built with a dummy key and a transport that raises on any request, so no network call can be made even by mistake.
  - Responses are stored in `results/cache/llm/<key>.json`. The key is a SHA-256 of the model name, the system prompt, all messages (these carry the earlier tool calls and tool results), the tool and answer definitions, and the model settings. The last two go beyond the list you gave: without them, changing a tool or the temperature would replay an answer to a different question. Run ids, conversation ids, timestamps and generated tool-call ids are left out, and call ids are renumbered.
  - An entry is written once and never changed, as JSON with sorted keys and LF bytes (`.gitattributes` marks it `-text`), so a replay reads the same bytes every time.
  - **`.gitignore` changed:** `results/cache/` was ignored, so cassettes would never be committed and the demo could not run offline on another machine. Only `results/cache/llm/` is un-ignored; the TabPFN cache stays ignored.
  - The "temperature dropped" warning of the live call is stored in the entry and repeated on replay. Without it, a replayed run would claim temperature was applied. A mutation check showed my first test did not prove this, because this PydanticAI version also warns in a step before the wrapper, so I added a direct test.
  - `--llm-mode {live,record,replay}` is on both entry points. The default is replay if the cache has entries, else live.
- **LLM-only baseline, `src/pfngate/evals/baseline_llm.py`.** The same model reads a text summary of the query features (the one the agent's `describe_query` returns: means, spreads and trends) and answers `alert` or `no_alert` with a probability. No tools, no PFN, no gate. It is traced the same way (the same trace fields, one `llm_call` step per call with tokens and an approximate cost). An unusable answer counts as `no_alert` and the record says so (`defaulted`).
- **Naive agent, `src/pfngate/agent/naive.py`.** My reading of "the agent with random context, no gate, no abstain option": the harness builds a random context at the largest allowed size before the agent starts, the agent only has `describe_query` and `run_pfn` (no `choose_context`), its verdict has no abstain, and no gate runs, so its verdict is the decision. The cap of 6 tool calls and the tracing are kept. If you meant that the agent may still pick a size, this needs a small change.
- **Evaluation entry points.** `python -m pfngate.evals.run_agent --variant {agent,naive}` and `python -m pfngate.evals.baseline_llm`, each with `--split {val,test,ood}`, `--llm-mode`, `--debug` (the cheaper model from `configs/agent.yaml`) and `--limit N` (a trial run). Results go to `results/agent_<split>.json`, `results/naive_agent_<split>.json` and `results/llm_only_<split>.json`.
  - A split is a whole dataset split, tipping runs and control runs together. The run-level metrics need both: at the end of the series every tipping run is positive and every control run is negative.
  - The test and out-of-prior splits need the gate tuned, frozen and committed, for all three variants, so no held-out run is looked at before the freeze. The agent and naive variants also check the gate's applicability, and the result records the frozen gate's hash, as `run_pfn` does.
- **Metrics, `src/pfngate/evals/agent_metrics.py`.** The truth of a run is the label of its last window (positive, negative, or the middle band, which is left out). Rates: abstention, recall of runs that end near tipping, alerts on runs that end far, alerts on control runs, precision, and AUROC when both classes are present, each with a bootstrap CI over runs. An abstention is not an alert. **These are run-level numbers. The PFN runs are scored per window, so the two sets are not the same quantity and must not be compared as if they were.**
- Shared parts: `agent/calls.py` (the deps and the tracing hook; a replayed call is traced as `llm_replay` and keeps the tokens and cost it had when recorded, so the cost of a run is known offline), `agent/runner.py` (runs an LLM and classifies how it stopped), `agent/summary.py`, `evals/llm_eval.py`. `httpx2` is now a declared dependency, because `replay.py` imports it directly.
- **Tests:** 61 new tests in `tests/test_replay.py`, `test_llm_variants.py`, `test_agent_metrics.py` and `test_llm_evals.py`, with helpers in `tests/agent_support.py`. They cover the three you named (replay is byte-for-byte deterministic, a missing key fails loudly, no connection leaves the machine in replay mode, with the socket module patched to refuse any non-loopback connection or name lookup), plus: record then replay of whole evaluations with the model unplugged, the held-out freeze rules, the metrics against hand-worked values, and that no variant sees an id or a label.
  - I broke the code on purpose 18 times (the replay mode, the key, write-once, the warning, the naive agent's tools and verdict, a leak into the baseline prompt, the freeze rule, the metrics and the truth), and a test failed each time. Two replay breaks were missed at first, and I added tests for them.
- **Not done, and not verified:**
  - **No real model call was made** (there is no API key here), so **`results/cache/llm/` does not exist yet**. The offline demo needs one `--llm-mode record` run with a key. How the real models use the tools, and how the three variants compare, is unknown.
  - Replay without a key is built for `anthropic:` models only. Other providers need a model object passed in.
  - The agent, naive and LLM-only results are not in `results/summary.md` yet.
  - The `results/cache/llm` cassettes will contain the prompts and responses, which hold feature summaries and no secrets, but check them before committing.

### 2026-10-28 (switch to OpenRouter)

- **The model you named cannot be used as it is.** `nvidia/nemotron-3.5-embed-1b:free` is an embedding model. OpenRouter lists it in its embeddings list (`/api/v1/embeddings/models`) with modality `text+image->embeddings` and output `embeddings`, and it is not in the chat-model list. It returns vectors, so it cannot answer, call a tool or fill in a verdict, which every LLM variant here needs.
- **What I used instead, until you decide:** `nvidia/nemotron-3.5-super-vl:free` for the agent and `nvidia/nemotron-3.5-lightning:free` for debug runs. They are the closest chat models in the same family. OpenRouter's public list (read on 2026-10-28) shows both free, with `tools`, `temperature` and `tool_choice` supported. It is a one-line change each in `configs/agent.yaml`.
- **The provider is now OpenRouter.**
  - Model strings are `openrouter:<OpenRouter id>`, for example `openrouter:nvidia/nemotron-3.5-super-vl:free`. The id holds a `:` and a `/` itself, and PydanticAI splits only at the first `:`.
  - Replay without a key builds an `OpenRouterModel` with a dummy key and a transport that raises on any request (the same two locks as before).
  - The config loader now checks that the part before the first `:` is a provider name. Before, any string with a `:` passed, and an OpenRouter id such as `nvidia/x:free` has one.
  - Prices in `configs/agent.yaml` are 0 for both models, from OpenRouter's list. A free model has a known cost of 0.0 in the trace, not null. A model that is not in the table still has no estimate.
- **Removed:** the `anthropic` extra and package (and the orphaned `docstring-parser`), `AnthropicProvider`, the Anthropic streaming mock and its tests, the Claude prices, and the comments that named them. A test fails if the word appears in `src/`, `configs/` or `pyproject.toml`. The dated log entries above are left as they were: they describe what was true then.
- **Temperature.** On the wire, the request now carries `temperature: 0.0`, so it is applied for these models. The code that records `temperature_applied` and replays a stored warning stays, because it is general (PydanticAI warns whenever a model drops sampling settings), and it is tested with a model that warns.
- **Tests:** a mock OpenRouter endpoint answers in the chat-completions format (including the `provider` field that OpenRouter adds and PydanticAI requires). It checks the request (URL, model id, `temperature: 0.0`, the four tools plus `final_result`, the answer schema, no leaked label), the tokens and the zero cost, and a 401 error that must not become an agent decision. 377 tests pass. I broke the code on purpose six more times (the network lock, the temperature request, the provider-name check, the price lookup, the provider guard, the trace backend) and a test failed each time.
- **Not done and not verified:**
  - **No live call has been made.** I have not called OpenRouter, so I do not know how these models handle our tools and the verdict schema, or how often the gate blocks them. `results/cache/llm/` still does not exist.
  - **Free models are rate limited.** I have not measured the limits for your account. One evaluation is about 5 calls per run for the agent, 3 for the naive agent and 1 for the baseline, so a split of dozens of runs may hit them. Use `--limit` for trials.
  - The agent, naive and LLM-only results are still not in `results/summary.md`.
- A mistake of mine, fixed: while rewriting `tests/agent_support.py` I cut off the `StatelessScript` helper at the end of the file. The tests caught it at collection, and I restored it.

### 2026-10-29 (five variants, trial runs, variants table)

- **Asked:** run classic, LLM-only, naive agent, full harness (agent + gate) and the family-matched oracle with `--limit`, `--llm-mode record`, 3 seeds, on in-prior test, out-of-prior and the controls; one row per variant in `results/summary.md`; a pass^3 column.
- **Done and in `results/summary.md` ("Variants, run level"):** classic, LLM-only and oracle, both splits (the naive agent was added on 2026-10-31, see below). Every cell is read from `results/{classic_ews,llm_only,naive_agent,agent,oracle}_{test,ood}.json`; a variant with no file shows "not run". The table has the nine columns asked for plus pass^3.
- **Not done: the naive agent and the full harness have no result.** OpenRouter's free tier allows 50 requests a day for all free models together (`free-models-per-day`, reset 00:00 UTC; a $10 credit lifts it to 1000). LLM-only used up most of it. The naive agent got through part of its first split and was then refused, so no result file was written for either agent variant. The table says "not run" for them. I did not fill them with anything.
  - The two agent variants need a few hundred requests in total at `--limit 3`, so they do not fit in one free day. Record mode keeps every finished response in `results/cache/llm/`, so the same command resumes where it stopped.
- **Bug found by the first live call, fixed.** The three tools that take no input (`describe_query`, `describe_context_pool`, `run_pfn`) had an empty schema. The free model answered every call with `{"additionalProperties": "{}"}`, which failed validation, so every agent run ended as bad output with zero tool calls. They now take an optional `reason` text that is accepted and not used. The naive agent's trace shows real tool calls after the change. The cassettes and result files of the failed attempt were deleted, because the tool schema is part of the cache key. Tests: no tool may have an empty input schema, and a tool accepts a `reason`.
- **Flaky test fixed.** The "model never sees ids" checks searched the raw message JSON for "2000", which also matched random digits in timestamps and ids once. They now remove timestamps and ids first (`shown_to_model` in `tests/agent_support.py`).
- **How to read the table.**
  - Run level: one verdict per trajectory, scored on the last window's label. These numbers are not comparable with the per-window PFN metrics.
  - `--limit 3` takes 3 scenarios of each kind (tipping and control), 3 seeds each, so a split has 18 runs. "Controls" are the control scenarios inside the test and out-of-prior splits.
  - The oracle cannot run on out-of-prior tipping families (not in its pool), so its out-of-prior AUROC is n/a. Its control rows come from a constant output (the ‡ note in the summary).
  - p95 latency: LLM calls as recorded plus PFN fit and predict time as first measured, both splits pooled.
  - pass^3 is the share of scenarios where the 3 seeds give the same decision. It measures consistency, not correctness.
  - 12 to 18 runs per split is a trial. The confidence intervals are wide and no ranking of the variants should be drawn from it.
- **Model.** `nvidia/nemotron-3.5-super-vl:free` stands in for the embedding model you named (see the entry above). Temperature 0 was applied in every recorded run (counted in the summary).
- **Tests:** 405 passed, 1 skipped; ruff clean. Nothing is committed. `results/cache/llm/` now holds cassettes that can be committed, but check them first.

### 2026-10-31 (all five variants run, pandas pin)

- **Asked:** run the two remaining variants after adding credits, update the files, say what changed. Then: wait for the 00:00 UTC reset, and when the 50 requests are used, do not retry; save and continue tomorrow.
- **Done:** all five variants now have both splits. The naive agent (`results/naive_agent_{test,ood}.json`) and the full harness (`results/agent_{test,ood}.json`) were added, and their rows in `results/summary.md` are filled from those files. The oracle has no out-of-prior tipping runs, as before.
- **How the full harness got finished.** The free tier cap (50 requests a day for all free models) was reached during `agent test`, and the run stopped at the first 429 without retrying (as asked). After the balance reached 10 credits, `is_free_tier` on the key turned `False` and a free-model request that the cap had refused went through. The same command then continued from the cached responses in `results/cache/llm/` and finished both splits.
- **Credits:** at 8 credits the account was still on the free tier. The threshold is 10.
- **Retries:** the free model gateway gives a 504 timeout now and then. A 504 is retried (the request usually works on the next try). A 429 for the daily cap is not retried.
- **pandas pinned to `>=3.0.6,<3.1`.** Windows Application Control blocked a file in pandas 3.1.0 (`pandas/_libs/testing.cp312-win_amd64.pyd`), and `uv sync` could not repair it. `uv.lock` was relocked. The classic, LLM-only and oracle results were produced on 3.1.0 and were not rerun on 3.0.6. The naive agent results were produced on 3.0.6. Tests: 405 passed, 1 skipped; ruff clean.
- **Reading the table (a trial, 18 runs per split).** Intervals are wide and none of this is a finding.
  - The naive agent (random context, no gate) scores as well as the full harness on AUROC. The trial cannot show a benefit of the agent's context choice or of the gate.
  - The gate blocked few of the BAD runs. With only a handful of BAD runs per split, no rate can be read from it.
  - The full harness has the lowest pass^3 (seed agreement) of all variants, and the highest latency and tool-call count. The cost of the agent loop is visible; its benefit is not.
  - All cells are in `results/summary.md`; none are typed here.
- **Offline replay check (network cut at the socket level, no API key, results written to a scratch folder).**
  - Classic, LLM-only and oracle rows came out the same. They do not call the agent code that was changed; classic and oracle were not rerun.
  - The naive agent row came out the same except the last digit of p95 latency, because the PFN time is measured again.
  - The full harness did **not** reproduce. 17 of 18 in-prior runs matched. One run (`hopf_L2_slow/3000`) took a different path, and one out-of-prior run stopped with a replay miss.
  - Cause: when the model asks for `choose_context` and `run_pfn` in one reply, PydanticAI runs the two tool calls at the same time, and the order is not fixed. `run_pfn` before `choose_context` fails, so the later requests differ from the recorded ones.
  - **Fixed.** The tools now run one at a time, in the order the model asked (`Tool(..., sequential=True)` in `agent/tools.py` and `agent/naive.py`). The cache key always records the flag as false, so earlier recordings stay valid (`agent/replay.py`). Tests: the flag is not in the key; every agent tool is sequential.
  - After the fix, the naive agent replays offline identically. A few full harness runs had followed a path that only existed because of the race, so `agent test` and `agent ood` were re-recorded (only the missing requests went live). The old files were replaced; the full harness row changed slightly (the summary has the new numbers). A final offline replay of both stages matches the result files run by run.
  - Tests: 410 passed, 1 skipped; ruff clean.
- **Agent checks for `interview.md`:** `scripts/agent_checks.py` writes `results/agent_checks.json` (runs per context strategy; control runs where the agent recommended alert, confidently, and where the final decision was alert). `interview.md` says to report both failures and to read the numbers from that file. Tests added in `tests/test_agent_checks.py`.
- Nothing is committed.

### 2026-10-31 (real_tipping: Daphnia, test only)

- **Asked:** add one small real dataset as a TEST-ONLY split `real_tipping` (Drake and Griffen 2010, Dryad doi:10.5061/dryad.q3p64). Real data never enters the context pool and never touches the frozen gate.
- **Done:**
  - Data: `data/raw/daphnia/` (git-ignored). Dryad needs a login, so you downloaded it. The archive has `data-and-code/`, not a folder called `Dataset`.
  - `docs/DAPHNIA.md`: what a row is, how populations map to trajectories, which series end in extinction and when, the surrogate label rule, and ten points I am unsure about. Counts are in `results/datasets/real_tipping_summary.json`, written by code.
  - `src/pfngate/data/daphnia.py` and `configs/real_tipping.yaml`: loader, same feature code as the simulators, windows of 24 weekly points (128 do not fit in at most 60), surrogate target `clip((t_extinction - t) / 100 days, 0, 1)`. Persisting populations have target 1 and are the real no-alarm controls. The settings were fixed from the series length before any result on this data.
  - Wiring: `build_dataset("real_tipping")` calls `assert_frozen()` before reading a row. New split flags `real` and `real_controls` in `run_pfn`, and `real` in `run_agent` and `baseline_llm`, all held out. `select_context` refuses a pool that holds real rows.
  - Tests: `tests/test_daphnia.py` (synthetic files). No real rows in the context pool, labels in [0, 1], a clear error when raw files are missing, the gate is checked before the data is read, and the tuner never mentions real data. Two existing tests that pin the split sets now include `real`. 419 passed, 1 skipped; ruff clean.
- **Found in the data:** five ids were started again on a later date and their census holds a first, dead population with a gap of months. The loader uses only the census from `Start`. Five ids with an `a` suffix have no census series and are left out. A few constant-treatment populations also go extinct and are treated as extinct (positives). All in `docs/DAPHNIA.md`.
- **Since then:** the variants were run on this split and the classic baseline is wired to it (see the next entry). The real windows are 24 weeks long and the context windows are 128 simulated points, so expect a shift between them. `--limit` does not shrink the real split; `--seeds-per-scenario` does (it keeps the first populations).
- Nothing is committed.

### 2026-10-31 (real_control: grid, and the single-command suite)

- **real_control (test only, false alarms only):** a slice of the Fingrid frequency data from Zenodo 3744121 (CC BY 4.0), 1 s values averaged to 60 s, one calendar day per trajectory, every window negative. It is normal operation, not a pre-blackout run, and has no AUROC. `docs/GRID.md` says so. Counts are in `results/datasets/real_control_summary.json`.
- **Licences:** the record has no per-file licence field. The licence is the `LICENSE.txt` in each area folder inside the zips: Fingrid and Nationalgrid are CC BY 4.0, TransnetBW has none. `scripts/fetch_grid.py` refuses any TransnetBW path in code and reads only the needed bytes of the zip with range requests. Source, access date, licence and SHA-256 are in `data/SOURCES.md` (also for Daphnia). The slice is the first stretch of five days that passed the stationarity rule (all values within 0.25 Hz of 50 Hz, at most 100 missing seconds a day); it starts on 1 January 2019. The rule was relaxed once, before any window was built, because ordinary days already reach 0.16 Hz.
- **Rules kept:** `assert_frozen()` runs first, the split is never in the context pool (`select_context` refuses it), and the tuner never mentions it. Tests: `tests/test_grid.py`.
- **Wiring:** the split is known to `build_dataset`, `run_pfn`, the LLM evals, the classic baseline (`data/load.py`) and `make_summary.py` (new "Real data (test only)" table).
- **`scripts/run_all.py` and `make full` / `make fast`:** ruff and pytest, the gate hash check, every variant on val, test, ood, real and real_control (replay when the cache has entries, record when a response is missing, never retried on the daily cap), the summary, and `results/run_manifest.json`. `--fast` uses 5 scenarios per family with 1 seed and writes to `results/fast/`; `--all` removes the limit. `make` is not installed on this machine, so the recipe was run directly. Tests: `tests/test_run_all.py`.
- **Bug found by the first full run and fixed:** a replay miss that record mode then filled was counted as a failed stage, so the manifest said `complete: false` and the exit code was 1. A stage now fails only if its last attempt failed.
- **Run:** the first full run recorded the missing val, real and real_control responses live; the second run replayed everything. Numbers: `results/summary.md` and `results/run_manifest.json`. Not typed here.
- **Caveats:** the manifest's wall-clock is from the replay run; latencies are the recorded live ones. The default profile has 3 scenarios x 3 seeds (18 runs per split), `--fast` has 10. Cost is 0 because the models are free-tier. Nothing is committed; `results/cache/llm/` and the per-population extinction days in `results/datasets/real_tipping_summary.json` should be looked at before a commit.

## Decisions

- Model version: v3.5. Keep the model version in the results JSON for every run. Never redistribute the weights.
- Features are scale-free only. No raw variance or raw level, so the model cannot tell systems apart by their units or noise level.
- The classic EWS baseline stays at K = 16 as it was configured before the first run. It is reported as weak and is not retuned after seeing test results.
- p_critical comes from the full predictive distribution when the backend exposes it, and from interpolating the quantiles otherwise. The method is recorded in every output.
- family_matched is an oracle. It is reported only as an upper bound, labelled as an oracle everywhere, and never as a deployable method.
- Alarm thresholds for false-alarm rates are set from validation controls only, in the same way for the baseline and every PFN run. These are not the gate's alert thresholds, which live in `configs/gate.yaml` and are not tuned yet.
- Runs on this CPU use `--max-context 200` and 3 seeds per scenario until a faster machine is available. The summary states the reduction.
- The gate is pure Python with frozen thresholds from configs/gate.yaml. Held-out splits refuse to run until that file is tuned on validation runs, frozen by hash and committed.
- The ood_share check measures distance to the whole pool, not to the selected context, so context selection cannot make a query look in-distribution.
- Gate tuning used 3 validation seeds per scenario, not the 10 of decision 4 in docs/GATE.md, because of the refit time. The tuner refuses fewer than 30 runs.
- The oracle's validation runs were briefly part of the tuning set (2026-10-28, at your request). They were taken out again the same day (docs/GATE.md, decision 9).
- When tuning stops, the gate stays provisional and no held-out gated run is made. Thresholds are not loosened after seeing validation results without a new decision written in docs/GATE.md.
- The interval width cap is an advisory: it is reported for every run and never blocks (docs/GATE.md, decision 7).
- knn_balanced keeps 25% of the context for near-tipping rows. The share is fixed by design (docs/GATE.md, decision 6) and is not tuned on results.
- The gate tuner maximises F1 with no minimum precision (docs/GATE.md, decision 12, 2026-10-28, at your request). The gate is tuned on random and knn validation runs and frozen. `near_tipping_support` and the width cap only report.
- The LLM agent only recommends. The decision of a run comes from the gate and the decision mapping, as the decision of the last window (abstain if the gate failed). Its tools return summaries from a server-side store and never see a label, a family or a trajectory id. Caps: 6 tool calls and 3 contexts, enforced in code.
- Agent settings (model names, temperature, limits, approximate prices) are in `configs/agent.yaml`, not in code. The models run through OpenRouter (`openrouter:<model id>`, key `OPENROUTER_API_KEY` in `.env`). Temperature 0 is requested on every call and reaches the request; the run record says whether the provider applied it. The model must be a chat model with tool calling, not an embedding model.
- LLM responses are recorded to and replayed from `results/cache/llm/` (the only part of `results/cache/` that is committed). A replay never calls the network and never needs a key. A response is written once and never changed.
- The LLM variants (agent, naive agent, LLM-only baseline) are scored at run level, on the label of the last window, with the same metrics. Held-out LLM evaluation needs a tuned, frozen and committed gate, for every variant.
- The summary's variants table is read only from `results/<variant>_{test,ood}.json`. A variant without a file is "not run"; a failed or rate-limited run is never written as a result.

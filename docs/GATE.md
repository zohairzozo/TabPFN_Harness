# Gate

Status: **approved (2026-10-27) with the five recommended decisions, and implemented** in `src/pfngate/gate/`. Tuning stopped in rounds 1 to 3, because no grid point reached the minimum precision. In round 4 (2026-10-28) the gate went back to round 2's checks, the precision floor was removed (decision 12), and `configs/gate.yaml` was **tuned on validation, frozen and committed before any held-out run**. The held-out results are in `results/summary.md` and `results/figs/reliability.png`.

**Revision 2** is at the end of this file. On 2026-10-28, four of its decisions were approved and implemented, and the sections above are updated for them:
- 6: the `knn_balanced` context;
- 7: the width cap only reports;
- 8: the `near_tipping_support` check;
- 9: the oracle leaves the tuning set.

Decisions 10 and 11 are still open.

**Round 4 (2026-10-28, at your request), the configuration that gets tuned and frozen:**
- **Back to round 2's blocking checks and tuning set.** Of the three rounds, round 2 gave the highest precision and recall on validation.
  - `near_tipping_support` (decision 8) is now an advisory, like the width cap: reported for every run, never blocking.
  - `knn_balanced` (decision 6) stays as a context strategy, but it is not tuned on and not in the gate's applicability block.
  - The tuning set is random and knn again.
- **Decision 12: no minimum precision.** The tuner picks the grid point with the highest F1 of blocking BAD runs. Ties go to higher precision, then lower abstention on good runs, then the looser point.
  - This replaces decision 3 (precision at least 0.6, highest recall).
  - It is a loosening made after seeing three stopped rounds, and the report says so.
  - The thresholds are still tuned on validation only, and frozen and committed before any held-out run.

The gate decides whether one model run may be trusted. It is plain Python. It reads frozen thresholds from `configs/gate.yaml`, it never sees the ground truth of the query trajectory, and nothing else can change what it returns. An LLM agent may read a gate result and recommend something. It cannot edit the result, the thresholds, or the decision.

Labels are defined in `docs/LABELS.md` and splits in `docs/SPLITS.md`.

## Terms

- **Run**: one query trajectory, one context selection, one backend fit, and predictions for all of its windows. The run is the unit the gate judges. All windows of a run share one gate decision.
- **Pool**: the context split, seed set A. Context rows are always drawn from it.
- **Validation runs**: runs whose query trajectory is in seed set B (in-prior validation, tipping and control trajectories). Gate thresholds are tuned on these and nothing else.
- **Held-out runs**: runs on set C (in-prior test), set D (out-of-prior test) and any real data.

## Ground truth: what a bad run is

This is used only to tune and to score the gate, and only in simulation. The gate itself never sees it.

**A run is BAD if the realised coverage of its 90% interval is below 0.80.**

- The 90% interval of a window is `[q05, q95]` from the backend output.
- Realised coverage is the share of the run's windows whose regression target lies inside that interval, ends included, with a tolerance of 1e-6. The tolerance matters because the target is clipped to [0, 1], so it has point masses at exactly 0 and 1, and a predicted q95 of 0.9999999 should still count as covering 1.
- **Which windows:** all kept windows of the run, meaning every pre-tip window that has a regression target, including the middle band that the binary task drops. For a control run, that is every window, and each has target 1. Coverage is a statement about the regression target, so there is no reason to drop windows that only the binary task excludes. **(Decision 1 below.)**
- The 0.80 is fixed here. It is part of the definition of BAD, it is not a gate threshold, and it is never tuned.

### Why this is a sensible release criterion

- **It tests the promise the model makes.** The model releases an interval and a probability (`p_critical` comes from the same predictive distribution). If the 90% interval covers far less than 90%, the model is overconfident and its stated risk cannot be trusted, whatever its AUROC. An overconfident early-warning model is the dangerous kind: it is sure the system is safe when it is not.
- **It is defined per run.** The gate decides per run, so the label it is scored against should be per run too. An aggregate calibration number over many trajectories would hide a few very bad runs.
- **It does not depend on the alarm threshold or the binary label gap.** It uses the continuous target, which is known exactly in simulation.
- **0.80 rather than 0.90 leaves room for chance.** Windows overlap heavily (window 128, stride 16), so a run of about 300 windows has far fewer independent windows. A well-calibrated run will often land below 0.90 by chance. The 0.10 margin keeps such runs out of BAD.

### Where it is weak

- **Coverage alone rewards wide intervals.** An interval of [0, 1] always covers. That is why the gate also reports whether the interval is wider than a cap. This is about usefulness rather than BAD, and since decision 7 it is an advisory that does not block.
- **Coverage is one-sided here.** An interval that is too wide (coverage near 1.0) is not BAD by this definition.
- **The easy windows dominate.** Most windows are far from the transition, where the target is near 1 and easy to cover. A run can be "good" while missing every window near the transition, which are the windows that matter for warning. The gate report will also show coverage on positive windows (target at most 0.2) as a secondary number, not as a criterion.
- **The label is noisy.** With few effective independent windows per run, the same model can be BAD on one seed and fine on the next. Some disagreement between gate and label is unavoidable.
- **Clipped point masses.** Controls have target exactly 1 everywhere, and delayed-tipping windows have target exactly 0. Coverage of a point mass depends on whether the predicted quantile reaches exactly 0 or 1, which is a property of the backend's output bins as much as of calibration. The tolerance helps but does not remove this.
- **Simulation only.** Real data has surrogate labels at best (`docs/LABELS.md`), so BAD cannot be computed on real data, and the gate metrics below cannot be reported there. The gate still runs on real data, because none of its checks needs query labels.

## What the gate may see

The gate takes a `GateInput` with:

- run settings: backend, model version, package version, strategy, `max_context`, data version;
- the context: row ids, size, and the context targets (these come from set A, never from the query);
- the query features (not the query targets);
- the backend output for the query: quantiles, `p_critical`, metadata (repairs, `p_critical_method`);
- the result of the context calibration refits (below).

**`GateInput` has no field for the query targets or labels.** A test checks the field list, so the rule "the gate decides without ground truth" is enforced by the type, not only by convention.

## Checks

Each check returns a value, a threshold, a comparison, a pass flag and a one-line reason. The gate passes only if every check passes.

Some thresholds are **fixed by design** in this file and are not tuned. The rest are **tuned on validation runs** using the protocol below. The reasons are given per check.

| # | Check | Value | Passes if | Threshold source |
|---|---|---|---|---|
| 1 | `min_context_size` | context rows | value ≥ threshold | fixed: 100 |
| 2 | `ood_share` | share of query windows far from the pool | value ≤ threshold | distance cut fixed from set A; share limit tuned |
| 3 | `interval_width_cap` | median of `q95 - q05` over query windows | value ≤ threshold | fixed: 0.6 (decision 2); **advisory, does not block** (decision 7) |
| 4 | `context_calibration` | 90% coverage on held-out context trajectories | low ≤ value ≤ high | tuned |
| 5 | `context_not_degenerate` | rows in context with target ≤ 0.2, and with target ≥ 0.6 | both ≥ threshold | fixed: 10 each |
| 6 | `output_integrity` | share of query rows with repaired quantiles; `p_critical_method` | share ≤ threshold and method is `full_distribution` | fixed: 0.01 |
| 7 | `near_tipping_support` (decision 8) | share of query windows whose 5th nearest near-tipping context row is beyond a cut | value ≤ threshold; fewer than 5 near-tipping rows fails | cut fixed from set A; limit fixed at 0.5; **advisory since round 4, does not block** |

The gate runs the blocking checks in the order 1, 2, 4, 5, 6. Checks 3 and 7 are advisories, reported apart from them.

### 1. min_context_size

The number of context rows. Below a floor, the PFN has too little data to condition on, and the calibration refits (check 4) become meaningless.

- **Fixed at 100, not tuned.** At the current settings every validation run has the same context size (the cap), so validation data carries no information about where this threshold should be. Tuning it would pick an arbitrary value.

### 2. ood_share

The share of query windows whose feature vector is far from the in-prior data.

**Recommendation: measure distance to the whole pool (set A), not to the selected context.** The knn strategy and the agent both pick context rows close to the query. Measured against the selected context, every query would look in-distribution because the context was chosen to look like it. Measured against the pool, the check is the same whatever the strategy or the agent chose, so it cannot be gamed by context selection.

How:

- Standardise the 8 features with the pool mean and standard deviation.
- For each query window, compute the distance to its 10th nearest pool row.
- A window is **far** if that distance is above the 99th percentile of the same statistic over pool rows, where each pool row's neighbours exclude its own trajectory (leave one trajectory out).
- The value is the share of far windows in the run.

The distance cut uses set A only and no labels, so it is fixed when the gate is built and stored in `configs/gate.yaml`. The share limit is tuned on a small grid.

- **Why k-nearest-neighbour distance and not max |z| or Mahalanobis:** some features are bounded (the Kendall taus) and some are heavy-tailed (kurtosis). Both z-scores and Mahalanobis distance assume a roughly elliptical shape and will flag ordinary heavy-tail windows. A percentile of nearest-neighbour distances needs no shape assumption. It reuses the KD-tree code from the knn strategy. Max |z| is the fallback if this turns out too slow.
- **Weakness:** validation runs are in-prior, so they contain almost no far windows. The share limit cannot learn much from validation data, and this check is exactly the one that matters on out-of-prior data. That is why its grid has no "off" value and its loosest value is 0.3.

### 3. interval_width_cap

The median width of the 90% interval over the run's query windows.

**Recommendation: fix this cap by usefulness, not by tuning against BAD.** Wide intervals almost always cover, so they are rarely BAD. If the tuner optimised this cap for blocking BAD runs, it would turn it off. Its job is different: an interval wider than the gap between the binary zones (target ≤ 0.2 is positive, ≥ 0.6 is negative) cannot tell a window near the transition from a safe one.

- **Proposed: 0.6** on the [0, 1] target. **(Decision 2.)**
- ~~Runs blocked by this check are counted separately in the gate metrics ("blocked as not useful"), so they do not distort precision on BAD runs.~~ Replaced by decision 7.
- **Since decision 7 (2026-10-28) this check is an advisory.** It is computed for every run and stored in `GateResult.advisories`, apart from the blocking `checks`. It never changes `passed` or the decision. The config must say `role: advisory`, so a file written before decision 7 does not load. The summary reports how many runs are over the cap.

### 4. context_calibration

Coverage measured on context data, where the labels are allowed.

- Split the run's context rows into 5 groups **by trajectory**, never by window. Groups are assigned deterministically from the sorted trajectory ids and the config seed.
- For each group: fit the backend on the other four groups, predict the held-out group, and record whether each held-out target is inside `[q05, q95]`.
- The value is the pooled coverage over all held-out rows.
- It passes if `low ≤ value ≤ high`. `low` catches overconfident intervals. `high` catches intervals so wide that they always cover.
- If the context has fewer than 5 distinct trajectories, the check fails with the reason "too few trajectories to calibrate".
- Refits are cached by the hash of the fold's context, like the main fit.

**Weaknesses:**

- It measures calibration on the context distribution, not the query distribution. That is fine in-prior. Out of prior it can pass while the query is badly miscalibrated, which is why it is paired with check 2.
- Each refit uses 80% of the context, so its intervals are slightly wider than the real fit's.
- It costs 5 extra fits and predicts per run, each on a small held-out set. The real cost will be measured and written to the trace, not estimated here.

### 5. context_not_degenerate (recommended addition)

The context must contain at least 10 rows with target ≤ 0.2 and at least 10 rows with target ≥ 0.6.

- **Reason:** the oracle runs on controls are the warning. Their context target was constant, TabPFN took its constant-target path, and every output was the same constant with zero width (`results/summary.md`, marked ‡). Such a run has perfect coverage on controls and says nothing. A context with no rows near the transition cannot express "close to tipping" at all.
- Fixed, not tuned: it is a structural requirement, not a matter of degree.

### 6. output_integrity (recommended addition)

At most 1% of query rows may have needed a quantile repair, and `p_critical_method` must be `full_distribution`.

- **Reason:** the adapter repairs crossing quantiles and falls back to interpolation or a step function when the full distribution is missing (see `src/pfngate/pfn/adapter.py`). Either means the backend output is not what the gate's other checks assume. This check makes those fallbacks visible as a gate failure instead of a line in the metadata.
- Fixed, not tuned.

### 7. near_tipping_support (revision 2, decision 8)

Does the context hold near-tipping examples that look like this query?
- **Near-tipping rows:** context rows with target ≤ 0.2, the binary positive zone.
- **Distance:** for each query window, the distance to its 5th nearest near-tipping context row, in the standardised features of `ood_share`.
- **Cut:** computed once from set A. For a sample of pool rows, take the distance to the 5th nearest near-tipping pool row of another trajectory, then take the 99th percentile. The number is stored in `configs/gate.yaml`.
- **Value:** the share of query windows beyond the cut.
- **Rule:** the run fails if the share is above the limit. A context with fewer than 5 near-tipping rows fails outright, with no value.
- **Why:** the stopped tuning showed that BAD runs cover their context well and fail near tipping on the query (revision 2). Context calibration cannot see that. This check asks about the context and the query together, and uses no query label.
- ~~**Tuned:** limit grid {0.1, 0.2, 0.3, 0.5}, with a floor of 0.5 in code.~~
- **Since round 4 it is an advisory:** reported, never blocking, with the limit fixed at 0.5. In round 3 it blocked almost every random and knn run, because its cut was fitted on the density of the whole pool and not on contexts of the size the gate sees.
- **Contrast with check 2:** `ood_share` measures distance to the whole pool so that context selection cannot game it. This check measures distance to the selected context on purpose, because the question is what the context holds.

**Not added:** a check on query `p_critical` spread or on the agent's reasoning. The first would mix the alarm decision into the trust decision. The second would put an LLM inside the gate.

## GateResult

A frozen pydantic model (`frozen=True`, tuples instead of lists):

- `run_id`, `trajectory_id`, `strategy`, `context_hash`;
- `checks`: a tuple of `CheckResult(name, value, threshold, comparison, passed, reason, source)`, where `source` is `fixed` or `tuned`;
- `passed`: true only if every check passed;
- `reasons`: the reasons of the failed checks, in check order;
- `gate_config_sha256`: the hash of the `configs/gate.yaml` that produced it;
- `gate_code_version`: the package version.

### Decision per window

The gate decides per run. The alert decides per window inside an accepted run.

- gate fails → **abstain** for every window of the run;
- gate passes and `p_critical ≥ alert_threshold` → **alert**;
- otherwise → **no_alert**.

`alert_threshold` is frozen in `configs/gate.yaml`, one value per strategy. It is set on validation control runs **that the gate accepted**, with the existing rule (at most the target share of their windows alarm, `alarm_threshold_for_rate` in `metrics.py`). One value per strategy, because the knn and random scores already differ and knn's threshold did not transfer (`results/summary.md`).

### Nothing may change a GateResult or a threshold

Python cannot make an object truly immutable, so this is defence in depth:

1. `GateResult` and the loaded thresholds are frozen models. Assignment raises.
2. `configs/gate.yaml` is loaded only through one function that checks its hash first (below). There is no API to set a threshold at runtime.
3. The agent module never imports the gate tuner or the gate config writer. A test scans the agent package's imports and fails if it does.
4. The agent receives a gate result as JSON text, after the decision is made. Its output is a recommendation in the trace and is never read back by the gate or the decision function.
5. The report recomputes every `GateResult` from the stored check values and the frozen thresholds and fails if anything differs. Editing a stored result therefore breaks the report.
6. One context selection per trajectory counts. The agent may not retry context selection until the gate passes, because "retry until it passes" would turn the gate into a filter that the agent can game. If retries are added later, every attempt is logged and the gate metrics count the first attempt.

## Tuning protocol

Tuning uses validation runs only. The grids, the objective and the minimum precision are fixed here, before any coverage is computed, so the choices cannot drift after looking at results.

1. **Runs.** Run strategies `random` and `knn` on all validation trajectories (round 3 added `knn_balanced`; round 4 took it out again), with the context calibration refits. That means `run_pfn --split val` for tipping runs and `--split val_controls` for controls. Pool the strategies into one tuning set, with one set of thresholds.
   - **Changed back on 2026-10-28 (decision 9):** the oracle is not in the tuning set, and the tuner does not open its files. Its runs are still gated and reported, labelled as oracle. The two notes below describe the short-lived earlier change.
   - **Changed on 2026-10-28, at your request:** the oracle's validation runs are included in tuning. The approved draft excluded them because the oracle is not a deployable method. They are still validation data, so this is not leakage, but the thresholds are partly fitted on runs that never happen in deployment.
   - The oracle's controls have a constant context, so `context_not_degenerate` blocks every one of them. Those runs are good (they cover their constant target), so they count against precision.
2. **Labels.** Compute realised coverage and BAD for each validation run.
3. **Stop rule.** If there are fewer than 30 validation runs, refuse to tune, write nothing, and say how to make more (added on 2026-10-28). If there are fewer than 20 BAD runs, or BAD runs are more than half of all runs, stop and report. Do not tune.
   - Too few BAD runs means precision and recall are noise.
   - A majority of BAD runs means the model is miscalibrated, which is a model problem a gate should not paper over.
4. **Grid** (the tuned thresholds only):
   - `ood_share` limit: {0.05, 0.1, 0.2, 0.3};
   - `context_calibration` low: {0.70, 0.75, 0.80, 0.85};
   - `context_calibration` high: {0.97, 0.99, 1.00}.

   That gives 48 combinations, kept small on purpose because the tuning set is small. (Round 3 added a `near_tipping_support` limit and 192 points; round 4 went back to 48.)
5. **Objective (decision 12, round 4).** Choose the combination with the highest F1 of blocking BAD runs, with no minimum precision. (Until round 3: precision at least 0.6, then the highest recall.) Every run counts. (Before decision 7, runs blocked only by the width cap were left out.)
6. **Ties** are broken by:
   - higher precision (added with decision 12);
   - lower abstention rate on good runs;
   - then the looser threshold, in the order the grid lists them.

   The procedure is deterministic.
7. **No feasible point.** If no combination has a defined F1 (nothing blocked, or no BAD runs), stop and report. (The old rule, "do not lower the minimum after seeing the data", was overridden at your request in round 4.)
8. **Optimism estimate.** Repeat steps 4 to 6 inside a 5-fold split of the validation runs. Group by trajectory, so a trajectory's random and knn runs stay in the same fold. Report out-of-fold precision and recall next to the in-sample numbers. The in-sample numbers are optimistic by construction.
9. **Alert thresholds.** Set one per strategy on accepted validation control runs (see above).
10. **Write and freeze**, then ask before committing (next section).

## Freezing

- The tuner writes `configs/gate.yaml` with:
  - every threshold, and whether it is fixed or tuned;
  - the `ood_share` distance cut;
  - the alert thresholds;
  - the BAD definition (0.80, the 90% interval);
  - an **applicability block** naming the settings it was tuned for: backend, model version, package version, strategies, `max_context`, and data version.
- It then writes the SHA-256 of the file's bytes to `configs/gate.frozen.sha256`.
- Line endings: this repository has `core.autocrlf=true` and no `.gitattributes`. A checkout could rewrite the YAML's line endings and change its hash. Add `configs/gate.yaml -text` to `.gitattributes` before freezing.
- The tuner refuses to overwrite an existing frozen pair. Re-tuning means deleting the hash file by hand and writing a PROGRESS.md entry saying why. Any held-out result computed under the old hash is void and is reported as such. Results record the hash they used, and `make_summary.py` flags results with a stale hash.
- The tuner refuses any run whose query trajectory is not in set B.

### Refusing to evaluate

Every evaluation entry point refuses to start on set C, set D, held-out controls or real data if any of these holds:

- `configs/gate.yaml` or the hash file is missing;
- the hash does not match;
- `configs/gate.yaml` has uncommitted changes in git (the rule in `docs/SPLITS.md`);
- the run settings differ from the applicability block. A gate tuned at `max_context` 200 says nothing about runs at 2000.

Validation runs need none of this, because they are the tuning data.

### Tests that fail if someone loosens the gate

- The SHA-256 of `configs/gate.yaml` equals `configs/gate.frozen.sha256`. The same hash is also written as a literal in the test file, so a loosening needs three edits that all show up in one diff.
- **Code-level floors.** The gate refuses to load a config looser than hard limits written in code, and a test pins them:
  - `min_context_size` ≥ 50;
  - `ood_share` limit ≤ 0.3;
  - `near_tipping_support` limit ≤ 0.5, and its near-tipping bound must equal the positive zone;
  - `interval_width_cap` ≤ 0.8;
  - `context_calibration` low ≥ 0.6;
  - degenerate-context floor ≥ 5;
  - repair share ≤ 0.05.
- For each check, a synthetic input just inside the threshold passes and one just outside fails.
- The gate passes only when every check passes (a property test over random check outcomes).
- `GateResult` assignment raises, and the same input gives byte-identical JSON.
- `GateInput` has no target or label field.
- Evaluation entry points refuse on a missing file, a changed hash, a dirty file, or mismatched settings. The tuner refuses non-B runs and refuses to overwrite a frozen pair.
- The agent package does not import the tuner or the config writer.

## Gate metrics

Reported per split (validation in-sample, validation out-of-fold, in-prior test, out-of-prior test), per strategy, and per family. Confidence intervals come from a bootstrap over trajectories.

- **Precision** of blocking: BAD runs among blocked runs.
- **Recall** of blocking: blocked BAD runs among all BAD runs.
- **Abstention rate:** blocked runs among all runs, also split by the check that failed. Runs over the width cap are counted on their own; since decision 7 they are not blocked.
- **Coverage:** mean realised 90% coverage, and the share of BAD runs, on accepted runs against all runs. Coverage on positive windows is shown as the secondary number.
- **AUROC** of `p_critical` (same balanced AUROC and trajectory bootstrap as before) on accepted runs against all runs.
  - These are different populations, so a higher AUROC on accepted runs can partly come from abstaining on hard trajectories.
  - The report shows the number of trajectories behind each AUROC and states this caveat.
- **False alarms:** window false-alarm rate on accepted control runs, at the frozen alert threshold.

The expected pattern, to be confirmed rather than assumed: low abstention in-prior and higher abstention out of prior. A gate that abstains equally everywhere is not detecting anything.

## Order of work after approval

1. Implement the gate, the checks, `GateResult`, the tuner, the freeze and the refusal logic. Test them with the fake backend. No real data yet.
2. Run validation runs with calibration refits, compute BAD on validation only, and apply the stop rule.
3. Tune, estimate optimism, write the frozen pair, and ask you before committing it.
4. Only after that commit: gate the in-prior test, out-of-prior test and held-out control runs, and write the gate metrics and report.

## Risks

1. **Too few validation runs.** At the current reduced settings (3 seeds per scenario), set B gives 81 runs per strategy (54 tipping and 27 control). The number of BAD runs among them is unknown. **Recommendation: run all 10 validation seeds before tuning (decision 4).** That is about three times the current validation run time; the times are in `results/summary.md`.
2. **Validation has no out-of-prior runs.** The gate is tuned where it matters least. Check 2 cannot be tuned meaningfully on in-prior data, and the gate may be blind to the failure modes that only appear out of prior. The out-of-prior test is the honest measure of this.
3. **Test and out-of-prior predictions already exist.** The PFN runs on sets C and D were made and their AUROC was seen before this gate existed. `docs/SPLITS.md` forbids tuning gate thresholds on C or D, and that has not happened. No per-run coverage or BAD label has been computed on C or D, and none will be until the gate is frozen. The tuner reads no cache entry or result file outside set B.
4. **Overfitting with several thresholds.** Mitigated by:
   - fixing four checks by design;
   - a pre-registered grid of 48 points;
   - the out-of-fold estimate.
5. **Dependent runs.** Pooling random and knn puts each validation trajectory in the tuning set twice. Bootstraps and folds group by trajectory.
6. **A noisy label** (see "Where it is weak"). It caps how well any gate can score.
7. **Context calibration is not query calibration.** This matters most out of prior.
8. **The hash guards against accidents, not intent.** Someone can edit the YAML, the hash file and the test literal together. The git history is the audit trail.
9. **Settings drift.** A gate is valid only for the backend, model version and `max_context` it was tuned on. The applicability block enforces this. It also means a full run at a larger context needs a new tuning on validation.
10. **Cost.** Context calibration multiplies backend calls per run. It will be measured on validation first. If it is too slow, the fold count drops from 5 to 3, decided on validation and before freezing.

## Decisions for you

1. **Windows used for BAD coverage:** all kept windows, including the middle band (recommended), or binary-labelled windows only.
2. **Interval width cap:** fixed at 0.6 by usefulness (recommended), or tuned like the others.
3. **Minimum precision for tuning:** 0.6 (recommended).
4. **Validation size:** run all 10 seeds of set B before tuning (recommended), or tune on the current 3 seeds and accept wide uncertainty.
   - **Not followed (2026-10-28):** tuning used 3 seeds. The context calibration refits made 10 seeds cost several hours on this CPU. The tuner refuses fewer than 30 runs and says how to add seeds.
5. **ood_share reference:** distance to the whole pool (recommended), or distance to the selected context as first written.

## Revision 2 (proposed 2026-10-28; decisions 6 to 9 approved and implemented)

Decisions 7 and 9 were approved and implemented on 2026-10-28, and decisions 6 and 8 later that day. Decisions 10 and 11 are open.

**Outcome of round 3 (2026-10-28): tuning stopped again.**
- knn_balanced has no BAD validation runs, but it ranks windows much worse than knn.
- `near_tipping_support` blocks almost every random and knn run, because its cut was fitted on the density of the whole pool, not on contexts of the size the gate sees.

The details are in PROGRESS.md and `results/summary.md`.

**The design was informed by validation seeds 2000 to 2002, and tuning uses the same seeds.** Its in-sample and out-of-fold numbers are therefore optimistic. Decision 10, with its fresh-seed report, is the remedy, and it is still open.

### What the stopped tuning showed

This uses validation seeds 2000 to 2002 only, read from `results/gate_tuning.json` and `results/pfn_*_val.json`. I did not look at any test, out-of-prior or held-out control result for this revision. No number is typed here. Proposal 7 writes them to a results file by code.

- **Almost all BAD runs are fold trajectories** under the random or knn context. The oracle's runs are almost never BAD, and its context holds the query's own family.
- **BAD runs fail near tipping.** They cover the windows far from tipping and miss most windows with target ≤ 0.2. Those are the windows an early-warning model exists for.
- **The cause is the context.** A BAD run's context has few rows near tipping. On these seeds, the smaller count in `context_not_degenerate` separates BAD from good runs well, but its fixed threshold of 10 is far too low to make use of that.
- **Context calibration cannot see this.** The held-out context trajectories come from families that the rest of the context covers well, so the refits look fine while the query fails.
- **The width cap blocks almost all random and knn runs** (median 90% interval wider than 0.6).
- **The oracle hurts precision.** Every oracle control run is blocked by `context_not_degenerate` (constant context) and counts against precision.

### Proposals

**1. A new deployable context strategy, `knn_balanced`.** This is outside the gate, and it is the main fix: it goes after the cause, not the symptom.
- It is knn as now, except that a fixed share of the context is reserved for pool rows near tipping (target ≤ 0.2). Those rows are chosen nearest to the query windows in the same standardised feature space.
- **Share: 25% of `max_context`, fixed by design, not tuned.**
- It uses pool targets only, which a context may use. It never uses the query's family or labels, so it is deployable. The oracle is not.
- **Risk:** near-tipping rows from other families may raise p_critical on controls. The false-alarm rate on validation controls measures this, and the alert threshold is set there.
- random and knn stay as baselines and are gated too.

**2. The width cap reports, but no longer blocks.**
- (a) **Keep it blocking at 0.6.** The gate's answer for TabPFN at max_context 200 is then "no-go: not useful", whatever the calibration. This is honest, but it ends the gate demo here.
- (b) **Raise it to 0.8**, the code floor. This would be chosen after seeing the widths, so it is a post-hoc loosening.
- (c) **Recommended: compute it and report it as an advisory in every `GateResult`, but do not block on it.**
  - The gate answers the trust question: will the interval cover? Usefulness is answered by AUROC and the false-alarm rate, which the report already shows next to the gate.
  - p_critical comes from the full distribution, so wide intervals do not stop the scores from ranking windows.
  - Precision and recall then count every run, and the "blocked only as not useful" exclusion goes away.

**3. A context-vs-query check, `near_tipping_support`, tuned.** It asks directly what the BAD runs get wrong: does the context hold near-tipping examples that look like this query?
- **Value:** the share of query windows whose 5th nearest near-tipping context row is farther than a cut. Distances use the standardised features of `ood_share`.
- **Cut:** computed once from set A, like the `ood_share` cut. For every pool row, take its distance to the 5th nearest near-tipping pool row of another trajectory, then take the 99th percentile.
- **Rule:** fail if the share is above the limit. A context with fewer than 5 near-tipping rows fails outright.
- It uses no query label. It needs no new TabPFN fit, so it can be recomputed on the existing validation runs.
- **Grid for the limit:** {0.1, 0.2, 0.3, 0.5}.
- Its predictive value has not been measured. It will be measured on the new seeds (proposal 5).
- **Alternative:** tune the count threshold of `context_not_degenerate` on {10, 20, 30, 50}.
  - It is simpler, and on seeds 2000 to 2002 it already separates BAD runs. But that was seen in-sample, so it is optimistic.
  - With `knn_balanced` the count always passes by construction. It would then guard only random and knn, while `near_tipping_support` still tests every strategy.
- `context_not_degenerate` stays fixed as a guard against constant contexts.
- `context_calibration` stays as it is. It is no longer expected to catch BAD runs, but it still guards against an overconfident backend (it caught the overconfident fake in the tests).

**4. The oracle leaves the tuning set.** This reverts the 2026-10-28 change. Its control contexts are constant by construction, so it adds blocked good runs and nothing else. Its runs are still gated and reported, labelled as oracle.

**5. More seeds, and a number from seeds the design never saw.**
- Revision 2 was designed by looking at seeds 2000 to 2002, so its tuning on those seeds is doubly optimistic.
- Run all 10 seeds of set B: drop `--seeds-per-scenario 3` for the validation runs. `configs/scenarios/_common.yaml` already lists 10.
- Tune on all 10 seeds. Report precision and recall twice:
  - out of fold on all runs;
  - out of fold on runs of seeds 2003 to 2009 only. Those runs played no part in the design, so this is the honest validation number.
- Cost: several hours on this CPU, because the new seeds need fits as well as refits. They run in the background.

**6. One more design round, then the result stands.** If the revision 2 tuning also stops, there is no revision 3 on this data. The report then says that TabPFN at this context size cannot be gated to the required precision on these families. This stops the design from being fitted to validation by trying again until it passes.

**7. Diagnostics written by code.** A script, `scripts/gate_diagnostics.py`, writes `results/gate_diagnostics.json`, and `make_summary.py` reads it. It holds:
- BAD runs by family and strategy;
- coverage on near-tipping and far windows;
- each check's value against BAD (AUROC);
- each check's value split by seed group (2000 to 2002 against 2003 to 2009).

### Unchanged

- The BAD definition. Changing the label after seeing which runs fail would move the goalposts.
- The minimum precision of 0.6.
- `ood_share`, the floors, and the freezing and refusal rules.
- The 30-run minimum.

### Order of work after approval

1. The diagnostics script (proposal 7), first on the existing runs.
2. `knn_balanced`, with tests on the fake backend.
3. `near_tipping_support`, the advisory width cap and the new tuning set, with tests.
4. Rebuild the provisional `configs/gate.yaml` for strategies random, knn and knn_balanced.
5. In the background: validation runs for seeds 2000 to 2009, tipping and controls, for random, knn and knn_balanced, with `--gate`. This uses the TabPFN cache.
6. Tune. If it stops, report it and stop (proposal 6).
7. Freeze, ask you before committing, then run the held-out gated runs, the summary and PROGRESS.md.

### Decisions for you (revision 2)

6. **Context strategy:** add `knn_balanced` with a fixed 25% near-tipping share (recommended), or no new strategy. **Approved and implemented** (`src/pfngate/pfn/context.py`; the share is `context.balanced_near_share` in `configs/pfn.yaml`, and the near-tipping bound is `critical_threshold`).
7. **Width cap:** advisory only (recommended), keep blocking at 0.6, or raise to 0.8. **Approved and implemented: advisory only.**
8. **Context-vs-query check:** `near_tipping_support`, tuned (recommended), or tune the count in `context_not_degenerate`. **Approved and implemented: `near_tipping_support`, check 7.**
9. **Oracle in tuning:** take it out (recommended), or keep it. **Approved and implemented: taken out.**
10. **Seeds:** all 10, with the fresh-seed report (recommended), or stay at 3.
11. **Rounds:** one more design round only (recommended).
12. **Objective without a floor (added 2026-10-28, round 4, at your request):** highest F1 and no minimum precision, with the round 2 checks and tuning set. **Approved and implemented.**

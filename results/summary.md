# Results summary

Written by `scripts/make_summary.py` from the files in `results/`.
Nothing here is typed by hand.

## Reduced runs

- **max_context was reduced** from the default 2000 to 200 because the CPU is slow: results/timing_tabpfn.json measured 41 s to predict 200 rows with 2000 context rows.
- **Only the first 3 seeds of each scenario were run** (seeds_per_scenario). All scenarios are included, but each has fewer trajectories than the design, so the confidence intervals are wider. This was done after max_context had been reduced, because a full run was still too slow on CPU.

## AUROC, score = p_critical (probability that the target is at or below the threshold)

| Method | AUROC in-prior val | AUROC in-prior test | AUROC out-of-prior | False alarms on held-out controls (window rate) | Median latency per trajectory (s) |
|---|---|---|---|---|---|
| Classic EWS baseline | 0.58 [0.56, 0.60] | 0.57 [0.55, 0.59] | 0.52 [0.48, 0.56] † | 0.045 | not measured |
| TabPFN random | 0.81 [0.77, 0.87] | 0.83 [0.79, 0.87] | 0.60 [0.55, 0.65] | 0.038 | 6.5 |
| TabPFN knn | 0.79 [0.72, 0.87] | 0.82 [0.77, 0.87] | 0.56 [0.52, 0.60] | 0.079 | 6.1 |
| TabPFN knn_balanced | 0.60 [0.51, 0.70] | not run | not run | not run | 6.8 |
| TabPFN family_matched (ORACLE) | 0.94 [0.91, 0.96] | 0.94 [0.92, 0.97] | n/a (no pool rows of this family) | 0.000 ‡ | 6.3 |

Higher is better and 0.5 is chance. Cells show AUROC with the bootstrap CI resampled over trajectories. † = the CI includes 0.5. ⚠ = AUROC above 0.98, check for leakage.

Compared with the baseline:

- in-prior val, TabPFN random: above the baseline (CIs do not overlap).
- in-prior val, TabPFN knn: above the baseline (CIs do not overlap).
- in-prior val, TabPFN knn_balanced: not clearly different from the baseline (CIs overlap).
- in-prior val, TabPFN family_matched (ORACLE): above the baseline (CIs do not overlap).
- in-prior test, TabPFN random: above the baseline (CIs do not overlap).
- in-prior test, TabPFN knn: above the baseline (CIs do not overlap).
- in-prior test, TabPFN family_matched (ORACLE): above the baseline (CIs do not overlap).
- out-of-prior, TabPFN random: not clearly different from the baseline (CIs overlap).
- out-of-prior, TabPFN knn: not clearly different from the baseline (CIs overlap).

<details><summary>AUROC per family</summary>

**in-prior val**

| Family | Classic EWS baseline | TabPFN random | TabPFN knn | TabPFN knn_balanced | TabPFN family_matched (ORACLE) |
|---|---|---|---|---|---|
| fold | 0.58 [0.55, 0.61] | 0.81 [0.76, 0.86] | 0.71 [0.65, 0.78] | 0.68 [0.59, 0.79] | 0.93 [0.90, 0.96] |
| hopf | 0.64 [0.61, 0.67] | 0.96 [0.93, 0.99] | 0.98 [0.97, 0.99] ⚠ | 0.55 [0.37, 0.78] † | 0.99 [0.98, 1.00] ⚠ |
| transcritical | 0.48 [0.45, 0.52] † | 0.82 [0.74, 0.88] | 0.84 [0.76, 0.89] | 0.62 [0.50, 0.73] † | 0.83 [0.75, 0.88] |

**in-prior test**

| Family | Classic EWS baseline | TabPFN random | TabPFN knn | TabPFN knn_balanced | TabPFN family_matched (ORACLE) |
|---|---|---|---|---|---|
| fold | 0.58 [0.55, 0.62] | 0.84 [0.79, 0.88] | 0.83 [0.79, 0.87] | not run | 0.96 [0.95, 0.97] |
| hopf | 0.61 [0.58, 0.65] | 0.98 [0.95, 0.99] | 0.97 [0.95, 0.99] | not run | 0.98 [0.96, 1.00] ⚠ |
| transcritical | 0.50 [0.46, 0.53] † | 0.77 [0.71, 0.82] | 0.79 [0.74, 0.84] | not run | 0.80 [0.74, 0.86] |

**out-of-prior**

| Family | Classic EWS baseline | TabPFN random | TabPFN knn | TabPFN knn_balanced | TabPFN family_matched (ORACLE) |
|---|---|---|---|---|---|
| noise_tipping | 0.51 [0.46, 0.56] † | 0.69 [0.63, 0.76] | 0.65 [0.58, 0.72] | not run | n/a (no pool rows of this family) |
| rate_tipping | 0.52 [0.47, 0.57] † | 0.58 [0.49, 0.66] † | 0.59 [0.49, 0.68] † | not run | n/a (no pool rows of this family) |

</details>

## AUROC, score = -median (minus the median of the predicted target)

| Method | AUROC in-prior val | AUROC in-prior test | AUROC out-of-prior | False alarms on held-out controls (window rate) | Median latency per trajectory (s) |
|---|---|---|---|---|---|
| Classic EWS baseline | 0.58 [0.56, 0.60] | 0.57 [0.55, 0.59] | 0.52 [0.48, 0.56] † | 0.045 | not measured |
| TabPFN random | 0.79 [0.74, 0.84] | 0.80 [0.76, 0.85] | 0.58 [0.54, 0.63] | 0.046 | 6.5 |
| TabPFN knn | 0.70 [0.62, 0.80] | 0.73 [0.66, 0.81] | 0.55 [0.50, 0.59] | 0.044 | 6.1 |
| TabPFN knn_balanced | 0.66 [0.59, 0.75] | not run | not run | not run | 6.8 |
| TabPFN family_matched (ORACLE) | 0.94 [0.92, 0.96] | 0.94 [0.92, 0.96] | n/a (no pool rows of this family) | 0.000 ‡ | 6.3 |

Higher is better and 0.5 is chance. Cells show AUROC with the bootstrap CI resampled over trajectories. † = the CI includes 0.5. ⚠ = AUROC above 0.98, check for leakage.

Compared with the baseline:

- in-prior val, TabPFN random: above the baseline (CIs do not overlap).
- in-prior val, TabPFN knn: above the baseline (CIs do not overlap).
- in-prior val, TabPFN knn_balanced: not clearly different from the baseline (CIs overlap).
- in-prior val, TabPFN family_matched (ORACLE): above the baseline (CIs do not overlap).
- in-prior test, TabPFN random: above the baseline (CIs do not overlap).
- in-prior test, TabPFN knn: above the baseline (CIs do not overlap).
- in-prior test, TabPFN family_matched (ORACLE): above the baseline (CIs do not overlap).
- out-of-prior, TabPFN random: not clearly different from the baseline (CIs overlap).
- out-of-prior, TabPFN knn: not clearly different from the baseline (CIs overlap).

<details><summary>AUROC per family</summary>

**in-prior val**

| Family | Classic EWS baseline | TabPFN random | TabPFN knn | TabPFN knn_balanced | TabPFN family_matched (ORACLE) |
|---|---|---|---|---|---|
| fold | 0.58 [0.55, 0.61] | 0.76 [0.71, 0.82] | 0.60 [0.51, 0.68] | 0.59 [0.50, 0.67] | 0.94 [0.91, 0.96] |
| hopf | 0.64 [0.61, 0.67] | 0.94 [0.88, 0.99] | 0.99 [0.98, 1.00] ⚠ | 0.95 [0.91, 0.99] | 0.99 [0.99, 1.00] ⚠ |
| transcritical | 0.48 [0.45, 0.52] † | 0.81 [0.72, 0.86] | 0.82 [0.74, 0.88] | 0.71 [0.61, 0.79] | 0.84 [0.76, 0.89] |

**in-prior test**

| Family | Classic EWS baseline | TabPFN random | TabPFN knn | TabPFN knn_balanced | TabPFN family_matched (ORACLE) |
|---|---|---|---|---|---|
| fold | 0.58 [0.55, 0.62] | 0.78 [0.72, 0.83] | 0.68 [0.62, 0.77] | not run | 0.96 [0.95, 0.97] |
| hopf | 0.61 [0.58, 0.65] | 0.98 [0.95, 0.99] | 0.98 [0.96, 0.99] | not run | 0.99 [0.97, 1.00] ⚠ |
| transcritical | 0.50 [0.46, 0.53] † | 0.77 [0.71, 0.82] | 0.76 [0.70, 0.81] | not run | 0.80 [0.75, 0.86] |

**out-of-prior**

| Family | Classic EWS baseline | TabPFN random | TabPFN knn | TabPFN knn_balanced | TabPFN family_matched (ORACLE) |
|---|---|---|---|---|---|
| noise_tipping | 0.51 [0.46, 0.56] † | 0.64 [0.56, 0.71] | 0.62 [0.55, 0.68] | not run | n/a (no pool rows of this family) |
| rate_tipping | 0.52 [0.47, 0.57] † | 0.58 [0.50, 0.66] | 0.60 [0.50, 0.70] | not run | n/a (no pool rows of this family) |

</details>

## False alarms on controls

| Method (score) | Threshold | Target | Validation controls | Held-out controls: windows | Held-out controls: trajectories with any alarm |
|---|---|---|---|---|---|
| Classic EWS baseline | 0.6417 | 0.050 | 0.049 | 0.045 | 0.972 |
| TabPFN random (p_critical) | 0.1899 | 0.050 | 0.050 | 0.038 | 0.667 |
| TabPFN random (neg_median) | -0.5778 | 0.050 | 0.050 | 0.046 | 0.685 |
| TabPFN knn (p_critical) | 0.1056 | 0.050 | 0.050 | 0.079 | 0.593 |
| TabPFN knn (neg_median) | -0.7515 | 0.050 | 0.050 | 0.044 | 0.593 |
| TabPFN knn_balanced (p_critical) | not run | | | | |
| TabPFN knn_balanced (neg_median) | not run | | | | |
| TabPFN family_matched (ORACLE) (p_critical) | 4.941e-324 | 0.050 | 0.000 | 0.000 | 0.000 |
| TabPFN family_matched (ORACLE) (neg_median) | -1 | 0.050 | 0.000 | 0.000 | 0.000 |

The alarm threshold is set on the validation controls only, so that at most the target share of their windows alarm. It is then applied unchanged to the held-out controls (test_in_prior and test_out_of_prior). A trajectory counts as an alarm if any of its windows alarms, so that rate is high for any threshold. The window rate is the informative one. The main table shows the window rate. ‡ = degenerate score on controls, see the checks below.

## TabPFN + gate

The gate (docs/GATE.md) decides per run whether to trust it. A blocked run abstains on all its windows. A run is BAD if its 90% interval covers less than 0.80 of its window targets. Thresholds are tuned on in-prior validation runs only and frozen before any held-out run. Scores are p_critical. CIs come from a bootstrap over runs (trajectories).

### Tuning

- Tuned on 162 in-prior validation runs (81 trajectories; random/val 54, random/val_controls 27, knn/val 54, knn/val_controls 27), of which 23 were BAD. Strategies: random, knn.
- Chosen point: ood_share limit 0.3, context_calibration [0.7, 1.0], out of 48 grid points. Rule: highest F1 of blocking BAD runs, no minimum precision (decision 12, 2026-10-28).
- In-sample on validation: precision 0.34, recall 0.43, F1 0.38. Out-of-fold (5 folds by trajectory): precision 0.34, recall 0.43. The in-sample numbers are optimistic by construction.
- Alert thresholds on p_critical, from accepted validation control runs: random 0.1899 (27 of 27 control runs accepted); knn 0.1427 (8 of 27 control runs accepted).
- Gate frozen with SHA-256 6e93455997a26538... at 2026-10-28T18:19:53+00:00.

### Gate on tipping runs

| Method | Split | Gate | Runs | Accepted runs | Abstention rate | Runs over the width cap (advisory) | AUROC, accepted runs | AUROC, all runs | 90% coverage, accepted runs | 90% coverage, all runs | BAD runs | Precision (blocking BAD) | Recall (blocking BAD) | Gate time per run, uncached refits (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TabPFN random + gate | in-prior val | tuned, in-sample | 54 | 54 | 0.00 | 53 | 0.81 [0.77, 0.87] | 0.81 [0.77, 0.87] | 0.90 | 0.90 | 12 | n/a (nothing blocked) | 0.00 [0.00, 0.00] | n/a (refits cached) |
| TabPFN random + gate | in-prior test | tuned | 54 | 54 | 0.00 | 54 | 0.83 [0.79, 0.87] | 0.83 [0.79, 0.87] | 0.88 | 0.88 | 14 | n/a (nothing blocked) | 0.00 [0.00, 0.00] | 18.6 |
| TabPFN random + gate | out-of-prior | tuned | 36 | 35 | 0.03 | 36 | 0.60 [0.55, 0.65] | 0.60 [0.55, 0.65] | 0.92 | 0.92 | 4 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 19.4 |
| TabPFN knn + gate | in-prior val | tuned, in-sample | 54 | 44 | 0.19 | 51 | 0.91 [0.86, 0.95] | 0.79 [0.72, 0.87] | 0.93 | 0.88 | 11 | 1.00 [1.00, 1.00] | 0.91 [0.67, 1.00] | n/a (refits cached) |
| TabPFN knn + gate | in-prior test | tuned | 54 | 44 | 0.19 | 52 | 0.90 [0.85, 0.93] | 0.82 [0.77, 0.87] | 0.92 | 0.88 | 14 | 0.90 [0.64, 1.00] | 0.64 [0.36, 0.90] | 19.1 |
| TabPFN knn + gate | out-of-prior | tuned | 36 | 34 | 0.06 | 28 | 0.56 [0.52, 0.61] | 0.56 [0.52, 0.60] | 0.75 | 0.74 | 19 | 1.00 [1.00, 1.00] | 0.11 [0.00, 0.27] | 19.1 |

AUROC on accepted runs and on all runs are computed on different sets of trajectories, so a higher AUROC on accepted runs can partly come from abstaining on hard trajectories. The interval width cap is an advisory: runs over it are counted and reported, but not blocked (docs/GATE.md, decision 7). The Gate column says which configs/gate.yaml produced the row: a provisional row shows the loosest grid point, not a tuned gate. AUROC on accepted runs from one or two accepted runs only ranks windows inside those trajectories.

### Gate on control runs

| Method | Held-out control runs | Abstention rate | Window false alarms (abstain = no alarm) | Window false alarms, accepted runs only | Window false alarms without the gate |
|---|---|---|---|---|---|
| TabPFN random + gate | 54 | 0.02 | 0.038 | 0.038 | 0.038 |
| TabPFN knn + gate | 54 | 0.61 | 0.028 | 0.071 | 0.079 |

### Does the gate carry over? (computed from the numbers above)

- TabPFN random: abstention out of prior (0.03) is higher than in-prior test (0.00).
- TabPFN random: recall of BAD runs out of prior is 0.00, the same as the in-sample validation recall (0.00).
- TabPFN random, in-prior val: accepting only gated runs did not change mean 90% coverage (0.90 on all runs, 0.90 on accepted runs).
- TabPFN random, in-prior test: accepting only gated runs did not change mean 90% coverage (0.88 on all runs, 0.88 on accepted runs).
- TabPFN random, out-of-prior: accepting only gated runs did not change mean 90% coverage (0.92 on all runs, 0.92 on accepted runs).
- TabPFN knn: abstention out of prior (0.06) is lower than in-prior test (0.19).
- TabPFN knn: recall of BAD runs out of prior is 0.11, lower than the in-sample validation recall (0.91).
- TabPFN knn, in-prior val: accepting only gated runs raised mean 90% coverage (0.88 on all runs, 0.93 on accepted runs).
- TabPFN knn, in-prior test: accepting only gated runs raised mean 90% coverage (0.88 on all runs, 0.92 on accepted runs).
- TabPFN knn, out-of-prior: accepting only gated runs raised mean 90% coverage (0.74 on all runs, 0.75 on accepted runs).

## Run settings

| Run | max_context (config default) | Seeds per scenario | Trajectories | Windows | Cache hits | Skipped | Wall time (min) | Model | Backend package |
|---|---|---|---|---|---|---|---|---|---|
| family_matched controls | 200 (2000) | 3 | 81 | 24705 | 0 | 0 | 0.7 | v3.5 | 9.2.0 |
| family_matched ood | 200 (2000) | 3 | 0 | 0 | 0 | 36 | 0.1 | v3.5 | 9.2.0 |
| family_matched test | 200 (2000) | 3 | 54 | 7769 | 0 | 0 | 6.2 | v3.5 | 9.2.0 |
| family_matched val | 200 (2000) | 3 | 54 | 7728 | 54 | 0 | 0.1 | v3.5 | 9.2.0 |
| knn controls | 200 (2000) | 3 | 81 | 24705 | 81 | 0 | 15.7 | v3.5 | 9.2.0 |
| knn ood | 200 (2000) | 3 | 36 | 7922 | 36 | 0 | 11.8 | v3.5 | 9.2.0 |
| knn test | 200 (2000) | 3 | 54 | 7769 | 54 | 0 | 17.5 | v3.5 | 9.2.0 |
| knn val | 200 (2000) | 3 | 54 | 7728 | 54 | 0 | 0.1 | v3.5 | 9.2.0 |
| knn_balanced val | 200 (2000) | 3 | 54 | 7728 | 54 | 0 | 0.1 | v3.5 | 9.2.0 |
| random controls | 200 (2000) | 3 | 81 | 24705 | 81 | 0 | 19.6 | v3.5 | 9.2.0 |
| random ood | 200 (2000) | 3 | 36 | 7922 | 36 | 0 | 11.8 | v3.5 | 9.2.0 |
| random test | 200 (2000) | 3 | 54 | 7769 | 54 | 0 | 17.3 | v3.5 | 9.2.0 |
| random val | 200 (2000) | 3 | 54 | 7728 | 54 | 0 | 0.1 | v3.5 | 9.2.0 |

## Checks computed from the results

- knn p_critical: held-out window false-alarm rate 0.079 is above 1.5 times the target 0.050. The validation threshold did not transfer.
- ‡ family_matched on controls: the p_critical alarm threshold is at the smallest positive float, so p_critical was exactly 0 on at least 95% of validation control windows. The score is degenerate on controls (a constant context target gives a constant prediction), so its false-alarm rate says nothing about the method.
- family_matched ood: 36 trajectories skipped (no pool rows in family 'noise_tipping'; no pool rows in family 'rate_tipping').

## Definitions

- **AUROC**: balanced AUROC on windows of tipping trajectories, positives and negatives as defined in `docs/LABELS.md` (the middle band is dropped). Controls are not in the AUROC. They are scored by the false-alarm rate.
- **Baseline**: mean Kendall tau of rolling variance and lag-1 autocorrelation over the previous K windows (`configs/baseline_ews.yaml`). Its settings were fixed before the first run.
- **TabPFN**: regression on the scale-free window features, with a context chosen from the in-prior context pool. The context never contains the query trajectory.
- **In-prior splits**: the context pool holds the same scenarios as the in-prior val and test splits, with other seeds (docs/SPLITS.md). In-prior AUROC is therefore easier than for a new system. The out-of-prior column is the test of generalisation.
- **knn_balanced**: knn, except that a fixed share of the context is kept for pool rows near tipping (target at or below the critical threshold), the ones nearest to the query. It reads pool targets only, never the query's family, so it is deployable (docs/GATE.md, decision 6).
- **family_matched (ORACLE)**: the context is sampled from the family of the query, which a real run would not know. It is an upper bound for context selection and not a method. It cannot run on out-of-prior families, which are not in the pool.
- **Latency per trajectory**: fit plus predict for all windows of one trajectory, median over the tipping trajectories of the strategy (val, test and ood runs), on this CPU. Control runs are left out. A cached result keeps the latency of the run that computed it. The baseline latency was not measured.


# Splits

Status: **approved design (2026-10-26).** No simulator code exists yet.

This file defines how simulated trajectories are divided into the context pool, validation and test sets, and how context episodes are drawn. Labels are defined in `docs/LABELS.md`. Seed sets and scenario specs are in `configs/scenarios/`.

## Rule: split by family and seed, never by window

The unit of splitting is the **trajectory**, `(scenario_id, seed)`. All windows cut from one trajectory go to the same split. Windows from one series overlap and share the same noise path, so splitting by window would leak the answer from context into the query.

Families decide **whether** a trajectory can appear in a split (in-prior, out-of-prior or control). Seed sets decide **which** split it goes to.

## Seed sets

| Set | Seeds | Used for |
|---|---|---|
| A | 1000-1019 (20) | Context pool |
| B | 2000-2009 (10) | In-prior validation: gate thresholds are tuned here and nowhere else |
| C | 3000-3009 (10) | In-prior test |
| D | 4000-4009 (10) | Out-of-prior test |

The RNG for a trajectory is `numpy SeedSequence([crc32(scenario_id), seed])`. The same seed number in two scenarios therefore gives independent noise, and no two trajectories share a noise path.

## Which families go where

| Split | Seed set | Fold, Hopf, transcritical | Rate, noise-induced | Controls |
|---|---|---|---|---|
| Context pool | A | yes | **never** | yes |
| In-prior validation | B | yes | **never** | yes |
| In-prior test | C | yes | no | yes |
| Out-of-prior test | D | no | yes | yes |

Controls appear in every split, so false alarms are measured everywhere. Out-of-prior families appear only in the out-of-prior test. They are never used for context, tuning or model choice.

## Scenario and trajectory counts

| Family | Scenarios | Seed sets | Trajectories |
|---|---|---|---|
| fold | 6 (3 noise x 2 ramp) | A, B, C | 240 |
| hopf | 6 | A, B, C | 240 |
| transcritical | 6 | A, B, C | 240 |
| rate_tipping | 6 | D | 60 |
| noise_tipping | 6 | D | 60 |
| control_ou | 3 (3 noise) | A, B, C, D | 150 |
| control_short_ramp | 3 (3 noise, slow ramp) | A, B, C, D | 150 |
| control_regime_change | 3 (3 noise) | A, B, C, D | 150 |
| **Total** | **39** | | **1290** |

Per split: A has 540 trajectories, B has 270, C has 270 and D has 210. These are planned counts. Window and class counts will be written to `results/` by code once the simulator exists.

## Gate thresholds are frozen before testing

1. Gate thresholds are tuned only on predictions for set B (with context from set A).
2. The tuned thresholds are written to `configs/gate.yaml` and committed.
3. Only then may code evaluate set C, set D or any real data. The evaluation code should refuse to run if `configs/gate.yaml` is missing or has uncommitted changes.
4. Results on C and D never feed back into the thresholds. If we change the thresholds after seeing C or D, those test results are void and must be reported as such.

## Context episodes

A **context episode** is the set of labelled windows given to the PFN together with a batch of query windows.

How an episode is drawn for a query trajectory `q = (scenario_id, seed)` from set B, C or D:

1. **Pool.** Candidates are windows from set A trajectories only, after the window rules in `docs/LABELS.md` (pre-tip only).
2. **Exclusion.** Any candidate whose trajectory key equals `q` is removed. With the seed sets above this never happens, because A is disjoint from B, C and D. The check is still run as an assertion, so a config mistake fails loudly instead of leaking.
3. **Seeding.** The episode RNG is `SeedSequence([crc32(scenario_id), seed, episode_index])`, so every episode can be reproduced exactly.
4. **Default composition.**
   - The windows are split equally across four groups: fold, Hopf, transcritical and controls.
   - Within each group, binary positives and negatives are balanced where possible.
   - Each group draws from at least 10 different trajectories, so one trajectory cannot dominate.
   - Default size is 1000 rows, and the hard cap is 5000 rows (the TabPFN CPU limit; see `results/timing_tabpfn.json` for cost).
5. **One episode per query trajectory.** All query windows from `q` share one episode. Different query trajectories get different episodes.
6. **Agent.** The context-selection agent may change the composition: group weights, size within the cap, noise-level filters. It can only choose from set A. It never sees query labels, every choice it makes is logged to `results/`, and it cannot read or change the gate.
7. **Diagnostics on set A.** If a query is ever taken from set A (for example, to debug the backend), the query's own trajectory is removed from the pool (leave one trajectory out).

## Leakage checks

These become unit tests once the code exists:

- Seed sets A, B, C and D are disjoint.
- No rate or noise-induced trajectory is in set A or B.
- Every split contains all three control families.
- No window appears in two splits, and no trajectory appears in two splits.
- Every context row comes from set A, and no context row shares a trajectory key with its query.
- `configs/gate.yaml` exists and is committed before any C, D or real-data evaluation runs.

## Real data

The one real dataset is the split `real_tipping` (Daphnia populations). It is test only, with surrogate labels. It is never context and never used for tuning, and reading it calls `assert_frozen()` first. See `docs/DAPHNIA.md`.

## What each test measures

- **In-prior test (C):** same families, noise levels and ramp speeds as the context, new noise paths. It measures generalisation across seeds, not across families.
- **Out-of-prior test (D):** tipping mechanisms the context never shows. It measures whether the model warns for the right reason, or for any reason at all.

## Open questions and risks (splits)

1. **Controls deviate from "2 ramp speeds per family".** OU and regime change have no ramp, and the short-ramp control uses the slow ramp only. That gives 3 scenarios per control family and 39 in total. Using both speeds for the short ramp would add 3 more (42, above the 40 limit).
2. **Family generalisation inside the prior.** Set C reuses the same scenarios as the context. A leave-one-family-out check (for example, context without fold, query on fold) would test family transfer within the prior. Should we add it as an extra, clearly labelled experiment?
3. **Seed counts.** 10 seeds per scenario in B may give few positive windows at noise level L3, where early tipping removes late windows. That makes the gate thresholds noisy. We may need 20 seeds in B.
4. **Context size against CPU time.** On this CPU, predicting 200 rows with 5000 context rows takes minutes (see `results/timing_tabpfn.json`). The default of 1000 rows keeps a full evaluation practical, but it may underuse the model.
5. **Channel count.** Hopf has two channels and every other family has one. The feature step must handle both, for example with per-channel features plus a missing-channel indicator. This should be decided before the features module is written.
6. **Same scenario in context and test.** Because set C shares scenarios with set A, in-prior test scores will look strong. The report must say that the out-of-prior test is the real generalisation test.

# Labels

Status: **approved design (2026-10-26).** No simulator code exists yet.

This file defines what the model is asked to predict for each window of a simulated time series. The scenario specs are in `configs/scenarios/`, and the splits are in `docs/SPLITS.md`.

We build our own small suite of simulated series, in the spirit of the kausable TipPFN paper (arXiv 2605.12308). We do **not** reproduce that paper, its prior, its labels or its results.

## Terms

- **Trajectory**: one simulated series, identified by `(scenario_id, seed)`.
- **Window**: 400 consecutive observed points (40 time units), taken with a stride of 50 points. A window is described by its **end time** `t`. All labels refer to the window end, because that is "now" for an early-warning model.
- **Critical time `t_c`**: for bifurcation families, the time the control parameter `p(t)` crosses its critical value `p_c`. It is known exactly from the spec.
- **Realised tip time `t_tip`**: the first time the simulated state actually tips, according to the family's `tip_detector`. This can differ from `t_c`. Noise can make the system tip early, and a fast ramp can make it tip late.

## Which windows are kept

1. Only windows that end **at or before `t_tip`** are kept. A window that contains the jump itself asks the model to detect a transition that already happened, not to warn about it.
2. Windows whose target falls between the two binary thresholds are dropped from the binary task. They are kept for the regression task.
3. Trajectories that are configured to tip but do not tip within the series cap are dropped. The number dropped is written to `results/`. This only applies to the noise-induced family in practice.

## Regression target: relative distance to criticality

The target is a number in [0, 1]. 1 means far from the transition and 0 means at the transition.

### Bifurcation families (fold, Hopf, transcritical)

`target(t) = clip((p_c - p(t)) / (p_c - p0), 0, 1)`

- **Why:** the ramp is linear, so this is the fraction of the parameter distance still left. It is known exactly, it needs no detector, and it maps every family onto the same scale whatever its `p0` and `p_c`.
- **Limit 1:** it measures distance in parameter space, not in time or resilience. On a slow ramp, a target of 0.2 means 80 time units before `t_c`. On a fast ramp it means 20 time units.
- **Limit 2:** with noise, the fold and subcritical Hopf can tip **before** `p_c`, because both are bistable near the transition. Those trajectories lose their late windows (rule 1 above), so the target never gets close to 0 for them.
- **Limit 3:** a fast ramp can carry `p` past `p_c` before the jump happens (delayed tipping). Windows in that gap have target 0 (clipped) and are kept as positives, which is correct: the system is past the point of no return.

### Rate-induced and noise-induced families

`target(t) = clip((t_tip - t) / H, 0, 1)` with a fixed horizon `H = 100` time units.

- **Why:** these families have no bifurcation parameter, so "distance to `p_c`" does not exist. Time to the realised tip is the only common measure.
- **Why `H = 100`:** it puts the binary positive zone (target ≤ 0.2) at the last 20 time units before the tip. That matches the positive zone of the fast-ramp bifurcation scenarios, so the binary task is equally hard in time.
- **Limit 1:** `t_tip` is only known after simulating the whole trajectory. That is fine for a label, but it means the label uses information from the future of the series, which a real model never has.
- **Limit 2:** for noise-induced tipping the tip time is random. Two windows that look the same can have very different labels. Some of this target is unpredictable by design.
- **Limit 3:** the two target types are on different scales (parameter fraction against time fraction). Scores on out-of-prior families are therefore not directly comparable with in-prior scores. The report must show them separately.

### Controls (OU, short ramp, regime change)

`target = 1` for every window.

- **Why:** the system never tips, so it is "far" in the only sense that matters for a warning.
- **Limit:** the short-ramp control is genuinely less resilient after its ramp than at the start (its bifurcation distance would be 0.5). Labelling it 1 rewards a model for **not** reporting that lower resilience. We accept this because the question we ask is "will it tip soon?", not "how resilient is it?". See open question 2.

## Binary label per window

- **Positive (1)** if the target at the window end is ≤ 0.2.
- **Negative (0)** if the target at the window end is ≥ 0.6.
- Windows in between are **dropped** from the binary task.
- **Controls:** every window is negative.

**Why a gap:** a window with target 0.4 is neither clearly safe nor clearly in danger. Forcing a label on it adds noise near the threshold and makes threshold tuning unstable. The cost is that the binary task never sees the hardest middle cases. The regression task still covers them.

**Class balance:** on a slow bifurcation ramp most windows are far from the transition, so negatives dominate. Controls add more negatives. The evaluation will report class counts per split from `results/`, and metrics must be balance-aware (for example AUROC and balanced accuracy, not plain accuracy).

## Families at a glance

| Family | Role | Target | Tip detector |
|---|---|---|---|
| fold | in-prior | bifurcation distance | x > 0 |
| hopf (subcritical) | in-prior | bifurcation distance | amplitude > 0.7 |
| transcritical | in-prior | bifurcation distance | p crosses p_c, or escape past the unstable branch |
| rate_tipping | out-of-prior | time to tip | x + lambda > 2 |
| noise_tipping | out-of-prior | time to tip | x > 0.5, then x > 0 for 10 time units |
| control_ou | control | 1 | none |
| control_short_ramp | control | 1 | none (asserts x < 0) |
| control_regime_change | control | 1 | none |

## Real data

There is no simulated `p(t)` in real data and usually no known tip time. **Labels for real data will be surrogates.** For example, a dated, documented transition can stand in for `t_tip`, and the time-to-tip target can then be used. These surrogates carry dating error and selection bias, because only transitions that happened get recorded. Real-data results must be reported as surrogate-label results and kept separate from simulated results.

## Open questions and risks (labels)

1. **Transcritical escape at L3.** With additive noise near `p_c`, the state can cross the unstable branch `x = p` and diverge. At noise level L3 this may happen in most trajectories, so they would end early. Options: keep it as a real failure mode (current choice), lower the noise for this family only, or add a stabilising term. We should decide after a first simulation reports the escape rate.
2. **Short-ramp control label.** Should it be 1 (current choice: it never tips) or its bifurcation distance of 0.5 (it is less resilient)? Choosing 1 makes it a strict false-alarm test.
3. **Subcritical or supercritical Hopf.** Subcritical gives an abrupt jump (current choice). Supercritical is the textbook case in many early-warning papers but has no jump, only a growing oscillation.
4. **Early tipping in bistable families.** At L3 the fold and subcritical Hopf may often tip well before `p_c`, which removes most positive windows from those trajectories. We should check the count of positives per scenario once simulated.
5. **Noise-induced family and variance.** Its noise rises over time, so variance rises too. A model that relies on variance may "warn" correctly for the wrong reason (more noise, not lower stability). This is worth reporting, not hiding.
6. **Window length.** 400 points (40 time units) is a guess. Critical slowing down is easier to see in long windows, but long windows reduce the number of windows on fast ramps.
7. **Citations.** The R-tipping form (Ashwin et al. 2012) and the noise-induced reference (Ditlevsen and Johnsen 2010) are cited from memory. Both must be checked against the papers before any code uses them.

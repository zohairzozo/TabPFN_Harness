# Daphnia data (the real_tipping split)

Status: **draft, 2026-10-31.** Written from the files and from the loader in `src/pfngate/data/daphnia.py`.

This is the one real dataset in the harness. It is **test only**. It is never part of the context pool, it never touches gate tuning (the gate is frozen in `configs/gate.yaml`), and reading it calls `assert_frozen()` first (docs/SPLITS.md, "Gate thresholds are frozen before testing"). Every count in this file is written by code to `results/datasets/real_tipping_summary.json`. Read the numbers there. This file does not repeat them.

## Source

- Drake, J.M. and Griffen, B.D. (2010). *Early warning signals of extinction in deteriorating environments.* Nature 467:456-459. doi:10.1038/nature09389.
- Data: Dryad, doi:10.5061/dryad.q3p64. Dryad needs a login to download it. The data is **not** in git (`data/raw/` is ignored) and the code never downloads it.
- To get it: download `data-and-code.zip`, unzip it into `data/raw/daphnia/`. The loader reads `data/raw/daphnia/data-and-code/extinctions.csv` and `timeseries.csv` (path set in `configs/real_tipping.yaml`). If a file is missing, the loader stops and says which one.
- Not used: `TrajectoryScores.csv` and `span*.csv` (the authors' hand scores of early-warning statistics), the R scripts and the figures.

## What the experiment is

Laboratory populations of *Daphnia magna*, each in a small two-chamber microcosm. In the "deteriorating" treatment the food supply was cut in steps, week by week, until the population died out. In the "constant" treatment the food stayed constant. The populations were counted once a week.

## What a row is

**`timeseries.csv`**: one row per population, chamber and census date. Columns: `ID` (population), `Subpop` (chamber, 1 or 2), `Date`, `sample1..3` (three counts of the same chamber on that date), the treatment columns and `FoodLevel`.

**`extinctions.csv`**: one row per population. Columns: `ID`, `Start`, `End`, `Censor`, treatment columns, `FoodAtExtinct`, `Notes`. `End` is the extinction date, or the date the series was cut off if the population was still alive (`Censor` is 1 for extinct and 0 for censored, as the README says).

## How replicates map to trajectories

- **One population (`ID`) is one trajectory.** All windows cut from it go to the same split, as in docs/SPLITS.md. Here that means all of them are in `real_tipping`.
- **The series of a population** is, for each census date, the sum over the two chambers of the mean of the three counts. Time is in days from the first census used.
- **Trajectory id:** `daphnia_<treatment>/<k>`, where `k` is the position of the population in the sorted list of ids. The `scenario_id` is `daphnia_deteriorating` or `daphnia_constant`.
- **A trajectory has one channel** and no hidden state. It is cut into windows with the same feature code as the simulators (`window_features`, `window_ends`, `window_matrix`).

## Which series end in extinction, and when

- **Extinct** (`Censor = 1`): the extinction time is `End`, in days from the first census used. The loader checks, and writes to the summary file, that the final count of every extinct population is zero and that `End` is within a week of the last census. Read the check lists in the summary file.
- **Persisting** (`Censor = 0`): alive at the end of the experiment. They have no extinction time. The loader checks that their final count is above zero.
- **Which treatment:** all deteriorating populations that have a series go extinct. Most constant populations persist. **A few constant-treatment populations also go extinct** (the summary file counts them by treatment and outcome).
- **When:** the summary file gives the earliest, median and latest extinction day. The experiment ran for about a year.
- **Roles in the harness:** a population that goes extinct has role `real` (a "tipping" run, with the surrogate target below). A population that persists has role `control` (target 1, every window negative), which is the real no-alarm control.

## The labels are surrogates

**These labels are not the true distance to a bifurcation.** There is no control parameter, no critical value and no bifurcation here. The populations die out by demographic chance as the food falls, and the harness has no way to tell how close a population is to a true tipping point. Nothing in this split measures that.

- **Extinct:** `target = clip((t_extinction - t) / H, 0, 1)` at the window end `t`, with the fixed horizon `H = 100` days. This is the same form as the time-to-tip target for the rate and noise families in docs/LABELS.md, and `H` is the simulators' 100, taken over as days. `H` was fixed before any result on this data and must not be changed after one.
- **Persisting:** `target = 1`.
- **Binary label:** positive if the target is at most 0.2 (the last 20 days), negative if it is at least 0.6, dropped in between. The thresholds come from `configs/scenarios/_common.yaml`, as for the simulated data.
- **Windows:** only windows that end at or before the extinction are kept (the rule of docs/LABELS.md). The time to extinction uses information from the future of the series, which a real model never has.
- **Limits:** extinction dates are weekly, so they carry up to a week of dating error. Only populations that went extinct have a tipping label, which is a selection effect (docs/LABELS.md, "Real data"). The target is on a time scale (days), not on the parameter scale of the bifurcation families, so scores here must be reported separately from simulated scores, as surrogate-label results.

## Windows

The series have at most a few dozen weekly points, so the simulators' 128-point windows do not fit. `configs/real_tipping.yaml` sets the window to 24 census points with a stride of 3, and a sub-window of 6 with a stride of 2 (the same ratios as `configs/features.yaml`). The feature code is the same. The settings were fixed from the series length, before any result on this data.

This means a real window covers 24 weeks, a simulated one 12.8 time units, and the features that depend on sub-window counts (the two Kendall tau features) are computed over fewer sub-windows. The features are scale-free, but the real windows are not the same kind of window as the ones in the context pool. Expect a shift, and report it as a limit.

## Things I am unsure about

1. **The folder.** The task said the data is in a sub folder called "Dataset". The archive has `data-and-code/` and no folder with that name. I used `data-and-code/`.
2. **Chambers.** I assume a population is the sum of its two chambers. For `HoleNum = 0` the chambers are not connected, and the two chambers differ clearly in their counts, so summing may not be what the authors treat as "the population". Not checked against the paper.
3. **Counts.** I average the three samples of a chamber. I do not know what volume they count, and I do not transform them (no log). The features are scale-free, so the unit does not matter for them.
4. **Constant treatment, extinct.** A few constant-treatment populations have `Censor = 1` and the note "Extinct during deterioration phase", although the treatment is constant. I treat them as extinct (positives) because the rule is that a series that ends in extinction is positive. The note may be a copy of the label of the other group.
5. **Restarted populations.** Five ids have `Start` later than their first census and a gap of months in the census (a first population died before the treatment began, and the id was used again). I use only the census from `Start`. This follows the dates in the two files and is not documented by the authors in the README.
6. **Populations without a series.** Five ids with an `a` suffix are in `extinctions.csv` (extinct before the treatment) and have no census series. They are left out. They are listed in the summary file.
7. **Extinction date.** I take `End` as the extinction date and keep windows whose last census is at or before it. The census is weekly, so the true extinction can be up to a week earlier.
8. **Header.** `timeseries.csv` has a column named `Date ` with a trailing space. The loader trims it, and also trims the ids (one id has a trailing space in `extinctions.csv`).
9. **Time zero.** Time is in days from the first census used, so it is not the same origin for every population. Only differences of time matter for the label and the features.
10. **The DOI.** The Dryad page lists the paper as doi:10.1038/nature09389. I did not check the paper itself, only the Dryad record and the files.

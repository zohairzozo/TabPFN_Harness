# Grid frequency data (the real_control split)

Status: **draft, 2026-10-31.**

**This split exists ONLY to measure false alarms.** Every window is labelled negative. It tells you how often the harness raises an alert on a real series where nothing is going wrong. It says nothing about whether the harness can warn before a real transition, because there is none in it.

**It contains normal operation, not a pre-blackout run.** The slice was chosen because it looks stationary, with no large event. It is not an example of a grid approaching a blackout, and no result on it should be described as detecting or missing one.

It is **test only**: it is never in the context pool, never used to tune the gate (the gate is frozen in `configs/gate.yaml`), and reading it calls `assert_frozen()` first. Counts are written by code to `results/datasets/real_control_summary.json`. Read the numbers there.

## Source and licence

- Kruse, J., Schaefer, B. and Witthaut, D., *Pre-Processed Power Grid Frequency Time Series*, Zenodo record 3744121, doi:10.5281/zenodo.3744121. Paper: "Predictability of Power Grid Frequency", arXiv:2004.09259.
- Used: **only the Fingrid (Nordic grid) files of `Data_cleansed.zip`**. The record text and the `LICENSE.txt` in that folder release the pre-processed National Grid and Fingrid files under **CC BY 4.0**. Attribution is required: cite the record and the paper.
- **Not used and never downloaded:** every TransnetBW file (Continental Europe). Their original data has no open licence, so the authors could not release them openly. The National Grid files carry the same CC BY 4.0 statement but were not needed.
- The record only offers the two big zip files. `scripts/fetch_grid.py` reads the zip directory with HTTP range requests and streams one yearly Fingrid file. TransnetBW files are refused in the code, and no byte of them is read.
- Source, access date, licence and SHA-256 of the slice are in `data/SOURCES.md`. The data is in `data/raw/grid/`, which git ignores.

## The slice

- **Resolution:** the record has 1-second data only (no 1-minute files), so the slice is 1-second data, cut to a few days. It is smaller than any full year, and only a few megabytes were read from Zenodo.
- **Selection rule, fixed in the script before any window was built:** the first run of consecutive calendar days in the 2019 file that are complete (one row per second) with at most 100 NaN seconds a day and no value farther than 0.25 Hz from 50 Hz. The first try used 0.15 Hz with no NaN, and found nothing. The Nordic grid already moves farther than that on ordinary days, so the rule was relaxed to the band above before the slice was fetched. The rule looks at the raw frequency only, never at any model output.
- **Windows:** the frequency is averaged over 60-second blocks, so a day has 1440 points. One calendar day is one trajectory, so all windows of a day stay together. Windows use the same 128-point length, stride and feature code as the simulators (`configs/features.yaml`).
- **NaN seconds** inside a day are filled by linear interpolation (a gap of at most 30 seconds). The summary file counts them.

## Labels

Target 1 and binary label 0 for every window. The loader stops with an error if any window is not negative. Because there is no positive, **AUROC is undefined** here. Report the false-alarm rate (the share of runs or windows that ended in an alert), with the number of runs.

## Limits and things I am unsure about

1. **Few days.** The slice is a few days, so a handful of trajectories. A false-alarm rate from it has a wide interval. It is a demonstration of a real-data control, not a measurement of a rate.
2. **The first stretch.** The rule takes the first stretch that qualifies, and that stretch starts on 1 January. New Year's Day may not be typical load behaviour. I did not check other stretches.
3. **Different from the simulated controls.** The simulated controls have a known form (OU, a short ramp, a regime change). Real grid frequency has daily cycles and operator actions. The features are scale-free, but the windows are 128 minutes long and not the same kind of window as the context pool. Expect a shift, and report it as a limit.
4. **Pre-processing by the record authors.** I use the "cleansed" file as published. I did not check their cleaning steps against the paper's supplement.
5. **Time stamps** are naive local time, as the record says. Only the order and spacing matter here.
6. **Selection effect.** Choosing a quiet stretch makes false alarms less likely than they would be over a whole year. The number is a lower bound for real operation.

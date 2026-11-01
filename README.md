# pfngate

pfngate is a release-gate harness for early-warning models, built as an interview demo for kausable, with TabPFN standing in for TipPFN. It exists to show a deterministic gate that decides, run by run, whether a model's output can be trusted, and to show the failures that gate does and does not catch.

## Results

The table compares five variants on the same trajectories. The classic baseline and the LLM-only model have no gate and no prediction interval, so their coverage and blocking cells are not applicable. The full harness is the only variant with a gate. The oracle uses the family of each query, which a real run cannot know, so it is an upper bound and not a method.

<!-- results:start -->
**Release verdict for the stand-in model: NO-GO**, with 2 of 4 required criteria met. 3 of 4 required results are undecided at 95% confidence, because their interval spans the threshold. The criteria are illustrative (configs/release_criteria.yaml). These numbers were produced from commit 118f5e8b0203 with 178 uncommitted files, so no commit reproduces them.

Model behind every number: tabpfn_local, model version v3.5, package version 9.2.0, context capped at 200 rows.

| Variant | In-prior AUROC | Out-of-prior AUROC | 90% coverage (in-prior / out-of-prior) | Calibration error (in-prior / out-of-prior) | False alarms on controls (in-prior / out-of-prior) | Share of BAD runs blocked (in-prior / out-of-prior) | p95 latency | Tool calls per verdict | pass^3: seeds that agree |
|---|---|---|---|---|---|---|---|---|---|
| Classic EWS baseline | 0.79 [0.52, 1.00] | 0.78 [0.53, 0.99] | n/a (no intervals) | n/a (no intervals) | 0.00 (n=9) / 0.00 (n=9) | n/a (no gate) | 0.00784 s | n/a (no tools) | 0.67 (8/12) |
| LLM only | 0.87 [0.65, 1.00] | 0.72 [0.43, 0.94] | n/a (no intervals) | n/a (no intervals) | 0.33 (n=9) / 0.22 (n=9) | n/a (no gate) | 43.2 s | 0.00 | 0.83 (10/12) |
| Naive agent (random context, no gate) | 0.99 [0.94, 1.00] | 0.99 [0.93, 1.00] | 0.94 / 0.96 | 0.230 / 0.216 | 0.00 (n=9) / 0.00 (n=9) | n/a (no gate) | 92.6 s | 2.00 | 0.83 (10/12) |
| Full harness (agent + gate) | 0.98 [0.91, 1.00] (n=16) | 0.91 [0.69, 1.00] (n=12) | 0.95 / 0.87 (accepted runs: 0.94 / 0.87) | 0.225 / 0.209 | 0.11 (n=9) / 0.00 (n=9) | 0.00 (0/2) / 0.33 (1/3) | 187 s | 5.72 | 0.25 (3/12) |
| Family-matched oracle (ORACLE) | 1.00 [1.00, 1.00] | n/a (tipping runs skipped: family not in the pool) | 0.96 / 1.00 | 0.280 / 0.500 | 0.00 (n=9) / 0.00 (n=9) | n/a (no gate) | 47.2 s | n/a (no tools) | 1.00 (9/9) |
<!-- results:end -->

The table scores one verdict per run, at the end of the series. That is a detection result, not an early-warning result. The next three tables ask the harder questions.

### Early warning

How long before the tip does the first alarm come? The alert thresholds are the ones frozen in the gate config. Nothing is tuned for this table.

<!-- early:start -->
| Split | Strategy | Tipping runs | Alarm before the tip | Lead time, median [q25, q75] | Gate-accepted runs: alarm before the tip | Gate-accepted runs: median lead time |
|---|---|---|---|---|---|---|
| In-prior test | random | 54 | 54 of 54 | 74.4 [33.7, 222.5] | 54 of 54 | 74.4 |
| In-prior test | knn | 54 | 49 of 54 | 92.8 [64.7, 255.4] | 43 of 44 | 92.8 |
| Out-of-prior | random | 36 | 35 of 36 | 330.9 [277.4, 416.6] | 34 of 35 | 329.3 |
| Out-of-prior | knn | 36 | 36 of 36 | 327.1 [255.5, 412.3] | 34 of 34 | 327.1 |

- Alert thresholds (frozen, from the gate config): random 0.1899, knn 0.1427.
- One window of the series spans 12.7 time units, so a lead time below that is no warning.
- The first alarm can be a false one: 36 of 54 held-out control trajectories (random context) also alarm in some window, so these lead times are an upper bound on useful warning.
- The first alarm can be a false one: 28 of 54 held-out control trajectories (knn context) also alarm in some window, so these lead times are an upper bound on useful warning.
<!-- early:end -->

### Sharpness

Coverage is cheap when the intervals are wide. This puts the PFN next to a prior that ignores the query.

<!-- sharp:start -->
| Split | Strategy | Median 90% interval width | Climatology width | Pinball skill vs climatology | Gate-accepted runs: pinball skill |
|---|---|---|---|---|---|
| In-prior test | random | 0.863 (0.97 of the climatology) | 0.892 | 0.244 | 0.244 |
| In-prior test | knn | 0.795 (0.89 of the climatology) | 0.892 | 0.298 | 0.383 |
| Out-of-prior | random | 0.931 (1.04 of the climatology) | 0.892 | -0.490 | -0.483 |
| Out-of-prior | knn | 0.840 (0.94 of the climatology) | 0.892 | -0.960 | -0.923 |
<!-- sharp:end -->

### Real data

Two small real sets, test only. They never enter the context pool and never tune the gate. The Daphnia labels are surrogates, and the grid slice is normal operation, so it measures false alarms only.

<!-- real:start -->
| Variant | Daphnia (surrogate labels): AUROC | Daphnia: alert on extinct populations | Daphnia: alert on persisting populations | Grid (normal operation): false-alarm rate |
|---|---|---|---|---|
| Classic EWS baseline | n/a (no run had a score) (all 9 runs decided no alert: the AUROC only ranks the confidence of one repeated decision) | 0.00 (n=6) | 0.00 (n=3) | 0.67 (n=3) |
| LLM only | 0.78 [0.42, 1.00] | 0.67 (n=6) | 0.00 (n=3) | 0.00 (n=3) |
| Naive agent (random context, no gate) | 0.06 [0.00, 0.25] (all 9 runs decided no alert: the AUROC only ranks the confidence of one repeated decision) | 0.00 (n=6) | 0.00 (n=3) | 0.67 (n=3) |
| Full harness (agent + gate) | n/a (no run had a score) (all 9 runs decided abstain: the AUROC only ranks the confidence of one repeated decision) | 0.00 (n=6) | 0.00 (n=3) | 0.67 (n=3) |
| Family-matched oracle (ORACLE) | n/a (tipping runs skipped: family not in the pool) | n/a (no such run) | n/a (no such run) | n/a (no such run) |

- Runs per variant on the Daphnia split: Classic EWS baseline: 9; LLM only: 9; Naive agent: 9; Full harness: 9; Family-matched oracle: 0.
- Runs per variant on the grid split: Classic EWS baseline: 3; LLM only: 3; Naive agent: 3; Full harness: 3; Family-matched oracle: 0.
- Decisions per variant on the Daphnia split: Classic EWS baseline: no alert 9; LLM only: no alert 5, alert 4; Naive agent: no alert 9; Full harness: abstain 9.
- Decisions per variant on the grid split: Classic EWS baseline: alert 2, no alert 1; LLM only: no alert 3; Naive agent: alert 2, no alert 1; Full harness: alert 2, no alert 1.
<!-- real:end -->

## Telling failures

Each line below is the top-ranked example of one kind of failure in `results/failures.json`. The three kinds are a false alarm on a control, a missed block by the gate, and a disagreement between the agent's recommendation and the gated decision. The failure gallery in the report shows the top two examples of every kind, with scenario plots and trace excerpts. Over-blocking, the fourth kind, is in the file and in the gallery but not here.

<!-- failures:start -->
- **False alarm:** the classic baseline alerted on the negative control `grid_fingrid/0` (real_control split), with a score of 0.925 (classic baseline score of the last window); see the [failure gallery](results/report/index.html#gallery).
- **Missed block:** the gate accepted TabPFN with knn context on `rate_L2_slow/4001` (ood split), where its prediction interval covered 0.462 of the window targets, below the line of 0.800; see the [failure gallery](results/report/index.html#gallery).
- **Disagreement:** the full harness recommended no alert with confidence 0.850 on `regime_L3/4000` (ood split), but the gate's decision was abstain; see the [failure gallery](results/report/index.html#gallery).
<!-- failures:end -->

## Running it

Create the environment with uv. It reads `pyproject.toml` and `uv.lock`.

```bash
uv sync
```

Keys go in `.env`, which git ignores. The file can hold `OPENROUTER_API_KEY` and `TABPFN_TOKEN`. The code reads them by name from the environment and never prints them. Replay needs no key.

Run the demo. It works offline, with the network disabled, because it replays recorded LLM answers and PFN outputs from `results/cache`.

```bash
make demo
```

It picks two held-out runs by rule, one the gate passed and one it blocked. It prints each step and saves the output to `results/demo_transcript.txt`. It needs the caches that `make full` leaves behind: the recorded LLM answers in `results/cache/llm`, which git tracks, and the PFN outputs in `results/cache/pfn`, which git does not. On a fresh clone, run `make full` once, with a network, before the demo works offline.

Run the full suite.

```bash
make full
```

It runs ruff, pytest, the frozen-gate check, every variant on every split, the summary and `results/run_manifest.json`. A recorded LLM answer is replayed. An answer that is not recorded is asked live, which needs `OPENROUTER_API_KEY`. `make fast` runs a smaller sample, for a quick check.

The real datasets are not in the repository. For the grid slice, read the options with `uv run python scripts/fetch_grid.py --help`. The Daphnia files are downloaded by hand, as `data/SOURCES.md` describes. Without them the real splits stop with a message that names what is missing.

Build the report.

```bash
make report
```

It writes `results/report/index.html`, one self-contained page with the verdict, the tables, the figures, the failure gallery, the method and the provenance. It fails if a number on the page is not in any result file.

`make` is not installed by default on Windows. Each target is one command, and you can run it directly: `uv run python scripts/demo.py`, `uv run python scripts/run_all.py` and `uv run python -m pfngate.report`. After new results arrive, refresh the tables in this file with `uv run python scripts/render_readme.py`.

## What is a stand-in

TabPFN stands in for TipPFN. The TipPFN weights and code are not public, so this repository cannot run TipPFN and does not reproduce it. `TipPFNBackend` in `src/pfngate/pfn/tippfn.py` is a stub. It raises `NotImplementedError` and documents the input and output that a real backend would need. Every result here comes from TabPFN. The TabPFN version is recorded in `results/run_manifest.json`.

TabPFN's newer weights are gated and non-commercial. They are downloaded on the user's own account and are not in this repository, and this repository does not redistribute them. The results here were produced with those gated weights (the model line above gives the version), so reproducing them means accepting the weights licence. This repository's own code is under the MIT licence (see `LICENSE`), which does not cover the weights. The `tabpfn` package declares an Apache licence for its code. The weights are covered by their own terms, which this repository does not quote: read them before any commercial use. Nothing in this project says how TipPFN would score.

## Limits

The data is simulated, apart from two small real sets. The simulated families are fold, subcritical Hopf and transcritical bifurcations in prior, noise-induced and rate-induced tipping out of prior, and three control families.

The two real sets are test only and never tune the gate. The Daphnia labels are surrogates: the time until extinction over a fixed horizon, not the distance to a bifurcation. The grid slice is normal operation with no tipping, so it measures false alarms only and has no AUROC.

The scenario set is small, and the runs are trial-sized. The context is reduced, each scenario has few seeds, and the language model is free. The intervals in the table are wide, and some cells are not applicable by design.

The gate is validated only on the families in the suite. The gate was tuned on in-prior validation runs only. Held-out simulated results had already been computed and read before the gate was designed, so the held-out numbers are not blind: `docs/AUDIT.md` has the timeline. Nothing shows that it holds for a system unlike those families.

The agent's tools show it some of the gate's own inputs: how far the query is from the pool, and how many near-tipping and far-from-tipping rows a context holds, at the gate's cut-offs. The agent can use that to pick a context that will pass. It still cannot change a gate result, a threshold or the freeze check.

The LLM results come from recorded answers of one free model. A replay shows what that recording produced. It is not a live test of the model.

The release criteria are illustrative, and they were written after the experiments. The verdict is a demonstration, not a decision fixed in advance.

## What I would do next with the real model

When TipPFN is available, implement its backend behind the existing interface. Run the same held-out splits without changing the gate or the splits.

Fix the release criteria before the first held-out run with the real model, and commit them first.

Run the full scenario set and the full context size, on a machine fast enough to do it.

Test the gate on families that the suite does not contain before trusting it. Its coverage is only as wide as the suite.

Keep the LLM agent only if it beats the naive agent on the held-out splits. Otherwise remove it.

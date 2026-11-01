# Audit (2026-10-31)

A skeptical review before the interview. No file other than this one was changed. Findings are ranked high, medium and low. Every claim points to a file and line, or to a command that was run.

## How the runs were done

- `make` is not installed on this laptop, and `make full --fast` is not a valid make call (the target is `make fast`). I ran the recipe directly: `scripts/run_all.py --fast`, then `scripts/demo.py`.
- To leave your tree untouched, both ran in a scratch copy of the repository (working tree with its uncommitted changes, `.git`, `data/raw`, `results/cache`), with the project's own Python.
- The network was disabled for every Python process, including the ones `run_all` starts, by a `sitecustomize` that refuses every non-loopback socket call and logs it. `UV_OFFLINE=1` was set. I did not switch the Wi-Fi off.

### Results

| Step | Result |
|---|---|
| ruff check, ruff format --check | pass |
| pytest | 498 passed, 2 failed, 1 skipped. The two failures (`tests/test_replay.py`, "network locked") are caused by my audit block: those tests install their own block and expect their own error message. Run normally they pass. |
| gate hash check | pass |
| classic baseline, all five splits | pass |
| LLM-only, naive agent, full harness | **fail offline.** The fast profile samples other trajectories than the recorded default profile, so there are no recorded answers. Every stage falls back to record mode, tries `openrouter.ai`, is blocked, and is retried. I stopped the run after 338 s; it would have spun for about half an hour and failed all LLM stages. 23 blocked attempts, all to `openrouter.ai`. |
| TabPFN with a cache miss | pass. One uncached fit and predict took 10.8 s with no network attempt (the weights are cached in the user profile, not in the repository). |
| demo | **pass.** Exit 0 in 5 s; both runs replay to the same decision, context and gate result as recorded. No network attempt by the demo. |

## High

**H1. Held-out results existed before the gate was frozen.** CLAUDE.md requires the gate to be frozen before any out-of-prior or real-data evaluation. The record shows otherwise:
- Commit `a873b25` (2026-10-26) added `results/baseline_ews.json` with out-of-prior AUROC. `scripts/run_baseline_ews.py:159` scores `test_out_of_prior` and calls no gate check at all.
- Traces `results/traces/tabpfn_local-{random,knn}-{test,ood,controls}-20261027T15*.jsonl` are ungated TabPFN runs on the in-prior test, out-of-prior and held-out controls on 2026-10-27.
- `PROGRESS.md:99-104` reports those test and out-of-prior AUROCs and the knn held-out false-alarm failure. The first gate entry is `PROGRESS.md:107`. `PROGRESS.md:121-122` acknowledges committed held-out results.
- The gate design decisions came on 2026-10-28, after that (`docs/GATE.md:412-418`). Decision 12 removed the minimum precision after tuning rounds 1 to 3 had failed to reach it.
- The tuner reads validation files only (`src/pfngate/gate/tune.py:224-226`), so no held-out data entered the thresholds. But the sentences "frozen before the held-out runs" (`README.md:83`) and "frozen before any held-out run" (`src/pfngate/report/page.py:454-455`) are not true as written, and an influence of the held-out numbers on the gate design cannot be ruled out.
- Fix: say exactly this in the README and the report, or run a fresh held-out seed set after the freeze.

**H2. Git cannot show the order of events, and no commit reproduces the numbers.** `configs/gate.yaml` and `configs/gate.frozen.sha256` first appear in commit `88d2335`, the same commit as the gated held-out results and as `assert_committed` itself. `results/run_manifest.json` records commit `118f5e8` with 178 uncommitted files, and the report prints that. Fix: commit, then rerun `make full` on a clean tree before quoting numbers.

**H3. The headline numbers measure the end of the series, not early warning.** The run-level table scores one verdict per run against the label of the last window (`results/summary.md`, "Variants, run level", notes). That is where the full harness gets 0.98 in prior and 0.91 out of prior. The window-level AUROC of the same model with p_critical is 0.83 in prior and 0.60 out of prior, and the out-of-prior CI overlaps the classic baseline (`results/summary.md`, first table). No lead-time metric exists anywhere. A physicist will ask how early the alarm comes. Fix: report AUROC or alarm rate against time to tip, and present the window-level numbers as the early-warning result.

## Medium

**M1. Number traceability holds by construction, but the test cannot catch a typed number.**
- I found no number in `README.md` or `results/report/index.html` that is not in a file in `results/`, apart from the label "90%" (low, L1).
- The README table and the report tables are verbatim copies of `results/summary.md`. The verdict comes from `results/report/criteria.json`, and the gallery and failure sentences from `results/failures.json`.
- The number check, though, accepts every two-decimal and three-decimal value between 0 and 1, and 98% of the integers up to 1000. It pools all numbers of every result file and trace (312,389 strings) at every rounding from 0 to 6 decimals and as percentages (`src/pfngate/report/check.py:39-45`). Restricted to the three files actually quoted, it still accepts 84% of two-decimal values.
- Fix: check each number against the specific field it is printed from, or say in the README that the check is a coarse guard.

**M2. The agent can see gate inputs before the gate runs.**
- `describe_context_pool` returns `share_of_windows_beyond_reference`, which is the `ood_share` value computed with the gate's own cut (`src/pfngate/agent/tools.py:75-80`).
- `choose_context` returns the near-tipping and far-from-tipping row counts at the exact cut-offs of `context_not_degenerate` (`tools.py:99-106`). With up to three contexts, the agent can pick one that will pass, and the verdict-named context is the one gated (`src/pfngate/agent/final.py`, `pick_context`).
- This does not change a GateResult, a threshold or `assert_frozen`. But the gate was tuned on contexts chosen by the harness, not by an agent that can see the checks.
- Gate integrity otherwise holds:
  - `GateResult` is frozen and refuses `model_copy` with changes (`src/pfngate/gate/gate.py:69-102`).
  - Thresholds come from the hashed file.
  - Every held-out entry point calls `assert_frozen`:
    - `src/pfngate/evals/run_pfn.py:268` for all held-out flags, controls included;
    - `src/pfngate/evals/llm_eval.py:109-113` for the LLM variants and the reference variants;
    - the real-data loaders, `src/pfngate/data/daphnia.py:266` and `src/pfngate/data/grid.py:150`;
    - the report gallery, `src/pfngate/report/gallery.py:161`.
  - No CLI flag points at another gate file.
  - The one exception is `scripts/run_baseline_ews.py`, which has no gate check (H1).

**M3. The gate is close to a single check, and coverage comes from wide intervals.**
- On held-out simulated runs, almost every block is `context_not_degenerate`: 2 in `agent_test.json` and 10 in `pfn_knn_test.json`. `ood_share` fired once per out-of-prior set. The gate never blocked a random-context run in prior (`pfn_random_test.json`).
- In-sample tuning gave precision 0.34 and recall 0.43 (`results/summary.md:137`). The full harness blocked 0 of 2 bad in-prior runs.
- The interval width advisory is exceeded on 54 of 54 random-context in-prior test runs (`results/summary.md:146`). The demo's passed run has a median 90% interval width of 0.78 on a target that lives in [0, 1].
- Coverage near 0.9 is therefore cheap. A physicist will see it.

**M4. Real data: the harness never answers on the real tipping data and false-alarms on normal grid days.**
- On all 9 Daphnia runs the gate failed `ood_share` and the harness abstained (`results/agent_real.json`).
- On the grid, which is also a system unlike anything in the pool, the gate passed all 3 days and the harness alerted on 2 (`results/agent_real_control.json`).
- The naive agent's Daphnia AUROC of 0.06 is an artefact. It said no alert on all 9 runs, and the AUROC only ranks its confidence among identical decisions (`src/pfngate/evals/agent_metrics.py:41-43`).

**M5. The pass and fail results rest on very few runs.** False alarms on simulated controls pass at 1 of 18; the exact 95% upper bound is about 0.27, well above the 0.10 threshold. Gate recall is 0 of 2. Coverage rests on 16 accepted runs. The criteria compare point estimates with no interval (`src/pfngate/report/criteria.py`), so a few runs could flip the verdict.

**M6. Fragility for a live demo.**
- `make` is not installed here, so `make demo` fails as typed. The recipe works.
- The demo needs `results/cache/pfn` (4,005 files, 47 MB), which git does not track. A fresh clone cannot run it offline.
- `scripts/run_all.py:77,170-177` retries any record failure five times with a 20 s sleep, not only provider timeouts. Offline, `make full` and `make fast` spin for about half an hour, and the fast profile cannot run offline at all.
- The tests that cover the offline claims skip themselves without local results and caches (`tests/test_demo.py:173`, `tests/test_render_readme.py:73-88`, `tests/test_report.py:325`). A green run on a fresh clone proves nothing about them.
- `make demo` rewrites the tracked `results/demo/` and `results/demo_transcript.txt`, which leaves the tree dirty. The transcript contains your Windows user path.
- The report's real-data plots need `data/raw`, which is not in git.
- `uv run --no-project` once failed here to query the Windows Store Python.

**M7. Unverified citation.** `docs/LABELS.md:7` and `src/pfngate/pfn/tippfn.py:14` cite TipPFN as arXiv 2605.12308, and `tippfn.py:18` says the paper was not read. The stub's input and output description is "as we understand it". It will be read by the people who wrote the paper. Fix: check the identifier and the interface against the paper, or remove both.

**M8. Licence wording.**
- `README.md:73` says TabPFN's newer weights are gated and non-commercial. It does not say that the results here were produced with such weights (v3.5, downloaded after accepting a licence, `PROGRESS.md`, 2026-10-26 entry).
- The `tabpfn` package metadata says Apache-2.0, which covers the code, not the weights. I did not read the weights licence text.
- The repository has no LICENSE file and no licence field in `pyproject.toml`.

## Low

- **L1.** "90%" is typed by hand in the gallery label (`src/pfngate/report/page.py:73`) and the criteria labels (`configs/release_criteria.yaml:21`). It matches `bad_run.interval` in `configs/gate.yaml`.
- **L2.** The report's provenance prints "wall clock (s)" from the replay rerun, not the recorded run (`page.py:526`). The latencies are the recorded ones.
- **L3.** knn chooses one context per trajectory from all its windows, the late ones included (`src/pfngate/pfn/context.py:146-175`). That is a mild look-ahead for the window-level AUROC tables, not for run-level verdicts.
- **L4.** The in-prior test uses the same scenario parameters as the context pool, with other seeds. It is documented, but in-prior numbers measure interpolation within known settings.
- **L5.** "Noise-induced tipping" ramps the noise level (`configs/scenarios/noise_tipping.yaml:19-37`), so variance rises before the escape. AUROC above chance there comes from growing noise, not critical slowing down. A physicist may object to the name.
- **L6.** The full harness gives the same decision on all three seeds in only 3 of 12 scenarios (pass^3, `results/summary.md`). The LLM is one free model, and the results are replays of one recording.
- **L7.** Two commit subjects are garbled (`71ea8a5`, `ed696a5`). `interview.docx` and the fast PFN cache are untracked now but remain in history.

## Checked and found clean

- **Leakage:**
  - A query trajectory is never in its own context (`context.py:103`). The pool is seed set A only.
  - `select_context` refuses real-data rows, and the tuner reads validation files only.
  - Window features use only the window's own observations, and windows after the tip are dropped (`src/pfngate/features/dataset.py:40-63`).
- **Secrets and data:** no `.env`, token, weight file or raw data file is tracked. The two values in `.env` appear in no tracked file. The only key-like match is an instruction string in `scripts/smoke_tabpfn.py:81`, and the replay and test API keys are placeholders. No tracked code holds an absolute path.
- **Claims:** no sentence says or implies that TipPFN is reproduced, available or used, or that the results show field performance. The README, report and stub all say the opposite.

## The five things most likely to embarrass you

1. **"Frozen before the held-out runs" is not true as written.** Held-out and out-of-prior results were computed and read a day before the gate existed, and git cannot show the order (H1, H2).
2. **The 0.98 and 0.91 headline is end-of-series detection.** The early-warning number out of prior is 0.60 with a CI that overlaps the classic baseline, and there is no lead time (H3).
3. **On real data the harness abstains on every tipping run and false-alarms on 2 of 3 normal grid days.** The naive agent's 0.06 AUROC on Daphnia is an artefact of identical decisions (M4).
4. **The gate's coverage comes from intervals that span most of the target range.** In practice the gate is one check about the context, not the query, and it blocked neither bad in-prior run (M3).
5. **An unverified arXiv identifier and interface description for TipPFN, in front of its authors (M7).**

Close behind: the demo on any machine other than this laptop (M6), and the number check, which cannot catch a typed number (M1).

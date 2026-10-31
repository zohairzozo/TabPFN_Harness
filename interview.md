# Interview notes

## Report these agent failures, even when the headline numbers look good

- **An agent that always picks the same strategy.** If the agent chooses the same context
  strategy (for example `knn`) on every run, it is not choosing. It is a fixed rule that has
  been given the cost of a language model. Report it as that.
- **An agent that confidently recommends `alert` on controls.** Control runs have nothing
  tipping. An `alert` recommendation there is a false alarm, and a confident one is the worst
  kind, because a person is more likely to act on it. Report it, and say whether the gate
  stopped it.

Both are computed by `scripts/agent_checks.py` and written to `results/agent_checks.json`.
Read the numbers from that file. Do not type them here.

- Strategy: the number of runs per strategy, for each agent variant and split. The naive agent
  does not choose its context, so only the full harness is meaningful here.
- Controls: the number of control runs where the agent recommended `alert`, the number where
  that recommendation was confident (confidence at or above the `confident_at` value in the
  file), and the number where the final decision, after the gate, was `alert`. A control run
  can also end with no recommendation at all, and that is counted separately.

How to read it: the agent's own recommendation and the final decision are different things.
The gate, written in plain Python, decides. An `alert` on a control that came from the gate's
decision, and not from the agent's recommendation, is a gate finding, not an agent finding.

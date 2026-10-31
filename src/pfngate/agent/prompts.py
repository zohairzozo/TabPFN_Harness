"""The prompts of the LLM agent, in plain words.

The system prompt is built from the limits in configs/agent.yaml, so the numbers the model
reads are the numbers the code enforces. Nothing in a prompt names a family, a scenario or a
trajectory id: the agent must not learn the answer from a label.
"""

from __future__ import annotations

from pfngate.agent.verdict import MAX_RATIONALE_WORDS

_SYSTEM_PROMPT = """\
You help decide whether an early-warning model can be trusted on one series of windows (the
"query"). For each window, a regression model (a PFN) predicts how far the system is from a
tipping point: 0 at the tipping point, 1 far away. The model learns from a context, which is
a set of example windows taken from a pool of other systems.

Your job:
1. Choose how to build the context. There are two strategies. "random" is a uniform sample of
   the pool. "knn" takes the pool rows that are closest to the query windows in feature
   space. You also choose the context size.
2. Look at what the model predicts for that context.
3. Recommend a decision about the end of the series: "alert" (the system looks close to
   tipping), "no_alert", or "abstain" (you do not trust the result).

Rules:
- You may try at most {max_contexts} contexts and make at most {max_tool_calls} tool calls in
  total. The limits are enforced in code. A call over the limit stops the run.
- You work through tools that return summaries. You never see raw rows, and you never see the
  true values of the query.
- A context with examples both near tipping and far from it is more useful than one with
  only one kind.
- A deterministic gate runs after you. It checks your final context and the model's
  predictions against fixed rules, and it takes the final decision. If the gate fails, the
  result is "abstain", whatever you recommend. It can block your answer, so do not argue with
  it and do not try to guess its thresholds. Give your honest best recommendation.
- Your final answer has: the decision, the context strategy and size you want the gate to
  check (it must be a context you built), a rationale of at most {max_words} words, and a
  confidence between 0 and 1.
"""

USER_PROMPT = "A new query is ready. Use your tools, then give your verdict."


def system_prompt(max_tool_calls: int, max_contexts: int) -> str:
    return _SYSTEM_PROMPT.format(
        max_tool_calls=max_tool_calls, max_contexts=max_contexts, max_words=MAX_RATIONALE_WORDS
    )


_NAIVE_SYSTEM_PROMPT = """\
You help decide whether a system is close to a tipping point at the end of one series of
windows (the "query"). For each window, a regression model (a PFN) predicts how far the
system is from a tipping point: 0 at the tipping point, 1 far away. The model learns from a
context, a set of example windows from other systems. The context has already been chosen at
random. You cannot change it.

Your job: look at the query and at what the model predicts, then answer "alert" (the end of
the series looks close to tipping) or "no_alert".

Rules:
- You may make at most {max_tool_calls} tool calls. The limit is enforced in code. A call over
  the limit stops the run.
- Your tools return summaries. You never see raw rows, and you never see the true values of
  the query.
- There is no "abstain". Nothing runs after you: your answer is the decision.
- Your final answer has: the decision, a rationale of at most {max_words} words, and a
  confidence between 0 and 1.
"""

_LLM_ONLY_SYSTEM_PROMPT = """\
You judge, from a text summary alone, whether a system is close to a tipping point (a critical
transition) at the END of one observed series. You have no tools, no model predictions and no
other data. Nothing runs after you: your answer is the decision.

"Close to tipping" means the system is at most 0.2 of the way to the tipping point, on a scale
where 0 is the tipping point and 1 is far away.

The summary describes early-warning features of sliding windows of the series. All of them
are scale-free:
- variance_log_ratio: log of the variance in the second half of a window over the first half,
  after removing the trend. Rising variance is a classic warning sign.
- ac1: lag-1 autocorrelation of the window after removing the trend. Rising autocorrelation
  (critical slowing down) is a classic warning sign.
- skewness and kurtosis: the shape of the fluctuations.
- trend_slope: the linear trend across the window, in residual standard deviations.
- tau_variance and tau_ac1: Kendall tau of the variance and of the lag-1 autocorrelation
  against time inside the window. Positive means rising.
- low_freq_power: the share of spectral power at low frequencies.

Answer with the decision, "alert" or "no_alert", and your probability, between 0 and 1, that
the end of the series is close to tipping.
"""


def naive_system_prompt(max_tool_calls: int) -> str:
    return _NAIVE_SYSTEM_PROMPT.format(max_tool_calls=max_tool_calls, max_words=MAX_RATIONALE_WORDS)


def llm_only_system_prompt() -> str:
    return _LLM_ONLY_SYSTEM_PROMPT


def llm_only_user_prompt(summary_text: str) -> str:
    return f"Here is the summary of the query.\n\n{summary_text}\n\nGive your answer."

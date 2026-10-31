"""Tests for the run-level metrics of the LLM variants, with values worked out by hand."""

from __future__ import annotations

from typing import Any

import pytest

from pfngate.evals.agent_metrics import RunOutcome, alert_score, end_label, source_metrics

KW = {"n_boot": 200, "seed": 0, "ci": 0.95}
O = RunOutcome  # noqa: E741


def example() -> list[RunOutcome]:
    return [
        O("t1", "in_prior", "positive", "alert", 0.9),
        O("t2", "in_prior", "positive", "no_alert", 0.4),
        O("t3", "in_prior", "positive", "abstain", None),
        O("t4", "in_prior", "middle", "alert", 0.7),
        O("c1", "control", "negative", "no_alert", 0.1),
        O("c2", "control", "negative", "alert", 0.6),
    ]


def test_labels_and_scores() -> None:
    assert (end_label(1), end_label(0), end_label(-1)) == ("positive", "negative", "middle")
    assert alert_score("alert", 0.8) == 0.8
    assert alert_score("no_alert", 0.8) == pytest.approx(0.2)


def test_the_rates_by_hand() -> None:
    m = source_metrics(example(), **KW)
    assert m["n_runs"] == 6 and m["decisions"] == {"alert": 3, "no_alert": 2, "abstain": 1}
    assert (m["n_positive_end"], m["n_negative_end"], m["n_middle_end"]) == (3, 2, 1)
    assert m["abstention"]["value"] == pytest.approx(1 / 6)
    assert m["recall_end"]["value"] == pytest.approx(1 / 3)  # t1 of t1, t2, t3: abstain is no alert
    assert m["recall_end"]["n"] == 3
    assert m["false_alarm_end"]["value"] == pytest.approx(1 / 2)  # c2 of c1, c2
    assert m["control_alert"]["value"] == pytest.approx(1 / 2)
    # Alerts are t1, t4 and c2. t4 ends in the middle band and is left out: 1 right of 2.
    assert m["precision_end"]["value"] == pytest.approx(1 / 2) and m["precision_end"]["n"] == 2


def test_auroc_by_hand() -> None:
    m = source_metrics(example(), **KW)
    # Scored runs that are not middle: positives 0.9 and 0.4, negatives 0.1 and 0.6.
    # Pairs won: 0.9>0.1, 0.9>0.6, 0.4>0.1. Lost: 0.4<0.6. So 3 of 4.
    assert m["n_scored"] == 4 and m["auroc_end"]["auroc"] == pytest.approx(0.75)


def test_auroc_needs_both_classes() -> None:
    only_positive = [o for o in example() if o.end_label == "positive"]
    assert source_metrics(only_positive, **KW)["auroc_end"] is None
    assert source_metrics([], **KW) == {"n_runs": 0}


def test_an_abstention_is_not_an_alert_and_has_no_score() -> None:
    runs = [O(f"t{i}", "in_prior", "positive", "abstain", None) for i in range(4)]
    runs += [O(f"c{i}", "control", "negative", "abstain", None) for i in range(4)]
    m = source_metrics(runs, **KW)
    assert m["abstention"]["value"] == 1.0
    assert m["recall_end"]["value"] == 0.0 and m["false_alarm_end"]["value"] == 0.0
    assert m["precision_end"]["value"] is None and m["auroc_end"] is None  # nothing to score


def test_the_intervals_are_inside_zero_one_and_fixed_by_the_seed() -> None:
    a, b = source_metrics(example(), **KW), source_metrics(example(), **KW)
    assert a == b  # the same seed, the same intervals
    for key in ("recall_end", "false_alarm_end", "control_alert", "precision_end", "abstention"):
        lo, hi = a[key]["ci"]
        assert 0.0 <= lo <= a[key]["value"] <= hi <= 1.0, key
    other = source_metrics(example(), n_boot=200, seed=1, ci=0.95)
    assert (
        other["recall_end"]["value"] == a["recall_end"]["value"]
    )  # the value does not depend on it


def test_a_perfect_variant_and_a_variant_that_always_alerts() -> None:
    perfect = [O(f"t{i}", "in_prior", "positive", "alert", 0.9) for i in range(5)]
    perfect += [O(f"c{i}", "control", "negative", "no_alert", 0.1) for i in range(5)]
    p: dict[str, Any] = source_metrics(perfect, **KW)
    assert p["recall_end"]["value"] == 1.0 and p["false_alarm_end"]["value"] == 0.0
    assert p["auroc_end"]["auroc"] == 1.0 and p["precision_end"]["value"] == 1.0
    always = [O(o.trajectory_id, o.role, o.end_label, "alert", 0.9) for o in perfect]
    a = source_metrics(always, **KW)
    assert a["recall_end"]["value"] == 1.0 and a["false_alarm_end"]["value"] == 1.0
    assert a["auroc_end"]["auroc"] == 0.5  # one score for everything ranks nothing


# The trial sample
def synthetic_queries(n_tipping: int, n_control: int, seeds: int = 3) -> list[Any]:
    import pandas as pd

    frames = []
    for role, n, prefix in (("in_prior", n_tipping, "t"), ("control", n_control, "c")):
        for i in range(n):
            for seed in range(seeds):
                frames.append(pd.DataFrame({"scenario_id": f"{prefix}{i}", "role": role,
                                            "trajectory_id": f"{prefix}{i}/{seed}",
                                            "window_end": [1, 2]}))  # fmt: skip
    return frames


def test_a_trial_sample_keeps_whole_scenarios_and_both_kinds() -> None:
    from pfngate.evals.llm_eval import select_scenarios

    queries = synthetic_queries(n_tipping=18, n_control=9)
    assert select_scenarios(queries, None) is queries
    picked = select_scenarios(queries, 3)
    ids = [str(q["trajectory_id"].iloc[0]) for q in picked]
    scenarios = []
    for i in ids:
        if i.split("/")[0] not in scenarios:
            scenarios.append(i.split("/")[0])
    assert scenarios == ["t0", "t8", "t17", "c0", "c4", "c8"]  # evenly spread (8.5 rounds to 8)
    assert len(ids) == 18 and all(ids.count(f"{s}/{k}") == 1 for s in scenarios for k in range(3))
    assert len(select_scenarios(queries, 99)) == len(queries)  # more than exist: all of them
    assert len(select_scenarios(synthetic_queries(4, 0), 2)) == 6  # no controls: tipping only
    with pytest.raises(ValueError, match="at least 1"):
        select_scenarios(queries, 0)

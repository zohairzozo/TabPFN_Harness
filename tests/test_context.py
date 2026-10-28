"""Tests for context selection strategies in src/pfngate/pfn/context.py."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pfngate.features import FEATURE_NAMES, build_dataset, load_feature_config
from pfngate.pfn import ContextSelection, load_pfn_config, select_context
from pfngate.pfn.context import (
    ORACLE_STRATEGIES,
    STRATEGIES,
    NoMatchingFamilyError,
    context_hash,
    context_xy,
    family_matched_context,
    knn_balanced_context,
    knn_context,
    random_context,
    row_ids,
)

CFG = load_pfn_config()
FAMILIES = ("fold", "hopf", "control_ou")
SPACING = 12.0  # distance between family centres in feature space, far above the noise


def make_pool(n_traj: int = 4, n_windows: int = 40, seed: int = 0) -> pd.DataFrame:
    """Synthetic context pool: each family sits around its own centre in feature space."""
    rng = np.random.default_rng(seed)
    frames = []
    for fi, fam in enumerate(FAMILIES):
        for t in range(n_traj):
            feats = SPACING * fi + rng.standard_normal((n_windows, len(FEATURE_NAMES)))
            df = pd.DataFrame(feats, columns=list(FEATURE_NAMES))
            df.insert(0, "trajectory_id", f"{fam}_s/{1000 + t}")
            df.insert(1, "family", fam)
            df.insert(2, "window_end", 127 + 16 * np.arange(n_windows))
            df.insert(3, "target", rng.random(n_windows))
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def make_query(
    family: str = "fold", n: int = 6, tid: str = "query/9999", seed: int = 1
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    centre = SPACING * FAMILIES.index(family)
    df = pd.DataFrame(
        centre + rng.standard_normal((n, len(FEATURE_NAMES))), columns=list(FEATURE_NAMES)
    )
    df.insert(0, "trajectory_id", tid)
    df.insert(1, "family", family)
    df.insert(2, "window_end", 127 + 16 * np.arange(n))
    return df


POOL = make_pool()
CAP = 50


# Caps
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_cap_is_respected(strategy: str) -> None:
    sel = select_context(strategy, POOL, make_query(), CFG, max_context=CAP)
    assert 0 < sel.size <= CAP
    assert len(sel.row_ids) == sel.size and sel.max_context == CAP
    assert sel.context["row_id"].is_unique


def test_cap_default_comes_from_the_config() -> None:
    assert CFG.max_context == 2000
    big = make_pool(n_traj=60, n_windows=40)  # 7200 rows
    assert select_context("random", big, make_query(), CFG).size == CFG.max_context


def test_small_pool_is_taken_whole_by_random_and_family_matched() -> None:
    sel = random_context(POOL, make_query(), max_context=10_000, seed=0)
    assert sel.size == len(POOL)
    fam = family_matched_context(POOL, make_query(), max_context=10_000, seed=0)
    assert fam.size == (POOL["family"] == "fold").sum()


# Determinism
def test_random_is_deterministic_for_a_seed_and_changes_with_it() -> None:
    a = random_context(POOL, make_query(), max_context=CAP, seed=3)
    b = random_context(POOL, make_query(), max_context=CAP, seed=3)
    c = random_context(POOL, make_query(), max_context=CAP, seed=4)
    assert a.row_ids == b.row_ids and a.context_hash == b.context_hash
    assert a.context_hash != c.context_hash


def test_random_gives_each_query_trajectory_its_own_draw() -> None:
    a = random_context(POOL, make_query(tid="query/1"), max_context=CAP, seed=0)
    b = random_context(POOL, make_query(tid="query/2"), max_context=CAP, seed=0)
    assert a.context_hash != b.context_hash


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_result_does_not_depend_on_pool_order(strategy: str) -> None:
    shuffled = POOL.sample(frac=1.0, random_state=7).reset_index(drop=True)
    a = select_context(strategy, POOL, make_query(), CFG, max_context=CAP)
    b = select_context(strategy, shuffled, make_query(), CFG, max_context=CAP)
    assert a.row_ids == b.row_ids and a.context_hash == b.context_hash


def test_knn_is_deterministic() -> None:
    a = knn_context(POOL, make_query(), max_context=CAP, knn_neighbors=10)
    b = knn_context(POOL, make_query(), max_context=CAP, knn_neighbors=10)
    assert a.row_ids == b.row_ids


# Hash
def test_context_hash_is_stable_and_order_free() -> None:
    assert context_hash(["a#1", "b#2"]) == context_hash(["b#2", "a#1"])
    assert context_hash(["a#1", "b#2"]) != context_hash(["a#1", "b#3"])
    assert len(context_hash(["a#1"])) == 16
    sel = random_context(POOL, make_query(), max_context=CAP, seed=0)
    assert sel.context_hash == context_hash(sel.row_ids)


def test_row_ids_are_unique_per_window() -> None:
    assert row_ids(POOL).is_unique


# knn
def test_knn_takes_rows_near_the_query() -> None:
    sel = knn_context(POOL, make_query("hopf"), max_context=30, knn_neighbors=10)
    assert set(sel.context["family"]) == {"hopf"}


def test_knn_fills_by_neighbour_rank_across_queries() -> None:
    # Two query windows far apart. With room for only two rows, each must get its own
    # nearest row, not both rows from one query.
    q = pd.concat([make_query("fold", n=1, tid="query/1"), make_query("control_ou", n=1)])
    q["trajectory_id"] = "query/1"
    sel = knn_context(POOL, q, max_context=2, knn_neighbors=20)
    assert sorted(sel.context["family"]) == ["control_ou", "fold"]


def test_knn_neighbour_is_the_true_nearest_in_standardised_space() -> None:
    q = make_query(n=1)
    sel = knn_context(POOL, q, max_context=1, knn_neighbors=5)
    x = POOL[list(FEATURE_NAMES)].to_numpy()
    z = (x - x.mean(0)) / x.std(0)
    zq = (q[list(FEATURE_NAMES)].to_numpy() - x.mean(0)) / x.std(0)
    best = POOL.iloc[int(np.argmin(((z - zq) ** 2).sum(1)))]
    assert sel.row_ids == (f"{best['trajectory_id']}#{best['window_end']}",)


# knn_balanced
def balanced(pool: pd.DataFrame, q: pd.DataFrame, cap: int = 40) -> ContextSelection:
    return knn_balanced_context(pool, q, max_context=cap, knn_neighbors=10, near_share=0.25,
                                near_max=0.2)  # fmt: skip


def test_knn_balanced_reserves_near_tipping_rows_for_a_query_far_from_them() -> None:
    # Only fold rows are near tipping. A control query gets its knn rows from its own
    # family, plus the reserved quota of near-tipping rows, which can only be fold rows.
    pool = POOL.copy()
    pool["target"] = np.where(pool["family"] == "fold", 0.1, 0.9)
    sel = balanced(pool, make_query("control_ou"))
    assert sel.size == 40 and sel.strategy == "knn_balanced" and not sel.oracle
    counts = sel.context["family"].value_counts().to_dict()
    assert counts == {"control_ou": 30, "fold": 10}
    assert knn_context(pool, make_query("control_ou"), max_context=40,
                       knn_neighbors=10).context["family"].eq("control_ou").all()  # fmt: skip


def test_knn_balanced_is_knn_when_the_pool_has_no_near_tipping_rows() -> None:
    pool = POOL.copy()
    pool["target"] = 0.9
    q = make_query("hopf")
    assert balanced(pool, q).row_ids == knn_context(pool, q, max_context=40,
                                                    knn_neighbors=10).row_ids  # fmt: skip


def test_knn_balanced_rows_are_near_tipping_or_knn_rows() -> None:
    q = make_query("hopf")
    sel = balanced(POOL, q)
    knn_ids = set(knn_context(POOL, q, max_context=40, knn_neighbors=10).row_ids)
    near = sel.context["target"] <= 0.2
    assert near.sum() >= 10  # the quota, round(0.25 * 40)
    assert set(sel.context.loc[~near, "row_id"]) <= knn_ids
    assert balanced(POOL, q).row_ids == sel.row_ids  # deterministic


def test_knn_balanced_never_reads_the_query_family() -> None:
    q = make_query("hopf")
    assert balanced(POOL, q.drop(columns="family")).row_ids == balanced(POOL, q).row_ids


def test_knn_balanced_share_comes_from_the_config() -> None:
    assert CFG.balanced_near_share == 0.25
    sel = select_context("knn_balanced", POOL, make_query(), CFG, max_context=CAP)
    assert (sel.context["target"] <= CFG.critical_threshold).sum() >= round(0.25 * CAP)


# family_matched is an oracle and only uses its own family
def test_family_matched_only_uses_its_family_and_is_marked_oracle() -> None:
    for fam in FAMILIES:
        sel = family_matched_context(POOL, make_query(fam), max_context=CAP, seed=0)
        assert set(sel.context["family"]) == {fam}
        assert sel.oracle is True and "ORACLE" in sel.label
    assert ORACLE_STRATEGIES == {"family_matched"}
    for strategy in ("random", "knn", "knn_balanced"):
        sel = select_context(strategy, POOL, make_query(), CFG, max_context=CAP)
        assert sel.oracle is False and "ORACLE" not in sel.label


def test_family_matched_with_unknown_family_raises() -> None:
    q = make_query()
    q["family"] = "rate_tipping"
    with pytest.raises(NoMatchingFamilyError):
        family_matched_context(POOL, q, max_context=CAP, seed=0)


# A query trajectory never contributes rows to its own context
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_query_trajectory_never_contributes_to_its_own_context(strategy: str) -> None:
    own = POOL[POOL["trajectory_id"] == "fold_s/1000"].copy()  # a trajectory that IS in the pool
    sel = select_context(strategy, POOL, own, CFG, max_context=10_000)
    assert "fold_s/1000" not in set(sel.context["trajectory_id"])
    assert sel.query_trajectory_id == "fold_s/1000"
    # Everything else is still available, so the exclusion is the only thing removed.
    others = (POOL["trajectory_id"] != "fold_s/1000").sum()
    if strategy == "random":
        assert sel.size == others


# Input checks
def test_query_must_be_one_trajectory_and_finite() -> None:
    two = pd.concat([make_query(tid="query/1"), make_query(tid="query/2")])
    with pytest.raises(ValueError, match="exactly one trajectory"):
        random_context(POOL, two, max_context=CAP, seed=0)
    bad = make_query()
    bad.loc[0, FEATURE_NAMES[0]] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        random_context(POOL, bad, max_context=CAP, seed=0)
    with pytest.raises(ValueError, match="missing columns"):
        random_context(POOL.drop(columns=["target"]), make_query(), max_context=CAP, seed=0)


def test_unknown_strategy_and_bad_cap_raise() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        select_context("nope", POOL, make_query(), CFG)
    with pytest.raises(ValueError, match="at least 1"):
        select_context("random", POOL, make_query(), CFG, max_context=0)


def test_context_xy_shapes_and_targets() -> None:
    sel = random_context(POOL, make_query(), max_context=CAP, seed=0)
    x, y = context_xy(sel)
    assert x.shape == (CAP, len(FEATURE_NAMES)) and y.shape == (CAP,)
    assert np.array_equal(y, sel.context["target"].to_numpy())
    assert isinstance(sel, ContextSelection)


# Real scenario data
@pytest.fixture(scope="module")
def real(tmp_path_factory: pytest.TempPathFactory) -> dict[str, pd.DataFrame]:
    tmp = tmp_path_factory.mktemp("cache")
    fc = dataclasses.replace(load_feature_config(), cache_dir=tmp, summary_dir=tmp)
    return {
        s: build_dataset(s, seeds_per_scenario=2, feature_config=fc)
        for s in ("context", "validation", "test_out_of_prior")
    }


def one_trajectory(df: pd.DataFrame, family: str) -> pd.DataFrame:
    sub = df[df["family"] == family]
    return sub[sub["trajectory_id"] == sub["trajectory_id"].iloc[0]]


def test_real_pool_has_no_out_of_prior_family(real: dict[str, pd.DataFrame]) -> None:
    assert set(real["context"]["role"]) <= {"in_prior", "control"}


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_real_caps_and_exclusion(strategy: str, real: dict[str, pd.DataFrame]) -> None:
    pool = real["context"]
    own = one_trajectory(pool, "hopf")  # a pool trajectory used as the query
    sel = select_context(strategy, pool, own, CFG, max_context=100)
    assert 0 < sel.size <= 100
    assert own["trajectory_id"].iloc[0] not in set(sel.context["trajectory_id"])


def test_real_family_matched_uses_only_its_family(real: dict[str, pd.DataFrame]) -> None:
    q = one_trajectory(real["validation"], "fold")
    sel = select_context("family_matched", real["context"], q, CFG, max_context=100)
    assert set(sel.context["family"]) == {"fold"} and sel.oracle


def test_real_family_matched_has_no_context_for_out_of_prior_family(
    real: dict[str, pd.DataFrame],
) -> None:
    q = one_trajectory(real["test_out_of_prior"], "rate_tipping")
    with pytest.raises(NoMatchingFamilyError):
        select_context("family_matched", real["context"], q, CFG)
    # The non-oracle strategies still work for it.
    assert select_context("knn", real["context"], q, CFG, max_context=100).size > 0


def test_real_selection_is_deterministic(real: dict[str, pd.DataFrame]) -> None:
    q = one_trajectory(real["validation"], "transcritical")
    for strategy in STRATEGIES:
        a = select_context(strategy, real["context"], q, CFG, max_context=100)
        b = select_context(strategy, real["context"], q, CFG, max_context=100)
        assert a.context_hash == b.context_hash, strategy


def test_real_pool_path_is_not_the_results_cache(tmp_path: Path) -> None:
    # Guard for the fixture above: it must write to a temp dir, never to results/cache.
    fc = dataclasses.replace(load_feature_config(), cache_dir=tmp_path, summary_dir=tmp_path)
    assert fc.cache_dir == tmp_path

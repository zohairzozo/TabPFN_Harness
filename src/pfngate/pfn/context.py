"""Context selection: which rows of the context pool the PFN sees for one query trajectory.

The pool is the table of windows from the context split (seed set A, see docs/SPLITS.md),
as returned by pfngate.features.build_dataset("context"). The query is a batch of windows
from ONE trajectory. Every strategy

  - drops the query trajectory from the pool first, so a trajectory never sees itself;
  - caps the context at max_context rows;
  - sorts rows by row id, so the result does not depend on the order of the pool;
  - returns the chosen row ids and a stable hash of them.

Context targets are the regression target of each window ("target"), for every window
including those the binary task drops, because the PFN is a regressor.
"""

from __future__ import annotations

import hashlib
import zlib
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from pfngate.data import REAL_SPLITS
from pfngate.features import FEATURE_NAMES
from pfngate.pfn.config import PFNConfig, load_pfn_config

STRATEGIES: tuple[str, ...] = ("random", "knn", "knn_balanced", "family_matched")
# Oracles read the answer (the family of the query) that a real run would not know.
ORACLE_STRATEGIES: frozenset[str] = frozenset({"family_matched"})

_TINY = 1e-12


class NoMatchingFamilyError(ValueError):
    """family_matched found no pool rows of the query family (e.g. an out-of-prior family)."""


@dataclass(frozen=True, eq=False)
class ContextSelection:
    strategy: str
    oracle: bool
    query_trajectory_id: str
    max_context: int
    context: pd.DataFrame  # chosen pool rows, sorted by row_id, with a row_id column
    row_ids: tuple[str, ...]
    context_hash: str  # 16 hex characters, see context_hash()

    @property
    def size(self) -> int:
        return len(self.context)

    @property
    def label(self) -> str:
        """Strategy name for traces and tables. Oracles are marked."""
        return f"{self.strategy} (ORACLE)" if self.oracle else self.strategy


def row_ids(df: pd.DataFrame) -> pd.Series:
    """One stable id per window: trajectory id and window end index."""
    return df["trajectory_id"].astype(str) + "#" + df["window_end"].astype(int).astype(str)


def context_hash(ids: Iterable[str]) -> str:
    """Hash of a set of row ids. The same rows give the same hash, in any order."""
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]


def context_xy(selection: ContextSelection) -> tuple[np.ndarray, np.ndarray]:
    """Features and regression target of the chosen context rows, ready for a backend."""
    df = selection.context
    return df[list(FEATURE_NAMES)].to_numpy(float), df["target"].to_numpy(float)


def query_x(query: pd.DataFrame) -> np.ndarray:
    return query[list(FEATURE_NAMES)].to_numpy(float)


def _prepare(pool: pd.DataFrame, query: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Check inputs, drop the query trajectory from the pool, sort by row id."""
    missing = {"trajectory_id", "window_end", "target", *FEATURE_NAMES} - set(pool.columns)
    if missing:
        raise ValueError(f"pool is missing columns {sorted(missing)}")
    if not {"trajectory_id", *FEATURE_NAMES} <= set(query.columns):
        raise ValueError("query needs trajectory_id and the feature columns")
    tids = query["trajectory_id"].unique()
    if len(tids) != 1:
        raise ValueError(f"query must come from exactly one trajectory, got {len(tids)}")
    if len(query) == 0:
        raise ValueError("query is empty")
    feats = list(FEATURE_NAMES)
    if not np.isfinite(pool[feats].to_numpy(float)).all():
        raise ValueError("pool has NaN or infinite features")
    if not np.isfinite(query[feats].to_numpy(float)).all():
        raise ValueError("query has NaN or infinite features")

    tid = str(tids[0])
    eligible = pool[pool["trajectory_id"] != tid].copy()
    eligible["row_id"] = row_ids(eligible)
    eligible = eligible.sort_values("row_id", kind="stable").reset_index(drop=True)
    return eligible, tid


def _finish(
    strategy: str, eligible: pd.DataFrame, positions: np.ndarray, tid: str, cap: int
) -> ContextSelection:
    context = eligible.iloc[np.sort(positions)].reset_index(drop=True)
    ids = tuple(context["row_id"])
    return ContextSelection(
        strategy=strategy,
        oracle=strategy in ORACLE_STRATEGIES,
        query_trajectory_id=tid,
        max_context=cap,
        context=context,
        row_ids=ids,
        context_hash=context_hash(ids),
    )


def _sample_positions(n: int, cap: int, seed: int, tid: str) -> np.ndarray:
    """Uniform sample of cap positions out of n, or all of them if n <= cap.

    The generator is seeded by the seed and the query trajectory id, so one seed gives
    each query trajectory its own draw, and the same draw every time it is repeated.
    """
    if n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(tid.encode())]))
    return rng.choice(n, size=cap, replace=False)


def random_context(
    pool: pd.DataFrame, query: pd.DataFrame, *, max_context: int, seed: int
) -> ContextSelection:
    """Uniform random sample of the pool."""
    eligible, tid = _prepare(pool, query)
    positions = _sample_positions(len(eligible), max_context, seed, tid)
    return _finish("random", eligible, positions, tid, max_context)


def knn_context(
    pool: pd.DataFrame, query: pd.DataFrame, *, max_context: int, knn_neighbors: int
) -> ContextSelection:
    """Nearest pool rows to the query windows, in standardised feature space.

    Features are standardised with the mean and standard deviation of the eligible pool.
    Each query window contributes its knn_neighbors nearest rows. The union is filled by
    neighbour rank: first the nearest row of every query window, then the second nearest,
    and so on, until max_context rows are chosen. The result can be smaller than the cap
    if the neighbours overlap. It is deterministic: no random numbers are used.
    """
    eligible, tid = _prepare(pool, query)
    z, q = _standardised(eligible, query)
    by_rank = _by_neighbour_rank(z, q, knn_neighbors)
    return _finish("knn", eligible, by_rank[:max_context], tid, max_context)


def _standardised(eligible: pd.DataFrame, query: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Pool and query features, standardised with the eligible pool's mean and std."""
    x = eligible[list(FEATURE_NAMES)].to_numpy(float)
    mean, sd = x.mean(axis=0), x.std(axis=0)
    sd = np.where(sd > _TINY, sd, 1.0)
    return (x - mean) / sd, (query_x(query) - mean) / sd


def _by_neighbour_rank(z: np.ndarray, q: np.ndarray, knn_neighbors: int) -> np.ndarray:
    """Positions in z: the nearest row of every query window first, then the second, ..."""
    if len(z) == 0:
        return np.array([], dtype=int)
    k = min(knn_neighbors, len(z))
    _, idx = cKDTree(z).query(q, k=k)
    idx = idx.reshape(len(q), k)  # cKDTree drops the neighbour axis when k == 1
    return pd.unique(idx.T.ravel())  # first-seen order


def knn_balanced_context(
    pool: pd.DataFrame,
    query: pd.DataFrame,
    *,
    max_context: int,
    knn_neighbors: int,
    near_share: float,
    near_max: float,
) -> ContextSelection:
    """knn, with a fixed share of the context reserved for pool rows near tipping.

    A knn context for a series that is still far from tipping holds few rows near tipping,
    so the PFN cannot express "close to tipping" (docs/GATE.md, decision 6). This strategy
    first takes round(near_share * max_context) pool rows with target <= near_max, picked
    by the knn neighbour-rank rule among those rows only. It then fills the rest of the cap
    with knn over all rows. It reads pool targets, which a context may use, and never the
    query's family or targets, so it is deployable. It uses no random numbers.
    """
    eligible, tid = _prepare(pool, query)
    z, q = _standardised(eligible, query)
    near = np.flatnonzero(eligible["target"].to_numpy(float) <= near_max)
    quota = min(int(round(near_share * max_context)), max_context)
    first = near[_by_neighbour_rank(z[near], q, knn_neighbors)][:quota]
    rest = _by_neighbour_rank(z, q, knn_neighbors)
    rest = rest[~np.isin(rest, first)][: max_context - len(first)]
    positions = np.concatenate([first, rest]).astype(int)
    return _finish("knn_balanced", eligible, positions, tid, max_context)


def family_matched_context(
    pool: pd.DataFrame, query: pd.DataFrame, *, max_context: int, seed: int
) -> ContextSelection:
    """ORACLE upper bound: a uniform sample from pool rows of the same family as the query.

    A real run does not know the family of a new series. This strategy reads it from the
    query table, so it shows how much a perfectly matched context could help. It must
    never be reported as a deployable method. The selection is marked oracle=True and its
    label carries "(ORACLE)".
    """
    if "family" not in query.columns or "family" not in pool.columns:
        raise ValueError("family_matched needs a family column in pool and query")
    families = query["family"].unique()
    if len(families) != 1:
        raise ValueError(f"query must come from exactly one family, got {list(families)}")
    eligible, tid = _prepare(pool, query)
    same = eligible[eligible["family"] == families[0]].reset_index(drop=True)
    if same.empty:
        raise NoMatchingFamilyError(f"no pool rows in family '{families[0]}' for {tid}")
    positions = _sample_positions(len(same), max_context, seed, tid)
    return _finish("family_matched", same, positions, tid, max_context)


def select_context(
    strategy: str,
    pool: pd.DataFrame,
    query: pd.DataFrame,
    cfg: PFNConfig | None = None,
    *,
    max_context: int | None = None,
) -> ContextSelection:
    """Run one strategy by name. Settings come from configs/pfn.yaml unless overridden."""
    cfg = cfg or load_pfn_config()
    cap = cfg.max_context if max_context is None else max_context
    if cap < 1:
        raise ValueError("max_context must be at least 1")
    if "split" in pool.columns and pool["split"].isin(REAL_SPLITS).any():
        raise ValueError("real data is test only and must never be part of the context pool")
    if strategy == "random":
        return random_context(pool, query, max_context=cap, seed=cfg.seed)
    if strategy == "knn":
        return knn_context(pool, query, max_context=cap, knn_neighbors=cfg.knn_neighbors)
    if strategy == "knn_balanced":
        return knn_balanced_context(
            pool,
            query,
            max_context=cap,
            knn_neighbors=cfg.knn_neighbors,
            near_share=cfg.balanced_near_share,
            near_max=cfg.critical_threshold,
        )
    if strategy == "family_matched":
        return family_matched_context(pool, query, max_context=cap, seed=cfg.seed)
    raise ValueError(f"unknown strategy '{strategy}', expected one of {list(STRATEGIES)}")

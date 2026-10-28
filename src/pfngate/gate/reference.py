"""Distances for two checks: query to the in-prior pool (ood_share), and query to the
near-tipping rows of its context (near_tipping_support).

For ood_share, distances are measured to the whole pool (seed set A), not to the selected
context. knn and the agent both choose context rows close to the query, so a distance to
the selected context would make every query look in-distribution. The pool is the same
whatever the strategy chose. See docs/GATE.md, check 2. near_tipping_support asks the
opposite question on purpose: whether the chosen context holds near-tipping rows close to
the query (check 7).

The statistic is the Euclidean distance to the k-th nearest pool row, in features
standardised with the pool mean and standard deviation. A percentile of nearest-neighbour
distances needs no assumption about the shape of the feature distribution, which matters
because the Kendall taus are bounded and the kurtosis is heavy-tailed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from pfngate.features import FEATURE_NAMES

_TINY = 1e-12


@dataclass(frozen=True)
class PoolReference:
    """What the gate stores about the pool: the transform and the distance cut."""

    k: int
    quantile: float
    n_sample: int
    distance_cut: float
    feature_mean: tuple[float, ...]
    feature_std: tuple[float, ...]
    n_pool_rows: int
    n_pool_trajectories: int


def _features(pool: pd.DataFrame) -> np.ndarray:
    x = pool[list(FEATURE_NAMES)].to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError("pool has NaN or infinite features")
    return x


def fit_pool_reference(
    pool: pd.DataFrame, *, k: int, quantile: float, n_sample: int, seed: int
) -> PoolReference:
    """The distance cut: a high percentile of the k-th neighbour distance of pool rows.

    Each sampled pool row looks for its neighbours among rows of OTHER trajectories (leave
    one trajectory out), because windows of one trajectory overlap and would sit unusually
    close to each other. Uses only features, never targets.
    """
    x = _features(pool)
    tids = pool["trajectory_id"].astype(str).to_numpy()
    if len(np.unique(tids)) < 2:
        raise ValueError("the pool needs at least two trajectories")
    mean = x.mean(axis=0)
    std = np.where(x.std(axis=0) > _TINY, x.std(axis=0), 1.0)
    z = (x - mean) / std

    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(z), size=min(n_sample, len(z)), replace=False))
    k_query = min(len(z), k + int(pd.Series(tids).value_counts().max()))
    dist, idx = cKDTree(z).query(z[rows], k=k_query)
    dist = np.where(tids[idx] == tids[rows][:, None], np.inf, dist)  # drop own trajectory
    kth = np.sort(dist, axis=1)[:, k - 1]
    if not np.isfinite(kth).all():
        raise ValueError("some pool rows have fewer than k neighbours in other trajectories")
    return PoolReference(
        k=k,
        quantile=quantile,
        n_sample=len(rows),
        distance_cut=float(np.quantile(kth, quantile)),
        feature_mean=tuple(float(v) for v in mean),
        feature_std=tuple(float(v) for v in std),
        n_pool_rows=len(z),
        n_pool_trajectories=int(len(np.unique(tids))),
    )


@dataclass(frozen=True)
class NearTippingReference:
    """The distance cut for near_tipping_support, fitted once on the pool (set A)."""

    k: int
    target_max: float
    quantile: float
    n_sample: int
    distance_cut: float
    n_near_rows: int


def fit_near_tipping_reference(
    pool: pd.DataFrame,
    *,
    feature_mean: tuple[float, ...],
    feature_std: tuple[float, ...],
    k: int,
    target_max: float,
    quantile: float,
    n_sample: int,
    seed: int,
) -> NearTippingReference:
    """The cut: a high percentile, over pool rows, of the distance to the k-th nearest
    near-tipping pool row (target <= target_max) of ANOTHER trajectory.

    It uses pool targets, which a context may use (set A), and the same standardisation as
    ood_share. See docs/GATE.md, check 7.
    """
    z = (_features(pool) - np.asarray(feature_mean)) / np.asarray(feature_std)
    tids = pool["trajectory_id"].astype(str).to_numpy()
    near = pool["target"].to_numpy(float) <= target_max
    zn, tn = z[near], tids[near]
    if len(np.unique(tn)) < 2 or len(zn) < k:
        raise ValueError("the pool needs near-tipping rows in at least two trajectories")
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(z), size=min(n_sample, len(z)), replace=False))
    k_query = min(len(zn), k + int(pd.Series(tn).value_counts().max()))
    dist, idx = cKDTree(zn).query(z[rows], k=k_query)
    dist = dist.reshape(len(rows), k_query)
    idx = idx.reshape(len(rows), k_query)
    dist = np.where(tn[idx] == tids[rows][:, None], np.inf, dist)  # drop own trajectory
    kth = np.sort(dist, axis=1)[:, k - 1]
    if not np.isfinite(kth).all():
        raise ValueError("some pool rows have fewer than k near-tipping rows elsewhere")
    return NearTippingReference(
        k=k,
        target_max=target_max,
        quantile=quantile,
        n_sample=len(rows),
        distance_cut=float(np.quantile(kth, quantile)),
        n_near_rows=int(near.sum()),
    )


def near_tipping_distances(
    context: pd.DataFrame,
    query_X: np.ndarray,
    *,
    feature_mean: tuple[float, ...],
    feature_std: tuple[float, ...],
    k: int,
    target_max: float,
) -> np.ndarray:
    """Per query window, the distance to the k-th nearest context row with target <=
    target_max. Infinite for every window when the context has fewer than k such rows."""
    mean, std = np.asarray(feature_mean, dtype=float), np.asarray(feature_std, dtype=float)
    near = context[context["target"].to_numpy(float) <= target_max]
    q = (np.asarray(query_X, dtype=float) - mean) / std
    if len(near) < k:
        return np.full(len(q), np.inf)
    dist, _ = cKDTree((_features(near) - mean) / std).query(q, k=k)
    return dist.reshape(len(q), k)[:, -1]


class PoolDistance:
    """k-th nearest pool row distance for query windows, with the stored transform."""

    def __init__(
        self,
        pool: pd.DataFrame,
        feature_mean: tuple[float, ...],
        feature_std: tuple[float, ...],
        k: int,
    ) -> None:
        self.mean = np.asarray(feature_mean, dtype=float)
        self.std = np.asarray(feature_std, dtype=float)
        self.k = k
        self.trajectories = frozenset(pool["trajectory_id"].astype(str))
        self.n_rows = len(pool)
        self._tree = cKDTree((_features(pool) - self.mean) / self.std)

    def distances(self, query_X: np.ndarray, query_trajectory_id: str) -> np.ndarray:
        if query_trajectory_id in self.trajectories:
            raise ValueError(f"query trajectory {query_trajectory_id} is in the pool")
        z = (np.asarray(query_X, dtype=float) - self.mean) / self.std
        dist, _ = self._tree.query(z, k=self.k)
        return dist.reshape(len(z), self.k)[:, -1]

"""Cut fixed-length windows from a series. A window is named by its last index."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def window_ends(n_points: int, length: int, stride: int) -> np.ndarray:
    """Index of the last point of every full window: length - 1, length - 1 + stride, ..."""
    if n_points < length:
        return np.empty(0, dtype=int)
    return np.arange(length - 1, n_points, stride)


def window_matrix(series: np.ndarray, ends: np.ndarray, length: int) -> np.ndarray:
    """Rows are the windows series[end - length + 1 : end + 1], one per end index."""
    views = sliding_window_view(series, length)
    return views[ends - length + 1]

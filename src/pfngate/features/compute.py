"""Early-warning features for a batch of windows, one window per row.

Every function works along the last axis and uses only the values in that row, so a
window's features never depend on data outside it. All features are scale-free: they
do not change if a series is multiplied by a positive constant or shifted, so context
from one system can be compared with another.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from pfngate.features.config import FeatureConfig

FEATURE_NAMES: tuple[str, ...] = (
    "variance_log_ratio",  # log(variance of the second half / first half), detrended
    "ac1",  # lag-1 autocorrelation of the detrended window
    "skewness",
    "kurtosis",  # excess kurtosis (0 for a Gaussian)
    "trend_slope",  # linear trend across the window, in residual standard deviations
    "tau_variance",  # Kendall tau of sub-window variance against time
    "tau_ac1",  # Kendall tau of sub-window lag-1 autocorrelation against time
    "low_freq_power",  # share of spectral power at low frequencies
)

# Ratios with a denominator at or below this are treated as undefined and set to 0.
_TINY = 1e-12


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    ok = den > _TINY
    return np.where(ok, num / np.where(ok, den, 1.0), 0.0)


def linear_detrend(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove a least-squares line from each row. Returns (residuals, slope per step)."""
    n = w.shape[-1]
    t = np.arange(n) - (n - 1) / 2.0
    centred = w - w.mean(axis=-1, keepdims=True)
    slope = centred @ t / (t @ t)
    return centred - slope[..., None] * t, slope


def lag1_autocorrelation(x: np.ndarray) -> np.ndarray:
    c = x - x.mean(axis=-1, keepdims=True)
    return _ratio((c[..., 1:] * c[..., :-1]).sum(axis=-1), (c * c).sum(axis=-1))


def kendall_tau_vs_time(v: np.ndarray) -> np.ndarray:
    """Kendall tau between each row and its time index (tau-a; ties count as 0)."""
    n = v.shape[-1]
    i, j = np.triu_indices(n, k=1)
    return np.sign(v[..., j] - v[..., i]).sum(axis=-1) / (n * (n - 1) / 2)


def window_features(w: np.ndarray, cfg: FeatureConfig) -> dict[str, np.ndarray]:
    """Features for windows w of shape (n_windows, window_length)."""
    r, slope = linear_detrend(w)
    var = (r * r).mean(axis=-1)
    sd = np.sqrt(var)

    half = r.shape[-1] // 2
    v_early = r[..., :half].var(axis=-1)
    v_late = r[..., half:].var(axis=-1)
    both = (v_early > _TINY) & (v_late > _TINY)
    variance_log_ratio = np.where(
        both, np.log(np.where(both, v_late, 1.0) / np.where(both, v_early, 1.0)), 0.0
    )

    subs = sliding_window_view(r, cfg.sub_length, axis=-1)[..., :: cfg.sub_stride, :]
    sub_var = subs.var(axis=-1)
    sub_ac1 = lag1_autocorrelation(subs)

    power = np.abs(np.fft.rfft(r, axis=-1)) ** 2
    power = power[..., 1:]  # drop the zero frequency (the mean, already removed)
    n_low = max(1, int(np.floor(cfg.low_frequency_fraction * power.shape[-1])))

    return {
        "variance_log_ratio": variance_log_ratio,
        "ac1": lag1_autocorrelation(r),
        "skewness": _ratio((r**3).mean(axis=-1), sd**3),
        "kurtosis": np.where(var > _TINY, _ratio((r**4).mean(axis=-1), var**2) - 3.0, 0.0),
        "trend_slope": _ratio(slope * (r.shape[-1] - 1), sd),
        "tau_variance": kendall_tau_vs_time(sub_var),
        "tau_ac1": kendall_tau_vs_time(sub_ac1),
        "low_freq_power": _ratio(power[..., :n_low].sum(axis=-1), power.sum(axis=-1)),
    }

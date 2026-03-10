"""Utilities for fitting Gaussian mixture patterns to time series data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from pattern_fill.pattern import DailyPattern, GaussianComponent
from pattern_fill.fitting import extract_daily_profile


def fit_gaussian_pattern(
    series: pd.Series,
    n_components: int = 3,
    resolution_minutes: int = 15,
    aggregation: str = "median",
    name: str = "fitted_gaussian",
    day_type: str = "all",
    baseline: float | None = None,
    min_width: float = 0.5,
    max_width: float = 6.0,
) -> DailyPattern:
    """Fit a Gaussian mixture pattern to time series data.

    Extracts the daily profile, detects peaks, and fits wrapped Gaussian
    components via least-squares optimization.

    Parameters
    ----------
    series : pd.Series
        Time series data with DatetimeIndex
    n_components : int
        Maximum number of Gaussian components to fit
    resolution_minutes : int
        Temporal resolution for daily profile extraction
    aggregation : str
        "median" or "mean" aggregation method
    name : str
        Pattern name
    day_type : str
        "all", "weekday", or "weekend"
    baseline : float, optional
        Fixed baseline (floor). If None, estimated as 5th percentile
    min_width : float
        Minimum Gaussian width in hours
    max_width : float
        Maximum Gaussian width in hours

    Returns
    -------
    DailyPattern
        Fitted pattern in gaussian mode
    """
    profile = extract_daily_profile(
        series,
        resolution_minutes=resolution_minutes,
        aggregation=aggregation,
    )

    hours = profile.index.values
    values = profile.values

    v_min, v_max = float(np.min(values)), float(np.max(values))
    if v_max - v_min > 0:
        values_norm = (values - v_min) / (v_max - v_min)
    else:
        values_norm = np.full_like(values, 0.5)

    if baseline is None:
        baseline = float(np.clip(np.percentile(values_norm, 5), 0.0, 1.0))

    excess = np.clip(values_norm - baseline, 0.0, None)

    # Pad circularly to catch midnight-spanning peaks
    pad = max(4, len(excess) // 8)
    padded = np.concatenate([excess[-pad:], excess, excess[:pad]])
    peak_indices_padded, peak_props = find_peaks(padded, height=0.01)
    # Shift back to original indices, deduplicate
    peak_indices = (peak_indices_padded - pad) % len(excess)
    heights = peak_props["peak_heights"]

    # Deduplicate (same index may appear from padding wrapping)
    seen = {}
    for idx, h in zip(peak_indices, heights):
        if idx not in seen or h > seen[idx]:
            seen[idx] = h

    sorted_peaks = sorted(seen.items(), key=lambda x: -x[1])[:n_components]

    # Build initial guesses
    x0_list = []
    for idx, h in sorted_peaks:
        center = float(hours[idx])
        x0_list.append([h, center, 2.0])

    # Pad with evenly-spaced fallbacks if not enough peaks
    while len(x0_list) < n_components:
        fallback_center = 24.0 * len(x0_list) / n_components
        x0_list.append([0.1, fallback_center, 2.0])

    x0 = np.array(x0_list).ravel()

    # Bounds: [amp, center, width] per component
    lb = np.tile([0.0, 0.0, min_width], n_components)
    ub = np.tile([1.0, 24.0, max_width], n_components)

    def residual(params):
        pred = np.full_like(hours, baseline, dtype=float)
        for i in range(n_components):
            amp, center, width = params[3 * i : 3 * i + 3]
            gc = GaussianComponent(amp, center, width)
            pred += gc.evaluate(hours % 24.0)
        pred = np.clip(pred, 0.0, 1.0)
        return pred - values_norm

    result = least_squares(residual, x0, bounds=(lb, ub))

    components = []
    for i in range(n_components):
        amp, center, width = result.x[3 * i : 3 * i + 3]
        if amp >= 0.01:
            components.append(GaussianComponent(float(amp), float(center), float(width)))

    if not components:
        components = [GaussianComponent(0.1, 0.0, 2.0)]

    return DailyPattern(
        gaussian_components=components,
        baseline=float(baseline),
        name=name,
        day_type=day_type,
    )

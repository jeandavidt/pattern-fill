from __future__ import annotations

import numpy as np
import pandas as pd

from pattern_fill.pattern import DailyPattern


def extract_daily_profile(
    series: pd.Series,
    resolution_minutes: int = 15,
    aggregation: str = "median",
) -> pd.Series:
    """Group a time series by fractional hour-of-day and aggregate.

    Returns a Series indexed by fractional hour (e.g. 8.25 for 08:15)
    with one value per bin.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series must have a DatetimeIndex")

    s = series.dropna()
    fractional_hour = s.index.hour + s.index.minute / 60.0 + s.index.second / 3600.0
    bin_edges = np.arange(0, 24 + resolution_minutes / 60, resolution_minutes / 60)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_idx = np.digitize(fractional_hour, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, len(bin_centers) - 1)

    grouped = pd.Series(s.values, index=bin_idx)
    if aggregation == "median":
        profile_values = grouped.groupby(level=0).median()
    elif aggregation == "mean":
        profile_values = grouped.groupby(level=0).mean()
    else:
        raise ValueError(f"aggregation must be 'median' or 'mean', got {aggregation!r}")

    result = pd.Series(
        index=bin_centers[profile_values.index.values],
        data=profile_values.values,
        name="daily_profile",
    )
    result.index.name = "hour"
    return result


def fit_pattern(
    series: pd.Series,
    n_control_points: int = 8,
    resolution_minutes: int = 15,
    aggregation: str = "median",
    name: str = "fitted",
    day_type: str = "all",
) -> DailyPattern:
    """Fit a DailyPattern from observed time series data.

    Extracts a daily profile, picks *n_control_points* evenly spaced along
    the 0-24 h axis, normalizes values to 0-1, and returns a DailyPattern.
    """
    profile = extract_daily_profile(
        series,
        resolution_minutes=resolution_minutes,
        aggregation=aggregation,
    )

    target_hours = np.linspace(0, 24, n_control_points + 1)[:-1]
    cp_values = np.interp(target_hours, profile.index.values, profile.values)

    v_min, v_max = cp_values.min(), cp_values.max()
    if v_max - v_min > 0:
        cp_norm = (cp_values - v_min) / (v_max - v_min)
    else:
        cp_norm = np.full_like(cp_values, 0.5)

    return DailyPattern(
        hours=target_hours.tolist(),
        values=cp_norm.tolist(),
        name=name,
        periodic=True,
        day_type=day_type,
    )

"""Utilities for fitting sine wave patterns to time series data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.optimize import least_squares

from pattern_fill.pattern import DailyPattern, SineComponent
from pattern_fill.fitting import extract_daily_profile


def fit_sine_pattern(
    series: pd.Series,
    n_components: int = 2,
    frequencies: list[float] | None = None,
    resolution_minutes: int = 15,
    aggregation: str = "median",
    name: str = "fitted_sine",
    day_type: str = "all",
    baseline: float | None = None,
) -> DailyPattern:
    """Fit a sine-based pattern to time series data using FFT.

    Extracts the daily profile, performs FFT to identify dominant frequencies,
    and fits sine wave components.

    Parameters
    ----------
    series : pd.Series
        Time series data with DatetimeIndex
    n_components : int
        Number of sine components to fit (ignored if frequencies specified)
    frequencies : list[float], optional
        Fixed frequencies to use (in cycles per day). If None, auto-detect
        using FFT. Common values: [1.0] (daily), [1.0, 2.0] (daily + twice-daily)
    resolution_minutes : int
        Temporal resolution for daily profile extraction
    aggregation : str
        "median" or "mean" aggregation method
    name : str
        Pattern name
    day_type : str
        "all", "weekday", or "weekend"
    baseline : float, optional
        Fixed baseline value. If None, estimated from data mean

    Returns
    -------
    DailyPattern
        Fitted pattern in sine mode

    Examples
    --------
    >>> # Auto-fit with 2 components (finds dominant frequencies)
    >>> pattern = fit_sine_pattern(series, n_components=2)

    >>> # Fit with fixed daily frequency
    >>> pattern = fit_sine_pattern(series, frequencies=[1.0])

    >>> # Fit daily + twice-daily pattern for wastewater
    >>> pattern = fit_sine_pattern(series, frequencies=[1.0, 2.0])
    """
    # Extract daily profile
    profile = extract_daily_profile(
        series,
        resolution_minutes=resolution_minutes,
        aggregation=aggregation,
    )

    # Estimate baseline from profile mean if not provided
    if baseline is None:
        baseline = float(np.clip(profile.mean(), 0.0, 1.0))

    # Get x (hours) and y (normalized values)
    hours = profile.index.values
    values = profile.values

    # Normalize values to roughly [0, 1] range
    v_min, v_max = values.min(), values.max()
    if v_max - v_min > 0:
        values_norm = (values - v_min) / (v_max - v_min)
    else:
        values_norm = np.full_like(values, 0.5)

    # Determine frequencies to fit
    if frequencies is None:
        frequencies = _detect_frequencies_fft(hours, values_norm, n_components)

    # Fit amplitude and phase for each frequency
    components = []
    for freq in frequencies:
        amplitude, phase = _fit_single_sine(hours, values_norm, freq, baseline)
        if amplitude > 0.01:  # Only include significant components
            components.append(SineComponent(amplitude, freq, phase))

    # If no significant components found, create a simple flat pattern
    if not components:
        components = [SineComponent(0.1, 1.0, 0.0)]

    return DailyPattern(
        sine_components=components,
        baseline=baseline,
        name=name,
        day_type=day_type,
    )


def _detect_frequencies_fft(
    hours: np.ndarray,
    values: np.ndarray,
    n_components: int,
) -> list[float]:
    """Detect dominant frequencies using FFT.

    Returns the top n_components frequencies in cycles per day.
    """
    # Perform FFT
    fft_values = rfft(values - values.mean())
    fft_freqs = rfftfreq(len(values), d=hours[1] - hours[0])

    # Convert frequencies from cycles/hour to cycles/day
    fft_freqs_per_day = fft_freqs * 24.0

    # Get magnitudes and find peaks
    magnitudes = np.abs(fft_values)

    # Exclude DC component (index 0) and find top peaks
    peak_indices = np.argsort(magnitudes[1:])[::-1][:n_components] + 1
    detected_freqs = fft_freqs_per_day[peak_indices]

    # Round to common fractions (0.5, 1.0, 1.5, 2.0, etc.)
    rounded_freqs = [round(f * 2) / 2 for f in detected_freqs]

    # Filter out zero or negative frequencies
    valid_freqs = [f for f in rounded_freqs if f > 0]

    return valid_freqs if valid_freqs else [1.0]


def _fit_single_sine(
    hours: np.ndarray,
    values: np.ndarray,
    frequency: float,
    baseline: float,
) -> tuple[float, float]:
    """Fit amplitude and phase for a single sine wave component.

    Given frequency and baseline, optimizes amplitude and phase to
    minimize residual error.

    Returns
    -------
    amplitude : float
    phase : float (in hours)
    """
    def residual(params):
        amp, phase = params
        # Use cosine for peak at phase hour
        predicted = baseline + amp * np.cos(
            2 * np.pi * frequency * (hours - phase) / 24.0
        )
        predicted = np.clip(predicted, 0.0, 1.0)
        return predicted - values

    # Initial guess: amplitude from std, phase from peak
    initial_amp = np.std(values - baseline)
    initial_phase = hours[np.argmax(values)]

    # Optimize
    result = least_squares(
        residual,
        x0=[initial_amp, initial_phase],
        bounds=([0, 0], [1.0, 24.0]),
    )

    amplitude, phase = result.x
    return float(amplitude), float(phase)

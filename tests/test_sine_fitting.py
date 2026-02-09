"""Tests for sine pattern fitting."""

import numpy as np
import pandas as pd
import pytest

from pattern_fill import fit_sine_pattern, DailyPattern


def _make_sine_series(
    amplitude=0.5,
    frequency=1.0,
    phase=6.0,
    baseline=0.5,
    noise_level=0.05,
):
    """Generate test series with known sine pattern."""
    rng = np.random.default_rng(42)
    index = pd.date_range("2024-01-01", periods=7 * 96, freq="15min")
    hours = index.hour + index.minute / 60.0

    # Generate sine pattern (using cosine for peak at phase hour)
    values = baseline + amplitude * np.cos(
        2 * np.pi * frequency * (hours - phase) / 24.0
    )

    # Add noise
    values += noise_level * rng.standard_normal(len(values))

    return pd.Series(values, index=index, name="test")


class TestFitSinePattern:
    def test_returns_daily_pattern(self):
        s = _make_sine_series()
        p = fit_sine_pattern(s)
        assert isinstance(p, DailyPattern)
        assert p.mode == "sine"

    def test_fits_single_frequency(self):
        # Generate series with known frequency
        s = _make_sine_series(amplitude=0.4, frequency=1.0, phase=8.0)
        p = fit_sine_pattern(s, frequencies=[1.0])

        # Should have one component near frequency 1.0
        assert len(p.sine_components) >= 1
        freqs = [c.frequency for c in p.sine_components]
        assert 1.0 in freqs

    def test_auto_detect_frequencies(self):
        # Create series with two frequency components
        s = _make_sine_series(amplitude=0.3, frequency=1.0, noise_level=0.02)
        p = fit_sine_pattern(s, n_components=2)

        # Should detect at least one component
        assert len(p.sine_components) >= 1

    def test_baseline_estimation(self):
        # Series with known baseline
        s = _make_sine_series(baseline=0.6)
        p = fit_sine_pattern(s)

        # Baseline should be close to 0.6 (normalized)
        # (May not be exact due to normalization)
        assert 0.3 <= p.baseline <= 0.8

    def test_handles_constant_data(self):
        """Pattern fitting handles constant data without errors."""
        index = pd.date_range("2024-01-01", periods=96, freq="15min")
        series = pd.Series(5.0, index=index, name="constant")

        # Should not raise an error
        p = fit_sine_pattern(series)
        assert isinstance(p, DailyPattern)
        assert p.mode == "sine"

    def test_respects_day_type(self):
        """Fitted pattern respects day_type parameter."""
        s = _make_sine_series()
        p = fit_sine_pattern(s, day_type="weekday")
        assert p.day_type == "weekday"

    def test_respects_name(self):
        """Fitted pattern respects name parameter."""
        s = _make_sine_series()
        p = fit_sine_pattern(s, name="custom_pattern")
        assert p.name == "custom_pattern"

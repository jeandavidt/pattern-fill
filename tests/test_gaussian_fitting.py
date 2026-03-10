"""Tests for Gaussian pattern fitting."""

import numpy as np
import pandas as pd
import pytest

from pattern_fill import fit_gaussian_pattern, DailyPattern, GaussianComponent


def _make_gaussian_series(
    amplitude=0.6,
    center=8.0,
    width=2.0,
    baseline=0.1,
    noise_level=0.02,
):
    """Generate test series with a known Gaussian peak."""
    rng = np.random.default_rng(42)
    index = pd.date_range("2024-01-01", periods=7 * 96, freq="15min")
    hours = index.hour + index.minute / 60.0

    gc = GaussianComponent(amplitude, center, width)
    values = baseline + gc.evaluate(hours.values % 24.0)
    values += noise_level * rng.standard_normal(len(values))
    values = np.clip(values, 0.0, 1.0)

    return pd.Series(values, index=index, name="test")


class TestFitGaussianPattern:
    def test_returns_daily_pattern(self):
        s = _make_gaussian_series()
        p = fit_gaussian_pattern(s)
        assert isinstance(p, DailyPattern)
        assert p.mode == "gaussian"

    def test_fitted_center_near_true(self):
        s = _make_gaussian_series(center=8.0, amplitude=0.6, noise_level=0.01)
        p = fit_gaussian_pattern(s, n_components=1)
        centers = [c.center for c in p.gaussian_components]
        assert any(abs(c - 8.0) < 2.0 for c in centers)

    def test_values_in_range(self):
        s = _make_gaussian_series()
        p = fit_gaussian_pattern(s)
        h = np.linspace(0, 24, 97)
        v = p.evaluate(h)
        assert v.min() >= 0.0
        assert v.max() <= 1.0

    def test_periodicity(self):
        s = _make_gaussian_series()
        p = fit_gaussian_pattern(s)
        assert p.evaluate(np.array([0.0]))[0] == pytest.approx(
            p.evaluate(np.array([24.0]))[0]
        )

    def test_handles_constant_data(self):
        index = pd.date_range("2024-01-01", periods=96, freq="15min")
        series = pd.Series(5.0, index=index, name="constant")
        p = fit_gaussian_pattern(series)
        assert isinstance(p, DailyPattern)
        assert p.mode == "gaussian"

    def test_respects_name(self):
        s = _make_gaussian_series()
        p = fit_gaussian_pattern(s, name="my_gaussian")
        assert p.name == "my_gaussian"

    def test_respects_day_type(self):
        s = _make_gaussian_series()
        p = fit_gaussian_pattern(s, day_type="weekday")
        assert p.day_type == "weekday"

    def test_respects_explicit_baseline(self):
        s = _make_gaussian_series()
        p = fit_gaussian_pattern(s, baseline=0.2)
        assert p.baseline == pytest.approx(0.2)

    def test_at_most_n_components(self):
        s = _make_gaussian_series()
        p = fit_gaussian_pattern(s, n_components=2)
        assert len(p.gaussian_components) <= 2

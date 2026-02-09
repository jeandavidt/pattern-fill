import numpy as np
import pandas as pd
import pytest

from pattern_fill.fitting import extract_daily_profile, fit_pattern
from pattern_fill.pattern import DailyPattern


def _make_sine_series(days: int = 7, freq: str = "15min") -> pd.Series:
    """Generate a sine-wave diurnal series (peak at 6h, trough at 18h)."""
    idx = pd.date_range("2024-01-01", periods=days * 24 * 4, freq=freq)
    frac_h = idx.hour + idx.minute / 60.0
    values = 10.0 + 5.0 * np.sin(2 * np.pi * frac_h / 24)
    return pd.Series(values, index=idx, name="test_signal")


class TestExtractDailyProfile:
    def test_returns_series(self):
        s = _make_sine_series()
        profile = extract_daily_profile(s)
        assert isinstance(profile, pd.Series)
        assert profile.index.name == "hour"

    def test_profile_shape(self):
        s = _make_sine_series()
        profile = extract_daily_profile(s, resolution_minutes=60)
        assert len(profile) == 24

    def test_peak_near_6h(self):
        s = _make_sine_series(days=30)
        profile = extract_daily_profile(s, resolution_minutes=60)
        peak_hour = profile.index[profile.values.argmax()]
        assert abs(peak_hour - 6.0) < 1.5

    def test_rejects_non_datetime_index(self):
        s = pd.Series([1, 2, 3], index=[0, 1, 2])
        with pytest.raises(TypeError, match="DatetimeIndex"):
            extract_daily_profile(s)

    def test_invalid_aggregation(self):
        s = _make_sine_series()
        with pytest.raises(ValueError, match="aggregation"):
            extract_daily_profile(s, aggregation="sum")

    def test_mean_aggregation(self):
        s = _make_sine_series()
        profile = extract_daily_profile(s, aggregation="mean")
        assert len(profile) > 0


class TestFitPattern:
    def test_returns_daily_pattern(self):
        s = _make_sine_series()
        p = fit_pattern(s)
        assert isinstance(p, DailyPattern)

    def test_correct_number_of_control_points(self):
        s = _make_sine_series()
        p = fit_pattern(s, n_control_points=12)
        assert len(p.hours) == 12
        assert len(p.values) == 12

    def test_values_normalized_0_1(self):
        s = _make_sine_series()
        p = fit_pattern(s)
        assert min(p.values) >= -0.01
        assert max(p.values) <= 1.01

    def test_day_type_propagated(self):
        s = _make_sine_series()
        p = fit_pattern(s, day_type="weekday")
        assert p.day_type == "weekday"

    def test_flat_series_gives_half(self):
        idx = pd.date_range("2024-01-01", periods=96, freq="15min")
        s = pd.Series(5.0, index=idx, name="flat")
        p = fit_pattern(s)
        np.testing.assert_allclose(p.values, 0.5, atol=0.01)

    def test_last_control_point_before_midnight_with_closure(self):
        """Control points end before 24 but periodic closure handles midnight."""
        s = _make_sine_series()
        p = fit_pattern(s, n_control_points=8)
        assert p.hours[-1] < 24, (
            f"Last control point should be before 24, "
            f"periodic closure at 24 is handled by DailyPattern"
        )
        assert p.hours[0] == 0, "First control point should be at 0"

    def test_no_extrapolation_beyond_control_points(self):
        """Pattern values should stay near 0-1 range without large overshoots."""
        s = _make_sine_series()
        p = fit_pattern(s, n_control_points=8)
        eval_hours = np.linspace(0, 24, 200)
        values = p.evaluate(eval_hours)
        assert values.min() >= -0.01, (
            f"Pattern undershoots significantly below 0 (min={values.min():.4f})"
        )
        assert values.max() <= 1.01, (
            f"Pattern overshoots significantly above 1 (max={values.max():.4f})"
        )

    def test_periodic_wraparound_smooth_at_midnight(self):
        """Pattern should be smooth across midnight, not have jumps or peaks."""
        s = _make_sine_series()
        p = fit_pattern(s, n_control_points=8)
        val_before = p.evaluate(np.array([23.9]))[0]
        val_at_0 = p.evaluate(np.array([0.0]))[0]
        val_after = p.evaluate(np.array([0.1]))[0]
        assert abs(val_before - val_at_0) < 0.02, (
            f"Pattern not continuous at midnight: "
            f"23.9h={val_before:.4f}, 0.0h={val_at_0:.4f}"
        )
        assert abs(val_at_0 - val_after) < 0.02, (
            f"Pattern not smooth after midnight: "
            f"0.0h={val_at_0:.4f}, 0.1h={val_after:.4f}"
        )
        assert abs(val_at_0 - 0.5) < 0.05, (
            f"Pattern value at 0h should be near 0.5 for sine pattern, "
            f"got {val_at_0:.4f}"
        )

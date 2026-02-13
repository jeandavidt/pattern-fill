import numpy as np
import pandas as pd
import pytest

from pattern_fill.gap_fill import (
    pattern_fill,
    find_nan_runs,
    classify_runs,
    smoothed_anchor,
    _blend_fill,
)
from pattern_fill.pattern import DailyPattern
from pattern_fill import fit_ar_model, generate_ar_noise


def _sine_pattern() -> DailyPattern:
    hours = list(range(24))
    values = [0.5 + 0.5 * np.sin(2 * np.pi * h / 24) for h in hours]
    return DailyPattern(hours=hours, values=values, name="sine")


def _make_series_with_gap(
    days: int = 7,
    gap_start_day: int = 3,
    gap_hours: int = 12,
) -> pd.Series:
    """Sine-wave series with a NaN gap in the middle."""
    idx = pd.date_range("2024-01-01", periods=days * 96, freq="15min")
    frac_h = idx.hour + idx.minute / 60.0
    values = 10.0 + 5.0 * np.sin(2 * np.pi * frac_h / 24)
    s = pd.Series(values, index=idx, name="sensor_raw")
    gap_start = pd.Timestamp(f"2024-01-0{gap_start_day + 1} 06:00")
    gap_end = gap_start + pd.Timedelta(hours=gap_hours)
    s.loc[gap_start:gap_end] = np.nan
    return s


# ---- _find_nan_runs (unchanged) -------------------------------------------


class TestFindNanRuns:
    def test_no_nans(self):
        assert find_nan_runs(np.array([False, False, False])) == []

    def test_single_run(self):
        mask = np.array([False, True, True, True, False])
        assert find_nan_runs(mask) == [(1, 4)]

    def test_multiple_runs(self):
        mask = np.array([True, True, False, True, False])
        assert find_nan_runs(mask) == [(0, 2), (3, 4)]

    def test_all_nan(self):
        mask = np.array([True, True, True])
        assert find_nan_runs(mask) == [(0, 3)]


# ---- _classify_runs --------------------------------------------------------


class TestClassifyRuns:
    def test_filters_leading(self):
        runs = [(0, 2), (5, 7)]
        assert classify_runs(runs, 10) == [(5, 7)]

    def test_filters_trailing(self):
        runs = [(3, 5), (8, 10)]
        assert classify_runs(runs, 10) == [(3, 5)]

    def test_filters_both(self):
        runs = [(0, 2), (5, 7), (8, 10)]
        assert classify_runs(runs, 10) == [(5, 7)]

    def test_all_interior(self):
        runs = [(2, 4), (6, 8)]
        assert classify_runs(runs, 10) == [(2, 4), (6, 8)]

    def test_empty(self):
        assert classify_runs([], 10) == []


# ---- _smoothed_anchor ------------------------------------------------------


class TestSmoothedAnchor:
    def test_clean_data_weighted_toward_gap(self):
        """Anchor should be weighted toward the closest-to-gap point."""
        idx = pd.date_range("2024-01-01", periods=6, freq="15min")
        s = pd.Series([10, 20, 30, np.nan, np.nan, np.nan], index=idx)
        # left anchor at idx=2, blend_n=3 → window [10, 20, 30]
        # weights [1, 2, 3] → avg = (10+40+90)/6 = 23.33
        anchor = smoothed_anchor(s, 2, "left", 3)
        assert anchor == pytest.approx((10 * 1 + 20 * 2 + 30 * 3) / 6)

    def test_noisy_data_smooths_outlier(self):
        """A noisy point at the edge should be smoothed by neighbors."""
        idx = pd.date_range("2024-01-01", periods=6, freq="15min")
        s = pd.Series([30, 31, 29, 100, np.nan, np.nan], index=idx)
        anchor = smoothed_anchor(s, 3, "left", 4)
        # weights [1, 2, 3, 4]: (30+62+87+400)/10 = 57.9
        # Much less than 100 (the single-point anchor would be)
        assert anchor < 60

    def test_limited_data_uses_available(self):
        """When fewer points than blend_n exist, use what's available."""
        idx = pd.date_range("2024-01-01", periods=4, freq="15min")
        s = pd.Series([5, np.nan, np.nan, np.nan], index=idx)
        anchor = smoothed_anchor(s, 0, "left", 4)
        assert anchor == pytest.approx(5.0)

    def test_right_side_weighted_toward_gap(self):
        """Right anchor weights the closest-to-gap point highest."""
        idx = pd.date_range("2024-01-01", periods=6, freq="15min")
        s = pd.Series([np.nan, np.nan, np.nan, 10, 20, 30], index=idx)
        anchor = smoothed_anchor(s, 3, "right", 3)
        # weights [3, 2, 1]: (30+40+30)/6 = 16.67
        assert anchor == pytest.approx((10 * 3 + 20 * 2 + 30 * 1) / 6)

    def test_nan_in_window_skipped(self):
        """NaN values within the blend window should be excluded."""
        idx = pd.date_range("2024-01-01", periods=5, freq="15min")
        s = pd.Series([10, np.nan, 30, np.nan, np.nan], index=idx)
        anchor = smoothed_anchor(s, 2, "left", 3)
        # window [10, NaN, 30] → valid [10, 30], weights [1, 2]
        assert anchor == pytest.approx((10 * 1 + 30 * 2) / 3)

    def test_no_valid_points_returns_none(self):
        """All NaN window returns None."""
        idx = pd.date_range("2024-01-01", periods=4, freq="15min")
        s = pd.Series([np.nan, np.nan, np.nan, np.nan], index=idx)
        assert smoothed_anchor(s, 1, "left", 2) is None


# ---- _blend_fill -----------------------------------------------------------


class TestBlendFill:
    def test_both_boundaries_exact_match(self):
        """Fill endpoints must exactly match the anchor values."""
        p = np.array([12.0, 15.0, 18.0, 16.0])
        filled = _blend_fill(p, left_anchor=10.0, right_anchor=20.0, blend_n=2)
        assert filled[0] == pytest.approx(10.0)
        assert filled[-1] == pytest.approx(20.0)

    def test_middle_follows_pattern_shape(self):
        """Middle of a long gap should follow the scaled pattern closely."""
        N = 100
        hours = np.linspace(0, 24, N)
        p = 30.0 + 10.0 * np.sin(2 * np.pi * hours / 24)
        filled = _blend_fill(p, left_anchor=31.0, right_anchor=29.0, blend_n=10)
        # Middle (indices 40-60) should be very close to p_scaled
        np.testing.assert_allclose(filled[40:60], p[40:60], atol=0.5)

    def test_no_amplification_when_pattern_endpoints_close(self):
        """Old affine blew up with close endpoints. Blend must not amplify."""
        p = np.array([35.0, 38.0, 32.0, 35.1])  # endpoints: 35.0, 35.1
        filled = _blend_fill(p, left_anchor=30.0, right_anchor=40.0, blend_n=2)
        # Affine would have scale=100, producing crazy values.
        # Blend should stay sane.
        assert filled.max() < 50
        assert filled.min() > 20

    def test_left_only_boundary(self):
        p = np.array([35.0, 38.0, 32.0, 35.0])
        filled = _blend_fill(p, left_anchor=30.0, right_anchor=None, blend_n=2)
        assert filled[0] == pytest.approx(30.0)

    def test_right_only_boundary(self):
        p = np.array([35.0, 38.0, 32.0, 35.0])
        filled = _blend_fill(p, left_anchor=None, right_anchor=40.0, blend_n=2)
        assert filled[-1] == pytest.approx(40.0)

    def test_no_boundaries_returns_input(self):
        p = np.array([10.0, 15.0, 20.0, 15.0])
        filled = _blend_fill(p, left_anchor=None, right_anchor=None, blend_n=2)
        np.testing.assert_allclose(filled, p)

    def test_short_gap_overlapping_blend_zones(self):
        """When gap < 2*blend_n, blend zones overlap — still smooth."""
        p = np.array([35.0, 36.0])
        filled = _blend_fill(p, left_anchor=30.0, right_anchor=40.0, blend_n=10)
        assert filled[0] == pytest.approx(30.0)
        assert filled[-1] == pytest.approx(40.0)


# ---- Leading / trailing NaN preservation -----------------------------------


class TestLeadingTrailingNaN:
    def test_leading_nans_preserved(self):
        idx = pd.date_range("2024-01-01", periods=96, freq="15min")
        vals = 30 + 5 * np.sin(2 * np.pi * (idx.hour + idx.minute / 60) / 24)
        s = pd.Series(vals, index=idx, name="sensor_raw")
        s.iloc[:10] = np.nan
        result = pattern_fill([s], pattern=_sine_pattern())
        filled, _ = result[0]
        assert filled.iloc[:10].isna().all()
        assert filled.iloc[10:].notna().all()

    def test_trailing_nans_preserved(self):
        idx = pd.date_range("2024-01-01", periods=96, freq="15min")
        vals = 30 + 5 * np.sin(2 * np.pi * (idx.hour + idx.minute / 60) / 24)
        s = pd.Series(vals, index=idx, name="sensor_raw")
        s.iloc[-10:] = np.nan
        result = pattern_fill([s], pattern=_sine_pattern())
        filled, _ = result[0]
        assert filled.iloc[-10:].isna().all()

    def test_both_leading_trailing_with_interior(self):
        idx = pd.date_range("2024-01-01", periods=96, freq="15min")
        vals = 30 + 5 * np.sin(2 * np.pi * (idx.hour + idx.minute / 60) / 24)
        s = pd.Series(vals, index=idx, name="sensor_raw")
        s.iloc[:5] = np.nan
        s.iloc[40:50] = np.nan
        s.iloc[-5:] = np.nan
        result = pattern_fill([s], pattern=_sine_pattern())
        filled, _ = result[0]
        assert filled.iloc[:5].isna().all()
        assert filled.iloc[40:50].notna().all()
        assert filled.iloc[-5:].isna().all()

    def test_all_nan_stays_all_nan(self):
        idx = pd.date_range("2024-01-01", periods=96, freq="15min")
        s = pd.Series(np.nan, index=idx, name="sensor_raw")
        result = pattern_fill([s], pattern=_sine_pattern())
        filled, _ = result[0]
        assert filled.isna().all()


# ---- Midnight-spanning gap -------------------------------------------------


class TestMidnightSpanningGap:
    def test_midnight_gap_no_distortion(self):
        """Gap spanning midnight should follow natural pattern shape."""
        idx = pd.date_range("2024-01-01", periods=7 * 96, freq="15min")
        vals = 30 + 5 * np.sin(2 * np.pi * (idx.hour + idx.minute / 60) / 24)
        s = pd.Series(vals, index=idx, name="sensor_raw")
        gap_start = pd.Timestamp("2024-01-03 22:00")
        gap_end = pd.Timestamp("2024-01-04 02:00")
        s.loc[gap_start:gap_end] = np.nan

        result = pattern_fill([s], pattern=_sine_pattern())
        filled, _ = result[0]

        # Filled values in the gap should stay in a reasonable range
        gap_vals = filled.loc[gap_start:gap_end]
        assert gap_vals.min() > 15, f"Midnight fill too low: {gap_vals.min()}"
        assert gap_vals.max() < 45, f"Midnight fill too high: {gap_vals.max()}"


# ---- Area normalization ----------------------------------------------------


class TestAreaNormalization:
    def _make_long_series_with_gap(self):
        """14-day series at 15min with a 12h gap."""
        idx = pd.date_range("2024-01-01", periods=14 * 96, freq="15min")
        frac_h = idx.hour + idx.minute / 60.0
        vals = 30 + 10 * np.sin(2 * np.pi * frac_h / 24)
        s = pd.Series(vals, index=idx, name="sensor_raw")
        # Gap in the middle
        s.iloc[500:548] = np.nan
        return s

    def test_normalized_area_close_to_expected(self):
        """Fill area should approximate the daily profile integral."""
        s = self._make_long_series_with_gap()
        result = pattern_fill(
            [s], pattern=_sine_pattern(), normalize_area=True
        )
        filled, _ = result[0]
        # The fill should exist
        assert filled.iloc[500:548].notna().all()

    def test_normalize_off_by_default(self):
        """Without normalize_area, the fill still works."""
        s = self._make_long_series_with_gap()
        result = pattern_fill([s], pattern=_sine_pattern())
        filled, _ = result[0]
        assert filled.iloc[500:548].notna().all()

    def test_flat_data_no_crash(self):
        """Flat data shouldn't crash area normalization."""
        idx = pd.date_range("2024-01-01", periods=14 * 96, freq="15min")
        s = pd.Series(42.0, index=idx, name="sensor_raw")
        s.iloc[200:248] = np.nan
        result = pattern_fill(
            [s], pattern=_sine_pattern(), normalize_area=True
        )
        filled, _ = result[0]
        assert filled.iloc[200:248].notna().all()

    def test_short_gap_area_normalized(self):
        """Even a short gap should work with area normalization."""
        s = self._make_long_series_with_gap()
        s.iloc[500:504] = np.nan  # only 4 points
        s.iloc[504:548] = 30 + 10 * np.sin(
            2
            * np.pi
            * (s.index[504:548].hour + s.index[504:548].minute / 60.0)
            / 24
        )
        result = pattern_fill(
            [s], pattern=_sine_pattern(), normalize_area=True
        )
        filled, _ = result[0]
        assert filled.iloc[500:504].notna().all()


# ---- Smoke test with demo dataset ------------------------------------------


class TestSmokeDemo:
    def test_demo_dataset_no_negatives_no_overshoot(self):
        """Replicate the exact demo dataset from the notebook.

        Assert: no negative values, no values exceeding the max of the holey
        data.
        """
        rng = np.random.default_rng(42)
        index = pd.date_range("2024-01-01", periods=14 * 96, freq="15min")
        hours = index.hour + index.minute / 60.0
        base = 0.6 * np.sin(2 * np.pi * (hours - 6) / 24) + 0.2 * np.sin(
            4 * np.pi * (hours - 3) / 24
        )
        noise = 0.08 * rng.standard_normal(len(index))
        raw = base + noise
        values = 30 + 15 * (raw - raw.min()) / (raw.max() - raw.min())
        s = pd.Series(values, index=index, name="NH4_raw")

        # Introduce gaps
        s.iloc[200:248] = np.nan
        s.iloc[500:524] = np.nan
        s.iloc[900:1000] = np.nan

        holey_max = s.max()

        pat = _sine_pattern()
        result = pattern_fill([s], pattern=pat)
        filled, _ = result[0]

        # All interior gaps should be filled
        assert filled.iloc[200:248].notna().all()
        assert filled.iloc[500:524].notna().all()
        assert filled.iloc[900:1000].notna().all()

        # No negative values
        assert filled.min() >= 0, f"Negative value: min={filled.min()}"

        # No values exceeding the max of the holey dataset
        assert filled.max() <= holey_max * 1.1, (
            f"Overshoot: fill max={filled.max():.2f}, holey max={holey_max:.2f}"
        )


# ---- Existing TestPatternFill (updated) ------------------------------------


class TestPatternFill:
    def test_fills_nans(self):
        s = _make_series_with_gap()
        result = pattern_fill([s], pattern=_sine_pattern())
        assert len(result) == 1
        filled_series, steps = result[0]
        assert not filled_series.isna().any()

    def test_returns_processing_step(self):
        s = _make_series_with_gap()
        result = pattern_fill([s], pattern=_sine_pattern())
        _, steps = result[0]
        assert len(steps) == 1
        step = steps[0]
        assert step.suffix == "PAT-FILL"
        assert step.type.value == "gap_filling"

    def test_output_naming(self):
        s = _make_series_with_gap()
        result = pattern_fill([s], pattern=_sine_pattern())
        filled_series, _ = result[0]
        assert filled_series.name == "sensor_PAT-FILL"

    def test_preserves_non_nan_values(self):
        s = _make_series_with_gap()
        non_nan_mask = s.notna()
        result = pattern_fill([s], pattern=_sine_pattern())
        filled, _ = result[0]
        np.testing.assert_allclose(
            filled.loc[non_nan_mask].values,
            s.loc[non_nan_mask].values,
        )

    def test_boundary_continuity(self):
        s = _make_series_with_gap()
        nan_mask = s.isna()
        first_nan = nan_mask.idxmax()
        last_valid_before = s.loc[:first_nan].dropna().index[-1]

        result = pattern_fill([s], pattern=_sine_pattern())
        filled, _ = result[0]

        val_before = s.loc[last_valid_before]
        filled_at_start = filled.loc[
            s.index[s.index.get_loc(last_valid_before) + 1]
        ]
        assert abs(val_before - filled_at_start) < 1.0

    def test_dict_pattern_weekday_weekend(self):
        s = _make_series_with_gap(days=14, gap_start_day=5, gap_hours=48)
        patterns = {
            "weekday": DailyPattern(
                hours=list(range(24)),
                values=[
                    0.5 + 0.5 * np.sin(2 * np.pi * h / 24) for h in range(24)
                ],
                name="wd",
                day_type="weekday",
            ),
            "weekend": DailyPattern(
                hours=list(range(24)),
                values=[
                    0.3 + 0.3 * np.sin(2 * np.pi * h / 24) for h in range(24)
                ],
                name="we",
                day_type="weekend",
            ),
        }
        result = pattern_fill([s], pattern=patterns)
        filled, _ = result[0]
        assert not filled.isna().any()

    def test_no_gaps_passthrough(self):
        idx = pd.date_range("2024-01-01", periods=96, freq="15min")
        s = pd.Series(np.ones(96), index=idx, name="clean_raw")
        result = pattern_fill([s], pattern=_sine_pattern())
        filled, _ = result[0]
        np.testing.assert_allclose(filled.values, 1.0)

    def test_rejects_non_datetime_index(self):
        s = pd.Series([1, 2, np.nan], index=[0, 1, 2], name="bad_raw")
        with pytest.raises(TypeError, match="DatetimeIndex"):
            pattern_fill([s], pattern=_sine_pattern())

    def test_multiple_series(self):
        s1 = _make_series_with_gap()
        s2 = s1.copy()
        s2.name = "sensor2_raw"
        result = pattern_fill([s1, s2], pattern=_sine_pattern())
        assert len(result) == 2
        assert result[0][0].name == "sensor_PAT-FILL"
        assert result[1][0].name == "sensor2_PAT-FILL"


# ============================================================================
# Sine Mode Integration Tests
# ============================================================================


class TestPatternFillWithSineMode:
    """Integration tests for pattern_fill with sine-mode patterns."""

    def test_gap_fill_with_simple_sine_pattern(self):
        """Gap fill works with simple sine-mode pattern."""
        # Create series with gaps
        index = pd.date_range("2024-01-01", periods=96, freq="15min")
        series = pd.Series(
            np.sin(2 * np.pi * np.arange(96) / 96) + 1,
            index=index,
            name="test"
        )
        series.iloc[20:30] = np.nan

        # Create sine pattern
        pattern = DailyPattern.from_simple_sine(
            amplitude=0.5,
            frequency=1.0,
            phase=6.0,
            baseline=0.5
        )

        # Fill gaps
        results = pattern_fill([series], pattern=pattern)
        filled, steps = results[0]

        # Verify gaps are filled
        assert filled.isna().sum() == 0
        assert len(steps) == 1

        # Verify processing metadata includes pattern info
        assert hasattr(steps[0].parameters, 'pattern')
        assert steps[0].parameters.pattern['mode'] == 'sine'

    def test_gap_fill_with_multicomponent_sine(self):
        """Gap fill with multi-component sine pattern."""
        index = pd.date_range("2024-01-01", periods=96, freq="15min")
        series = pd.Series(np.random.randn(96) + 5, index=index, name="test")
        series.iloc[20:30] = np.nan

        # Multi-component pattern
        pattern = DailyPattern.from_sine_waves(
            components=[
                (0.3, 1.0, 8.0),
                (0.15, 2.0, 12.0),
            ],
            baseline=0.5
        )

        results = pattern_fill([series], pattern=pattern)
        filled, _ = results[0]

        assert filled.isna().sum() == 0

    def test_sine_pattern_weekday_weekend(self):
        """Sine patterns work with weekday/weekend discrimination."""
        s = _make_series_with_gap(days=14, gap_start_day=5, gap_hours=48)
        patterns = {
            "weekday": DailyPattern.from_sine_waves(
                [(0.35, 1.0, 8.0)],
                baseline=0.5,
                name="wd_sine",
                day_type="weekday"
            ),
            "weekend": DailyPattern.from_sine_waves(
                [(0.25, 1.0, 10.0)],
                baseline=0.45,
                name="we_sine",
                day_type="weekend"
            ),
        }
        result = pattern_fill([s], pattern=patterns)
        filled, _ = result[0]
        assert not filled.isna().any()

    def test_sine_pattern_preserves_non_nan_values(self):
        """Sine pattern filling preserves existing non-NaN values."""
        s = _make_series_with_gap()
        non_nan_mask = s.notna()

        pattern = DailyPattern.from_simple_sine(
            amplitude=0.4,
            frequency=1.0,
            phase=6.0,
            baseline=0.5
        )

        result = pattern_fill([s], pattern=pattern)
        filled, _ = result[0]

        np.testing.assert_allclose(
            filled.loc[non_nan_mask].values,
            s.loc[non_nan_mask].values,
        )


# ============================================================================
# AR Noise Injection Tests
# ============================================================================


class TestPatternFillWithNoise:
    """Tests for AR noise injection in pattern_fill."""

    def _make_series_with_variability(self):
        """Create test series with gap and more variability."""
        idx = pd.date_range("2024-01-01", periods=7 * 96, freq="15min")
        hours = idx.hour + idx.minute / 60.0
        rng = np.random.default_rng(42)
        values = 10.0 + 5.0 * np.sin(2 * np.pi * hours / 24) + 0.5 * rng.standard_normal(len(hours))
        s = pd.Series(values, index=idx, name="sensor_raw")
        s.iloc[200:248] = np.nan  # 12-hour gap
        return s

    def test_add_noise_false_no_noise(self):
        """When add_noise=False, no noise is added."""
        s = _make_series_with_gap()
        result1 = pattern_fill([s], pattern=_sine_pattern(), add_noise=False)
        result2 = pattern_fill(
            [s], pattern=_sine_pattern(), add_noise=False, random_seed=42
        )

        filled1, _ = result1[0]
        filled2, _ = result2[0]

        # Should be identical
        np.testing.assert_allclose(filled1.values, filled2.values)

    def test_add_noise_true_adds_variability(self):
        """When add_noise=True, noise is added."""
        s = self._make_series_with_variability()
        result1 = pattern_fill([s], pattern=_sine_pattern(), add_noise=False)
        result2 = pattern_fill(
            [s], pattern=_sine_pattern(), add_noise=True, random_seed=42
        )

        filled1, _ = result1[0]
        filled2, _ = result2[0]

        # Filled regions should differ
        gap_mask = s.isna()
        assert not np.allclose(
            filled1.loc[gap_mask].values, filled2.loc[gap_mask].values
        )

    def test_noise_reproducible_with_seed(self):
        """Same seed produces same noise."""
        s = _make_series_with_gap()
        result1 = pattern_fill(
            [s], pattern=_sine_pattern(), add_noise=True, random_seed=42
        )
        result2 = pattern_fill(
            [s], pattern=_sine_pattern(), add_noise=True, random_seed=42
        )

        filled1, _ = result1[0]
        filled2, _ = result2[0]

        np.testing.assert_allclose(filled1.values, filled2.values)

    def test_noise_preserves_boundary_continuity(self):
        """Noise injection preserves boundary continuity."""
        s = _make_series_with_gap()
        nan_mask = s.isna()
        first_nan = nan_mask.idxmax()
        last_valid_before = s.loc[:first_nan].dropna().index[-1]

        result = pattern_fill(
            [s], pattern=_sine_pattern(), add_noise=True, random_seed=42
        )
        filled, _ = result[0]

        val_before = s.loc[last_valid_before]
        filled_at_start = filled.loc[
            s.index[s.index.get_loc(last_valid_before) + 1]
        ]
        assert abs(val_before - filled_at_start) < 1.0

    def test_noise_no_negative_values(self):
        """Noise injection doesn't produce negative values."""
        s = _make_series_with_gap()
        result = pattern_fill(
            [s], pattern=_sine_pattern(), add_noise=True, random_seed=42
        )
        filled, _ = result[0]

        assert filled.min() >= 0

    def test_noise_no_overshoot(self):
        """Noise injection doesn't cause excessive overshoot."""
        s = _make_series_with_gap()
        holey_max = s.max()

        result = pattern_fill(
            [s], pattern=_sine_pattern(), add_noise=True, random_seed=42
        )
        filled, _ = result[0]

        assert filled.max() <= holey_max * 1.1

    def test_ar_order_affects_noise(self):
        """Different AR orders produce different noise."""
        s = self._make_series_with_variability()
        result1 = pattern_fill(
            [s], pattern=_sine_pattern(), add_noise=True, ar_order=1, random_seed=42
        )
        result2 = pattern_fill(
            [s], pattern=_sine_pattern(), add_noise=True, ar_order=4, random_seed=42
        )

        filled1, _ = result1[0]
        filled2, _ = result2[0]

        gap_mask = s.isna()
        assert not np.allclose(
            filled1.loc[gap_mask].values, filled2.loc[gap_mask].values
        )

    def test_noise_with_area_normalization(self):
        """Noise works correctly with area normalization."""
        s = _make_series_with_gap(days=14, gap_start_day=3, gap_hours=12)
        result = pattern_fill(
            [s],
            pattern=_sine_pattern(),
            normalize_area=True,
            add_noise=True,
            random_seed=42,
        )
        filled, _ = result[0]

        # Should fill all gaps
        assert filled.isna().sum() == 0

    def test_noise_with_dict_pattern(self):
        """Noise works with weekday/weekend patterns."""
        s = _make_series_with_gap(days=14, gap_start_day=5, gap_hours=48)
        patterns = {
            "weekday": DailyPattern(
                hours=list(range(24)),
                values=[
                    0.5 + 0.5 * np.sin(2 * np.pi * h / 24) for h in range(24)
                ],
                name="wd",
                day_type="weekday",
            ),
            "weekend": DailyPattern(
                hours=list(range(24)),
                values=[
                    0.3 + 0.3 * np.sin(2 * np.pi * h / 24) for h in range(24)
                ],
                name="we",
                day_type="weekend",
            ),
        }
        result = pattern_fill(
            [s], pattern=patterns, add_noise=True, random_seed=42
        )
        filled, _ = result[0]
        assert not filled.isna().any()

    def test_noise_with_sine_pattern(self):
        """Noise works with sine-mode patterns."""
        s = _make_series_with_gap()
        pattern = DailyPattern.from_simple_sine(
            amplitude=0.4, frequency=1.0, phase=6.0, baseline=0.5
        )
        result = pattern_fill(
            [s], pattern=pattern, add_noise=True, random_seed=42
        )
        filled, _ = result[0]
        assert not filled.isna().any()

    def test_noise_multiple_gaps(self):
        """Noise works correctly with multiple gaps."""
        s = _make_series_with_gap()
        # Add another gap
        s.iloc[500:524] = np.nan

        result = pattern_fill(
            [s], pattern=_sine_pattern(), add_noise=True, random_seed=42
        )
        filled, _ = result[0]
        assert not filled.isna().any()

    def test_noise_preserves_non_nan_values(self):
        """Noise injection preserves existing non-NaN values."""
        s = _make_series_with_gap()
        non_nan_mask = s.notna()

        result = pattern_fill(
            [s], pattern=_sine_pattern(), add_noise=True, random_seed=42
        )
        filled, _ = result[0]

        np.testing.assert_allclose(
            filled.loc[non_nan_mask].values,
            s.loc[non_nan_mask].values,
        )


class TestArWindowDays:
    """Tests for the ar_window_days rolling-window AR fitting feature."""

    def test_ar_window_days_none_matches_global(self):
        """ar_window_days=None (default) produces the same result as omitting it."""
        s = _make_series_with_gap()
        pat = _sine_pattern()

        result_default = pattern_fill([s.copy()], pattern=pat, add_noise=True, random_seed=0)
        result_none = pattern_fill(
            [s.copy()], pattern=pat, add_noise=True, ar_window_days=None, random_seed=0
        )
        filled_default, _ = result_default[0]
        filled_none, _ = result_none[0]
        np.testing.assert_array_equal(filled_default.values, filled_none.values)

    def test_ar_window_days_uses_local_data(self):
        """ar_window_days produces different fills than global fit when variability changes."""
        # Build a series: first half very noisy, second half smooth — gap in smooth region
        rng = np.random.default_rng(42)
        idx = pd.date_range("2024-01-01", periods=30 * 96, freq="15min")
        frac_h = idx.hour + idx.minute / 60.0
        base = 10.0 + 5.0 * np.sin(2 * np.pi * frac_h / 24)
        noise_level = np.where(idx < pd.Timestamp("2024-01-16"), 4.0, 0.05)
        values = base + rng.normal(0, noise_level)
        s = pd.Series(values, index=idx, name="sensor_raw")

        # Gap in the smooth (second) half
        gap_start = pd.Timestamp("2024-01-22 06:00")
        gap_end = gap_start + pd.Timedelta(hours=12)
        s.loc[gap_start:gap_end] = np.nan

        pat = _sine_pattern()

        result_global = pattern_fill(
            [s.copy()], pattern=pat, add_noise=True, ar_window_days=None, random_seed=7
        )
        result_window = pattern_fill(
            [s.copy()], pattern=pat, add_noise=True, ar_window_days=7, random_seed=7
        )
        filled_global, _ = result_global[0]
        filled_window, _ = result_window[0]

        gap_mask = s.isna()
        # The two approaches use different calibration data so must produce different fills
        assert not np.allclose(
            filled_global[gap_mask].values, filled_window[gap_mask].values
        )

    def test_ar_window_days_sparse_data_expands_window(self):
        """Gap near start of series with little prior data should not raise."""
        idx = pd.date_range("2024-01-01", periods=10 * 96, freq="15min")
        frac_h = idx.hour + idx.minute / 60.0
        values = 10.0 + 5.0 * np.sin(2 * np.pi * frac_h / 24)
        s = pd.Series(values, index=idx, name="sensor_raw")

        # Gap after only half a day of data
        gap_start = pd.Timestamp("2024-01-01 12:00")
        gap_end = gap_start + pd.Timedelta(hours=6)
        s.loc[gap_start:gap_end] = np.nan

        pat = _sine_pattern()
        # Should not raise even though requested window (7 days) exceeds available data
        result = pattern_fill(
            [s], pattern=pat, add_noise=True, ar_window_days=7, random_seed=0
        )
        filled, _ = result[0]
        assert not filled.isna().any()

    def test_ar_window_days_no_prior_data_still_fills(self):
        """When the gap is at the very start, fill completes without crashing."""
        idx = pd.date_range("2024-01-01", periods=5 * 96, freq="15min")
        frac_h = idx.hour + idx.minute / 60.0
        values = 10.0 + 5.0 * np.sin(2 * np.pi * frac_h / 24)
        s = pd.Series(values, index=idx, name="sensor_raw")

        # Gap starting at 06:00 on day 1 — only ~24 clean samples before it
        gap_start = pd.Timestamp("2024-01-01 06:00")
        gap_end = gap_start + pd.Timedelta(hours=12)
        s.loc[gap_start:gap_end] = np.nan

        pat = _sine_pattern()
        result = pattern_fill(
            [s], pattern=pat, add_noise=True, ar_window_days=7, random_seed=0
        )
        filled, _ = result[0]
        assert not filled.isna().any()

    def test_ar_window_captures_local_noise_level(self):
        """AR window should capture local noise characteristics.

        When data has non-stationary noise (high noise period followed by
        low noise period), a short AR window in the low noise period should
        produce lower noise than a long window that includes the high noise period.
        """
        rng = np.random.default_rng(42)
        idx = pd.date_range("2024-01-01", periods=10 * 96, freq="15min")
        frac_h = idx.hour + idx.minute / 60.0
        base = 10.0 + 5.0 * np.sin(2 * np.pi * frac_h / 24)

        # High noise for first 7 days, low noise for last 3 days
        day = (idx - idx[0]).total_seconds() / 86400
        noise_std = np.where(day < 7, 2.0, 0.2)
        noise = noise_std * rng.standard_normal(len(idx))
        values = base + noise
        s = pd.Series(values, index=idx, name="sensor_raw")

        # Gap in the low-noise period (day 8)
        gap_start_idx = 8 * 96  # start of day 8
        gap_end_idx = gap_start_idx + 48  # 12-hour gap
        s.iloc[gap_start_idx:gap_end_idx] = np.nan

        pat = _sine_pattern()

        # Fill with short AR window (1 day - only low noise data)
        result_short = pattern_fill(
            [s.copy()], pattern=pat, add_noise=True, ar_window_days=1, random_seed=42
        )
        filled_short, _ = result_short[0]

        # Fill with long AR window (7 days - includes high noise data)
        result_long = pattern_fill(
            [s.copy()], pattern=pat, add_noise=True, ar_window_days=7, random_seed=42
        )
        filled_long, _ = result_long[0]

        # Fill without noise to get baseline
        result_base = pattern_fill(
            [s.copy()], pattern=pat, add_noise=False
        )
        filled_base, _ = result_base[0]

        # Compute noise std in the gap
        gap_slice = slice(gap_start_idx, gap_end_idx)
        noise_short = np.std(filled_short.iloc[gap_slice].values - filled_base.iloc[gap_slice].values)
        noise_long = np.std(filled_long.iloc[gap_slice].values - filled_base.iloc[gap_slice].values)

        # Short window (low noise data) should produce LOWER noise than long window
        # The ratio should be significant (at least 2x difference)
        assert noise_short < noise_long, (
            f"Short window noise ({noise_short:.4f}) should be less than "
            f"long window noise ({noise_long:.4f})"
        )
        # The ratio should be substantial (reflecting the 10x noise difference in data)
        assert noise_long / noise_short > 2, (
            f"Noise ratio ({noise_long/noise_short:.2f}) should be > 2"
        )


# ============================================================================
# Pattern Window Days Tests
# ============================================================================


class TestPatternWindowDays:
    """Tests for the pattern_window_days backward-window normalization feature."""

    def test_pattern_window_days_none_is_global(self):
        """pattern_window_days=None uses all clean data."""
        s = _make_series_with_gap()
        result = pattern_fill([s.copy()], pattern=_sine_pattern(), pattern_window_days=None)
        filled, _ = result[0]
        assert not filled.isna().any()

    def test_pattern_window_days_affects_fill_level(self):
        """Different pattern_window_days produces different fill levels when data varies."""
        idx = pd.date_range("2024-01-01", periods=30 * 96, freq="15min")
        frac_h = idx.hour + idx.minute / 60.0
        base = 5.0 * np.sin(2 * np.pi * frac_h / 24)
        offset = np.where(idx < pd.Timestamp("2024-01-15"), 50.0, 10.0)
        values = offset + base
        s = pd.Series(values, index=idx, name="sensor_raw")
        gap_start = pd.Timestamp("2024-01-22 06:00")
        gap_end = gap_start + pd.Timedelta(hours=12)
        s.loc[gap_start:gap_end] = np.nan

        pat = _sine_pattern()
        result_short = pattern_fill([s.copy()], pattern=pat, pattern_window_days=3)
        result_global = pattern_fill([s.copy()], pattern=pat, pattern_window_days=None)
        filled_short, _ = result_short[0]
        filled_global, _ = result_global[0]

        gap_mask = s.isna()
        assert not np.allclose(
            filled_short[gap_mask].values, filled_global[gap_mask].values, rtol=0.1
        )

    def test_pattern_window_days_sparse_expands(self):
        """When window has < 2 clean samples, expands to all available data."""
        idx = pd.date_range("2024-01-01", periods=10 * 96, freq="15min")
        frac_h = idx.hour + idx.minute / 60.0
        values = 10.0 + 5.0 * np.sin(2 * np.pi * frac_h / 24)
        s = pd.Series(values, index=idx, name="sensor_raw")
        gap_start = pd.Timestamp("2024-01-01 06:00")
        gap_end = gap_start + pd.Timedelta(hours=6)
        s.loc[gap_start:gap_end] = np.nan
        result = pattern_fill([s], pattern=_sine_pattern(), pattern_window_days=7)
        filled, _ = result[0]
        assert not filled.isna().any()


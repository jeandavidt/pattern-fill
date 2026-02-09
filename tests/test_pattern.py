import json

import numpy as np
import pandas as pd
import pytest

from pattern_fill.pattern import DailyPattern, SineComponent


def _simple_pattern() -> DailyPattern:
    """A sine-wave pattern with 24 hourly control points."""
    hours = list(range(24))
    values = [0.5 + 0.5 * np.sin(2 * np.pi * h / 24) for h in hours]
    return DailyPattern(hours=hours, values=values, name="sine")


class TestDailyPatternInit:
    def test_basic_creation(self):
        p = _simple_pattern()
        assert p.name == "sine"
        assert p.day_type == "all"
        assert p.periodic is True

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            DailyPattern(hours=[0, 6, 12], values=[0.5, 1.0])

    def test_too_few_points(self):
        with pytest.raises(ValueError, match="at least 2"):
            DailyPattern(hours=[0], values=[0.5])

    def test_invalid_day_type(self):
        with pytest.raises(ValueError, match="day_type"):
            DailyPattern(hours=[0, 12], values=[0, 1], day_type="holiday")


class TestEvaluate:
    def test_returns_correct_shape(self):
        p = _simple_pattern()
        h = np.linspace(0, 24, 100)
        result = p.evaluate(h)
        assert result.shape == (100,)

    def test_periodic_wrapping(self):
        p = _simple_pattern()
        val_0 = p.evaluate(np.array([0.0]))[0]
        val_24 = p.evaluate(np.array([24.0]))[0]
        assert abs(val_0 - val_24) < 1e-10

    def test_interpolates_between_points(self):
        p = _simple_pattern()
        val = p.evaluate(np.array([6.0]))[0]
        # sine peaks at 6h → should be near 1.0
        assert val == pytest.approx(1.0, abs=0.05)

    def test_values_clamped_to_0_1(self):
        """Pattern values should be clamped to [0,1] to prevent spline overshoot."""
        p = DailyPattern(
            hours=[0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0],
            values=[0.037, 0.340, 0.487, 0.619, 0.930, 1.000, 0.516, 0.000],
        )
        test_hours = np.linspace(0, 24, 500)
        results = p.evaluate(test_hours)
        assert results.min() >= 0.0, "Values should not go below 0"
        assert results.max() <= 1.0, "Values should not go above 1"


class TestToSeries:
    def test_produces_series_with_matching_index(self):
        p = _simple_pattern()
        idx = pd.date_range("2024-01-01", periods=96, freq="15min")
        s = p.to_series(idx)
        assert isinstance(s, pd.Series)
        assert len(s) == 96
        assert s.index.equals(idx)

    def test_values_follow_pattern(self):
        p = _simple_pattern()
        idx = pd.date_range("2024-01-01 06:00", periods=1, freq="h")
        s = p.to_series(idx)
        assert s.iloc[0] == pytest.approx(1.0, abs=0.05)


class TestSerialization:
    def test_roundtrip_dict(self):
        p = _simple_pattern()
        d = p.to_dict()
        p2 = DailyPattern.from_dict(d)
        assert p2.hours == p.hours
        assert p2.values == p.values
        assert p2.name == p.name
        assert p2.day_type == p.day_type

    def test_roundtrip_json(self):
        p = _simple_pattern()
        j = p.to_json()
        p2 = DailyPattern.from_json(j)
        np.testing.assert_allclose(
            p2.evaluate(np.linspace(0, 24, 50)),
            p.evaluate(np.linspace(0, 24, 50)),
        )

    def test_json_is_valid(self):
        p = _simple_pattern()
        d = json.loads(p.to_json())
        assert set(d.keys()) == {"hours", "values", "name", "periodic", "day_type", "mode"}


# ============================================================================
# Sine Mode Tests
# ============================================================================


class TestSineModeInit:
    def test_sine_mode_basic_creation(self):
        """Sine mode pattern can be created with sine_components."""
        comps = [SineComponent(amplitude=0.5, frequency=1.0, phase=6.0)]
        pattern = DailyPattern(sine_components=comps, baseline=0.5)
        assert pattern.mode == "sine"
        assert pattern.baseline == 0.5
        assert len(pattern.sine_components) == 1

    def test_cannot_mix_modes(self):
        """Cannot specify both spline and sine parameters."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            DailyPattern(
                hours=[0, 12],
                values=[0.5, 0.8],
                sine_components=[SineComponent(0.5, 1.0, 0.0)]
            )

    def test_must_specify_mode(self):
        """Must specify either spline or sine parameters."""
        with pytest.raises(ValueError, match="Must specify either"):
            DailyPattern()

    def test_empty_sine_components(self):
        """Cannot create pattern with empty sine_components list."""
        with pytest.raises(ValueError, match="cannot be empty"):
            DailyPattern(sine_components=[], baseline=0.5)

    def test_invalid_baseline(self):
        """Baseline must be in [0, 1] range."""
        comps = [SineComponent(0.5, 1.0, 0.0)]
        with pytest.raises(ValueError, match="baseline"):
            DailyPattern(sine_components=comps, baseline=1.5)
        with pytest.raises(ValueError, match="baseline"):
            DailyPattern(sine_components=comps, baseline=-0.1)


class TestSineModeEvaluate:
    def test_evaluate_single_component(self):
        """Sine pattern evaluates correctly with single component."""
        comps = [SineComponent(amplitude=0.5, frequency=1.0, phase=6.0)]
        pattern = DailyPattern(sine_components=comps, baseline=0.5, name="test")

        # At phase hour (6), baseline + amplitude = 1.0
        # At opposite (18), baseline - amplitude = 0.0
        result = pattern.evaluate(np.array([6.0, 18.0]))
        assert result[0] == pytest.approx(1.0, abs=0.01)
        assert result[1] == pytest.approx(0.0, abs=0.01)

    def test_evaluate_multiple_components(self):
        """Sine pattern sums multiple components correctly."""
        comps = [
            SineComponent(amplitude=0.3, frequency=1.0, phase=6.0),
            SineComponent(amplitude=0.1, frequency=2.0, phase=12.0),
        ]
        pattern = DailyPattern(sine_components=comps, baseline=0.5)

        hours = np.linspace(0, 24, 100)
        result = pattern.evaluate(hours)

        assert len(result) == 100
        assert result.min() >= 0.0  # Clipped to [0, 1]
        assert result.max() <= 1.0

    def test_clipping_prevents_overflow(self):
        """Values are clipped to [0, 1] even if sum exceeds."""
        # Create components that would exceed [0, 1] without clipping
        comps = [SineComponent(amplitude=0.8, frequency=1.0, phase=0.0)]
        pattern = DailyPattern(sine_components=comps, baseline=0.5)

        hours = np.linspace(0, 24, 100)
        result = pattern.evaluate(hours)

        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_periodic_wrapping_sine(self):
        """Sine pattern wraps correctly at midnight."""
        comps = [SineComponent(amplitude=0.4, frequency=1.0, phase=6.0)]
        pattern = DailyPattern(sine_components=comps, baseline=0.5)

        val_0 = pattern.evaluate(np.array([0.0]))[0]
        val_24 = pattern.evaluate(np.array([24.0]))[0]
        assert abs(val_0 - val_24) < 1e-10


class TestFactoryMethods:
    def test_from_sine_waves_with_tuples(self):
        """from_sine_waves factory method works with tuples."""
        pattern = DailyPattern.from_sine_waves(
            components=[(0.5, 1.0, 6.0), (0.2, 2.0, 12.0)],
            baseline=0.4
        )

        assert pattern.mode == "sine"
        assert len(pattern.sine_components) == 2
        assert pattern.baseline == 0.4
        assert pattern.sine_components[0].amplitude == 0.5
        assert pattern.sine_components[0].frequency == 1.0
        assert pattern.sine_components[0].phase == 6.0

    def test_from_sine_waves_with_objects(self):
        """from_sine_waves factory accepts SineComponent objects."""
        comps = [
            SineComponent(0.3, 1.0, 8.0),
            SineComponent(0.15, 2.0, 12.0),
        ]
        pattern = DailyPattern.from_sine_waves(components=comps, baseline=0.5)

        assert pattern.mode == "sine"
        assert len(pattern.sine_components) == 2

    def test_from_simple_sine(self):
        """from_simple_sine creates single-component pattern."""
        pattern = DailyPattern.from_simple_sine(
            amplitude=0.4,
            frequency=1.0,
            phase=6.0,
            baseline=0.5
        )

        assert pattern.mode == "sine"
        assert len(pattern.sine_components) == 1
        assert pattern.sine_components[0].amplitude == 0.4
        assert pattern.sine_components[0].frequency == 1.0
        assert pattern.sine_components[0].phase == 6.0
        assert pattern.baseline == 0.5


class TestSineModeSerialization:
    def test_to_dict_includes_mode(self):
        """Sine pattern serialization includes mode field."""
        comps = [SineComponent(0.3, 1.0, 8.0)]
        pattern = DailyPattern(sine_components=comps, baseline=0.45, name="test")

        d = pattern.to_dict()
        assert d["mode"] == "sine"
        assert "sine_components" in d
        assert "baseline" in d
        assert len(d["sine_components"]) == 1

    def test_roundtrip_dict(self):
        """Sine pattern serializes and deserializes correctly."""
        comps = [
            SineComponent(amplitude=0.3, frequency=1.0, phase=8.0),
            SineComponent(amplitude=0.15, frequency=2.0, phase=12.0),
        ]
        pattern = DailyPattern(
            sine_components=comps,
            baseline=0.45,
            name="test_sine",
            day_type="weekday"
        )

        # to_dict / from_dict
        d = pattern.to_dict()
        pattern2 = DailyPattern.from_dict(d)

        assert pattern2.mode == "sine"
        assert len(pattern2.sine_components) == 2
        assert pattern2.baseline == 0.45
        assert pattern2.name == "test_sine"
        assert pattern2.day_type == "weekday"

        # Evaluate both and compare
        hours = np.linspace(0, 24, 100)
        np.testing.assert_allclose(
            pattern.evaluate(hours),
            pattern2.evaluate(hours),
            rtol=1e-10
        )

    def test_roundtrip_json(self):
        """Sine pattern JSON serialization roundtrip."""
        pattern = DailyPattern.from_sine_waves(
            components=[(0.4, 1.0, 7.0)],
            baseline=0.5,
            name="json_test"
        )

        json_str = pattern.to_json()
        pattern2 = DailyPattern.from_json(json_str)

        hours = np.linspace(0, 24, 100)
        np.testing.assert_allclose(
            pattern.evaluate(hours),
            pattern2.evaluate(hours),
            rtol=1e-10
        )

    def test_backward_compatibility_spline_dict(self):
        """Old dict format without 'mode' defaults to spline."""
        old_dict = {
            "hours": [0, 6, 12, 18],
            "values": [0.5, 0.8, 0.9, 0.6],
            "name": "old_pattern",
            "periodic": True,
            "day_type": "all",
        }

        pattern = DailyPattern.from_dict(old_dict)
        assert pattern.mode == "spline"
        assert pattern.hours == [0, 6, 12, 18]

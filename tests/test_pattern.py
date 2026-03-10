import json

import numpy as np
import pandas as pd
import pytest

from pattern_fill.pattern import DailyPattern, SineComponent, GaussianComponent


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
        with pytest.raises(ValueError, match="more than one mode"):
            DailyPattern(
                hours=[0, 12],
                values=[0.5, 0.8],
                sine_components=[SineComponent(0.5, 1.0, 0.0)]
            )

    def test_must_specify_mode(self):
        """Must specify either spline or sine parameters."""
        with pytest.raises(ValueError, match="Must specify"):
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


# ---------------------------------------------------------------------------
# Gaussian mode tests
# ---------------------------------------------------------------------------

class TestGaussianModeInit:
    def test_basic_creation(self):
        p = DailyPattern(
            gaussian_components=[GaussianComponent(0.8, 8.0, 2.0)],
            baseline=0.1,
        )
        assert p.mode == "gaussian"
        assert p.baseline == 0.1
        assert len(p.gaussian_components) == 1

    def test_mode_mixing_error(self):
        with pytest.raises(ValueError, match="more than one mode"):
            DailyPattern(
                hours=[0, 12],
                values=[0.0, 1.0],
                gaussian_components=[GaussianComponent(0.5, 6.0, 2.0)],
            )

    def test_empty_components_error(self):
        with pytest.raises(ValueError, match="empty"):
            DailyPattern(gaussian_components=[])

    def test_invalid_baseline_error(self):
        with pytest.raises(ValueError, match="baseline"):
            DailyPattern(
                gaussian_components=[GaussianComponent(0.5, 6.0, 2.0)],
                baseline=1.5,
            )


class TestGaussianModeEvaluate:
    def _pattern(self):
        return DailyPattern(
            gaussian_components=[GaussianComponent(0.7, 8.0, 2.0)],
            baseline=0.1,
        )

    def test_peak_above_baseline(self):
        p = self._pattern()
        val_at_peak = p.evaluate(np.array([8.0]))[0]
        assert val_at_peak > 0.1

    def test_values_clipped_to_01(self):
        p = DailyPattern(
            gaussian_components=[GaussianComponent(1.0, 6.0, 0.5)],
            baseline=0.5,
        )
        h = np.linspace(0, 24, 97)
        v = p.evaluate(h)
        assert v.min() >= 0.0
        assert v.max() <= 1.0

    def test_periodicity(self):
        p = self._pattern()
        assert p.evaluate(np.array([0.0]))[0] == pytest.approx(
            p.evaluate(np.array([24.0]))[0]
        )

    def test_midnight_spanning_symmetry(self):
        p = DailyPattern(
            gaussian_components=[GaussianComponent(0.6, 23.0, 2.0)],
            baseline=0.0,
        )
        v_before = p.evaluate(np.array([22.0]))[0]
        v_after = p.evaluate(np.array([0.0]))[0]
        assert v_before == pytest.approx(v_after, rel=1e-6)


class TestGaussianModeSerialization:
    def _pattern(self):
        return DailyPattern(
            gaussian_components=[GaussianComponent(0.7, 8.0, 2.0)],
            baseline=0.1,
            name="my_gaussian",
            day_type="weekday",
        )

    def test_to_dict_mode(self):
        d = self._pattern().to_dict()
        assert d["mode"] == "gaussian"
        assert "gaussian_components" in d
        assert d["baseline"] == 0.1

    def test_from_dict_roundtrip(self):
        p = self._pattern()
        p2 = DailyPattern.from_dict(p.to_dict())
        assert p2.mode == "gaussian"
        assert p2.name == p.name
        assert p2.day_type == p.day_type
        assert p2.baseline == pytest.approx(p.baseline)

    def test_json_roundtrip(self):
        p = self._pattern()
        p2 = DailyPattern.from_json(p.to_json())
        assert p2.mode == "gaussian"
        assert len(p2.gaussian_components) == 1
        assert p2.gaussian_components[0].amplitude == pytest.approx(0.7)

    def test_from_dict_baseline_default(self):
        d = {
            "mode": "gaussian",
            "gaussian_components": [{"amplitude": 0.5, "center": 6.0, "width": 2.0}],
        }
        p = DailyPattern.from_dict(d)
        assert p.baseline == 0.0


class TestFromGaussiansFactory:
    def test_from_tuples(self):
        p = DailyPattern.from_gaussians([(0.8, 8.0, 2.0), (0.5, 18.0, 1.5)])
        assert p.mode == "gaussian"
        assert len(p.gaussian_components) == 2

    def test_from_objects(self):
        comps = [GaussianComponent(0.6, 10.0, 3.0)]
        p = DailyPattern.from_gaussians(comps)
        assert p.mode == "gaussian"

    def test_default_baseline_zero(self):
        p = DailyPattern.from_gaussians([(0.5, 6.0, 2.0)])
        assert p.baseline == 0.0

    def test_custom_name_and_day_type(self):
        p = DailyPattern.from_gaussians(
            [(0.5, 6.0, 2.0)], name="g", day_type="weekend"
        )
        assert p.name == "g"
        assert p.day_type == "weekend"

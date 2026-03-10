"""Tests for GaussianComponent class."""

import numpy as np
import pytest

from pattern_fill import GaussianComponent


class TestGaussianComponent:
    def test_basic_creation(self):
        comp = GaussianComponent(amplitude=0.8, center=8.0, width=2.0)
        assert comp.amplitude == 0.8
        assert comp.center == 8.0
        assert comp.width == 2.0

    def test_center_normalization(self):
        comp = GaussianComponent(amplitude=0.5, center=25.0, width=1.0)
        assert comp.center == pytest.approx(1.0)

        comp2 = GaussianComponent(amplitude=0.5, center=-1.0, width=1.0)
        assert comp2.center == pytest.approx(23.0)

    def test_validation_negative_amplitude(self):
        with pytest.raises(ValueError, match="amplitude"):
            GaussianComponent(amplitude=-0.1, center=6.0, width=2.0)

    def test_validation_zero_width(self):
        with pytest.raises(ValueError, match="width"):
            GaussianComponent(amplitude=0.5, center=6.0, width=0.0)

    def test_validation_negative_width(self):
        with pytest.raises(ValueError, match="width"):
            GaussianComponent(amplitude=0.5, center=6.0, width=-1.0)

    def test_evaluate_peak_at_center(self):
        comp = GaussianComponent(amplitude=1.0, center=8.0, width=2.0)
        hours = np.array([8.0])
        result = comp.evaluate(hours)
        # At center, k=0 dominates; exp(0) = 1 plus tiny k=±1 contributions
        assert result[0] > 0.99

    def test_evaluate_periodicity(self):
        comp = GaussianComponent(amplitude=0.7, center=6.0, width=2.0)
        assert comp.evaluate(np.array([0.0]))[0] == pytest.approx(
            comp.evaluate(np.array([24.0]))[0]
        )

    def test_evaluate_symmetry(self):
        comp = GaussianComponent(amplitude=0.5, center=12.0, width=2.0)
        left = comp.evaluate(np.array([10.0]))[0]
        right = comp.evaluate(np.array([14.0]))[0]
        assert left == pytest.approx(right, rel=1e-6)

    def test_evaluate_midnight_spanning(self):
        # Component centered at 23 should be symmetric around midnight
        comp = GaussianComponent(amplitude=0.5, center=23.0, width=2.0)
        # One hour before midnight and one hour after should be equal
        v_before = comp.evaluate(np.array([22.0]))[0]
        v_after = comp.evaluate(np.array([0.0]))[0]  # 24 mod 24 = 0, 1h after midnight
        # Actually center=23, so 22 is 1h before and 0 (=24) is 1h after
        assert v_before == pytest.approx(v_after, rel=1e-6)

    def test_serialization_roundtrip(self):
        comp = GaussianComponent(amplitude=0.6, center=14.0, width=3.0)
        d = comp.to_dict()
        assert d == {"amplitude": 0.6, "center": 14.0, "width": 3.0}

        comp2 = GaussianComponent.from_dict(d)
        assert comp2.amplitude == comp.amplitude
        assert comp2.center == comp.center
        assert comp2.width == comp.width

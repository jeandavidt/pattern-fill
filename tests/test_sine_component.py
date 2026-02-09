"""Tests for SineComponent class."""

import numpy as np
import pytest

from pattern_fill import SineComponent


class TestSineComponent:
    def test_basic_creation(self):
        comp = SineComponent(amplitude=0.5, frequency=1.0, phase=0.0)
        assert comp.amplitude == 0.5
        assert comp.frequency == 1.0
        assert comp.phase == 0.0

    def test_phase_normalization(self):
        # Phase > 24 should wrap
        comp = SineComponent(amplitude=0.5, frequency=1.0, phase=25.5)
        assert comp.phase == 1.5

        # Negative phase should wrap
        comp = SineComponent(amplitude=0.5, frequency=1.0, phase=-2.0)
        assert comp.phase == 22.0

    def test_validation_negative_amplitude(self):
        with pytest.raises(ValueError, match="amplitude"):
            SineComponent(amplitude=-0.1, frequency=1.0)

    def test_validation_zero_frequency(self):
        with pytest.raises(ValueError, match="frequency"):
            SineComponent(amplitude=0.5, frequency=0.0)

    def test_validation_negative_frequency(self):
        with pytest.raises(ValueError, match="frequency"):
            SineComponent(amplitude=0.5, frequency=-1.0)

    def test_evaluate_basic(self):
        # Daily sine wave, peak at hour 6
        comp = SineComponent(amplitude=1.0, frequency=1.0, phase=6.0)
        hours = np.array([6.0, 18.0])
        result = comp.evaluate(hours)

        # At phase hour (6), should be at peak (+1)
        # At opposite phase (18), should be at trough (-1)
        assert result[0] == pytest.approx(1.0, abs=0.01)
        assert result[1] == pytest.approx(-1.0, abs=0.01)

    def test_evaluate_twice_daily(self):
        # Twice-daily cycle
        comp = SineComponent(amplitude=0.5, frequency=2.0, phase=0.0)
        hours = np.array([0.0, 6.0, 12.0, 18.0])
        result = comp.evaluate(hours)

        # Peaks at 0 and 12, troughs at 6 and 18
        assert result[0] == pytest.approx(0.5, abs=0.01)
        assert result[1] == pytest.approx(-0.5, abs=0.01)
        assert result[2] == pytest.approx(0.5, abs=0.01)
        assert result[3] == pytest.approx(-0.5, abs=0.01)

    def test_serialization(self):
        comp = SineComponent(amplitude=0.3, frequency=2.0, phase=12.0)
        d = comp.to_dict()

        assert d == {"amplitude": 0.3, "frequency": 2.0, "phase": 12.0}

        comp2 = SineComponent.from_dict(d)
        assert comp2.amplitude == comp.amplitude
        assert comp2.frequency == comp.frequency
        assert comp2.phase == comp.phase

    def test_serialization_default_phase(self):
        d = {"amplitude": 0.5, "frequency": 1.0}  # No phase
        comp = SineComponent.from_dict(d)
        assert comp.phase == 0.0

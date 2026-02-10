"""Tests for AR model fitting and noise generation."""

import numpy as np
import pandas as pd
import pytest

from pattern_fill import (
    add_ar_noise,
    DailyPattern,
    fit_ar_model,
    generate_ar_noise,
)


class TestFitArModel:
    def test_returns_coefficients(self):
        """AR model fitting returns coefficient array."""
        rng = np.random.default_rng(42)
        residuals = rng.normal(0, 1, 100)
        coeffs = fit_ar_model(residuals, order=2)
        assert isinstance(coeffs, np.ndarray)
        assert len(coeffs) == 2

    def test_order_1(self):
        """AR(1) model fitting works."""
        rng = np.random.default_rng(42)
        residuals = rng.normal(0, 1, 100)
        coeffs = fit_ar_model(residuals, order=1)
        assert len(coeffs) == 1
        assert -1 < coeffs[0] < 1  # Stationary AR(1)

    def test_order_4(self):
        """AR(4) model fitting works."""
        rng = np.random.default_rng(42)
        residuals = rng.normal(0, 1, 100)
        coeffs = fit_ar_model(residuals, order=4)
        assert len(coeffs) == 4

    def test_order_5(self):
        """AR(5) model fitting works."""
        rng = np.random.default_rng(42)
        residuals = rng.normal(0, 1, 100)
        coeffs = fit_ar_model(residuals, order=5)
        assert len(coeffs) == 5

    def test_insufficient_data_raises_error(self):
        """Not enough data raises ValueError."""
        residuals = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Need at least"):
            fit_ar_model(residuals, order=3)

    def test_deterministic_with_seed(self):
        """Same input produces same coefficients."""
        rng = np.random.default_rng(42)
        residuals = rng.normal(0, 1, 100)
        coeffs1 = fit_ar_model(residuals, order=2)
        coeffs2 = fit_ar_model(residuals, order=2)
        np.testing.assert_allclose(coeffs1, coeffs2)

    def test_handles_constant_residuals(self):
        """Constant residuals don't crash."""
        residuals = np.ones(50)
        coeffs = fit_ar_model(residuals, order=2)
        assert len(coeffs) == 2


class TestGenerateArNoise:
    def test_returns_correct_length(self):
        """Generated noise has correct length."""
        coeffs = np.array([0.5])
        noise = generate_ar_noise(50, coeffs, noise_std=1.0, seed=42)
        assert len(noise) == 50

    def test_reproducible_with_seed(self):
        """Same seed produces same noise."""
        coeffs = np.array([0.5, 0.2])
        noise1 = generate_ar_noise(100, coeffs, noise_std=1.0, seed=42)
        noise2 = generate_ar_noise(100, coeffs, noise_std=1.0, seed=42)
        np.testing.assert_allclose(noise1, noise2)

    def test_different_seeds_different_noise(self):
        """Different seeds produce different noise."""
        coeffs = np.array([0.5])
        noise1 = generate_ar_noise(100, coeffs, noise_std=1.0, seed=42)
        noise2 = generate_ar_noise(100, coeffs, noise_std=1.0, seed=43)
        assert not np.allclose(noise1, noise2)

    def test_noise_std_affects_magnitude(self):
        """Higher noise_std produces larger magnitude noise."""
        coeffs = np.array([0.5])
        noise1 = generate_ar_noise(100, coeffs, noise_std=0.1, seed=42)
        noise2 = generate_ar_noise(100, coeffs, noise_std=1.0, seed=42)
        assert np.std(noise2) > np.std(noise1)

    def test_higher_order_produces_correlated_noise(self):
        """Higher AR order produces more correlated noise."""
        coeffs1 = np.array([0.1])
        coeffs2 = np.array([0.8, 0.1, 0.05])
        noise1 = generate_ar_noise(100, coeffs1, noise_std=1.0, seed=42)
        noise2 = generate_ar_noise(100, coeffs2, noise_std=1.0, seed=42)
        # Higher order should have higher autocorrelation
        acf1 = np.correlate(noise1, noise1, mode="full")
        acf2 = np.correlate(noise2, noise2, mode="full")
        assert acf2.max() > acf1.max()

    def test_zero_noise_std_returns_zeros(self):
        """Zero noise_std produces zero noise."""
        coeffs = np.array([0.5])
        noise = generate_ar_noise(50, coeffs, noise_std=0.0, seed=42)
        np.testing.assert_allclose(noise, np.zeros(50))

    def test_single_point(self):
        """Can generate single point of noise."""
        coeffs = np.array([0.5])
        noise = generate_ar_noise(1, coeffs, noise_std=1.0, seed=42)
        assert len(noise) == 1


class TestAddArNoise:
    def _make_test_series(self):
        """Create test series with gap."""
        idx = pd.date_range("2024-01-01", periods=7 * 96, freq="15min")
        hours = idx.hour + idx.minute / 60.0
        values = 10.0 + 5.0 * np.sin(2 * np.pi * hours / 24)
        s = pd.Series(values, index=idx, name="test")
        s.iloc[200:248] = np.nan  # 12-hour gap
        return s

    def test_returns_noise_array(self):
        """add_ar_noise returns noise array."""
        s = self._make_test_series()
        pattern = DailyPattern.from_simple_sine(
            amplitude=0.5, frequency=1.0, phase=6.0, baseline=0.5
        )
        p_scaled = np.ones(48) * 10.0

        noise = add_ar_noise(
            p_scaled=p_scaled,
            col=s,
            gap_start=200,
            gap_stop=248,
            pattern=pattern,
            scaling_params=(5.0, 15.0),
            ar_order=2,
            blend_n=4,
            seed=42,
        )

        assert isinstance(noise, np.ndarray)
        assert len(noise) == 48

    def test_noise_tapered_at_edges(self):
        """Noise is tapered to zero at boundaries."""
        # Create series with noise to ensure non-zero residuals
        idx = pd.date_range("2024-01-01", periods=7 * 96, freq="15min")
        hours = idx.hour + idx.minute / 60.0
        rng = np.random.default_rng(42)
        values = 10.0 + 5.0 * np.sin(2 * np.pi * hours / 24) + 0.5 * rng.standard_normal(len(hours))
        s = pd.Series(values, index=idx, name="test")
        s.iloc[200:248] = np.nan  # 12-hour gap

        pattern = DailyPattern.from_simple_sine(
            amplitude=0.5, frequency=1.0, phase=6.0, baseline=0.5
        )
        p_scaled = np.ones(48) * 10.0

        noise = add_ar_noise(
            p_scaled=p_scaled,
            col=s,
            gap_start=200,
            gap_stop=248,
            pattern=pattern,
            scaling_params=(5.0, 15.0),
            ar_order=2,
            blend_n=4,
            seed=42,
        )

        # Edges should be zero (taper = 0 at boundaries)
        assert abs(noise[0]) < 1e-10
        assert abs(noise[-1]) < 1e-10
        # Interior should have non-zero noise
        assert np.abs(noise[4:-4]).max() > 0.1

    def test_insufficient_data_returns_zeros(self):
        """Insufficient clean data returns zero noise."""
        idx = pd.date_range("2024-01-01", periods=10, freq="15min")
        s = pd.Series(np.ones(10), index=idx, name="test")
        s.iloc[2:8] = np.nan

        pattern = DailyPattern.from_simple_sine(
            amplitude=0.5, frequency=1.0, phase=6.0, baseline=0.5
        )
        p_scaled = np.ones(6) * 10.0

        noise = add_ar_noise(
            p_scaled=p_scaled,
            col=s,
            gap_start=2,
            gap_stop=8,
            pattern=pattern,
            scaling_params=(0.0, 1.0),
            ar_order=4,  # Higher than available data
            blend_n=1,
            seed=42,
        )

        np.testing.assert_allclose(noise, np.zeros(6))

    def test_reproducible_with_seed(self):
        """Same seed produces same noise."""
        s = self._make_test_series()
        pattern = DailyPattern.from_simple_sine(
            amplitude=0.5, frequency=1.0, phase=6.0, baseline=0.5
        )
        p_scaled = np.ones(48) * 10.0

        noise1 = add_ar_noise(
            p_scaled=p_scaled,
            col=s,
            gap_start=200,
            gap_stop=248,
            pattern=pattern,
            scaling_params=(5.0, 15.0),
            ar_order=2,
            blend_n=4,
            seed=42,
        )

        noise2 = add_ar_noise(
            p_scaled=p_scaled,
            col=s,
            gap_start=200,
            gap_stop=248,
            pattern=pattern,
            scaling_params=(5.0, 15.0),
            ar_order=2,
            blend_n=4,
            seed=42,
        )

        np.testing.assert_allclose(noise1, noise2)

    def test_different_ar_orders_produce_different_noise(self):
        """Different AR orders produce different noise."""
        # Create series with more variability
        idx = pd.date_range("2024-01-01", periods=7 * 96, freq="15min")
        hours = idx.hour + idx.minute / 60.0
        rng = np.random.default_rng(42)
        values = 10.0 + 5.0 * np.sin(2 * np.pi * hours / 24) + 0.5 * rng.standard_normal(len(hours))
        s = pd.Series(values, index=idx, name="test")
        s.iloc[200:248] = np.nan  # 12-hour gap

        pattern = DailyPattern.from_simple_sine(
            amplitude=0.5, frequency=1.0, phase=6.0, baseline=0.5
        )
        p_scaled = np.ones(48) * 10.0

        noise1 = add_ar_noise(
            p_scaled=p_scaled,
            col=s,
            gap_start=200,
            gap_stop=248,
            pattern=pattern,
            scaling_params=(5.0, 15.0),
            ar_order=1,
            blend_n=4,
            seed=42,
        )

        noise2 = add_ar_noise(
            p_scaled=p_scaled,
            col=s,
            gap_start=200,
            gap_stop=248,
            pattern=pattern,
            scaling_params=(5.0, 15.0),
            ar_order=4,
            blend_n=4,
            seed=42,
        )

        assert not np.allclose(noise1, noise2)

    def test_zero_range_scaling(self):
        """Handles zero data range gracefully."""
        s = self._make_test_series()
        pattern = DailyPattern.from_simple_sine(
            amplitude=0.5, frequency=1.0, phase=6.0, baseline=0.5
        )
        p_scaled = np.ones(48) * 10.0

        noise = add_ar_noise(
            p_scaled=p_scaled,
            col=s,
            gap_start=200,
            gap_stop=248,
            pattern=pattern,
            scaling_params=(10.0, 10.0),  # Zero range
            ar_order=2,
            blend_n=4,
            seed=42,
        )

        assert len(noise) == 48

    def test_short_gap(self):
        """Handles short gaps correctly."""
        s = self._make_test_series()
        pattern = DailyPattern.from_simple_sine(
            amplitude=0.5, frequency=1.0, phase=6.0, baseline=0.5
        )
        p_scaled = np.ones(4) * 10.0

        noise = add_ar_noise(
            p_scaled=p_scaled,
            col=s,
            gap_start=200,
            gap_stop=204,
            pattern=pattern,
            scaling_params=(5.0, 15.0),
            ar_order=2,
            blend_n=4,
            seed=42,
        )

        assert len(noise) == 4

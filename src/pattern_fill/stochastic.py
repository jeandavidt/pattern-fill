"""Stochastic process utilities for AR noise generation.

This module provides functions for fitting autoregressive (AR) models to
residuals and generating AR noise for gap-filling applications.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pattern_fill.pattern import DailyPattern


def fit_ar_model(residuals: np.ndarray, order: int) -> np.ndarray:
    """Fit AR(p) model using Yule-Walker equations.

    Parameters
    ----------
    residuals : np.ndarray
        Residuals (clean data minus pattern prediction)
    order : int
        AR order (1-5)

    Returns
    -------
    np.ndarray
        AR coefficients [phi_1, phi_2, ..., phi_p]

    Raises
    ------
    ValueError
        If insufficient data for the specified order

    Notes
    -----
    Uses the Yule-Walker equations to solve for AR coefficients:
    R * phi = r

    where R is the Toeplitz autocorrelation matrix and r is the
    autocorrelation vector.
    """
    n = len(residuals)
    if n < order + 1:
        raise ValueError(
            f"Need at least {order + 1} points for AR({order}), got {n}"
        )

    # Center residuals
    r_centered = residuals - residuals.mean()

    # Compute autocorrelation for lags 0 to order
    # r(k) = (1/n) * sum(t=0 to n-k-1) of x[t] * x[t+k]
    autocorr = np.array([
        np.sum(r_centered[:n - k] * r_centered[k:]) / n
        for k in range(order + 1)
    ])

    # Build Toeplitz matrix R from autocorrelations
    # R[i,j] = r(|i-j|)
    R = np.array([
        [autocorr[abs(i - j)] for j in range(order)]
        for i in range(order)
    ])

    # Build vector r (autocorrelations 1 to order)
    r_vec = autocorr[1:order + 1]

    # Check for singular matrix (e.g., constant residuals)
    # Add small regularization if needed
    try:
        phi = np.linalg.solve(R, r_vec)
    except np.linalg.LinAlgError:
        # Add small regularization to handle singular matrix
        R_reg = R + np.eye(order) * 1e-10
        phi = np.linalg.solve(R_reg, r_vec)

    return phi


def generate_ar_noise(
    n_points: int,
    ar_coefficients: np.ndarray,
    noise_std: float,
    seed: int | None = None,
) -> np.ndarray:
    """Generate AR(p) noise realization.

    Parameters
    ----------
    n_points : int
        Length of noise to generate
    ar_coefficients : np.ndarray
        AR coefficients from fit_ar_model()
    noise_std : float
        Standard deviation of white noise
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        AR noise realization of length n_points

    Notes
    -----
    The AR process is generated as:
    x[t] = sum(phi[i] * x[t-i-1]) + epsilon[t]

    where epsilon[t] ~ N(0, noise_std^2).

    A warm-up period of 2*order is used to reach stationarity.
    """
    order = len(ar_coefficients)
    rng = np.random.default_rng(seed)

    # Warm-up period to reach stationarity
    warmup = 2 * order
    total_len = n_points + warmup

    # Generate white noise
    epsilon = rng.normal(0, noise_std, total_len)

    # Generate AR process
    x = np.zeros(total_len)
    for t in range(order, total_len):
        # x[t] = phi[0]*x[t-1] + phi[1]*x[t-2] + ... + phi[p-1]*x[t-p] + epsilon[t]
        x[t] = np.sum(ar_coefficients * x[t - order:t][::-1]) + epsilon[t]

    # Return only the non-warmup portion
    return x[warmup:]


def add_ar_noise(
    p_scaled: np.ndarray,
    col: pd.Series,
    gap_start: int,
    gap_stop: int,
    pattern: DailyPattern,
    scaling_params: tuple[float, float],
    ar_order: int,
    blend_n: int,
    seed: int | None = None,
    ar_coefficients: np.ndarray | None = None,
    noise_std: float | None = None,
) -> np.ndarray:
    """Add AR noise to scaled pattern values.

    This function computes residuals between clean data and pattern prediction
    near a gap, fits an AR model to those residuals, generates AR noise for
    the gap length, and applies a cosine taper so the noise goes to zero at
    the boundaries.

    Parameters
    ----------
    p_scaled : np.ndarray
        Scaled pattern values for the gap
    col : pd.Series
        Original series with gaps
    gap_start : int
        Gap start index
    gap_stop : int
        Gap end index
    pattern : DailyPattern
        Pattern object used for prediction
    scaling_params : tuple[float, float]
        (data_min, data_max) for pattern scaling
    ar_order : int
        AR order (1-5)
    blend_n : int
        Blend window size (in samples)
    seed : int, optional
        Random seed for reproducibility
    ar_coefficients : np.ndarray, optional
        Pre-fitted AR coefficients. If provided, these will be used instead
        of fitting a new model to local residuals.
    noise_std : float, optional
        Pre-computed noise standard deviation. If provided, this will be used
        instead of computing from local residuals.

    Returns
    -------
    np.ndarray
        Noise array to add to p_scaled (same length as p_scaled)
    """
    data_min, data_max = scaling_params
    dr = data_max - data_min if abs(data_max - data_min) > 1e-12 else 1.0

    # Use pre-fitted AR coefficients if provided, otherwise fit locally
    if ar_coefficients is None or noise_std is None:
        # Get clean data near gap (blend_n samples on each side)
        left_start = max(0, gap_start - blend_n)
        right_end = min(len(col), gap_stop + blend_n)

        # Extract clean data (excluding the gap itself)
        clean_mask = np.ones(len(col), dtype=bool)
        clean_mask[gap_start:gap_stop] = False
        clean_idx = np.where(clean_mask)[0]

        # Get clean data near gap
        nearby_mask = (clean_idx >= left_start) & (clean_idx < right_end)
        nearby_idx = clean_idx[nearby_mask]

        # If insufficient data, return zero noise
        if len(nearby_idx) < ar_order + 1:
            return np.zeros_like(p_scaled)

        # Compute pattern prediction on clean data
        nearby_times = col.index[nearby_idx]
        frac_hours = (
            nearby_times.hour
            + nearby_times.minute / 60.0
            + nearby_times.second / 3600.0
        )
        pat_pred = pattern.evaluate(frac_hours.values) * dr + data_min

        # Compute residuals
        clean_values = col.iloc[nearby_idx].values
        residuals = clean_values - pat_pred

        # Fit AR model
        if ar_coefficients is None:
            ar_coefficients = fit_ar_model(residuals, ar_order)

        # Estimate noise_std from residuals
        if noise_std is None:
            noise_std = np.std(residuals)

    # Generate AR noise for gap length
    gap_noise = generate_ar_noise(
        len(p_scaled), ar_coefficients, noise_std, seed
    )

    # Apply cosine taper at boundaries
    # Taper goes from 0 at edges to 1 at blend_n samples inward
    # This ensures noise is zero at boundaries for continuity, but present in the gap interior
    positions = np.arange(len(gap_noise), dtype=float)
    edge_dist = np.minimum(
        positions, float(len(gap_noise) - 1) - positions
    )
    t = np.clip(edge_dist / max(blend_n, 1), 0.0, 1.0)
    # Use sine taper: 0 at edges, 1 at blend_n samples inward
    taper = np.sin(np.pi * t / 2.0)

    return gap_noise * taper

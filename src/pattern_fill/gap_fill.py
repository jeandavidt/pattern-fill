from __future__ import annotations

import datetime
from typing import Any

import numpy as np
import pandas as pd
from meteaudata.types import (
    FunctionInfo,
    Parameters,
    ProcessingStep,
    ProcessingType,
    Signal,
    TimeSeries,
)

from pattern_fill.fitting import extract_daily_profile
from pattern_fill.pattern import DailyPattern


def find_nan_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, stop) index pairs for contiguous True runs in *mask*."""
    if not mask.any():
        return []
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, len(mask)]
    return list(zip(starts.tolist(), ends.tolist()))


def classify_runs(
    runs: list[tuple[int, int]], series_len: int
) -> list[tuple[int, int]]:
    """Return only interior NaN runs, excluding leading and trailing."""
    return [
        (start, stop)
        for start, stop in runs
        if start != 0 and stop != series_len
    ]


def select_pattern(
    pattern: DailyPattern | dict[str, DailyPattern],
    timestamp: pd.Timestamp,
) -> DailyPattern:
    """Pick the right pattern for a timestamp based on day_type."""
    if isinstance(pattern, DailyPattern):
        return pattern
    dow = timestamp.dayofweek  # 0=Monday … 6=Sunday
    key = "weekday" if dow < 5 else "weekend"
    if key in pattern:
        return pattern[key]
    if "all" in pattern:
        return pattern["all"]
    raise KeyError(
        f"No pattern for day_type={key!r} or 'all' in pattern dict "
        f"(available keys: {list(pattern.keys())})"
    )


def infer_freq_minutes(index: pd.DatetimeIndex) -> float:
    """Infer the sampling frequency in minutes from a DatetimeIndex."""
    freq = pd.infer_freq(index)
    if freq is not None:
        offset = pd.tseries.frequencies.to_offset(freq)
        return offset.nanos / 1e9 / 60
    # Fallback: median of first diffs
    n = min(10, len(index))
    diffs = pd.Series(index[:n]).diff().dropna()
    if len(diffs) == 0:
        return 15.0
    return diffs.median().total_seconds() / 60


def smoothed_anchor(
    col: pd.Series,
    gap_edge_idx: int,
    side: str,
    blend_n: int,
) -> float | None:
    """Compute a noise-resistant anchor value near a gap edge.

    Returns the weighted average of up to *blend_n* non-NaN points
    adjacent to the gap.  Weights increase linearly toward the gap edge
    (the closest point gets the highest weight).
    """
    if side == "left":
        start = max(0, gap_edge_idx - blend_n + 1)
        window = col.iloc[start : gap_edge_idx + 1]
    else:
        end = min(len(col), gap_edge_idx + blend_n)
        window = col.iloc[gap_edge_idx:end]

    valid = window.dropna()
    if len(valid) == 0:
        return None

    k = len(valid)
    if side == "left":
        weights = np.arange(1, k + 1, dtype=float)
    else:
        weights = np.arange(k, 0, -1, dtype=float)

    return float(np.average(valid.values, weights=weights))


def collect_window_idx(
    clean_mask: np.ndarray,
    col_index: pd.DatetimeIndex,
    before_time: pd.Timestamp,
    window_days: float | None,
    min_count: int = 2,
) -> np.ndarray:
    """Indices of clean samples in the backward window before before_time.

    If window_days is None, returns ALL clean samples before before_time.
    If window_days is set, starts with [before_time - window_days, before_time);
    expands further back if fewer than min_count clean samples are found.
    """
    if window_days is not None:
        window_start = before_time - pd.Timedelta(days=window_days)
        idx = np.where(
            (col_index >= window_start)
            & (col_index < before_time)
            & clean_mask
        )[0]
        if len(idx) < min_count:
            idx = np.where((col_index < before_time) & clean_mask)[0]
    else:
        idx = np.where(clean_mask)[0]
    return idx


def _blend_fill(
    p_scaled: np.ndarray,
    left_anchor: float | None,
    right_anchor: float | None,
    blend_n: int,
) -> np.ndarray:
    """Apply cosine-decay boundary corrections to pre-scaled pattern values.

    When both boundaries are available, linearly interpolates between the
    anchors and blends with the scaled pattern using a cosine edge-weight
    (1 at each edge, 0 at ``blend_n`` steps inward).  This guarantees exact
    boundary continuity even when blend zones overlap in short gaps.

    For single-boundary cases an additive correction is used instead.
    """
    N = len(p_scaled)
    positions = np.arange(N, dtype=float)

    if left_anchor is not None and right_anchor is not None:
        # Linearly interpolate between anchors
        alpha = positions / max(N - 1, 1)
        anchor_interp = (1.0 - alpha) * left_anchor + alpha * right_anchor

        # Cosine edge weight: 1 at the edges, 0 beyond blend_n from edge
        edge_dist = np.minimum(positions, float(N - 1) - positions)
        t = np.clip(edge_dist / max(blend_n, 1), 0.0, 1.0)
        w = 0.5 * (1.0 + np.cos(np.pi * t))

        return w * anchor_interp + (1.0 - w) * p_scaled

    # Single-boundary: additive correction that decays inward
    result = p_scaled.copy()

    if left_anchor is not None:
        r_L = left_anchor - p_scaled[0]
        t_L = np.clip(positions / max(blend_n, 1), 0.0, 1.0)
        w_L = 0.5 * (1.0 + np.cos(np.pi * t_L))
        result += w_L * r_L

    if right_anchor is not None:
        r_R = right_anchor - p_scaled[-1]
        dist_from_right = float(N - 1) - positions
        t_R = np.clip(dist_from_right / max(blend_n, 1), 0.0, 1.0)
        w_R = 0.5 * (1.0 + np.cos(np.pi * t_R))
        result += w_R * r_R

    return result


def _compute_expected_area(
    col: pd.Series,
    gap_idx: pd.DatetimeIndex,
    pattern: DailyPattern | dict[str, DailyPattern],
) -> float | None:
    """Expected sum of values across a gap, based on the daily profile of clean data.

    Splits clean data by day type when *pattern* is a dict, so the expected
    area respects weekday/weekend differences.
    """
    clean = col.dropna()
    if len(clean) < 10:
        return None

    uses_day_types = isinstance(pattern, dict)

    if uses_day_types:
        profiles: dict[str, pd.Series] = {}
        for dtype in ("weekday", "weekend"):
            mask = (
                clean.index.dayofweek < 5
                if dtype == "weekday"
                else clean.index.dayofweek >= 5
            )
            subset = clean[mask]
            profiles[dtype] = extract_daily_profile(
                subset if len(subset) > 10 else clean, aggregation="mean"
            )
    else:
        profile_all = extract_daily_profile(clean, aggregation="mean")

    expected = np.empty(len(gap_idx))
    for i, ts in enumerate(gap_idx):
        frac_h = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
        if uses_day_types:
            dtype = "weekday" if ts.dayofweek < 5 else "weekend"
            profile = profiles[dtype]
        else:
            profile = profile_all
        expected[i] = np.interp(frac_h, profile.index.values, profile.values)

    total = expected.sum()
    return total if abs(total) > 1e-12 else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pattern_fill(
    input_series: list[pd.Series],
    pattern: DailyPattern | dict[str, DailyPattern],
    pattern_window_days: float | None = None,
    blend_minutes: int = 60,
    normalize_area: bool = False,
    add_noise: bool = False,
    ar_order: int = 4,
    ar_window_days: float | None = None,
    random_seed: int | None = None,
    *args: Any,
    **kwargs: Any,
) -> list[tuple[pd.Series, list[ProcessingStep]]]:
    """Fill NaN gaps using a daily diurnal pattern.

    Conforms to metEAUdata's ``SignalTransformFunctionProtocol``.

    Parameters
    ----------
    pattern_window_days : float, optional
        When set, pattern scaling uses only clean data from this many days
        before each gap start.  If fewer than 2 clean samples exist in the
        window, all available prior clean data is used.  When None, all clean
        data in the series is used for scaling (global default).
    blend_minutes : int
        Width (in minutes) of the smoothing window used for anchor
        computation and the cosine blend zone inside each gap.
    normalize_area : bool
        When True, the fill's area is normalized to match the expected daily
        profile computed from the clean portions of the series.
    add_noise : bool
        When True, adds AR(p) noise to the filled values to better represent
        variability.  The noise is fitted to residuals between the pattern
        and clean data near each gap.
    ar_order : int
        Autoregressive order for noise generation (1-5).  Only used when
        add_noise=True.
    ar_window_days : float, optional
        When set, the AR model is fitted on a rolling window of available
        (non-NaN) clean data of up to this many days immediately before each
        gap.  If fewer clean samples exist than needed, all available prior
        clean data is used.  Only used when add_noise=True.
    random_seed : int, optional
        Random seed for reproducible noise generation.  Only used when
        add_noise=True.
    """
    if isinstance(pattern, DailyPattern):
        pattern_meta = pattern.to_dict()
    else:
        pattern_meta = {k: v.to_dict() for k, v in pattern.items()}

    func_info = FunctionInfo(
        name="pattern_fill",
        version="0.2.0",
        author="pattern-fill",
        reference="https://github.com/jeandavidt/pattern-fill",
    )
    parameters = Parameters(
        pattern=pattern_meta,
        pattern_window_days=pattern_window_days,
        blend_minutes=blend_minutes,
        normalize_area=normalize_area,
        add_noise=add_noise,
        ar_order=ar_order,
        ar_window_days=ar_window_days,
    )
    processing_step = ProcessingStep(
        type=ProcessingType.GAP_FILLING,
        function_info=func_info,
        parameters=parameters,
        description=(
            "Gap-filling using a daily diurnal pattern with "
            "cosine-blended boundary matching"
        ),
        run_datetime=datetime.datetime.now(),
        requires_calibration=False,
        input_series_names=[str(col.name) for col in input_series],
        suffix="PAT-FILL",
    )

    outputs: list[tuple[pd.Series, list[ProcessingStep]]] = []
    for col in input_series:
        col = col.copy()
        signal_name, _, _ = Signal.extract_ts_base_and_number(str(col.name))

        if not isinstance(col.index, pd.DatetimeIndex):
            raise TypeError(
                f"Series {col.name} must have a DatetimeIndex, "
                f"got {type(col.index)}"
            )

        freq_min = infer_freq_minutes(col.index)
        blend_n = max(1, int(round(blend_minutes / freq_min)))

        # Original clean mask (before any filling)
        original_clean_mask = ~col.isna().values
        original_col = col.copy()  # keep original for AR residuals

        nan_mask = col.isna().values
        runs = find_nan_runs(nan_mask)
        interior_runs = classify_runs(runs, len(col))

        # Store scaling params for each gap (needed for AR noise in Pass 2)
        gap_scaling_params: dict[tuple[int, int], tuple[float, float, float]] = {}

        # Pass 1: pattern fill
        for start, stop in interior_runs:
            gap_idx = col.index[start:stop]
            gap_start_time = col.index[start]

            # 1. Collect backward window of clean data
            win_idx = collect_window_idx(
                original_clean_mask, col.index, gap_start_time,
                pattern_window_days, min_count=2
            )
            win_values = (
                col.iloc[win_idx].values if len(win_idx) >= 2
                else col[original_clean_mask].values
            )
            if len(win_values) < 2:
                data_min, data_max = 0.0, 1.0
            else:
                data_min = float(win_values.min())
                data_max = float(win_values.max())
            dr = data_max - data_min if abs(data_max - data_min) > 1e-12 else 1.0

            # 2. Evaluate pattern over gap
            pat_vals = np.array([
                select_pattern(pattern, ts).evaluate(
                    np.array([ts.hour + ts.minute / 60.0 + ts.second / 3600.0])
                )[0]
                for ts in gap_idx
            ])
            p_scaled = pat_vals * dr + data_min

            # Store scaling params for AR noise (Pass 2)
            gap_scaling_params[(start, stop)] = (data_min, data_max, dr)

            # 3. Optional area normalization
            if normalize_area:
                expected = _compute_expected_area(col, gap_idx, pattern)
                actual = p_scaled.sum()
                if expected is not None and abs(actual) > 1e-12:
                    p_scaled *= expected / actual

            # 4. Cosine blend at boundaries
            left_anchor = (
                smoothed_anchor(col, start - 1, "left", blend_n)
                if start > 0
                else None
            )
            right_anchor = (
                smoothed_anchor(col, stop, "right", blend_n)
                if stop < len(col)
                else None
            )
            filled = _blend_fill(p_scaled, left_anchor, right_anchor, blend_n)
            col.iloc[start:stop] = filled

        # Pass 2: AR noise
        if add_noise:
            from pattern_fill.stochastic import fit_ar_model, generate_ar_noise

            for start, stop in interior_runs:
                gap_start_time = col.index[start]

                # Get the scaling params used for this gap in Pass 1
                # These are needed to scale the noise to match the fill
                scaling_params = gap_scaling_params.get((start, stop))
                if scaling_params is None:
                    continue  # shouldn't happen, but be safe
                fill_min, fill_max, fill_dr = scaling_params

                # 1. Collect backward window of clean residuals
                win_idx = collect_window_idx(
                    original_clean_mask, col.index, gap_start_time,
                    ar_window_days, min_count=ar_order + 1
                )

                if len(win_idx) < ar_order + 1:
                    continue  # not enough data — skip noise for this gap

                # 2. Compute residuals = actual - pattern_prediction
                # Use LOCAL scaling from the AR window to capture true local noise
                win_times = col.index[win_idx]
                win_data = original_col.iloc[win_idx].values
                win_hours = (
                    win_times.hour + win_times.minute / 60.0 + win_times.second / 3600.0
                ).values
                
                # Compute local scaling for the AR window
                local_min = float(win_data.min())
                local_max = float(win_data.max())
                local_dr = local_max - local_min if abs(local_max - local_min) > 1e-12 else 1.0
                
                # Compute pattern prediction with LOCAL scaling
                win_pred_local = np.array([
                    select_pattern(pattern, ts).evaluate(np.array([h]))[0]
                    for ts, h in zip(win_times, win_hours)
                ]) * local_dr + local_min
                
                # Residuals in local scale
                win_residuals = win_data - win_pred_local
                
                # 3. Fit AR model to local residuals
                ar_coeffs = fit_ar_model(win_residuals, ar_order)
                local_noise_std = float(np.std(win_residuals))
                
                # Scale noise std from local to fill scale
                # This ensures noise magnitude matches the filled values
                noise_std = local_noise_std * (fill_dr / local_dr)
                
                gap_noise = generate_ar_noise(
                    stop - start, ar_coeffs, noise_std, random_seed
                )

                # 4. Apply sine taper (zero at boundaries)
                n = stop - start
                positions = np.arange(n, dtype=float)
                edge_dist = np.minimum(positions, float(n - 1) - positions)
                taper = np.sin(np.pi * np.clip(edge_dist / max(blend_n, 1), 0, 1) / 2)
                col.iloc[start:stop] += gap_noise * taper

        col.name = f"{signal_name}_{processing_step.suffix}"
        outputs.append((col, [processing_step]))

    return outputs


def pattern_fill_dataset(
    input_signals: list[Signal],
    input_series_names: list[str],
    patterns: list[DailyPattern | dict[str, DailyPattern]],
    mode: str = "load",
    blend_minutes: int = 60,
    pattern_window_days: float | None = None,
    *args: Any,
    **kwargs: Any,
) -> list[Signal]:
    """Fill NaN gaps with area normalization at the dataset level.

    Conforms to metEAUdata's ``DatasetTransformFunctionProtocol``.

    Parameters
    ----------
    mode : str
        ``"concentration"`` or ``"flow"`` — single signal, area normalized to
        its own daily profile.  ``"load"`` — two signals (concentration first,
        flow second), both filled, then concentration normalized so that
        ``conc × flow`` matches the expected daily load.
    pattern_window_days : float, optional
        When set, pattern scaling uses only clean data from this many days
        before each gap start.  Passed through to ``pattern_fill()``.
    """
    valid_modes = ("concentration", "flow", "load")
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got {mode!r}")

    if mode == "load":
        if len(input_signals) != 2 or len(patterns) != 2:
            raise ValueError(
                "load mode requires exactly 2 signals and 2 patterns "
                f"(concentration, flow); got {len(input_signals)} signals "
                f"and {len(patterns)} patterns"
            )
    else:
        if len(input_signals) != 1 or len(patterns) != 1:
            raise ValueError(
                f"{mode} mode requires exactly 1 signal and 1 pattern; "
                f"got {len(input_signals)} signals and {len(patterns)} patterns"
            )

    func_info = FunctionInfo(
        name="pattern_fill_dataset",
        version="0.2.0",
        author="pattern-fill",
        reference="https://github.com/jeandavidt/pattern-fill",
    )

    # ---- concentration / flow modes (single signal, area-normalized) --------
    if mode in ("concentration", "flow"):
        signal = input_signals[0]
        ts_name = input_series_names[0]
        series = signal.time_series[ts_name].series
        pat = patterns[0]

        results = pattern_fill(
            [series],
            pattern=pat,
            pattern_window_days=pattern_window_days,
            blend_minutes=blend_minutes,
            normalize_area=True,
        )
        filled_series, steps = results[0]

        ts_obj = TimeSeries(series=filled_series, processing_steps=steps)
        out_signal = Signal(
            input_data=ts_obj,
            name=Signal.extract_ts_base_and_number(str(filled_series.name))[0],
            provenance=signal.provenance,
            units=signal.units,
        )
        return [out_signal]

    # ---- load mode (two signals: concentration + flow) ----------------------
    conc_signal, flow_signal = input_signals
    conc_ts_name, flow_ts_name = input_series_names
    conc_series = conc_signal.time_series[conc_ts_name].series
    flow_series = flow_signal.time_series[flow_ts_name].series
    conc_pat, flow_pat = patterns

    # Step 1: fill both signals (without area normalization)
    conc_results = pattern_fill(
        [conc_series],
        pattern=conc_pat,
        pattern_window_days=pattern_window_days,
        blend_minutes=blend_minutes,
        normalize_area=False,
    )
    flow_results = pattern_fill(
        [flow_series],
        pattern=flow_pat,
        pattern_window_days=pattern_window_days,
        blend_minutes=blend_minutes,
        normalize_area=False,
    )
    filled_conc, conc_steps = conc_results[0]
    filled_flow, flow_steps = flow_results[0]

    # Step 2: compute daily load profile from clean data
    conc_clean = conc_series.dropna()
    flow_clean = flow_series.dropna()
    common_idx = conc_clean.index.intersection(flow_clean.index)

    if len(common_idx) > 20:
        load_clean = pd.Series(
            conc_clean.loc[common_idx].values * flow_clean.loc[common_idx].values,
            index=common_idx,
            name="load",
        )
        uses_day_types = isinstance(conc_pat, dict)

        if uses_day_types:
            load_profiles: dict[str, pd.Series] = {}
            for dtype in ("weekday", "weekend"):
                mask = (
                    load_clean.index.dayofweek < 5
                    if dtype == "weekday"
                    else load_clean.index.dayofweek >= 5
                )
                subset = load_clean[mask]
                load_profiles[dtype] = extract_daily_profile(
                    subset if len(subset) > 10 else load_clean,
                    aggregation="mean",
                )
        else:
            load_profile_all = extract_daily_profile(
                load_clean, aggregation="mean"
            )

        # Step 3: normalize concentration fills so load matches expected
        nan_mask = conc_series.isna().values
        runs = find_nan_runs(nan_mask)
        interior_runs = classify_runs(runs, len(conc_series))

        for gap_start, gap_stop in interior_runs:
            gap_idx = conc_series.index[gap_start:gap_stop]

            expected_load = np.empty(len(gap_idx))
            for i, ts in enumerate(gap_idx):
                frac_h = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
                if uses_day_types:
                    dtype = "weekday" if ts.dayofweek < 5 else "weekend"
                    lp = load_profiles[dtype]
                else:
                    lp = load_profile_all
                expected_load[i] = np.interp(
                    frac_h, lp.index.values, lp.values
                )

            expected_sum = expected_load.sum()

            gap_conc = filled_conc.iloc[gap_start:gap_stop].values
            gap_flow = filled_flow.iloc[gap_start:gap_stop].values
            actual_sum = (gap_conc * gap_flow).sum()

            if abs(actual_sum) > 1e-12 and abs(expected_sum) > 1e-12:
                ratio = expected_sum / actual_sum
                filled_conc.iloc[gap_start:gap_stop] = gap_conc * ratio

    # Build processing step for load normalization
    load_step = ProcessingStep(
        type=ProcessingType.GAP_FILLING,
        function_info=func_info,
        parameters=Parameters(
            mode=mode,
            pattern_window_days=pattern_window_days,
            blend_minutes=blend_minutes,
        ),
        description=(
            "Gap-filling with load-normalized daily pattern "
            "(concentration adjusted so conc × flow matches expected load)"
        ),
        run_datetime=datetime.datetime.now(),
        requires_calibration=False,
        input_series_names=input_series_names,
        suffix="PAT-FILL",
    )

    conc_ts = TimeSeries(
        series=filled_conc,
        processing_steps=conc_steps + [load_step],
    )
    flow_ts = TimeSeries(
        series=filled_flow,
        processing_steps=flow_steps,
    )

    out_conc = Signal(
        input_data=conc_ts,
        name=Signal.extract_ts_base_and_number(str(filled_conc.name))[0],
        provenance=conc_signal.provenance,
        units=conc_signal.units,
    )
    out_flow = Signal(
        input_data=flow_ts,
        name=Signal.extract_ts_base_and_number(str(filled_flow.name))[0],
        provenance=flow_signal.provenance,
        units=flow_signal.units,
    )
    return [out_conc, out_flow]

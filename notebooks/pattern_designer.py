import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from wigglystuff import ChartPuck
    from pattern_fill import (
        DailyPattern,
        SineComponent,
        fit_pattern,
        fit_sine_pattern,
        pattern_fill,
    )
    import io
    import json

    return (
        ChartPuck,
        DailyPattern,
        SineComponent,
        fit_pattern,
        fit_sine_pattern,
        io,
        json,
        mo,
        np,
        pattern_fill,
        pd,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Pattern Designer

    Design daily diurnal patterns for gap-filling time series data.
    Upload your own data or use the built-in demo, then fit a pattern
    automatically or shape one by hand.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 🎨 Pattern Building Mode

    Choose how you want to create your daily pattern:
    - **Spline**: Drag control points (pucks) to shape a smooth curve
    - **Sine Waves**: Combine multiple sine waves with adjustable amplitude, frequency, and phase
    """)
    return


@app.cell
def _(mo):
    pattern_mode = mo.ui.radio(
        options=["Spline", "Sine Waves"],
        value="Spline",
        label="Pattern type",
    )
    pattern_mode
    return (pattern_mode,)


@app.cell
def _(np, pd):
    def generate_demo_data():
        rng = np.random.default_rng(42)
        index = pd.date_range("2024-01-01", periods=14 * 96, freq="15min")
        hours = index.hour + index.minute / 60.0
        base = 0.6 * np.sin(2 * np.pi * (hours - 6) / 24) + 0.2 * np.sin(
            4 * np.pi * (hours - 3) / 24
        )
        noise = 0.08 * rng.standard_normal(len(index))
        raw = base + noise
        values = 30 + 15 * (raw - raw.min()) / (raw.max() - raw.min())
        series = pd.Series(values, index=index, name="NH4")
        series.iloc[200:248] = np.nan
        series.iloc[500:524] = np.nan
        series.iloc[900:1000] = np.nan
        return series.to_frame()

    return (generate_demo_data,)


@app.cell
def _(mo):
    data_source = mo.ui.radio(
        options=["demo", "upload"],
        value="demo",
        label="Data source",
    )
    file_upload = mo.ui.file(
        filetypes=[".csv", ".parquet", ".xlsx", ".xls"],
        kind="area",
        label="Upload a CSV, Parquet, or Excel file",
    )
    data_source
    return data_source, file_upload


@app.cell
def _(data_source, file_upload, mo):
    mo.stop(data_source.value != "upload")
    file_upload
    return


@app.cell
def _(data_source, file_upload, generate_demo_data, io, mo, pd):
    # Debug: Check what data_source.value actually is
    _debug_msg = f"data_source.value = {repr(data_source.value)}"

    if data_source.value == "demo":
        try:
            df = generate_demo_data()
        except Exception as e:
            mo.stop(
                True,
                mo.callout(mo.md(f"Error generating demo data: {e}"), kind="danger"),
            )
            df = pd.DataFrame()
    else:
        mo.stop(
            not file_upload.value,
            mo.callout(
                mo.md(f"Upload a file to continue. ({_debug_msg})"), kind="warn"
            ),
        )
        _uploaded = file_upload.value[0]
        _name = _uploaded.name.lower()
        _buf = io.BytesIO(_uploaded.contents)
        if _name.endswith(".csv"):
            df = pd.read_csv(_buf)
        elif _name.endswith(".parquet"):
            df = pd.read_parquet(_buf)
        elif _name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(_buf)
        else:
            mo.stop(
                True,
                mo.callout(mo.md("Unsupported file type."), kind="danger"),
            )
            df = pd.DataFrame()
    return (df,)


@app.cell
def _(df, mo, pd):
    _cols = list(df.columns)
    _time_options = {}
    if isinstance(df.index, pd.DatetimeIndex):
        _time_options["(index)"] = "(index)"
    for _c in _cols:
        _time_options[_c] = _c
    _default_time = (
        "(index)" if "(index)" in _time_options else (_cols[0] if _cols else None)
    )
    time_col = mo.ui.dropdown(
        options=_time_options,
        value=_default_time,
        label="Datetime column",
    )
    _num_cols = [c for c in _cols if pd.api.types.is_numeric_dtype(df[c])]
    value_col = mo.ui.dropdown(
        options={c: c for c in _num_cols},
        value=_num_cols[0] if _num_cols else None,
        label="Value column",
    )
    mo.hstack([time_col, value_col], justify="start", gap=1)
    return time_col, value_col


@app.cell
def _(df, mo, value_col):
    mo.md(f"""
    **Debug info:**
    - DataFrame columns: {list(df.columns)}
    - DataFrame shape: {df.shape}
    - value_col.value: {value_col.value}
    - value_col.value is None: {value_col.value is None}
    """)
    return


@app.cell
def _(df, mo, pd, time_col, value_col):
    mo.stop(
        value_col.value is None,
        mo.callout(mo.md("Select a value column."), kind="warn"),
    )
    if time_col.value == "(index)":
        series = df[value_col.value].copy()
        if not isinstance(series.index, pd.DatetimeIndex):
            series.index = pd.to_datetime(series.index)
    else:
        _tmp = df.copy()
        _tmp[time_col.value] = pd.to_datetime(_tmp[time_col.value])
        _tmp = _tmp.set_index(time_col.value)
        series = _tmp[value_col.value]
    series.name = value_col.value
    return (series,)


@app.cell
def _(pd, plt, series):
    _fig, _ax = plt.subplots(figsize=(12, 3))
    _valid = series.dropna()
    _ax.plot(_valid.index, _valid.values, linewidth=0.8, color="steelblue")
    _nan_mask = series.isna()
    if _nan_mask.any():
        _changes = _nan_mask.astype(int).diff().fillna(0)
        _starts = series.index[_changes == 1]
        _ends = series.index[_changes == -1]
        if _nan_mask.iloc[0]:
            _starts = _starts.insert(0, series.index[0])
        if _nan_mask.iloc[-1]:
            _ends = _ends.append(pd.DatetimeIndex([series.index[-1]]))
        for _s, _e in zip(_starts, _ends):
            _ax.axvspan(_s, _e, alpha=0.2, color="red")
    _ax.set_title("Raw series (red = gaps)")
    _ax.set_xlabel("Time")
    _ax.set_ylabel(series.name)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo, pattern_mode):
    """Configure day type and count for pattern editing."""

    # Day type selector: "all days" = single pattern, "weekdays + weekends" = weekday/weekend patterns
    day_type_radio = mo.ui.radio(
        options=["all days", "weekdays + weekends"],
        value="all days",
        label="Day type",
    )

    # Fit observations checkbox - when checked, auto-fit the data
    # Default to False for Sine mode (manual control), True for Spline mode
    _default_fit = False if pattern_mode.value == "Sine Waves" else True
    fit_checkbox = mo.ui.checkbox(
        label="Fit observations",
        value=_default_fit,
    )

    # Normalize checkbox - when checked, normalize the data to 0-1 range
    normalize_checkbox = mo.ui.checkbox(
        label="Normalize data",
        value=True,
    )

    # Blend minutes slider - controls the tapering window size
    blend_minutes_slider = mo.ui.slider(
        start=0,
        stop=240,
        step=15,
        value=60,
        label="Blend window (minutes)",
        show_value=True,
    )

    # Create count sliders based on mode
    n_points_slider = None
    n_components_slider = None

    if pattern_mode.value == "Spline":
        # Number of control points (pucks) slider
        n_points_slider = mo.ui.slider(
            start=4,
            stop=24,
            step=1,
            value=8,
            label="Number of pucks",
            show_value=True,
        )

        _out = mo.hstack(
            [
                day_type_radio,
                fit_checkbox,
                normalize_checkbox,
                blend_minutes_slider,
                n_points_slider,
            ],
            justify="start",
            gap=1.5,
            wrap=True,
        )
    else:  # Sine Waves mode
        # Number of sine wave components
        n_components_slider = mo.ui.slider(
            start=1,
            stop=5,
            step=1,
            value=2,
            label="Number of sine waves",
            show_value=True,
        )

        _out = mo.hstack(
            [
                day_type_radio,
                fit_checkbox,
                normalize_checkbox,
                blend_minutes_slider,
                n_components_slider,
            ],
            justify="start",
            gap=1.5,
            wrap=True,
        )

    _out
    return (
        blend_minutes_slider,
        day_type_radio,
        fit_checkbox,
        n_components_slider,
        n_points_slider,
        normalize_checkbox,
    )


@app.cell
def _(day_type_radio, mo, n_components_slider, pattern_mode):
    """Create sliders for sine wave parameters - weekday and optional weekend."""
    mo.stop(pattern_mode.value != "Sine Waves" or n_components_slider is None)

    _n = n_components_slider.value

    sine_amplitude_sliders = mo.ui.array(
        [
            mo.ui.slider(
                start=0.0,
                stop=0.5,
                step=0.01,
                value=0.3 if i == 0 else 0.15,
                label=f"Wave {i + 1} Amplitude",
                show_value=True,
            )
            for i in range(_n)
        ]
    )

    sine_frequency_sliders = mo.ui.array(
        [
            mo.ui.slider(
                start=0.5,
                stop=4.0,
                step=0.5,
                value=1.0 if i == 0 else 2.0,
                label=f"Wave {i + 1} Frequency",
                show_value=True,
            )
            for i in range(_n)
        ]
    )

    sine_phase_sliders = mo.ui.array(
        [
            mo.ui.slider(
                start=0.0,
                stop=24.0,
                step=0.5,
                value=8.0 if i == 0 else 13.0,
                label=f"Wave {i + 1} Phase (hour)",
                show_value=True,
            )
            for i in range(_n)
        ]
    )

    sine_baseline_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.01,
        value=0.5,
        label="Baseline",
        show_value=True,
    )

    weekend_sine_amplitude_sliders = mo.ui.array(
        [
            mo.ui.slider(
                start=0.0,
                stop=0.5,
                step=0.01,
                value=0.3 if i == 0 else 0.15,
                label=f"Weekend Wave {i + 1} Amplitude",
                show_value=True,
            )
            for i in range(_n)
        ]
    )

    weekend_sine_frequency_sliders = mo.ui.array(
        [
            mo.ui.slider(
                start=0.5,
                stop=4.0,
                step=0.5,
                value=1.0 if i == 0 else 2.0,
                label=f"Weekend Wave {i + 1} Frequency",
                show_value=True,
            )
            for i in range(_n)
        ]
    )

    weekend_sine_phase_sliders = mo.ui.array(
        [
            mo.ui.slider(
                start=0.0,
                stop=24.0,
                step=0.5,
                value=8.0 if i == 0 else 13.0,
                label=f"Weekend Wave {i + 1} Phase (hour)",
                show_value=True,
            )
            for i in range(_n)
        ]
    )

    weekend_sine_baseline_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.01,
        value=0.5,
        label="Weekend Baseline",
        show_value=True,
    )

    _weekday_rows = []
    for i in range(_n):
        _weekday_rows.append(
            mo.hstack(
                [
                    sine_amplitude_sliders[i],
                    sine_frequency_sliders[i],
                    sine_phase_sliders[i],
                ],
                gap=1,
            )
        )

    _weekend_rows = []
    for i in range(_n):
        _weekend_rows.append(
            mo.hstack(
                [
                    weekend_sine_amplitude_sliders[i],
                    weekend_sine_frequency_sliders[i],
                    weekend_sine_phase_sliders[i],
                ],
                gap=1,
            )
        )

    _elements = [
        mo.md(
            """
            ### 🌊 Sine Wave Parameters

            Configure each sine wave component. Amplitude controls strength,
            frequency controls cycles per day (1.0 = daily, 2.0 = twice daily),
            and phase controls the hour of peak value.
            """
        ),
        mo.md("**Weekday Pattern:**"),
        *_weekday_rows,
        sine_baseline_slider,
    ]

    if day_type_radio.value == "weekdays + weekends":
        _elements.extend(
            [
                mo.md("**Weekend Pattern:**"),
                *_weekend_rows,
                weekend_sine_baseline_slider,
            ]
        )

    slider_display = mo.vstack(_elements, gap=1)
    slider_display
    return (
        sine_amplitude_sliders,
        sine_baseline_slider,
        sine_frequency_sliders,
        sine_phase_sliders,
        weekend_sine_amplitude_sliders,
        weekend_sine_baseline_slider,
        weekend_sine_frequency_sliders,
        weekend_sine_phase_sliders,
    )


@app.cell
def _(day_type_radio, np, pd, series):
    def get_profile_stats(s, resolution_minutes=15):
        if s is None or not isinstance(s.index, pd.DatetimeIndex):
            return None

        _s = s.dropna()
        if len(_s) == 0:
            return None

        _frac_hour = _s.index.hour + _s.index.minute / 60.0 + _s.index.second / 3600.0

        # Binning
        _bins = np.arange(0, 24 + resolution_minutes / 60.0, resolution_minutes / 60.0)
        _centers = (_bins[:-1] + _bins[1:]) / 2.0

        # Digitize
        _bin_idx = np.digitize(_frac_hour, _bins) - 1
        _bin_idx = np.clip(_bin_idx, 0, len(_centers) - 1)

        _grouped = pd.Series(_s.values, index=_bin_idx).groupby(level=0)
        _mean = _grouped.mean()
        _std = _grouped.std()

        _stats = pd.DataFrame({"mean": _mean, "std": _std})
        # Map back to hours
        _stats.index = _centers[_stats.index]
        _stats.index.name = "hour"
        return _stats

    profiles = {}
    if day_type_radio.value == "all days":
        _prof = get_profile_stats(series)
        profiles["all days"] = _prof
    else:  # weekdays + weekends
        _wd_mask = series.index.dayofweek < 5
        profiles["weekday"] = get_profile_stats(series[_wd_mask])
        profiles["weekend"] = get_profile_stats(series[~_wd_mask])
    return (profiles,)


@app.cell
def _(day_type_radio, mo, np, pattern_mode, profiles):
    """Pre-compute normalized silhouette data for spline charts."""
    mo.stop(pattern_mode.value != "Spline")

    silhouette_data = {}

    def prepare_silhouette(stats_df):
        """Convert stats to normalized coordinates for fast plotting."""
        if stats_df is None or stats_df.empty:
            return None

        mean = stats_df["mean"].values
        std = stats_df["std"].fillna(0).values
        hours = stats_df.index.values

        # Normalize to 0-1 range
        pmin, pmax = mean.min(), mean.max()
        if pmax - pmin > 0:
            norm_mean = (mean - pmin) / (pmax - pmin)
            norm_std = std / (pmax - pmin)
        else:
            norm_mean = np.full_like(mean, 0.5)
            norm_std = np.zeros_like(std)

        return {
            "hours": hours,
            "mean": norm_mean,
            "std_upper": np.clip(norm_mean + 2 * norm_std, 0, 1),
            "std_lower": np.clip(norm_mean - 2 * norm_std, 0, 1),
        }

    if day_type_radio.value == "all days":
        silhouette_data["all days"] = prepare_silhouette(profiles.get("all days"))
    else:
        silhouette_data["weekday"] = prepare_silhouette(profiles.get("weekday"))
        silhouette_data["weekend"] = prepare_silhouette(profiles.get("weekend"))
    return (silhouette_data,)


@app.cell
def _(
    DailyPattern,
    SineComponent,
    day_type_radio,
    fit_checkbox,
    fit_pattern,
    fit_sine_pattern,
    mo,
    n_components_slider,
    n_points_slider,
    np,
    pattern_mode,
    series,
    sine_amplitude_sliders,
    sine_baseline_slider,
    sine_frequency_sliders,
    sine_phase_sliders,
    weekend_sine_amplitude_sliders,
    weekend_sine_baseline_slider,
    weekend_sine_frequency_sliders,
    weekend_sine_phase_sliders,
):
    fitted_patterns = {}
    _is_fit_mode = fit_checkbox.value

    if pattern_mode.value == "Spline":
        mo.stop(n_points_slider is None)
        _n_points = n_points_slider.value

        if day_type_radio.value == "all days":
            if _is_fit_mode:
                fitted_patterns["all days"] = fit_pattern(
                    series,
                    n_control_points=_n_points,
                    day_type="all",
                )
            else:
                # Initialize with uniform values when not fitting
                _hours = (
                    np.linspace(0, 24 - 24.0 / (_n_points - 1), _n_points)
                    if _n_points > 1
                    else [0]
                )
                _values = np.full(_n_points, 0.5)
                fitted_patterns["all days"] = DailyPattern(
                    hours=list(_hours),
                    values=list(_values),
                    name="manual",
                    day_type="all",
                )
        else:  # weekdays + weekends
            _wd_mask = series.index.dayofweek < 5
            if _is_fit_mode:
                fitted_patterns["weekday"] = fit_pattern(
                    series[_wd_mask],
                    n_control_points=_n_points,
                    day_type="weekday",
                )
                fitted_patterns["weekend"] = fit_pattern(
                    series[~_wd_mask],
                    n_control_points=_n_points,
                    day_type="weekend",
                )
            else:
                _hours = (
                    np.linspace(0, 24 - 24.0 / (_n_points - 1), _n_points)
                    if _n_points > 1
                    else [0]
                )
                _values = np.full(_n_points, 0.5)
                fitted_patterns["weekday"] = DailyPattern(
                    hours=list(_hours),
                    values=list(_values),
                    name="manual_weekday",
                    day_type="weekday",
                )
                fitted_patterns["weekend"] = DailyPattern(
                    hours=list(_hours),
                    values=list(_values),
                    name="manual_weekend",
                    day_type="weekend",
                )

    else:  # Sine Waves mode
        mo.stop(n_components_slider is None)
        _n_components = n_components_slider.value

        if day_type_radio.value == "all days":
            if _is_fit_mode:
                fitted_patterns["all days"] = fit_sine_pattern(
                    series,
                    n_components=_n_components,
                    day_type="all",
                )
            else:
                # Use manual slider values
                _components = [
                    SineComponent(
                        amplitude=sine_amplitude_sliders[i].value,
                        frequency=sine_frequency_sliders[i].value,
                        phase=sine_phase_sliders[i].value,
                    )
                    for i in range(_n_components)
                ]
                fitted_patterns["all days"] = DailyPattern(
                    sine_components=_components,
                    baseline=sine_baseline_slider.value,
                    name="manual_sine",
                    day_type="all",
                )
        else:  # weekdays + weekends
            _wd_mask = series.index.dayofweek < 5
            if _is_fit_mode:
                fitted_patterns["weekday"] = fit_sine_pattern(
                    series[_wd_mask],
                    n_components=_n_components,
                    day_type="weekday",
                )
                fitted_patterns["weekend"] = fit_sine_pattern(
                    series[~_wd_mask],
                    n_components=_n_components,
                    day_type="weekend",
                )
            else:
                # Use manual slider values for weekday
                _components = [
                    SineComponent(
                        amplitude=sine_amplitude_sliders[i].value,
                        frequency=sine_frequency_sliders[i].value,
                        phase=sine_phase_sliders[i].value,
                    )
                    for i in range(_n_components)
                ]
                fitted_patterns["weekday"] = DailyPattern(
                    sine_components=_components,
                    baseline=sine_baseline_slider.value,
                    name="manual_sine_weekday",
                    day_type="weekday",
                )
                # Use separate weekend sliders for weekend pattern
                _weekend_components = [
                    SineComponent(
                        amplitude=weekend_sine_amplitude_sliders[i].value,
                        frequency=weekend_sine_frequency_sliders[i].value,
                        phase=weekend_sine_phase_sliders[i].value,
                    )
                    for i in range(_n_components)
                ]
                fitted_patterns["weekend"] = DailyPattern(
                    sine_components=_weekend_components,
                    baseline=weekend_sine_baseline_slider.value,
                    name="manual_sine_weekend",
                    day_type="weekend",
                )
    return (fitted_patterns,)


@app.cell
def _(fitted_patterns, mo):
    """Create mo.state to hold committed puck positions.

    Initialized with the auto-fitted patterns. Updated when the user clicks
    "Apply Pattern" in the display cell. Downstream cells depend on
    get_committed_patterns rather than the puck widget directly, so they
    only re-execute on explicit Apply, not on every mouse-move during drag.
    """
    get_committed_patterns, set_committed_patterns = mo.state(fitted_patterns)
    return get_committed_patterns, set_committed_patterns


@app.cell
def _(
    ChartPuck,
    DailyPattern,
    day_type_radio,
    fitted_patterns,
    mo,
    n_points_slider,
    np,
    pattern_mode,
    silhouette_data,
):
    """Create ChartPuck widgets with real-time spline updates."""
    mo.stop(pattern_mode.value != "Spline" or n_points_slider is None)

    def make_draw_fn(silhouette, title_suffix="", dt_key="all"):
        """Factory to create draw callback with silhouette in closure."""

        def draw_chart(ax, widget):
            """Called on init and every puck move - must be fast!"""
            # Draw pre-computed silhouette (fast - just plotting cached data)
            if silhouette is not None:
                ax.fill_between(
                    silhouette["hours"],
                    silhouette["std_lower"],
                    silhouette["std_upper"],
                    color="lightgray",
                    alpha=0.6,
                    label="±2 std dev",
                )
                ax.plot(
                    silhouette["hours"],
                    silhouette["mean"],
                    "-",
                    color="gray",
                    linewidth=2,
                    alpha=0.7,
                    label="Mean profile",
                )

            # Create pattern from current puck positions
            if len(widget.x) >= 2:
                pattern = DailyPattern(
                    hours=list(widget.x),
                    values=list(widget.y),
                    name=f"preview_{dt_key}",
                    day_type=dt_key if dt_key in ["weekday", "weekend"] else "all",
                )
                # Draw spline curve (fast - 200 point evaluation)
                h = np.linspace(0, 24, 200)
                ax.plot(
                    h,
                    pattern.evaluate(h),
                    "-",
                    color="coral",
                    linewidth=2.5,
                    label="Spline",
                    zorder=3,
                )

            # Styling
            ax.set_xlabel("Hour of day")
            ax.set_ylabel("Normalized value (0–1)")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_title(
                f"Pattern Editor ({n_points_slider.value} pucks){title_suffix}"
            )

        return draw_chart

    puck_all = None
    puck_weekday = None
    puck_weekend = None

    if day_type_radio.value == "all days":
        _pat = fitted_patterns.get("all days")
        _sil = silhouette_data.get("all days")
        if _pat and len(_pat.hours) > 0:
            puck_all = ChartPuck.from_callback(
                draw_fn=make_draw_fn(_sil, "", "all"),
                x_bounds=(0, 24),
                y_bounds=(0, 1),
                figsize=(4, 3),
                x=list(_pat.hours),
                y=list(_pat.values),
                drag_x_bounds=(0, 24),
                drag_y_bounds=(0, 1),
                puck_radius=12,
                puck_color="#e63946",
            )
    else:  # weekdays + weekends
        for dt_key in ["weekday", "weekend"]:
            _pat = fitted_patterns.get(dt_key)
            _sil = silhouette_data.get(dt_key)
            if _pat and len(_pat.hours) > 0:
                _puck = ChartPuck.from_callback(
                    draw_fn=make_draw_fn(_sil, f" – {dt_key}", dt_key),
                    x_bounds=(0, 24),
                    y_bounds=(0, 1),
                    figsize=(4, 3),
                    x=list(_pat.hours),
                    y=list(_pat.values),
                    drag_x_bounds=(0, 24),
                    drag_y_bounds=(0, 1),
                    puck_radius=12,
                    puck_color="#e63946",
                )
                if dt_key == "weekday":
                    puck_weekday = _puck
                else:
                    puck_weekend = _puck
    return puck_all, puck_weekday, puck_weekend


@app.cell
def _(
    DailyPattern,
    day_type_radio,
    mo,
    n_points_slider,
    puck_all,
    puck_weekday,
    puck_weekend,
    set_committed_patterns,
):
    """Display PuckChart widgets with Apply button.

    Charts auto-update as pucks are dragged via from_callback.
    Apply button commits current positions to mo.state for gap-fill preview.
    """

    def _apply_patterns(_):
        """Read current puck positions and commit to mo.state."""
        _patterns = {}
        if day_type_radio.value == "all days" and puck_all is not None:
            _patterns["all days"] = DailyPattern(
                hours=list(puck_all.x),
                values=list(puck_all.y),
                name="manual",
                day_type="all",
            )
        else:
            if puck_weekday is not None:
                _patterns["weekday"] = DailyPattern(
                    hours=list(puck_weekday.x),
                    values=list(puck_weekday.y),
                    name="manual_weekday",
                    day_type="weekday",
                )
            if puck_weekend is not None:
                _patterns["weekend"] = DailyPattern(
                    hours=list(puck_weekend.x),
                    values=list(puck_weekend.y),
                    name="manual_weekend",
                    day_type="weekend",
                )
        set_committed_patterns(_patterns)

    _apply_button = mo.ui.button(
        label="✅ Apply Pattern",
        on_click=_apply_patterns,
    )

    # Display logic
    if day_type_radio.value == "all days":
        if puck_all is None:
            _out = mo.callout(
                mo.md("No pattern available. Please load data first."), kind="warn"
            )
        else:
            _out = mo.vstack(
                [
                    mo.md(f"""
                ### 📊 Puck Chart ({n_points_slider.value} control points)

                **Drag the control points** to shape the pattern. The spline
                updates in **real-time** as you drag! Click **Apply Pattern** when
                you're happy with the shape to update the gap-fill preview below.

                *The coral spline shows the interpolated curve through your pucks.
                The gray silhouette shows the mean ±2 standard deviations of your data.*
                """),
                    mo.ui.anywidget(puck_all),
                    _apply_button,
                ]
            )
    elif day_type_radio.value == "weekdays + weekends":
        _parts = []
        if puck_weekday is not None:
            _parts.append(
                mo.vstack(
                    [
                        mo.md(
                            "### 📅 Weekday Pattern\n\nDrag pucks to adjust the weekday pattern."
                        ),
                        mo.ui.anywidget(puck_weekday),
                    ],
                    gap=0.5,
                )
            )
        if puck_weekend is not None:
            _parts.append(
                mo.vstack(
                    [
                        mo.md(
                            "### 🏠 Weekend Pattern\n\nDrag pucks to adjust the weekend pattern."
                        ),
                        mo.ui.anywidget(puck_weekend),
                    ],
                    gap=0.5,
                )
            )

        if not _parts:
            _out = mo.callout(mo.md("No patterns available."), kind="warn")
        else:
            _out = mo.vstack(
                [
                    mo.md(f"""
                ### 📊 Puck Charts ({n_points_slider.value} control points each)

                **Drag the control points** to shape patterns. The splines update
                in **real-time** as you drag! Click **Apply** when ready to update
                the gap-fill preview below.
                """),
                    mo.hstack(_parts, gap=1, wrap=True),
                    _apply_button,
                ]
            )
    else:
        _out = mo.callout(mo.md("Unknown mode selected."), kind="warn")

    _out
    return


@app.cell
def _(
    day_type_radio,
    fitted_patterns,
    mo,
    n_components_slider,
    np,
    pattern_mode,
    plt,
    profiles,
    set_committed_patterns,
):
    """Display sine wave patterns with Apply button.

    Charts update automatically when sliders change (marimo reactivity).
    Apply button commits patterns to mo.state for gap-fill preview.
    """
    mo.stop(pattern_mode.value != "Sine Waves" or n_components_slider is None)

    def _render_sine_chart(stats_df, pattern, title_suffix=""):
        """Render chart with sine wave pattern."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlim(0, 24)
        ax.set_ylim(-0.1, 1.1)

        # Plot data silhouette
        if stats_df is not None and not stats_df.empty:
            _mean = stats_df["mean"]
            _std = stats_df["std"].fillna(0)
            _pmin, _pmax = _mean.min(), _mean.max()
            _range = _pmax - _pmin

            if _range > 0:
                _norm = lambda v: (v - _pmin) / _range
            else:
                _norm = lambda v: np.full_like(v, 0.5)

            ax.fill_between(
                stats_df.index,
                _norm((_mean - 2 * _std).values),
                _norm((_mean + 2 * _std).values),
                color="lightgray",
                alpha=0.6,
                label="±2 std dev",
            )
            ax.plot(
                stats_df.index,
                _norm(_mean.values),
                "-",
                color="gray",
                linewidth=2,
                alpha=0.7,
                label="Mean profile",
            )

        # Plot sine wave pattern
        if pattern is not None:
            _h = np.linspace(0, 24, 200)
            _values = pattern.evaluate(_h)
            ax.plot(
                _h,
                _values,
                "-",
                color="coral",
                linewidth=2.5,
                label="Sine pattern",
                zorder=3,
            )

            # Show individual components (if multiple)
            if pattern.mode == "sine" and len(pattern.sine_components) > 1:
                for i, comp in enumerate(pattern.sine_components):
                    _comp_vals = pattern.baseline + comp.evaluate(_h)
                    ax.plot(
                        _h,
                        np.clip(_comp_vals, 0, 1),
                        "--",
                        alpha=0.4,
                        linewidth=1.5,
                        label=f"Wave {i + 1}",
                    )

        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Normalized value (0–1)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f"Sine Pattern ({n_components_slider.value} components){title_suffix}"
        )
        plt.tight_layout()
        return fig

    def _apply_sine_patterns(_):
        """Commit current sine patterns to state."""
        _patterns = {}
        if day_type_radio.value == "all days":
            _patterns["all days"] = fitted_patterns.get("all days")
        else:
            _patterns["weekday"] = fitted_patterns.get("weekday")
            _patterns["weekend"] = fitted_patterns.get("weekend")
        set_committed_patterns(_patterns)

    _apply_button = mo.ui.button(
        label="✅ Apply Pattern",
        on_click=_apply_sine_patterns,
    )

    # Render and display charts
    if day_type_radio.value == "all days":
        _pat = fitted_patterns.get("all days")
        _prof = profiles.get("all days")
        if _pat:
            _fig = _render_sine_chart(_prof, _pat)
            _out = mo.vstack(
                [
                    mo.md("""
                ### 🌊 Sine Wave Pattern Preview

                The chart shows your sine wave pattern (coral) overlaid on
                the data profile (gray). Individual components shown as dashed lines.
                **Adjust the sliders above** to modify the pattern in real-time,
                then click **Apply Pattern** to use it for gap filling.
                """),
                    _fig,
                    _apply_button,
                ]
            )
            plt.close(_fig)
        else:
            _out = mo.callout(mo.md("No sine pattern available."), kind="warn")
    else:  # weekdays + weekends
        _figs = []
        for _dt_key, _label in [("weekday", "Weekday"), ("weekend", "Weekend")]:
            _pat = fitted_patterns.get(_dt_key)
            _prof = profiles.get(_dt_key)
            if _pat:
                _fig = _render_sine_chart(_prof, _pat, f" – {_label}")
                _figs.append(_fig)

        if _figs:
            _out = mo.vstack(
                [
                    mo.md("""
                ### 🌊 Sine Wave Pattern Preview

                Charts show your sine patterns overlaid on data profiles.
                **Adjust sliders above** to modify patterns in real-time,
                then click **Apply** to use them for gap filling.
                """),
                    mo.hstack(_figs, gap=1),
                    _apply_button,
                ]
            )
            for _f in _figs:
                plt.close(_f)
        else:
            _out = mo.callout(mo.md("No sine patterns available."), kind="warn")

    _out
    return


@app.cell
def _(get_committed_patterns, mo):
    """Read committed patterns from mo.state.

    This cell re-executes only when set_committed_patterns is called
    (i.e., when the user clicks Apply Pattern), not during puck dragging.
    The state is initialized with auto-fitted patterns so the gap-fill
    preview is available immediately on load.
    """
    active_patterns = get_committed_patterns()
    mo.stop(not active_patterns)
    return (active_patterns,)


@app.cell
def _(
    active_patterns,
    blend_minutes_slider,
    mo,
    normalize_checkbox,
    pattern_fill,
    pd,
    plt,
    series,
):
    mo.stop(
        not active_patterns,
        mo.callout(mo.md("No pattern configured yet."), kind="warn"),
    )

    if len(active_patterns) == 1 and "all days" in active_patterns:
        _pat_arg = active_patterns["all days"]
    else:
        _pat_arg = active_patterns

    _results = pattern_fill(
        [series],
        pattern=_pat_arg,
        scaling="local",
        normalize=normalize_checkbox.value,
        blend_minutes=blend_minutes_slider.value,
    )
    _filled, _steps = _results[0]

    _fig, _axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    _ax = _axes[0]
    _ax.plot(series.index, series.values, linewidth=0.8, color="steelblue")
    _nan_mask = series.isna()
    if _nan_mask.any():
        _changes = _nan_mask.astype(int).diff().fillna(0)
        _starts = series.index[_changes == 1]
        _ends = series.index[_changes == -1]
        if _nan_mask.iloc[0]:
            _starts = _starts.insert(0, series.index[0])
        if _nan_mask.iloc[-1]:
            _ends = _ends.append(pd.DatetimeIndex([series.index[-1]]))
        for _s, _e in zip(_starts, _ends):
            _ax.axvspan(_s, _e, alpha=0.15, color="red")
    _ax.set_title("Before (with gaps)")
    _ax.set_ylabel(series.name)

    _ax = _axes[1]
    _ax.plot(_filled.index, _filled.values, linewidth=0.8, color="steelblue")
    _filled_mask = series.isna() & _filled.notna()
    if _filled_mask.any():
        _ax.plot(
            _filled.index[_filled_mask],
            _filled.values[_filled_mask],
            linestyle="none",
            marker=".",
            markersize=2,
            color="coral",
            label="Pattern-filled",
        )
    _ax.set_title("After (gaps filled)")
    _ax.set_ylabel(_filled.name)
    _ax.set_xlabel("Time")
    _ax.legend(fontsize=8)

    plt.tight_layout()
    _fig
    return


@app.cell
def _(active_patterns, json, mo):
    mo.stop(not active_patterns)

    if len(active_patterns) == 1:
        _pat = list(active_patterns.values())[0]
        _export = _pat.to_dict()
        _snippet = (
            "from pattern_fill import DailyPattern, pattern_fill\n\n"
            f"pattern = DailyPattern.from_json('{_pat.to_json()}')\n\n"
            "# Use with metEAUdata:\n"
            "# signal.process(\n"
            '#     input_time_series_names=["raw"],\n'
            "#     transform_function=pattern_fill,\n"
            "#     pattern=pattern,\n"
            "# )"
        )
    else:
        _export = {k: v.to_dict() for k, v in active_patterns.items()}
        _pjson = json.dumps(_export)
        _snippet = (
            "import json\n"
            "from pattern_fill import DailyPattern, pattern_fill\n\n"
            f"raw = json.loads('{_pjson}')\n"
            "patterns = {k: DailyPattern.from_dict(v) for k, v in raw.items()}\n\n"
            "# Use with metEAUdata:\n"
            "# signal.process(\n"
            '#     input_time_series_names=["raw"],\n'
            "#     transform_function=pattern_fill,\n"
            "#     pattern=patterns,\n"
            "# )"
        )

    _json_str = json.dumps(_export, indent=2)
    mo.vstack(
        [
            mo.md("## Export"),
            mo.md("### Pattern JSON"),
            mo.md(f"```json\n{_json_str}\n```"),
            mo.md("### Usage snippet"),
            mo.md(f"```python\n{_snippet}\n```"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()

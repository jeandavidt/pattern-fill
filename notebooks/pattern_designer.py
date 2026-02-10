import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")


@app.cell
async def _():
    """Install packages not in Pyodide when running in WASM."""
    import sys

    # Check if we're running in Pyodide (WASM environment)
    if "pyodide" in sys.modules:
        import micropip

        # Install dependencies from PyPI
        print("📦 Installing dependencies from PyPI...")
        await micropip.install("typing-extensions>=4.12.0")
        await micropip.install("wigglystuff>=0.2.21")
        await micropip.install("pattern-fill")
        print("✅ All packages installed successfully!")
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
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
    import os
    import tempfile
    import zipfile
    import shutil
    import base64
    from meteaudata import Signal, Dataset, TimeSeries

    return (
        ChartPuck,
        DailyPattern,
        Dataset,
        Signal,
        SineComponent,
        alt,
        base64,
        fit_pattern,
        fit_sine_pattern,
        io,
        json,
        mo,
        np,
        os,
        pattern_fill,
        pd,
        plt,
        shutil,
        tempfile,
        zipfile,
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
        options=["demo", "upload_csv", "upload_meteaudata"],
        value="demo",
        label="Data source",
    )
    file_upload = mo.ui.file(
        filetypes=[".csv", ".parquet", ".xlsx", ".xls"],
        kind="area",
        label="Upload a CSV, Parquet, or Excel file",
    )
    meteaudata_upload = mo.ui.file(
        filetypes=[".zip"],
        kind="area",
        label="Upload a metEAUdata zip file (Signal or Dataset)",
    )
    data_source
    return data_source, file_upload, meteaudata_upload


@app.cell
def _(data_source, file_upload, meteaudata_upload, mo):
    mo.stop(data_source.value == "demo")
    _upload_widget = file_upload if data_source.value == "upload_csv" else meteaudata_upload
    _upload_widget
    return


@app.cell
def _(
    Dataset,
    Signal,
    data_source,
    file_upload,
    generate_demo_data,
    io,
    meteaudata_upload,
    mo,
    os,
    pd,
    tempfile,
    zipfile,
):
    _debug_msg = f"data_source.value = {repr(data_source.value)}"

    original_signal = None
    original_dataset = None

    if data_source.value == "demo":
        try:
            df = generate_demo_data()
            import_type = "demo"
        except Exception as e:
            mo.stop(
                True,
                mo.callout(mo.md(f"Error generating demo data: {e}"), kind="danger"),
            )
            df = pd.DataFrame()
            import_type = None
    elif data_source.value == "upload_csv":
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
        import_type = "csv"
    else:
        mo.stop(
            not meteaudata_upload.value,
            mo.callout(
                mo.md(f"Upload a metEAUdata zip file to continue. ({_debug_msg})"),
                kind="warn",
            ),
        )
        _uploaded = meteaudata_upload.value[0]
        _name = _uploaded.name
        _buf = io.BytesIO(_uploaded.contents)

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = f"{tmpdir}/upload.zip"
            with open(zip_path, "wb") as f:
                f.write(_buf.getvalue())

            # Extract the uploaded ZIP
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdir)

            dir_items = os.listdir(tmpdir)

            # Check if there are nested ZIP files (metEAUdata exports often contain nested ZIPs)
            nested_zips = [f for f in dir_items if f.endswith(".zip")]
            if nested_zips:
                # Extract all nested ZIP files
                for nested_zip in nested_zips:
                    nested_zip_path = os.path.join(tmpdir, nested_zip)
                    with zipfile.ZipFile(nested_zip_path, "r") as zip_ref:
                        zip_ref.extractall(tmpdir)
                # Refresh directory listing after extracting nested ZIPs
                dir_items = os.listdir(tmpdir)

            # Check if it's a Dataset (has dataset.json) or Signal(s)
            has_dataset_json = "dataset.json" in dir_items

            if has_dataset_json:
                # Load as Dataset
                dataset = Dataset.load_from_directory(tmpdir)

                # Create DataFrame from all time series across all signals
                df = pd.DataFrame()
                for sig_name, sig in dataset.signals.items():
                    for ts_name, ts in sig.time_series.items():
                        # Use fully qualified name: signal_name.time_series_name
                        col_name = f"{sig_name}.{ts_name}"
                        df[col_name] = ts.series

                import_type = "meteaudata"
                original_signal = None  # It's a Dataset, not a single Signal
                original_dataset = dataset
            else:
                # Load as Signal(s)
                # metEAUdata Signals are stored in directories like "signal_name#1_data"
                # where #1 is the version number
                signal_dirs = [
                    d
                    for d in dir_items
                    if d.endswith("_data") and os.path.isdir(os.path.join(tmpdir, d))
                ]
                # Extract signal names by removing "_data" suffix
                # e.g., "pattern_filled#1_data" -> "pattern_filled#1"
                signal_names = [d.replace("_data", "") for d in signal_dirs]

                if not signal_names:
                    mo.stop(
                        True,
                        mo.callout(
                            mo.md(
                                "No Signal or Dataset found in zip file. Expected format: `signal_name#version_data/` directory (e.g., `pattern_filled#1_data/`) or `dataset.json` file."
                            ),
                            kind="danger",
                        ),
                    )
                    df = pd.DataFrame()
                    import_type = None
                else:
                    signal_name = signal_names[0]

                    # Try to load the Signal
                    try:
                        signal = Signal.load_from_directory(tmpdir, signal_name)

                        df = pd.DataFrame()
                        for ts_name, ts in signal.time_series.items():
                            df[ts_name] = ts.series

                        original_signal = signal
                    except ValueError:
                        # Workaround for metEAUdata parsing issue with multiple # in time series names
                        # Load CSV files manually and create a Signal object
                        data_dir = os.path.join(tmpdir, f"{signal_name}_data")
                        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

                        df = pd.DataFrame()
                        for csv_file in csv_files:
                            csv_path = os.path.join(data_dir, csv_file)
                            ts_data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                            # Use the CSV filename (without .csv) as the column name
                            # Keep original metEAUdata names - don't modify version numbers
                            col_name = csv_file.replace('.csv', '')
                            loaded_series = ts_data.iloc[:, 0]
                            df[col_name] = loaded_series

                        # Don't create a Signal object here - parsing issues with multiple #
                        # Export cell will create Signal from DataFrame to preserve all series
                        original_signal = None

                    import_type = "meteaudata"
                    original_dataset = None
    original_signal = (
        original_signal
        if "original_signal" in dir() or "original_signal" in locals()
        else None
    )
    original_dataset = (
        original_dataset
        if "original_dataset" in dir() or "original_dataset" in locals()
        else None
    )
    return df, import_type, original_dataset, original_signal


@app.cell
def _(df, import_type, mo, pd):
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
        label="Value column" + (" (TimeSeries)" if import_type == "meteaudata" else ""),
    )
    mo.hstack([time_col, value_col], justify="start", gap=1)
    return time_col, value_col


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
def _(alt, pd, series):
    # Prepare data for Altair
    _df = pd.DataFrame({
        "time": series.index,
        "value": series.values,
        "is_gap": series.isna()
    })

    # Calculate gap regions for highlighting
    _nan_mask = series.isna()
    _gap_rects = []
    if _nan_mask.any():
        _changes = _nan_mask.astype(int).diff().fillna(0)
        _starts = series.index[_changes == 1]
        _ends = series.index[_changes == -1]
        if _nan_mask.iloc[0]:
            _starts = _starts.insert(0, series.index[0])
        if _nan_mask.iloc[-1]:
            _ends = _ends.append(pd.DatetimeIndex([series.index[-1]]))
        for _s, _e in zip(_starts, _ends):
            _gap_rects.append({"start": _s, "end": _e})

    # Base chart with valid data
    _valid_df = _df[_df["is_gap"] == False].copy()

    _chart = alt.Chart(_valid_df).mark_line(
        color="steelblue",
        strokeWidth=0.8
    ).encode(
        x=alt.X("time:T", title="Time"),
        y=alt.Y("value:Q", title=series.name)
    )

    # Add gap regions as highlighted rectangles
    if _gap_rects:
        _gap_df = pd.DataFrame(_gap_rects)
        _gap_chart = alt.Chart(_gap_df).mark_rect(
            color="red",
            opacity=0.2
        ).encode(
            x=alt.X("start:T"),
            x2=alt.X2("end:T")
        )
        _chart = _chart + _gap_chart

    _chart = _chart.properties(
        title="Raw series (red = gaps)",
        height=100
    ).configure_axis(
        labelFontSize=10,
        titleFontSize=11
    )

    _chart
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
        # Number of control points (pucks) input
        n_points_slider = mo.ui.number(
            start=4,
            stop=24,
            step=1,
            value=8,
            label="Number of pucks",
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
        n_components_slider = mo.ui.number(
            start=1,
            stop=5,
            step=1,
            value=2,
            label="Number of sine waves",
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
def _(mo, n_components_slider, pattern_mode):
    """State management for sine wave parameters.

    This cell always runs (no mo.stop) to ensure state variables are always available,
    even when not in Sine mode.
    """
    # Determine n_components for default initialization
    # Use 2 as default if slider doesn't exist yet or we're not in Sine mode
    if pattern_mode.value == "Sine Waves" and n_components_slider is not None:
        _n = n_components_slider.value
    else:
        _n = 2

    # Default values for initial state
    _default_amplitudes = [0.3 if i == 0 else 0.15 for i in range(_n)]
    _default_frequencies = [1.0 if i == 0 else 2.0 for i in range(_n)]
    _default_phases = [8.0 if i == 0 else 13.0 for i in range(_n)]
    _default_baseline = 0.5

    # Weekday parameter states
    get_sine_amplitudes, set_sine_amplitudes = mo.state(_default_amplitudes)
    get_sine_frequencies, set_sine_frequencies = mo.state(_default_frequencies)
    get_sine_phases, set_sine_phases = mo.state(_default_phases)
    get_sine_baseline, set_sine_baseline = mo.state(_default_baseline)

    # Weekend parameter states
    get_weekend_amplitudes, set_weekend_amplitudes = mo.state(_default_amplitudes)
    get_weekend_frequencies, set_weekend_frequencies = mo.state(_default_frequencies)
    get_weekend_phases, set_weekend_phases = mo.state(_default_phases)
    get_weekend_baseline, set_weekend_baseline = mo.state(_default_baseline)
    return (
        get_sine_amplitudes,
        get_sine_baseline,
        get_sine_frequencies,
        get_sine_phases,
        get_weekend_amplitudes,
        get_weekend_baseline,
        get_weekend_frequencies,
        get_weekend_phases,
        set_sine_amplitudes,
        set_sine_baseline,
        set_sine_frequencies,
        set_sine_phases,
        set_weekend_amplitudes,
        set_weekend_baseline,
        set_weekend_frequencies,
        set_weekend_phases,
    )


@app.cell
def _(
    day_type_radio,
    get_sine_amplitudes,
    get_sine_baseline,
    get_sine_frequencies,
    get_sine_phases,
    get_weekend_amplitudes,
    get_weekend_baseline,
    get_weekend_frequencies,
    get_weekend_phases,
    mo,
    n_components_slider,
    pattern_mode,
    set_sine_amplitudes,
    set_sine_baseline,
    set_sine_frequencies,
    set_sine_phases,
    set_weekend_amplitudes,
    set_weekend_baseline,
    set_weekend_frequencies,
    set_weekend_phases,
):
    """Create sliders for sine wave parameters - values driven by state."""
    mo.stop(pattern_mode.value != "Sine Waves" or n_components_slider is None)

    _n = n_components_slider.value

    # Get current state values
    _amp_vals = get_sine_amplitudes()
    _freq_vals = get_sine_frequencies()
    _phase_vals = get_sine_phases()
    _baseline_val = get_sine_baseline()

    _weekend_amp_vals = get_weekend_amplitudes()
    _weekend_freq_vals = get_weekend_frequencies()
    _weekend_phase_vals = get_weekend_phases()
    _weekend_baseline_val = get_weekend_baseline()

    # Ensure state has correct length (handle n_components changes)
    if len(_amp_vals) != _n:
        # Preserve existing values and add defaults for new components
        _amp_vals = list(_amp_vals[:_n]) + [0.15] * max(0, _n - len(_amp_vals))
        _freq_vals = list(_freq_vals[:_n]) + [2.0] * max(0, _n - len(_freq_vals))
        _phase_vals = list(_phase_vals[:_n]) + [13.0] * max(0, _n - len(_phase_vals))
        set_sine_amplitudes(_amp_vals)
        set_sine_frequencies(_freq_vals)
        set_sine_phases(_phase_vals)

        # Do same for weekend
        _weekend_amp_vals = list(_weekend_amp_vals[:_n]) + [0.15] * max(
            0, _n - len(_weekend_amp_vals)
        )
        _weekend_freq_vals = list(_weekend_freq_vals[:_n]) + [2.0] * max(
            0, _n - len(_weekend_freq_vals)
        )
        _weekend_phase_vals = list(_weekend_phase_vals[:_n]) + [13.0] * max(
            0, _n - len(_weekend_phase_vals)
        )
        set_weekend_amplitudes(_weekend_amp_vals)
        set_weekend_frequencies(_weekend_freq_vals)
        set_weekend_phases(_weekend_phase_vals)

    # Create on_change handlers
    def make_amplitude_handler(index):
        def handler(value):
            _vals = list(get_sine_amplitudes())
            _vals[index] = value
            set_sine_amplitudes(_vals)

        return handler

    def make_frequency_handler(index):
        def handler(value):
            _vals = list(get_sine_frequencies())
            _vals[index] = value
            set_sine_frequencies(_vals)

        return handler

    def make_phase_handler(index):
        def handler(value):
            _vals = list(get_sine_phases())
            _vals[index] = value
            set_sine_phases(_vals)

        return handler

    # Similar handlers for weekend
    def make_weekend_amplitude_handler(index):
        def handler(value):
            _vals = list(get_weekend_amplitudes())
            _vals[index] = value
            set_weekend_amplitudes(_vals)

        return handler

    def make_weekend_frequency_handler(index):
        def handler(value):
            _vals = list(get_weekend_frequencies())
            _vals[index] = value
            set_weekend_frequencies(_vals)

        return handler

    def make_weekend_phase_handler(index):
        def handler(value):
            _vals = list(get_weekend_phases())
            _vals[index] = value
            set_weekend_phases(_vals)

        return handler

    # Weekday sliders
    sine_amplitude_sliders = mo.ui.array(
        [
            mo.ui.slider(
                start=0.0,
                stop=1.0,
                step=0.01,
                value=_amp_vals[i],
                label=f"Wave {i + 1} Amplitude",
                show_value=True,
                on_change=make_amplitude_handler(i),
            )
            for i in range(_n)
        ]
    )

    sine_frequency_sliders = mo.ui.array(
        [
            mo.ui.slider(
                start=0.08333,
                stop=12.0,
                step=0.125,
                value=_freq_vals[i],
                label=f"Wave {i + 1} Frequency",
                show_value=True,
                on_change=make_frequency_handler(i),
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
                value=_phase_vals[i],
                label=f"Wave {i + 1} Phase (hour)",
                show_value=True,
                on_change=make_phase_handler(i),
            )
            for i in range(_n)
        ]
    )

    sine_baseline_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.01,
        value=_baseline_val,
        label="Baseline",
        show_value=True,
        on_change=set_sine_baseline,
    )

    # Weekend sliders
    weekend_sine_amplitude_sliders = mo.ui.array(
        [
            mo.ui.slider(
                start=0.0,
                stop=0.5,
                step=0.01,
                value=_weekend_amp_vals[i],
                label=f"Weekend Wave {i + 1} Amplitude",
                show_value=True,
                on_change=make_weekend_amplitude_handler(i),
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
                value=_weekend_freq_vals[i],
                label=f"Weekend Wave {i + 1} Frequency",
                show_value=True,
                on_change=make_weekend_frequency_handler(i),
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
                value=_weekend_phase_vals[i],
                label=f"Weekend Wave {i + 1} Phase (hour)",
                show_value=True,
                on_change=make_weekend_phase_handler(i),
            )
            for i in range(_n)
        ]
    )

    weekend_sine_baseline_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.01,
        value=_weekend_baseline_val,
        label="Weekend Baseline",
        show_value=True,
        on_change=set_weekend_baseline,
    )

    # Display layout (same as before)
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
    return


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
    get_sine_amplitudes,
    get_sine_baseline,
    get_sine_frequencies,
    get_sine_phases,
    get_weekend_amplitudes,
    get_weekend_baseline,
    get_weekend_frequencies,
    get_weekend_phases,
    mo,
    n_components_slider,
    n_points_slider,
    np,
    pattern_mode,
    series,
    set_sine_amplitudes,
    set_sine_baseline,
    set_sine_frequencies,
    set_sine_phases,
    set_weekend_amplitudes,
    set_weekend_baseline,
    set_weekend_frequencies,
    set_weekend_phases,
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
                # Note: editor_patterns state keeps puck positions separately,
                # so this doesn't reset the pucks (they read from editor_patterns, not fitted_patterns)
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
                # Initialize with uniform values when not fitting
                # Note: editor_patterns state keeps puck positions separately,
                # so this doesn't reset the pucks (they read from editor_patterns, not fitted_patterns)
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
                # Fit the pattern
                fitted_patterns["all days"] = fit_sine_pattern(
                    series,
                    n_components=_n_components,
                    day_type="all",
                )

                # Extract fitted parameters and update state
                _fitted_pat = fitted_patterns["all days"]
                _new_amps = [c.amplitude for c in _fitted_pat.sine_components]
                _new_freqs = [c.frequency for c in _fitted_pat.sine_components]
                _new_phases = [c.phase for c in _fitted_pat.sine_components]

                set_sine_amplitudes(_new_amps)
                set_sine_frequencies(_new_freqs)
                set_sine_phases(_new_phases)
                set_sine_baseline(_fitted_pat.baseline)
            else:
                # Read from state (already updated by sliders)
                _amps = get_sine_amplitudes()
                _freqs = get_sine_frequencies()
                _phases = get_sine_phases()
                _baseline = get_sine_baseline()

                # Ensure state has enough values (pad with defaults if needed)
                while len(_amps) < _n_components:
                    _amps = list(_amps) + [0.15]
                    _freqs = list(_freqs) + [2.0]
                    _phases = list(_phases) + [13.0]

                _components = [
                    SineComponent(
                        amplitude=_amps[i],
                        frequency=_freqs[i],
                        phase=_phases[i],
                    )
                    for i in range(_n_components)
                ]
                fitted_patterns["all days"] = DailyPattern(
                    sine_components=_components,
                    baseline=_baseline,
                    name="manual_sine",
                    day_type="all",
                )
        else:  # weekdays + weekends
            _wd_mask = series.index.dayofweek < 5
            if _is_fit_mode:
                # Fit both patterns
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

                # Extract and update state for weekday
                _wd_pat = fitted_patterns["weekday"]
                set_sine_amplitudes([c.amplitude for c in _wd_pat.sine_components])
                set_sine_frequencies([c.frequency for c in _wd_pat.sine_components])
                set_sine_phases([c.phase for c in _wd_pat.sine_components])
                set_sine_baseline(_wd_pat.baseline)

                # Extract and update state for weekend
                _we_pat = fitted_patterns["weekend"]
                set_weekend_amplitudes([c.amplitude for c in _we_pat.sine_components])
                set_weekend_frequencies([c.frequency for c in _we_pat.sine_components])
                set_weekend_phases([c.phase for c in _we_pat.sine_components])
                set_weekend_baseline(_we_pat.baseline)
            else:
                # Read from state for weekday
                _amps = get_sine_amplitudes()
                _freqs = get_sine_frequencies()
                _phases = get_sine_phases()
                _baseline = get_sine_baseline()

                # Ensure state has enough values (pad with defaults if needed)
                while len(_amps) < _n_components:
                    _amps = list(_amps) + [0.15]
                    _freqs = list(_freqs) + [2.0]
                    _phases = list(_phases) + [13.0]

                _components = [
                    SineComponent(
                        amplitude=_amps[i],
                        frequency=_freqs[i],
                        phase=_phases[i],
                    )
                    for i in range(_n_components)
                ]
                fitted_patterns["weekday"] = DailyPattern(
                    sine_components=_components,
                    baseline=_baseline,
                    name="manual_sine_weekday",
                    day_type="weekday",
                )

                # Read from state for weekend
                _weekend_amps = get_weekend_amplitudes()
                _weekend_freqs = get_weekend_frequencies()
                _weekend_phases = get_weekend_phases()
                _weekend_baseline = get_weekend_baseline()

                # Ensure state has enough values (pad with defaults if needed)
                while len(_weekend_amps) < _n_components:
                    _weekend_amps = list(_weekend_amps) + [0.15]
                    _weekend_freqs = list(_weekend_freqs) + [2.0]
                    _weekend_phases = list(_weekend_phases) + [13.0]

                _weekend_components = [
                    SineComponent(
                        amplitude=_weekend_amps[i],
                        frequency=_weekend_freqs[i],
                        phase=_weekend_phases[i],
                    )
                    for i in range(_n_components)
                ]
                fitted_patterns["weekend"] = DailyPattern(
                    sine_components=_weekend_components,
                    baseline=_weekend_baseline,
                    name="manual_sine_weekend",
                    day_type="weekend",
                )
    return (fitted_patterns,)


@app.cell
def _(fit_checkbox, fitted_patterns, mo):
    """Create mo.state to hold editor UI state and committed patterns.

    editor_patterns: Holds the UI state for spline mode (puck positions).
                     For sine mode, dedicated parameter states are used instead
                     (see sine parameter state cell).
                     When autofit toggles ON, it's updated to the fitted values.
                     When autofit toggles OFF, it stays where it is.

    committed_patterns: The patterns that were last applied for gap-fill preview.
    """
    get_editor_patterns, set_editor_patterns = mo.state(fitted_patterns)
    get_committed_patterns, set_committed_patterns = mo.state(fitted_patterns)

    # Update editor state when fit_checkbox changes to True
    # (fitted_patterns already contains the new values)
    if fit_checkbox.value:
        set_editor_patterns(fitted_patterns)
    return (
        get_committed_patterns,
        get_editor_patterns,
        set_committed_patterns,
        set_editor_patterns,
    )


@app.cell
def _(
    ChartPuck,
    DailyPattern,
    day_type_radio,
    get_editor_patterns,
    mo,
    n_points_slider,
    np,
    pattern_mode,
    silhouette_data,
):
    """Create ChartPuck widgets with real-time spline updates.

    Pucks are initialized from editor_patterns, which persists user edits
    even when autofit is toggled off.
    """
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

    # Use editor_patterns (persisted state) instead of fitted_patterns
    editor_patterns = get_editor_patterns()

    puck_all = None
    puck_weekday = None
    puck_weekend = None

    if day_type_radio.value == "all days":
        _pat = editor_patterns.get("all days")
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
            _pat = editor_patterns.get(dt_key)
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
    set_editor_patterns,
):
    """Display PuckChart widgets with Apply button.

    Charts auto-update as pucks are dragged via from_callback.
    Apply button commits current positions to both mo.state and editor_patterns.
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
        # Also update editor state so if user toggles autofit again, they start from here
        set_editor_patterns(_patterns)

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
    set_editor_patterns,
):
    """Display sine wave patterns with Apply button.

    Charts update automatically when sliders change (marimo reactivity).
    Apply button commits patterns to both committed and editor states.
    """
    mo.stop(pattern_mode.value != "Sine Waves" or n_components_slider is None)

    def _render_sine_chart(stats_df, pattern, title_suffix=""):
        """Render matplotlib chart with sine wave pattern."""
        fig, ax = plt.subplots(figsize=(6, 3))

        # Generate pattern curve
        _h = np.linspace(0, 24, 200)
        _values = pattern.evaluate(_h)

        # Plot pattern
        ax.plot(_h, _values, "-", color="coral", linewidth=2.5, label="Pattern")

        # Add data silhouette if available
        if stats_df is not None and not stats_df.empty:
            _mean = stats_df["mean"]
            _std = stats_df["std"].fillna(0)
            _pmin, _pmax = _mean.min(), _mean.max()
            _range = _pmax - _pmin

            if _range > 0:
                _norm = lambda v: (v - _pmin) / _range
            else:
                _norm = lambda v: np.full_like(v, 0.5)

            # Plot silhouette
            _hours = stats_df.index.values
            _mean_norm = _norm(_mean.values)
            _std_lower = _norm((_mean - 2 * _std).values)
            _std_upper = _norm((_mean + 2 * _std).values)

            ax.fill_between(_hours, _std_lower, _std_upper, color="lightgray", alpha=0.6, label="±2 std dev")
            ax.plot(_hours, _mean_norm, "-", color="gray", linewidth=2, alpha=0.7, label="Mean profile")

        # Add individual components if multiple
        if pattern.mode == "sine" and len(pattern.sine_components) > 1:
            for i, comp in enumerate(pattern.sine_components):
                _comp_vals = pattern.baseline + comp.evaluate(_h)
                ax.plot(_h, np.clip(_comp_vals, 0, 1), "--", color="gray", linewidth=1.5, alpha=0.4, label=f"Wave {i+1}")

        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Normalized value (0–1)")
        ax.set_xlim(0, 24)
        ax.set_ylim(-0.1, 1.1)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Sine Pattern ({n_components_slider.value} components){title_suffix}")

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
        # Also update editor state so if user toggles autofit again, they start from here
        set_editor_patterns(_patterns)

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
    alt,
    blend_minutes_slider,
    mo,
    normalize_checkbox,
    pattern_fill,
    pd,
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

    # Prepare data for Altair charts
    # Before chart: original series with gaps highlighted
    _before_df = pd.DataFrame({
        "time": series.index,
        "value": series.values,
        "is_gap": series.isna()
    })

    # Calculate gap regions
    _nan_mask = series.isna()
    _gap_rects = []
    if _nan_mask.any():
        _changes = _nan_mask.astype(int).diff().fillna(0)
        _starts = series.index[_changes == 1]
        _ends = series.index[_changes == -1]
        if _nan_mask.iloc[0]:
            _starts = _starts.insert(0, series.index[0])
        if _nan_mask.iloc[-1]:
            _ends = _ends.append(pd.DatetimeIndex([series.index[-1]]))
        for _s, _e in zip(_starts, _ends):
            _gap_rects.append({"start": _s, "end": _e})

    # Before chart
    _valid_before_df = _before_df[_before_df["is_gap"] == False].copy()

    # Calculate dynamic y-axis range
    _before_values = _valid_before_df["value"].dropna()
    if len(_before_values) > 0:
        _y_min = _before_values.min()
        _y_max = _before_values.max()
        _y_range = _y_max - _y_min
        _y_extent = 0.2 * _y_range if _y_range > 0 else 1
        _y_domain_min = _y_min - _y_extent
        _y_domain_max = _y_max + _y_extent
    else:
        _y_domain_min = 0
        _y_domain_max = 1

    _before_chart = alt.Chart(_valid_before_df).mark_line(
        color="steelblue",
        strokeWidth=0.8
    ).encode(
        x=alt.X("time:T", title="Time"),
        y=alt.Y("value:Q", title=series.name, scale=alt.Scale(domain=[_y_domain_min, _y_domain_max]))
    )

    if _gap_rects:
        _gap_df = pd.DataFrame(_gap_rects)
        _gap_chart = alt.Chart(_gap_df).mark_rect(
            color="red",
            opacity=0.15
        ).encode(
            x=alt.X("start:T"),
            x2=alt.X2("end:T")
        )
        _before_chart = _before_chart + _gap_chart

    _before_chart = _before_chart.properties(
        title="Before (with gaps)",
        height=150
    ).configure_axis(
        labelFontSize=10,
        titleFontSize=11
    )

    # After chart: filled series with filled points highlighted
    _after_df = pd.DataFrame({
        "time": _filled.index,
        "value": _filled.values
    })

    _filled_mask = series.isna() & _filled.notna()

    # Calculate dynamic y-axis range based on both before and after data
    _after_values = _after_df["value"].dropna()
    if len(_before_values) > 0 and len(_after_values) > 0:
        _y_min = min(_before_values.min(), _after_values.min())
        _y_max = max(_before_values.max(), _after_values.max())
        _y_range = _y_max - _y_min
        _y_extent = 0.2 * _y_range if _y_range > 0 else 1
        _y_domain_min = _y_min - _y_extent
        _y_domain_max = _y_max + _y_extent
    elif len(_after_values) > 0:
        _y_min = _after_values.min()
        _y_max = _after_values.max()
        _y_range = _y_max - _y_min
        _y_extent = 0.2 * _y_range if _y_range > 0 else 1
        _y_domain_min = _y_min - _y_extent
        _y_domain_max = _y_max + _y_extent
    else:
        _y_domain_min = 0
        _y_domain_max = 1

    _after_chart = alt.Chart(_after_df).mark_line(
        color="steelblue",
        strokeWidth=0.8
    ).encode(
        x=alt.X("time:T", title="Time"),
        y=alt.Y("value:Q", title=_filled.name, scale=alt.Scale(domain=[_y_domain_min, _y_domain_max]))
    )

    if _filled_mask.any():
        _filled_pts_df = pd.DataFrame({
            "time": _filled.index[_filled_mask],
            "value": _filled.values[_filled_mask]
        })
        _filled_pts_chart = alt.Chart(_filled_pts_df).mark_point(
            color="coral",
            size=10
        ).encode(
            x=alt.X("time:T"),
            y=alt.Y("value:Q")
        )
        _after_chart = _after_chart + _filled_pts_chart

    _after_chart = _after_chart.properties(
        title="After (gaps filled)",
        height=150
    ).configure_axis(
        labelFontSize=10,
        titleFontSize=11
    ).configure_legend(
        labelFontSize=8
    )

    _out = mo.vstack([
        _before_chart,
        _after_chart
    ])

    _out
    return


@app.cell
def _(
    Signal,
    active_patterns,
    base64,
    blend_minutes_slider,
    df,
    import_type,
    mo,
    normalize_checkbox,
    original_dataset,
    original_signal,
    pattern_fill,
    series,
    shutil,
    tempfile,
):
    mo.stop(
        not active_patterns,
        mo.callout(mo.md("No pattern configured yet."), kind="warn"),
    )

    if len(active_patterns) == 1 and "all days" in active_patterns:
        _pat_arg = active_patterns["all days"]
    else:
        _pat_arg = active_patterns

    def create_meteaudata_download(
        series,
        df_all,
        pattern_arg,
        normalize,
        blend_minutes,
        import_type,
        original_signal,
        original_dataset,
    ):
        if import_type == "meteaudata" and original_dataset is not None:
            # Handle Dataset case
            # series.name format: "signal_name.time_series_name"
            if "." in series.name:
                sig_name, ts_name = series.name.split(".", 1)
            else:
                sig_name = list(original_dataset.signals.keys())[0]
                ts_name = series.name

            # Get the specific signal from the dataset
            target_signal = original_dataset.signals[sig_name].model_copy(deep=True)

            # Apply pattern_fill to the signal
            target_signal = target_signal.process(
                [ts_name],
                pattern_fill,
                pattern=pattern_arg,
                scaling="local",
                normalize=normalize,
                blend_minutes=blend_minutes,
            )

            # Update the dataset with the processed signal
            dataset = original_dataset.model_copy(deep=True)
            dataset._signals[sig_name] = target_signal

            # Save and export the entire dataset
            with tempfile.TemporaryDirectory() as tmpdir:
                dataset.save(tmpdir)
                zip_path = f"{tmpdir}/{dataset.name}.zip"
                shutil.make_archive(
                    zip_path.replace(".zip", ""),
                    "zip",
                    tmpdir,
                )

                with open(f"{zip_path}", "rb") as f:
                    zip_contents = f.read()

                b64 = base64.b64encode(zip_contents).decode()
                href = f"data:application/zip;base64,{b64}"

                download_link = mo.md(
                    f'<a href="{href}" download="{dataset.name}_filled.zip" style="'
                    "background-color: #4CAF50; color: white; padding: 10px 20px; "
                    'text-decoration: none; border-radius: 4px; font-size: 16px;">'
                    "📥 Download metEAUdata Dataset (ZIP)</a>"
                )

                return mo.vstack(
                    [
                        mo.md("### Download Processed Data"),
                        mo.md(
                            "Click the button below to download the processed data as a metEAUdata Dataset:"
                        ),
                        download_link,
                    ]
                )

        # Handle Signal case (existing Signal or create new one)
        # NOTE: Signal name must not contain underscores - metEAUdata uses underscore as delimiter
        signal_name = original_signal.name if original_signal else "pattern-filled"

        if import_type == "meteaudata" and original_signal is not None:
            # Use the original Signal - it already has all the data
            signal = original_signal.model_copy(deep=True)
            # Use the series name from the original data
            ts_name_to_process = series.name
        else:
            # Create a new Signal from the selected column only
            # When creating from scratch (CSV/DataFrame), just process the selected series

            # Clean the series name to avoid metEAUdata parsing issues with multiple #
            # metEAUdata expects names like "base#version", not "base#1#2"
            clean_series = df_all[series.name].copy()
            if '#' in str(series.name):
                # Extract just the base name before the first #
                base_name = str(series.name).split('#')[0]
                clean_series.name = base_name
            else:
                clean_series.name = str(series.name)

            signal = Signal(
                input_data=clean_series,
                name=signal_name,
                units="unknown",
            )
            ts_name_to_process = clean_series.name

        # Apply pattern_fill using Signal.process()
        signal = signal.process(
            [ts_name_to_process],
            pattern_fill,
            pattern=pattern_arg,
            scaling="local",
            normalize=normalize,
            blend_minutes=blend_minutes,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            signal.save(f"{tmpdir}/{signal_name}")
            zip_path = f"{tmpdir}/{signal_name}.zip"
            shutil.make_archive(
                zip_path.replace(".zip", ""),
                "zip",
                f"{tmpdir}/{signal_name}",
            )

            with open(f"{zip_path}", "rb") as f:
                zip_contents = f.read()

            b64 = base64.b64encode(zip_contents).decode()
            href = f"data:application/zip;base64,{b64}"

            download_link = mo.md(
                f'<a href="{href}" download="{signal_name}_filled.zip" style="'
                "background-color: #4CAF50; color: white; padding: 10px 20px; "
                'text-decoration: none; border-radius: 4px; font-size: 16px;">'
                "📥 Download metEAUdata Signal (ZIP)</a>"
            )

            return mo.vstack(
                [
                    mo.md("### Download Processed Data"),
                    mo.md(
                        "Click the button below to download the processed data as a metEAUdata Signal:"
                    ),
                    download_link,
                ]
            )

    create_meteaudata_download(
        series,
        df,
        _pat_arg,
        normalize_checkbox.value,
        blend_minutes_slider.value,
        import_type,
        original_signal,
        original_dataset,
    )
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

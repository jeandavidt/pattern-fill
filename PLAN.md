pattern-fill: Architecture & Development Roadmap
Context
Wastewater sensor data follows diurnal (daily) patterns — morning peak, slight workday decrease, dinner-time increase (5-8pm). Gaps in this data are common. This package provides gap-filling functions that leverage daily patterns, with tools for users to design those patterns interactively (fitting from data or manual spline design). The gap-filling function conforms to metEAUdata's SignalTransformFunctionProtocol, making it composable and metadata-tracked.

Package Structure

pattern-fill/
├── src/
│   └── pattern_fill/
│       ├── __init__.py          # Public API exports
│       ├── pattern.py           # DailyPattern dataclass + spline evaluation
│       ├── fitting.py           # Fit DailyPattern from observed time series
│       └── gap_fill.py          # pattern_fill() — SignalTransformFunctionProtocol
├── notebooks/
│   └── pattern_designer.py      # Marimo notebook (interactive pattern design)
├── tests/
│   ├── test_pattern.py
│   ├── test_fitting.py
│   └── test_gap_fill.py
├── .github/
│   └── workflows/
│       └── deploy-pages.yml     # GitHub Actions: WASM export + Pages deploy
├── pyproject.toml
├── .gitignore
├── .python-version
└── README.md
Core Design
DailyPattern (pattern.py)
A dataclass representing a periodic 24-hour curve via spline control points.

Fields: hours: list[float] (0-24), values: list[float] (0-1 normalized), name: str, periodic: bool = True, day_type: str = "all" (one of "all", "weekday", "weekend")
Spline: Uses scipy.interpolate.CubicSpline with bc_type='periodic' to ensure smooth midnight wrap-around
Key methods:
evaluate(hours: np.ndarray) -> np.ndarray — evaluate at arbitrary hour-of-day values
to_series(index: pd.DatetimeIndex) -> pd.Series — project pattern onto a real datetime index (bridge to gap filler)
to_dict() / from_dict() / to_json() / from_json() — full serialization for pipeline reuse
Control points are intentionally minimal (as few as 4, as many as 24). The cubic spline handles smoothness.
fitting.py
Extract a DailyPattern from existing data.

extract_daily_profile(series, resolution_minutes=15, aggregation="median") -> pd.Series — groups by fractional hour-of-day, computes median/mean per bin. Separated out because the notebook needs the raw profile for visualization.
fit_pattern(series, n_control_points=8, ...) -> DailyPattern — extracts profile, picks n evenly-spaced control points, normalizes to 0-1, returns a DailyPattern.
gap_fill.py
The main deliverable: pattern_fill() conforming to SignalTransformFunctionProtocol.

Signature (follows same pattern as metEAUdata's resample, linear_interpolation, replace_ranges):


def pattern_fill(
    input_series: list[pd.Series],
    pattern: DailyPattern | dict[str, DailyPattern],  # single or keyed by day_type
    scaling: str = "local",   # "local", "global", or "none"
    window: str = "24h",
    *args, **kwargs,
) -> list[tuple[pd.Series, list[ProcessingStep]]]:
pattern accepts either a single DailyPattern (applied to all days) or a dict keyed by day_type (e.g. {"weekday": pat1, "weekend": pat2}). When a dict is provided, each gap timestamp is routed to the appropriate pattern based on its day of week.

Gap-filling algorithm per series:

Identify contiguous NaN regions
For each gap, select the appropriate DailyPattern per timestamp (by day_type if dict provided)
Evaluate pattern.to_series(gap_index) to get the normalized shape
Boundary matching: compute an affine transform (scale + offset) so the pattern passes through the last valid value before the gap and the first valid value after the gap, ensuring continuity at edges
If only one boundary exists (gap at start or end of series), fall back to:
"local": use non-NaN data within window of the available boundary
"global": use the series' overall min/max
"none": raw 0-1 values
Fill NaN positions with transformed pattern values
metEAUdata integration: Creates FunctionInfo, Parameters (with pattern.to_dict() for full traceability), and ProcessingStep with type=ProcessingType.GAP_FILLING, suffix="PAT-FILL". The pattern's control points are stored in metadata, enabling full reproducibility.

Usage with metEAUdata:


signal.process(
    input_time_series_names=["raw"],
    transform_function=pattern_fill,
    pattern=my_pattern,
    scaling="local",
)
pyproject.toml changes
Add [build-system] using hatchling (uv delegates to it via uv build)
Add [tool.hatch.build.targets.wheel] for src/ layout
Core deps: numpy, pandas, scipy, meteaudata>=0.11.0
Optional [project.optional-dependencies]:
interactive: marimo>=0.19.8, wigglystuff>=0.2.21, matplotlib
dev: pytest, pytest-cov
Build/install with uv build and uv pip install -e ".[dev,interactive]"
Marimo Notebook (notebooks/pattern_designer.py)
Interactive notebook with this flow:

Upload data — mo.ui.file() (CSV/Parquet/Excel), column selection dropdowns
Visualize raw series — plot with NaN gaps highlighted
Day type selection — radio: "All days", "Separate weekday/weekend". If separate, the fit/design steps run for each day type with tabs.
Mode selection — radio: "Fit from data" vs "Design manually"
Fit path — slider for n_control_points, shows fitted pattern overlaid on daily profile
Manual design path — ChartPuck.from_callback() with draggable anchor points on a 0-24h / 0-1 plot. Spline redraws live as anchors move. Daily profile from data shown as background reference.
Gap-fill preview — applies pattern_fill(), shows before/after comparison
Export — download pattern(s) as JSON, display code snippet for pipeline usage
ChartPuck is anywidget-based, so it works in marimo via mo.ui.anywidget(puck). Marimo's reactivity means downstream cells (preview, export) automatically update when anchors are dragged.

GitHub Pages Deployment
Use marimo export html-wasm to produce a standalone WASM-powered HTML file (runs Python via Pyodide in the browser)
GitHub Actions workflow on push to main: install deps, export notebook, deploy via actions/deploy-pages@v4
scipy, numpy, pandas, matplotlib are all Pyodide-supported
wigglystuff (anywidget-based) works in WASM context
pattern-fill itself can be embedded via --include-code flag or published to PyPI
Development Roadmap
Phase 1: Core Library
Set up src/pattern_fill/ package structure, update pyproject.toml with build system and deps
Implement DailyPattern in pattern.py (dataclass, spline, serialization)
Implement extract_daily_profile() and fit_pattern() in fitting.py
Implement pattern_fill() in gap_fill.py (protocol-conforming gap filler)
Write tests with synthetic diurnal data (sine wave + noise + NaN gaps)
Phase 2: Interactive Notebook
Create notebooks/pattern_designer.py as a marimo notebook
Build data upload + parsing + column selection cells
Build auto-fit path (slider, profile visualization)
Build manual design path (ChartPuck + live spline)
Build gap-fill preview and JSON export cells
Phase 3: GitHub Pages
Create .github/workflows/deploy-pages.yml
Test WASM export locally
Handle Pyodide packaging for pattern-fill
Deploy and test end-to-end
Phase 4: Polish
README with installation, quick-start, link to live demo
Edge cases: timezone handling, irregular frequencies, very short/long gaps
Additional scaling strategies (e.g., linearly interpolating scale factor across the gap for even smoother transitions)
Verification
Unit tests: pytest tests/ — synthetic data round-trips, serialization, protocol conformance
Notebook: uv run marimo edit notebooks/pattern_designer.py — interactive testing with real or synthetic data
WASM export: uv run marimo export html-wasm notebooks/pattern_designer.py -o _site/ --mode run — verify all widgets work in browser
metEAUdata integration: create a Signal, call .process() with pattern_fill, verify ProcessingStep metadata is correctly stored
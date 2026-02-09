# pattern-fill Project Memory

## Project Overview
Small Python package for time series gap-filling using daily diurnal patterns, focused on wastewater treatment sensor data. Plan: `.claude/plans/rustling-hatching-quail.md`

## Key Architecture Decisions
- `src/pattern_fill/` layout with hatchling build backend, built via `uv build`
- `DailyPattern` dataclass with `day_type` field ("all"/"weekday"/"weekend") for weekday/weekend differentiation
- `pattern_fill()` accepts single pattern or `dict[str, DailyPattern]` keyed by day_type
- Boundary matching: affine transform to ensure gap fills are continuous at edges (not deferred)
- Core deps: numpy, pandas, scipy, meteaudata. Interactive extras: marimo, wigglystuff, matplotlib

## metEAUdata Protocol
- `SignalTransformFunctionProtocol` in `meteaudata/types.py` (NOT "SignalTransformationFunctionProtocol")
- Signature: `__call__(input_series: list[pd.Series], *args, **kwargs) -> list[tuple[pd.Series, list[ProcessingStep]]]`
- Must create: `FunctionInfo`, `Parameters`, `ProcessingStep` objects
- Reference implementations: see `meteaudata/processing_steps/univariate/` (resample.py, interpolate.py, replace.py)
- `Signal.process()` passes extra kwargs through to the transform function

## Phase 1 Status: COMPLETE
- `pattern.py`: DailyPattern dataclass with periodic CubicSpline, JSON serialization
- `fitting.py`: extract_daily_profile() (bin by fractional hour), fit_pattern() (evenly-spaced control points, 0-1 normalization)
- `gap_fill.py`: pattern_fill() conforming to SignalTransformFunctionProtocol, affine boundary matching, weekday/weekend dict support
- 39 tests passing

## Phase 2 Status: COMPLETE
- `notebooks/pattern_designer.py`: marimo notebook (18 cells)
- Flow: data source (demo/upload) → column selection → raw plot → config (day_type, mode, n_control_points) → fit or manual design → gap-fill preview → export JSON
- ChartPuck: basic `ChartPuck(fig)` with static background (silhouette + initial spline rendered once)
- **Performance fix**: replaced `from_callback` (re-rendered matplotlib on every mousemove) with basic constructor + `mo.state` + `mo.ui.button` pattern. Pucks drag smoothly via JS canvas; downstream cells only re-execute on "Apply Pattern" button click via `set_committed_patterns`.
- Weekday/weekend: separate puck_all/puck_weekday/puck_weekend raw ChartPuck objects; wrapped in `mo.ui.anywidget()` locally in display cell (not returned) so trait changes don't trigger reactivity
- `mo.state(fitted_patterns)` initialized with auto-fitted patterns for immediate gap-fill preview; updated via button on_click callback
- `mo.stop()` for conditional display without blocking variable returns
- Run with: `uv run --extra interactive marimo edit notebooks/pattern_designer.py`

## Gotcha: CubicSpline periodic BC
- `CubicSpline(bc_type="periodic")` requires y[0] == y[-1] exactly
- Fix: in `_build_spline`, append first value at hour 24 to close the loop before fitting

## Gotcha: marimo anywidget reactivity
- Pucks must be returned as separate named variables (not in a dict) for marimo to track value changes
- Always create pucks even when mode is "fit" (just don't display them); use `mo.stop()` in display-only cells
- `uv run --extra interactive` needed for matplotlib/wigglystuff/marimo deps
- **Performance**: Never use `ChartPuck.from_callback()` — it re-renders matplotlib on every mousemove via traitlet observer. Use basic `ChartPuck(fig)` with a pre-rendered static image instead.
- **Performance**: Use `mo.state` + `mo.ui.button(on_click=...)` to decouple puck widget display from downstream computation. Create `mo.ui.anywidget()` locally in display cell (don't return it) so trait changes don't propagate. Button callback reads raw `puck.x`/`puck.y` and commits to state. Note: `mo.ui.form()` does NOT work with anywidgets (NotImplementedError: widgets cannot be copied).

## wigglystuff Key Widget
- `ChartPuck.from_callback(draw_fn, x_bounds, y_bounds, ...)` — draggable anchors on matplotlib
- anywidget-based, works in marimo via `mo.ui.anywidget(puck)`
- draw_fn signature: `(ax, widget)` — widget.x / widget.y are lists of puck positions

## marimo Deployment
- `marimo export html-wasm` for GitHub Pages (Pyodide/WASM)
- `MarimoIslandGenerator` for embedding individual cells

# Pattern Fill

A metadata-aware implementation of pattern fill algorithms for gap-filling time series data using daily diurnal patterns.

## 🕹️ 📊 Try it Online

**[Launch Interactive Pattern Designer →](https://jeandavidt.github.io/pattern-fill/)**

Design and test patterns directly in your browser using our WASM-powered marimo notebook. No installation required!

## Features

- **Dual pattern modes**: Spline-based (traditional) and Sine wave-based (new)
- **Metadata awareness**: Track processing steps, preserve data provenance
- **Weekday/weekend patterns**: Different patterns for different day types
- **FFT-based auto-fitting**: Automatically extract sine components from data
- **Flexible API**: Easy-to-use factory methods and serialization

## Installation

```bash
uv pip install pattern-fill
```

Or with pip:
```bash
pip install pattern-fill
```

## Quick Start

### Sine Wave Patterns

Define patterns using intuitive sine wave parameters:

```python
from pattern_fill import DailyPattern, pattern_fill
import pandas as pd
import numpy as np

# Create a simple daily pattern peaking at 8 AM
pattern = DailyPattern.from_simple_sine(
    amplitude=0.4,    # Variation strength (0-1)
    frequency=1.0,    # Cycles per day
    phase=8.0,        # Peak time (hours)
    baseline=0.5      # Center value (0-1)
)

# Create time series with gaps
index = pd.date_range("2024-01-01", periods=96, freq="15min")
series = pd.Series(np.random.randn(96) + 10, index=index)
series.iloc[20:30] = np.nan

# Fill gaps using the pattern
results = pattern_fill([series], pattern=pattern)
filled_series, processing_steps = results[0]
```

### Complex Multi-Component Patterns

For wastewater treatment or environmental monitoring:

```python
# Complex pattern: daily + twice-daily variations
pattern = DailyPattern.from_sine_waves(
    components=[
        (0.35, 1.0, 8.0),   # Daily cycle, peak at 8 AM
        (0.15, 2.0, 13.0),  # Twice-daily, peaks at 1 PM and 1 AM
        (0.05, 1/7, 0.0),   # Weekly variation
    ],
    baseline=0.45,
    name="nh4_pattern"
)
```

### Auto-Fit from Data

```python
from pattern_fill import fit_sine_pattern

# Automatically extract sine components using FFT
pattern = fit_sine_pattern(
    clean_series,
    n_components=2,  # Number of components to fit
)

# Or specify frequencies explicitly
pattern = fit_sine_pattern(
    clean_series,
    frequencies=[1.0, 2.0],  # Daily + twice-daily
)
```

### Spline-Based Patterns (Traditional)

The traditional approach using control points:

```python
pattern = DailyPattern(
    hours=[0, 6, 12, 18],
    values=[0.2, 0.8, 0.9, 0.5],
    name="flow_pattern"
)
```

## Sine Wave Parameters

- **amplitude**: Controls the strength of variation (0-1 range)
- **frequency**: Cycles per day
  - `1.0` = once per day (24-hour cycle)
  - `2.0` = twice per day (12-hour cycle)
  - `0.5` = once every 2 days (48-hour cycle)
- **phase**: Time of peak in hours (0-24)
  - `0.0` = peak at midnight
  - `6.0` = peak at 6 AM
  - `12.0` = peak at noon
- **baseline**: Center value around which the sine oscillates (0-1 range)

## Weekday/Weekend Patterns

```python
patterns = {
    "weekday": DailyPattern.from_sine_waves(
        [(0.35, 1.0, 8.0)],
        baseline=0.5,
        day_type="weekday"
    ),
    "weekend": DailyPattern.from_sine_waves(
        [(0.25, 1.0, 10.0)],
        baseline=0.45,
        day_type="weekend"
    ),
}

results = pattern_fill([series], pattern=patterns)
```

## Serialization

Save and load patterns:

```python
# To JSON
json_str = pattern.to_json()

# From JSON
pattern2 = DailyPattern.from_json(json_str)

# To dict
d = pattern.to_dict()

# From dict
pattern3 = DailyPattern.from_dict(d)
```

## Benefits of Sine Patterns over Splines

- **Intuitive**: Parameters directly map to physical phenomena
- **Readable**: `DailyPattern.from_sine_waves([(0.35, 1.0, 8.0)])` immediately communicates "daily cycle peaking at 8 AM"
- **Composable**: Easily combine multiple periodicities
- **Natural**: Perfect for wastewater treatment and environmental monitoring patterns

## API Reference

### Main Functions

- `pattern_fill(input_series, pattern, ...)` - Fill gaps in time series
- `fit_pattern(series, n_control_points, ...)` - Auto-fit spline pattern from data
- `fit_sine_pattern(series, n_components, ...)` - Auto-fit sine pattern using FFT

### Classes

- `DailyPattern` - Main pattern class supporting both spline and sine modes
- `SineComponent` - Individual sine wave component

### Factory Methods

- `DailyPattern.from_sine_waves(components, baseline, ...)` - Create from sine components
- `DailyPattern.from_simple_sine(amplitude, frequency, phase, baseline)` - Create simple sine pattern

## Examples

See the [notebooks/](notebooks/) directory for interactive examples using the pattern designer.

## Development

```bash
# Install dependencies
uv sync

# Run tests
pytest tests/

# Run specific test file
pytest tests/test_sine_component.py -v
```

## References

- [Spline interpolation: Scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_interp_spline.html)
- [FFT-based sine fitting: Scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html)
- [AR model fitting:L Statsmodel](https://www.statsmodels.org/stable/generated/statsmodels.tsa.ar_model.AutoReg.html)
- [Stochastic gap filling: in *Influent generator: Towards realistic modelling of wastewater flowrate and water quality using machine-learning methods.*
PhD Thesis. Département de génie civil et de génie des eaux, Université Laval, Québec, QC, Canada. 161pp.](https://4e9b4375-0b81-4185-8a04-40043079b54e.usrfiles.com/ugd/4e9b43_c3a48614df824166b319bf6f14da6319.pdf)
- [Smooth daily pattern insertion in wastewater time series: in *Suivi, compréhension et modélisation d’une technologie à biofilm pour l’augmentation de la capacité des étangs aérés*
PhD Thesis. Département de génie civil et de génie des eaux, Université Laval, Québec, Canada. 214pp.](https://4e9b4375-0b81-4185-8a04-40043079b54e.usrfiles.com/ugd/4e9b43_d3f4086026c843c58f8a7b1585d36155.pdf)

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or pull request.

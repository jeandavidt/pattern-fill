"""Demo: Sine Wave Pattern Fill

This script demonstrates the new sine wave pattern feature for gap filling.
"""

import numpy as np
import pandas as pd
from pattern_fill import DailyPattern, pattern_fill, fit_sine_pattern

print("=" * 70)
print("Sine Wave Pattern Fill - Demo")
print("=" * 70)

# Create a time series with gaps
print("\n1. Creating test time series with gaps...")
index = pd.date_range("2024-01-01", periods=96, freq="15min")
series = pd.Series(
    np.sin(2 * np.pi * np.arange(96) / 96) * 5 + 10,
    index=index,
    name="NH4_concentration"
)
# Introduce gaps
series.iloc[20:30] = np.nan
series.iloc[60:70] = np.nan

print(f"   - Total points: {len(series)}")
print(f"   - Missing values: {series.isna().sum()}")
print(f"   - Data range: [{series.min():.2f}, {series.max():.2f}]")

# Method 1: Simple sine pattern
print("\n2. Creating simple sine pattern (peak at 8 AM)...")
simple_pattern = DailyPattern.from_simple_sine(
    amplitude=0.4,
    frequency=1.0,  # Once per day
    phase=8.0,      # Peak at 8 AM
    baseline=0.5,
    name="simple_daily"
)
print(f"   - Mode: {simple_pattern.mode}")
print(f"   - Components: {len(simple_pattern.sine_components)}")
print(f"   - Baseline: {simple_pattern.baseline}")

# Method 2: Multi-component pattern
print("\n3. Creating multi-component pattern...")
complex_pattern = DailyPattern.from_sine_waves(
    components=[
        (0.35, 1.0, 8.0),   # Daily cycle, peak at 8 AM
        (0.15, 2.0, 13.0),  # Twice-daily, peaks at 1 PM/AM
    ],
    baseline=0.45,
    name="wastewater_nh4"
)
print(f"   - Mode: {complex_pattern.mode}")
print(f"   - Components: {len(complex_pattern.sine_components)}")
for i, comp in enumerate(complex_pattern.sine_components, 1):
    print(f"     Wave {i}: amplitude={comp.amplitude:.2f}, "
          f"frequency={comp.frequency:.1f}, phase={comp.phase:.1f}h")

# Method 3: Auto-fit from data
print("\n4. Auto-fitting pattern from clean data using FFT...")
clean_data = series.dropna()
fitted_pattern = fit_sine_pattern(
    clean_data,
    n_components=2,
    name="auto_fitted"
)
print(f"   - Mode: {fitted_pattern.mode}")
print(f"   - Components detected: {len(fitted_pattern.sine_components)}")
print(f"   - Baseline: {fitted_pattern.baseline:.3f}")
for i, comp in enumerate(fitted_pattern.sine_components, 1):
    print(f"     Wave {i}: amplitude={comp.amplitude:.3f}, "
          f"frequency={comp.frequency:.1f}, phase={comp.phase:.1f}h")

# Fill gaps using the complex pattern
print("\n5. Filling gaps with multi-component pattern...")
results = pattern_fill([series], pattern=complex_pattern, scaling="local")
filled_series, processing_steps = results[0]

print(f"   - Gaps filled: {series.isna().sum()}")
print(f"   - Result name: {filled_series.name}")
print(f"   - Processing steps: {len(processing_steps)}")
print(f"   - Filled range: [{filled_series.min():.2f}, {filled_series.max():.2f}]")

# Serialization demo
print("\n6. Pattern serialization...")
json_str = complex_pattern.to_json()
print(f"   - JSON length: {len(json_str)} chars")
print(f"   - JSON preview: {json_str[:100]}...")

# Load from JSON
loaded_pattern = DailyPattern.from_json(json_str)
print(f"   - Loaded mode: {loaded_pattern.mode}")
print(f"   - Loaded components: {len(loaded_pattern.sine_components)}")

# Verify patterns are equivalent
test_hours = np.linspace(0, 24, 100)
np.testing.assert_allclose(
    complex_pattern.evaluate(test_hours),
    loaded_pattern.evaluate(test_hours),
    rtol=1e-10
)
print("   - ✓ Roundtrip successful!")

print("\n" + "=" * 70)
print("Demo complete! Sine wave patterns work perfectly.")
print("=" * 70)

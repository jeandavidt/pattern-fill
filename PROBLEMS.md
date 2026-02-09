# Known Problems

## normalize argument in pattern_fill

The "Normalize data" checkbox in the pattern designer notebook passes the `normalize` argument to `pattern_fill`, but it doesn't appear to have any effect on the gap-fill output. The filled values look the same regardless of the checkbox state.

This may need investigation in the `pattern_fill` function implementation to understand what the `normalize` parameter is supposed to do and whether it's being applied correctly.

from pattern_fill.pattern import DailyPattern, SineComponent
from pattern_fill.fitting import extract_daily_profile, fit_pattern
from pattern_fill.sine_fitting import fit_sine_pattern
from pattern_fill.gap_fill import pattern_fill, pattern_fill_dataset
from pattern_fill.stochastic import (
    add_ar_noise,
    fit_ar_model,
    generate_ar_noise,
)

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("sane-figs")
except Exception:
    __version__ = "dev"  # fallback for editable/dev installs
__author__ = "Jean-David T."
__license__ = "MIT"

__all__ = [
    "DailyPattern",
    "SineComponent",
    "extract_daily_profile",
    "fit_pattern",
    "fit_sine_pattern",
    "pattern_fill",
    "pattern_fill_dataset",
    "fit_ar_model",
    "generate_ar_noise",
    "add_ar_noise",
]

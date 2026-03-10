from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


@dataclass
class SineComponent:
    """A single sine wave component for building daily patterns.

    Parameters
    ----------
    amplitude : float
        The amplitude of the sine wave (normalized, 0-1 range recommended)
    frequency : float
        Frequency as cycles per day. Examples:
        - 1.0 = once per day (24-hour cycle)
        - 2.0 = twice per day (12-hour cycle)
        - 0.5 = once every 2 days (48-hour cycle)
    phase : float
        Phase offset in hours from midnight (0-24)
        Examples:
        - 0 = peak at midnight
        - 6 = peak at 6:00 AM
        - 12 = peak at noon
    """

    amplitude: float
    frequency: float
    phase: float = 0.0

    def __post_init__(self):
        if self.amplitude < 0:
            raise ValueError("amplitude must be non-negative")
        if self.frequency <= 0:
            raise ValueError("frequency must be positive")
        # Normalize phase to [0, 24) range
        self.phase = self.phase % 24.0

    def evaluate(self, hours: np.ndarray) -> np.ndarray:
        """Evaluate this sine component at given hours.

        Formula: amplitude * sin(2π * frequency * (hours - phase)/24 + π/2)
        This gives a peak at hour = phase.
        """
        hours = np.asarray(hours, dtype=float)
        # Shift so peak occurs at phase hour, using cosine (sine shifted by π/2)
        return self.amplitude * np.cos(
            2 * np.pi * self.frequency * (hours - self.phase) / 24.0
        )

    def to_dict(self) -> dict:
        return {
            "amplitude": self.amplitude,
            "frequency": self.frequency,
            "phase": self.phase,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SineComponent:
        return cls(
            amplitude=d["amplitude"],
            frequency=d["frequency"],
            phase=d.get("phase", 0.0),
        )


@dataclass
class GaussianComponent:
    """A single wrapped Gaussian component for building daily patterns.

    Parameters
    ----------
    amplitude : float
        Peak height above baseline (>= 0)
    center : float
        Hour of peak, normalized to [0, 24)
    width : float
        Standard deviation in hours (> 0)
    """

    amplitude: float
    center: float
    width: float

    def __post_init__(self):
        if self.amplitude < 0:
            raise ValueError("amplitude must be non-negative")
        if self.width <= 0:
            raise ValueError("width must be positive")
        self.center = self.center % 24.0

    def evaluate(self, hours: np.ndarray) -> np.ndarray:
        """Evaluate this wrapped Gaussian at given hours.

        Uses three shifted copies (k = -1, 0, 1) to ensure periodicity at 0/24.
        """
        hours = np.asarray(hours, dtype=float)
        result = np.zeros_like(hours)
        for k in (-1, 0, 1):
            result += np.exp(-0.5 * ((hours - self.center - 24.0 * k) / self.width) ** 2)
        return self.amplitude * result

    def to_dict(self) -> dict:
        return {"amplitude": self.amplitude, "center": self.center, "width": self.width}

    @classmethod
    def from_dict(cls, d: dict) -> GaussianComponent:
        return cls(amplitude=d["amplitude"], center=d["center"], width=d["width"])


@dataclass
class DailyPattern:
    """A periodic 24-hour curve defined by spline control points or sine waves.

    The pattern is normalized to the 0-1 range and can be created in two ways:

    **Spline Mode**:
        Cubic spline through user-defined control points (hours, values).

    **Sine Mode**:
        Sum of sine wave components with configurable amplitude, frequency, and phase.

    Examples
    --------
    Spline mode:

    >>> pattern = DailyPattern(
    ...     hours=[0, 6, 12, 18],
    ...     values=[0.2, 0.8, 0.9, 0.5]
    ... )

    Sine mode - simple single wave:

    >>> pattern = DailyPattern.from_simple_sine(
    ...     amplitude=0.4,
    ...     frequency=1.0,  # Once per day
    ...     phase=6.0,      # Peak at 6 AM
    ...     baseline=0.5
    ... )

    Sine mode - complex multi-component:

    >>> pattern = DailyPattern.from_sine_waves(
    ...     components=[
    ...         (0.35, 1.0, 8.0),   # Daily cycle, peak at 8 AM
    ...         (0.15, 2.0, 13.0),  # Twice-daily, peaks at 1 PM/AM
    ...     ],
    ...     baseline=0.45
    ... )
    """

    # Spline mode parameters (existing)
    hours: list[float] | None = None
    values: list[float] | None = None

    # Sine mode parameters
    sine_components: list[SineComponent] | None = None
    baseline: float = 0.5  # Baseline offset for sine/gaussian waves (0-1 range)

    # Gaussian mode parameters
    gaussian_components: list[GaussianComponent] | None = None

    # Common parameters
    name: str = "pattern"
    periodic: bool = True
    day_type: str = "all"  # "all", "weekday", or "weekend"

    # Mode tracking
    mode: str = field(init=False, repr=True)  # "spline", "sine", or "gaussian"
    _spline: CubicSpline | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        # Determine mode
        has_spline_params = self.hours is not None and self.values is not None
        has_sine_params = self.sine_components is not None
        has_gaussian_params = self.gaussian_components is not None

        n_modes = sum([has_spline_params, has_sine_params, has_gaussian_params])
        if n_modes > 1:
            raise ValueError(
                "Cannot specify parameters from more than one mode. Use one mode only."
            )
        if n_modes == 0:
            raise ValueError(
                "Must specify spline parameters (hours/values), sine_components, "
                "or gaussian_components"
            )

        if has_spline_params:
            self.mode = "spline"
            self._validate_spline_params()
            self._build_spline()
        elif has_sine_params:
            self.mode = "sine"
            self._validate_sine_params()
        else:
            self.mode = "gaussian"
            self._validate_gaussian_params()

        if self.day_type not in ("all", "weekday", "weekend"):
            raise ValueError(
                f"day_type must be 'all', 'weekday', or 'weekend', got {self.day_type!r}"
            )

    def _validate_spline_params(self) -> None:
        if len(self.hours) != len(self.values):
            raise ValueError(
                f"hours and values must have the same length, "
                f"got {len(self.hours)} and {len(self.values)}"
            )
        if len(self.hours) < 2:
            raise ValueError("Need at least 2 control points")

    def _validate_sine_params(self) -> None:
        if not self.sine_components:
            raise ValueError("sine_components cannot be empty")
        if not (0.0 <= self.baseline <= 1.0):
            raise ValueError(f"baseline must be in [0,1], got {self.baseline}")

    def _validate_gaussian_params(self) -> None:
        if not self.gaussian_components:
            raise ValueError("gaussian_components cannot be empty")
        if not (0.0 <= self.baseline <= 1.0):
            raise ValueError(f"baseline must be in [0,1], got {self.baseline}")

    def _build_spline(self) -> None:
        h = np.asarray(self.hours)
        v = np.asarray(self.values)
        order = np.argsort(h)
        h = h[order]
        v = v[order]

        if self.periodic:
            # CubicSpline requires y[0] == y[-1] for periodic BC.
            # Append the first value at hour 24 to close the loop.
            h = np.append(h, 24.0)
            v = np.append(v, v[0])
            self._spline = CubicSpline(h, v, bc_type="periodic")
        else:
            self._spline = CubicSpline(h, v)

    def evaluate(self, hours: np.ndarray) -> np.ndarray:
        """Evaluate the pattern at arbitrary hour-of-day values (0-24)."""
        hours = np.asarray(hours, dtype=float)

        if self.mode == "spline":
            result = self._spline(hours % 24.0)
        elif self.mode == "sine":
            result = np.full_like(hours, self.baseline, dtype=float)
            for component in self.sine_components:
                result += component.evaluate(hours % 24.0)
        else:  # gaussian mode
            result = np.full_like(hours, self.baseline, dtype=float)
            for component in self.gaussian_components:
                result += component.evaluate(hours % 24.0)

        # Clip to [0, 1] range (prevents spline overshoot and sine sum overflow)
        return np.clip(result, 0.0, 1.0)

    def to_series(self, index: pd.DatetimeIndex) -> pd.Series:
        """Project the pattern onto a real DatetimeIndex."""
        fractional_hours = index.hour + index.minute / 60.0 + index.second / 3600.0
        values = self.evaluate(fractional_hours.values)
        return pd.Series(values, index=index, name=self.name)

    # -- serialization --

    def to_dict(self) -> dict:
        """Serialize to dictionary, including mode information."""
        base = {
            "mode": self.mode,
            "name": self.name,
            "periodic": self.periodic,
            "day_type": self.day_type,
        }

        if self.mode == "spline":
            base.update({
                "hours": list(self.hours),
                "values": list(self.values),
            })
        elif self.mode == "sine":
            base.update({
                "sine_components": [c.to_dict() for c in self.sine_components],
                "baseline": self.baseline,
            })
        else:  # gaussian mode
            base.update({
                "gaussian_components": [c.to_dict() for c in self.gaussian_components],
                "baseline": self.baseline,
            })

        return base

    @classmethod
    def from_dict(cls, d: dict) -> DailyPattern:
        """Deserialize from dictionary, supporting both modes."""
        mode = d.get("mode", "spline")  # Default to spline for backward compatibility

        if mode == "spline":
            return cls(
                hours=d["hours"],
                values=d["values"],
                name=d.get("name", "pattern"),
                periodic=d.get("periodic", True),
                day_type=d.get("day_type", "all"),
            )
        elif mode == "sine":
            components = [SineComponent.from_dict(c) for c in d["sine_components"]]
            return cls(
                sine_components=components,
                baseline=d.get("baseline", 0.5),
                name=d.get("name", "pattern"),
                periodic=d.get("periodic", True),
                day_type=d.get("day_type", "all"),
            )
        else:  # gaussian mode
            components = [GaussianComponent.from_dict(c) for c in d["gaussian_components"]]
            return cls(
                gaussian_components=components,
                baseline=d.get("baseline", 0.0),
                name=d.get("name", "pattern"),
                periodic=d.get("periodic", True),
                day_type=d.get("day_type", "all"),
            )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> DailyPattern:
        return cls.from_dict(json.loads(s))

    # -- factory methods for sine mode --

    @classmethod
    def from_gaussians(
        cls,
        components: list[tuple[float, float, float]] | list[GaussianComponent],
        baseline: float = 0.0,
        name: str = "gaussian_pattern",
        day_type: str = "all",
    ) -> DailyPattern:
        """Create a DailyPattern from wrapped Gaussian components.

        Parameters
        ----------
        components : list of tuples or GaussianComponent objects
            If tuples: each is (amplitude, center, width)
            If GaussianComponent objects: used directly
        baseline : float
            Floor offset (0-1 range)
        name : str
            Pattern name
        day_type : str
            "all", "weekday", or "weekend"
        """
        gauss_comps = [
            c if isinstance(c, GaussianComponent) else GaussianComponent(*c)
            for c in components
        ]
        return cls(
            gaussian_components=gauss_comps,
            baseline=baseline,
            name=name,
            day_type=day_type,
        )

    @classmethod
    def from_sine_waves(
        cls,
        components: list[tuple[float, float, float]] | list[SineComponent],
        baseline: float = 0.5,
        name: str = "sine_pattern",
        day_type: str = "all",
    ) -> DailyPattern:
        """Create a DailyPattern from sine wave components.

        Parameters
        ----------
        components : list of tuples or SineComponent objects
            If tuples: each is (amplitude, frequency, phase)
            If SineComponent objects: used directly
        baseline : float
            Baseline offset (0-1 range)
        name : str
            Pattern name
        day_type : str
            "all", "weekday", or "weekend"

        Examples
        --------
        # Simple daily cycle with peak at 6 AM
        pattern = DailyPattern.from_sine_waves(
            components=[(0.5, 1.0, 6.0)],
            baseline=0.5
        )

        # Complex pattern: daily + twice-daily components
        pattern = DailyPattern.from_sine_waves(
            components=[
                (0.3, 1.0, 8.0),   # Daily cycle, peak at 8 AM
                (0.2, 2.0, 12.0),  # Twice-daily, peaks at noon and midnight
            ],
            baseline=0.4
        )
        """
        # Convert tuples to SineComponent objects if needed
        sine_comps = []
        for comp in components:
            if isinstance(comp, SineComponent):
                sine_comps.append(comp)
            else:
                amplitude, frequency, phase = comp
                sine_comps.append(SineComponent(amplitude, frequency, phase))

        return cls(
            sine_components=sine_comps,
            baseline=baseline,
            name=name,
            day_type=day_type,
        )

    @classmethod
    def from_simple_sine(
        cls,
        amplitude: float = 0.5,
        frequency: float = 1.0,
        phase: float = 6.0,
        baseline: float = 0.5,
        name: str = "simple_sine",
        day_type: str = "all",
    ) -> DailyPattern:
        """Create a simple single-sine-wave pattern.

        Convenience method for the most common case.

        Examples
        --------
        # Daily pattern peaking at 6 AM
        pattern = DailyPattern.from_simple_sine(
            amplitude=0.4,
            frequency=1.0,
            phase=6.0,
            baseline=0.5
        )
        """
        return cls.from_sine_waves(
            components=[(amplitude, frequency, phase)],
            baseline=baseline,
            name=name,
            day_type=day_type,
        )

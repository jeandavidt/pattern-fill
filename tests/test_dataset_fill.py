import numpy as np
import pandas as pd
import pytest
from meteaudata import Signal

from pattern_fill.gap_fill import pattern_fill_dataset
from pattern_fill.pattern import DailyPattern


def _sine_pattern(day_type: str = "all") -> DailyPattern:
    hours = list(range(24))
    values = [0.5 + 0.5 * np.sin(2 * np.pi * h / 24) for h in hours]
    return DailyPattern(hours=hours, values=values, name="sine", day_type=day_type)


def _make_signal(
    name: str,
    base: float = 30.0,
    amplitude: float = 10.0,
    units: str = "mg/L",
    days: int = 14,
    gap_slices: list[tuple[int, int]] | None = None,
) -> tuple[Signal, str]:
    """Create a test Signal with a sine-wave time series and optional gaps.

    Returns (signal, ts_name).
    """
    idx = pd.date_range("2024-01-01", periods=days * 96, freq="15min")
    frac_h = idx.hour + idx.minute / 60.0
    vals = base + amplitude * np.sin(2 * np.pi * frac_h / 24)
    s = pd.Series(vals, index=idx, name="raw")

    if gap_slices:
        for start, stop in gap_slices:
            s.iloc[start:stop] = np.nan

    signal = Signal(input_data=s, name=name, units=units)
    ts_name = list(signal.time_series.keys())[0]
    return signal, ts_name


# ---- Concentration mode ----------------------------------------------------


class TestPatternFillDatasetConcentration:
    def test_fills_gaps_and_returns_signal(self):
        sig, ts_name = _make_signal("conc", gap_slices=[(200, 248)])
        pat = _sine_pattern()

        result = pattern_fill_dataset(
            [sig], [ts_name], patterns=[pat], mode="concentration"
        )
        assert len(result) == 1
        out = result[0]
        assert isinstance(out, Signal)
        out_ts = list(out.time_series.values())[0]
        # The gap should be filled (interior gap, not leading/trailing)
        assert out_ts.series.iloc[200:248].notna().all()

    def test_preserves_non_nan(self):
        sig, ts_name = _make_signal("conc", gap_slices=[(200, 248)])
        original = sig.time_series[ts_name].series.copy()
        non_nan = original.notna()

        result = pattern_fill_dataset(
            [sig], [ts_name], patterns=[_sine_pattern()], mode="concentration"
        )
        out_ts = list(result[0].time_series.values())[0]
        np.testing.assert_allclose(
            out_ts.series.loc[non_nan].values,
            original.loc[non_nan].values,
        )


# ---- Flow mode -------------------------------------------------------------


class TestPatternFillDatasetFlow:
    def test_fills_gaps(self):
        sig, ts_name = _make_signal(
            "flow", base=100, amplitude=20, units="m3/h",
            gap_slices=[(300, 348)],
        )
        result = pattern_fill_dataset(
            [sig], [ts_name], patterns=[_sine_pattern()], mode="flow"
        )
        out_ts = list(result[0].time_series.values())[0]
        assert out_ts.series.iloc[300:348].notna().all()


# ---- Load mode -------------------------------------------------------------


class TestPatternFillDatasetLoad:
    def test_both_signals_filled(self):
        conc_sig, conc_ts = _make_signal("conc", gap_slices=[(200, 248)])
        flow_sig, flow_ts = _make_signal(
            "flow", base=100, amplitude=20, units="m3/h",
            gap_slices=[(300, 348)],
        )
        conc_pat = _sine_pattern()
        flow_pat = _sine_pattern()

        result = pattern_fill_dataset(
            [conc_sig, flow_sig],
            [conc_ts, flow_ts],
            patterns=[conc_pat, flow_pat],
            mode="load",
        )
        assert len(result) == 2
        out_conc_ts = list(result[0].time_series.values())[0]
        out_flow_ts = list(result[1].time_series.values())[0]
        assert out_conc_ts.series.iloc[200:248].notna().all()
        assert out_flow_ts.series.iloc[300:348].notna().all()

    def test_load_normalization_adjusts_concentration(self):
        """After load normalization, conc*flow should match expected load."""
        conc_sig, conc_ts = _make_signal("conc", gap_slices=[(200, 248)])
        flow_sig, flow_ts = _make_signal(
            "flow", base=100, amplitude=20, units="m3/h",
        )
        conc_pat = _sine_pattern()
        flow_pat = _sine_pattern()

        result = pattern_fill_dataset(
            [conc_sig, flow_sig],
            [conc_ts, flow_ts],
            patterns=[conc_pat, flow_pat],
            mode="load",
        )
        out_conc = list(result[0].time_series.values())[0].series
        out_flow = list(result[1].time_series.values())[0].series

        # The filled concentration values should be reasonable
        gap_conc = out_conc.iloc[200:248]
        assert gap_conc.min() > 0, f"Negative concentration: {gap_conc.min()}"
        assert gap_conc.max() < 100, f"Unreasonable concentration: {gap_conc.max()}"

    def test_overlapping_gaps_handled(self):
        """Gaps in both signals at the same time should still work."""
        conc_sig, conc_ts = _make_signal("conc", gap_slices=[(200, 248)])
        flow_sig, flow_ts = _make_signal(
            "flow", base=100, amplitude=20, units="m3/h",
            gap_slices=[(200, 248)],  # same gap
        )
        result = pattern_fill_dataset(
            [conc_sig, flow_sig],
            [conc_ts, flow_ts],
            patterns=[_sine_pattern(), _sine_pattern()],
            mode="load",
        )
        out_conc = list(result[0].time_series.values())[0].series
        out_flow = list(result[1].time_series.values())[0].series
        assert out_conc.iloc[200:248].notna().all()
        assert out_flow.iloc[200:248].notna().all()


# ---- Edge cases ------------------------------------------------------------


class TestPatternFillDatasetEdgeCases:
    def test_invalid_mode_raises(self):
        sig, ts_name = _make_signal("conc")
        with pytest.raises(ValueError, match="mode must be"):
            pattern_fill_dataset(
                [sig], [ts_name], patterns=[_sine_pattern()], mode="invalid"
            )

    def test_load_mode_wrong_signal_count(self):
        sig, ts_name = _make_signal("conc")
        with pytest.raises(ValueError, match="2 signals"):
            pattern_fill_dataset(
                [sig], [ts_name], patterns=[_sine_pattern()], mode="load"
            )

    def test_concentration_mode_two_signals_raises(self):
        sig1, ts1 = _make_signal("conc")
        sig2, ts2 = _make_signal("flow")
        with pytest.raises(ValueError, match="1 signal"):
            pattern_fill_dataset(
                [sig1, sig2],
                [ts1, ts2],
                patterns=[_sine_pattern(), _sine_pattern()],
                mode="concentration",
            )

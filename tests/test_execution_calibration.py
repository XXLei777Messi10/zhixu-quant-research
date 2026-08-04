from __future__ import annotations

import pandas as pd

from quant.execution.calibration import (
    apply_calibration,
    calibrate_from_oos_signals,
)

CONFIG = {
    "calibration": {
        "lookback_years": 3,
        "minimum_peer_samples": 50,
        "minimum_oos_model_score": 0.90,
        "quantiles": {
            "entry_low": 0.20,
            "entry_high": 0.70,
            "chase": 0.90,
            "add": 0.50,
            "favorable": 0.70,
            "favorable_high": 0.90,
            "adverse": 0.80,
            "hard_exit": 0.95,
        },
    }
}


def calibration_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    dates = pd.bdate_range("2025-01-01", periods=180)
    rows: list[dict] = []
    predictions: list[dict] = []
    for symbol_index, symbol in enumerate(("SH600000", "SZ000001")):
        for index, trade_date in enumerate(dates):
            close = 10.0 + symbol_index + index * 0.01
            gap = ((index % 7) - 3) * 0.002
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "hfq_open": close * (1.0 + gap),
                    "hfq_high": close * (1.015 + (index % 3) * 0.001),
                    "hfq_low": close * (0.985 - (index % 4) * 0.001),
                    "hfq_close": close,
                }
            )
            predictions.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "model_score": 0.95,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(predictions), dates[-1]


def test_completed_oos_signals_produce_calibration() -> None:
    bars, predictions, as_of = calibration_inputs()
    result = calibrate_from_oos_signals(bars, predictions, as_of, CONFIG)

    assert result["calibration_status"] == "WALK_FORWARD_PASS"
    assert result["sample_count"] >= CONFIG["calibration"]["minimum_peer_samples"]
    assert result["uses_only_completed_oos_signals"] is True
    assert result["parameters"]["entry_low_atr"] > 0
    assert result["parameters"]["peer_mfe_q70"] > 0
    assert result["parameters"]["peer_mfe_q90"] >= result["parameters"]["peer_mfe_q70"]
    assert result["parameters"]["peer_mae_q80"] > 0


def test_calibration_ignores_bars_after_as_of() -> None:
    bars, predictions, as_of = calibration_inputs()
    expected = calibrate_from_oos_signals(bars, predictions, as_of, CONFIG)
    future = pd.DataFrame(
        [
            {
                "symbol": "SH600000",
                "trade_date": pd.Timestamp("2030-01-01"),
                "hfq_open": 1000.0,
                "hfq_high": 2000.0,
                "hfq_low": 0.01,
                "hfq_close": 1000.0,
            }
        ]
    )
    actual = calibrate_from_oos_signals(
        pd.concat([bars, future], ignore_index=True),
        predictions,
        as_of,
        CONFIG,
    )

    assert actual == expected


def test_insufficient_calibration_does_not_mark_contexts_as_passed() -> None:
    bars, predictions, as_of = calibration_inputs()
    config = {
        "calibration": {
            **CONFIG["calibration"],
            "minimum_peer_samples": 1_000_000,
        }
    }
    result = calibrate_from_oos_signals(bars, predictions, as_of, config)
    contexts = {"SH600000": {"reference_close": 10.0}}

    assert result["calibration_status"] == "INSUFFICIENT_OOS_SAMPLES"
    assert apply_calibration(contexts, result) == contexts

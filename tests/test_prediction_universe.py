import pandas as pd

from quant.cli import _eligible_prediction_symbols


def test_prediction_excludes_symbols_without_point_in_time_history() -> None:
    calendar = pd.bdate_range("2025-01-01", periods=250)
    mature = pd.DataFrame(
        {
            "symbol": "SH600000",
            "trade_date": calendar,
            "amount": 50_000_000.0,
            "is_trading": True,
            "in_universe": True,
        }
    )
    incomplete = pd.DataFrame(
        {
            "symbol": "SH601111",
            "trade_date": [calendar[-1]],
            "amount": [100_000_000.0],
            "is_trading": [True],
            "in_universe": [True],
        }
    )
    config = {
        "fallback_universe": {
            "min_history_days": 250,
            "liquidity_window": 60,
            "min_median_amount_cny": 20_000_000,
            "max_symbols": 300,
        }
    }

    result = _eligible_prediction_symbols(
        pd.concat([mature, incomplete], ignore_index=True),
        calendar[-1],
        config,
    )

    assert result == {"SH600000"}

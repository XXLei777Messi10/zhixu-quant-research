import pandas as pd

from quant.data.universe import dynamic_universe


def test_dynamic_universe_uses_only_trailing_information() -> None:
    dates = pd.bdate_range("2020-01-01", periods=300)
    frame = pd.DataFrame(
        [
            {"trade_date": day, "symbol": symbol, "amount": amount, "is_trading": True}
            for day in dates
            for symbol, amount in (("SH600000", 30_000_000), ("SZ000001", 10_000_000))
        ]
    )
    result = dynamic_universe(frame, min_history_days=250, liquidity_window=60, max_symbols=300)
    assert set(result.symbol) == {"SH600000"}
    assert result.trade_date.min() >= dates[249]

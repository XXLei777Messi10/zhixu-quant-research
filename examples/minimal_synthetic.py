"""Run a small offline simulation without downloading market data."""

from __future__ import annotations

import pandas as pd

from quant.backtest.engine import AShareSimulator


def make_bars() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-06", periods=10)
    rows: list[dict[str, object]] = []
    for index, day in enumerate(dates):
        for symbol, base in (("SH600000", 10.0), ("SZ000001", 20.0)):
            open_price = base + index * 0.01
            close_price = base + index * 0.02
            rows.append(
                {
                    "trade_date": day,
                    "symbol": symbol,
                    "open": open_price,
                    "high": max(open_price, close_price) + 0.10,
                    "low": min(open_price, close_price) - 0.10,
                    "close": close_price,
                    "is_trading": True,
                    "is_st": False,
                }
            )
    return pd.DataFrame(rows)


def make_predictions(bars: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for day in sorted(bars["trade_date"].unique()):
        rows.extend(
            [
                {"trade_date": day, "symbol": "SH600000", "model_score": 0.9},
                {"trade_date": day, "symbol": "SZ000001", "model_score": 0.8},
            ]
        )
    return pd.DataFrame(rows)


def main() -> None:
    bars = make_bars()
    predictions = make_predictions(bars)
    config = {
        "initial_cash": 100_000.0,
        "top_k": 2,
        "max_weight": 0.45,
        "lot_size": 100,
        "commission_rate": 0.0003,
        "minimum_commission": 5.0,
        "transfer_fee_rate": 0.00002,
        "slippage_rate": 0.0005,
        "stamp_duty": [
            {"start": "1900-01-01", "end": "2099-12-31", "rate": 0.001}
        ],
    }
    result = AShareSimulator(config).run(bars, predictions)
    assert not result.nav.empty
    assert not result.trades.empty
    print("status: ok")
    print(f"trading_days: {len(result.nav)}")
    print(f"filled_trades: {len(result.trades)}")
    print(f"orders: {len(result.orders)}")
    print("live_data: false")


if __name__ == "__main__":
    main()

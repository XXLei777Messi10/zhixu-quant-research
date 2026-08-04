import pandas as pd

from quant.backtest.engine import Account, AShareSimulator

CONFIG = {
    "initial_cash": 1_000_000,
    "top_k": 2,
    "max_weight": 0.10,
    "lot_size": 100,
    "commission_rate": 0.0003,
    "minimum_commission": 5.0,
    "transfer_fee_rate": 0.00002,
    "slippage_rate": 0.0005,
    "stamp_duty": [{"start": "1900-01-01", "end": "2099-12-31", "rate": 0.001}],
}


def _bars() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=12)
    rows = []
    for day_index, day in enumerate(dates):
        for symbol, base in (("SH600000", 10.0), ("SZ000001", 20.0), ("SH600001", 30.0)):
            rows.append(
                {
                    "trade_date": day,
                    "symbol": symbol,
                    "open": base + day_index * 0.01,
                    "close": base + day_index * 0.02,
                    "high": base + 0.2,
                    "low": base - 0.2,
                    "is_trading": True,
                    "is_st": False,
                }
            )
    return pd.DataFrame(rows)


def _predictions() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=12)
    return pd.DataFrame(
        [
            {"trade_date": day, "symbol": symbol, "model_score": score}
            for day in dates
            for symbol, score in (("SH600000", 0.9), ("SZ000001", 0.8), ("SH600001", 0.1))
        ]
    )


def test_weights_lots_costs_and_cash() -> None:
    result = AShareSimulator(CONFIG).run(_bars(), _predictions())
    assert not result.trades.empty
    assert (result.trades["shares"] % 100 == 0).all()
    assert (result.trades["fees"] > 0).all()
    assert (result.nav["cash"] >= -1e-6).all()
    latest = result.holdings[result.holdings.trade_date == result.holdings.trade_date.max()]
    assert (latest.market_value / result.nav.iloc[-1].nav <= 0.101).all()


def test_t_plus_one_account_lots() -> None:
    account = Account(1000)
    account.add("SH600000", 100, pd.Timestamp("2026-01-05").date())
    assert account.sellable_shares("SH600000", pd.Timestamp("2026-01-05").date()) == 0
    assert account.sellable_shares("SH600000", pd.Timestamp("2026-01-06").date()) == 100


def test_suspension_and_limit_generate_rejections() -> None:
    bars = _bars()
    dates = sorted(bars.trade_date.unique())
    execution_day = dates[5]
    bars.loc[(bars.trade_date == execution_day) & (bars.symbol == "SH600000"), "is_trading"] = False
    result = AShareSimulator(CONFIG).run(bars, _predictions())
    assert "SUSPENDED" in set(result.orders.reason)


def test_market_high_risk_blocks_new_weekly_positions() -> None:
    predictions = _predictions()
    signal_dates = AShareSimulator._weekly_signal_dates(
        pd.DatetimeIndex(_bars().trade_date.unique())
    )
    first_signal = sorted(signal_dates)[0]
    predictions["market_state"] = "NORMAL"
    predictions["sector_state"] = "NORMAL"
    predictions.loc[predictions.trade_date.eq(first_signal), "market_state"] = "HIGH_RISK"

    result = AShareSimulator(CONFIG).run(_bars(), predictions)

    assert "RISK_FILTER" in set(result.orders.reason)
    first_execution = pd.Timestamp(first_signal) + pd.offsets.BDay(1)
    buys = result.trades[result.trades.side.eq("BUY")]
    assert not buys.empty
    assert pd.to_datetime(buys.trade_date).min() > first_execution


def test_sector_high_risk_blocks_only_affected_candidate() -> None:
    predictions = _predictions()
    predictions["market_state"] = "NORMAL"
    predictions["sector_state"] = "NORMAL"
    predictions.loc[predictions.symbol.eq("SH600000"), "sector_state"] = "HIGH_RISK"

    result = AShareSimulator(CONFIG).run(_bars(), predictions)
    bought = set(result.trades.loc[result.trades.side.eq("BUY"), "symbol"])

    assert "SH600000" not in bought
    assert "SZ000001" in bought
    assert "RISK_FILTER" in set(result.orders.reason)


def test_risk_filter_never_adds_to_existing_position() -> None:
    simulator = AShareSimulator({**CONFIG, "max_weight": 0.50, "top_k": 1})
    account = Account(9_000)
    account.add("SH600000", 100, pd.Timestamp("2026-01-05").date())
    day = pd.DataFrame(
        [
            {
                "symbol": "SH600000",
                "open": 5.0,
                "close": 5.0,
                "is_trading": True,
                "is_st": False,
            }
        ]
    ).set_index("symbol")
    trades: list[dict] = []
    orders: list[dict] = []

    simulator._rebalance(
        account,
        ["SH600000"],
        {"SH600000"},
        day,
        {"SH600000": 5.0},
        pd.Timestamp("2026-01-06").date(),
        trades,
        orders,
    )

    assert account.shares("SH600000") == 100
    assert not trades
    assert any(order["reason"] == "RISK_FILTER" for order in orders)

from __future__ import annotations

import pandas as pd

from quant.research.decision_attribution import HorizonCohortSimulator


def _execution() -> dict:
    return {
        "daily_open_simulation": {"initial_cash": 1_000_000},
        "max_daily_new_positions": 5,
        "commission_rate": 0.0003,
        "minimum_commission": 5.0,
        "transfer_fee_rate": 0.00002,
        "sell_tax_rate": [
            {"start": "1900-01-01", "end": "2099-12-31", "rate": 0.001}
        ],
        "slippage_rate": 0.0005,
        "lot_size": 100,
        "max_liquidity_participation": 0.01,
        "portfolio_policy": {
            "max_gross_exposure": 0.80,
            "minimum_cash_reserve": 0.20,
            "max_sector_exposure": 0.20,
        },
    }


def _research() -> dict:
    return {
        "top_pool_size": 20,
        "new_positions_per_day": 3,
        "holding_trading_days": 5,
        "cohort_position_cap": 15,
        "target_position_weight": 0.05,
        "staged_entry_fractions": [0.50, 0.25, 0.25],
    }


def _market(high_risk: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2020-01-02", periods=15)
    symbols = [f"SH600{index:03d}" for index in range(24)]
    bars = []
    predictions = []
    for day_index, day in enumerate(dates):
        for symbol_index, symbol in enumerate(symbols):
            price = 10.0 * (1.01**day_index) + symbol_index * 0.01
            bars.append(
                {
                    "trade_date": day,
                    "symbol": symbol,
                    "open": price,
                    "close": price,
                    "volume": 10_000_000,
                    "is_trading": True,
                    "is_st": False,
                }
            )
            predictions.append(
                {
                    "trade_date": day,
                    "symbol": symbol,
                    "model_score": 1.0 - symbol_index / 100,
                    "market_state": "HIGH_RISK" if high_risk else "NORMAL",
                    "sector_state": "NORMAL",
                    "sector_name": f"sector-{symbol_index // 4}",
                }
            )
    return pd.DataFrame(bars), pd.DataFrame(predictions)


def _run(
    *,
    staged: bool,
    constraints: bool,
    gate: bool,
    high_risk: bool = False,
):
    bars, predictions = _market(high_risk)
    return HorizonCohortSimulator(
        _research(),
        _execution(),
        staged_entry=staged,
        trading_constraints=constraints,
        risk_gate=gate,
        variant_name="test",
    ).run(bars, predictions)


def test_daily_cohort_holds_from_t_plus_1_open_to_t_plus_6_open() -> None:
    result = _run(staged=False, constraints=False, gate=False)
    first_buy = result.trades[result.trades["side"].eq("BUY")].iloc[0]
    first_sell = result.trades[
        result.trades["side"].eq("SELL")
        & result.trades["symbol"].eq(first_buy["symbol"])
    ].iloc[0]
    dates = pd.bdate_range("2020-01-02", periods=15)

    assert pd.Timestamp(first_buy["trade_date"]) == dates[1]
    assert pd.Timestamp(first_sell["trade_date"]) == dates[6]
    assert result.holdings.groupby("trade_date")["symbol"].nunique().max() <= 15


def test_staged_entry_uses_three_pre_registered_fractions() -> None:
    result = _run(staged=True, constraints=False, gate=False)
    first_symbol = result.trades[result.trades["side"].eq("BUY")].iloc[0]["symbol"]
    buys = result.trades[
        result.trades["side"].eq("BUY")
        & result.trades["symbol"].eq(first_symbol)
    ].sort_values("trade_date")

    assert buys["action"].head(3).tolist() == ["OPEN", "ADD", "ADD"]
    assert len(buys.head(3)) == 3


def test_trading_constraints_apply_lots_and_costs() -> None:
    result = _run(staged=False, constraints=True, gate=False)
    buys = result.trades[result.trades["side"].eq("BUY")]

    assert not buys.empty
    assert (buys["shares"] % 100 == 0).all()
    assert buys["fees"].gt(0).all()
    assert buys["slippage"].gt(0).all()


def test_hard_risk_gate_blocks_new_cohorts() -> None:
    result = _run(
        staged=False,
        constraints=True,
        gate=True,
        high_risk=True,
    )

    assert result.trades.empty
    assert result.orders["reason"].eq("RISK_GATE").any()

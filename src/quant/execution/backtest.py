from __future__ import annotations

from dataclasses import replace
from datetime import datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from quant.backtest.engine import Account, BacktestResult
from quant.execution.calibration import apply_calibration, calibrate_from_oos_signals
from quant.execution.daily import evaluate_daily_open
from quant.execution.planner import build_execution_plans, price_contexts_from_bars
from quant.execution.policy import execution_signal, portfolio_policy
from quant.execution.simulator import simulate_order


class ProductionMirrorSimulator:
    """Replay the same plan, open decision and fill functions used by daily simulation."""

    def __init__(
        self,
        execution_config: dict[str, Any],
        *,
        apply_risk_gate: bool,
        calibration_cache: dict[pd.Timestamp, dict[str, Any]] | None = None,
        exit_rank: int | None = None,
        risk_weight_scales: dict[str, dict[str, float]] | None = None,
        risk_scale_refresh: str = "daily",
        variant_name: str | None = None,
    ):
        self.config = execution_config
        self.apply_risk_gate = apply_risk_gate
        self.policy = portfolio_policy(execution_config)
        self.variant = variant_name or ("gated" if apply_risk_gate else "ungated")
        self.exit_rank = (
            int(exit_rank)
            if exit_rank is not None
            else int(self.policy["candidate_count"])
        )
        if self.exit_rank < int(self.policy["candidate_count"]):
            raise ValueError("Exit rank cannot be below the entry candidate count")
        self.risk_weight_scales = risk_weight_scales
        if risk_scale_refresh not in {"daily", "weekly"}:
            raise ValueError("Risk scale refresh must be daily or weekly")
        self.risk_scale_refresh = risk_scale_refresh
        self.calibration_cache = (
            calibration_cache if calibration_cache is not None else {}
        )

    def _risk_weight_scale(self, market_state: str, sector_state: str) -> float:
        if self.risk_weight_scales is None:
            return 1.0
        market = float(
            self.risk_weight_scales.get("market", {}).get(market_state, 0.0)
        )
        sector = float(
            self.risk_weight_scales.get("sector", {}).get(sector_state, 0.0)
        )
        return min(max(market, 0.0), max(sector, 0.0), 1.0)

    def _target_risk_scale(
        self,
        symbol: str,
        market_state: str,
        sector_state: str,
        *,
        pool_refreshed: bool,
        cache: dict[str, float],
    ) -> float:
        if self.risk_weight_scales is None:
            return 1.0
        current = self._risk_weight_scale(market_state, sector_state)
        if self.risk_scale_refresh == "daily":
            return current
        if pool_refreshed or symbol not in cache:
            cache[symbol] = current
        return cache[symbol]

    @staticmethod
    def _weekly_signal_dates(calendar: pd.DatetimeIndex) -> set[pd.Timestamp]:
        result: set[pd.Timestamp] = set()
        for index, current in enumerate(calendar):
            if index == len(calendar) - 1:
                continue
            current_iso = current.isocalendar()
            next_iso = calendar[index + 1].isocalendar()
            if (current_iso.year, current_iso.week) != (next_iso.year, next_iso.week):
                result.add(pd.Timestamp(current))
        return result

    @staticmethod
    def _marked_value(
        account: Account,
        prices: dict[str, float],
    ) -> tuple[float, float]:
        market_value = sum(
            account.shares(symbol) * float(prices.get(symbol, 0.0))
            for symbol in account.lots
        )
        return account.cash + market_value, market_value

    def run(self, bars: pd.DataFrame, predictions: pd.DataFrame) -> BacktestResult:
        required = {
            "symbol",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "is_trading",
            "is_st",
            "adjust_factor",
            "hfq_open",
            "hfq_high",
            "hfq_low",
            "hfq_close",
        }
        if missing := required - set(bars):
            raise ValueError(f"Production mirror bars missing: {sorted(missing)}")
        frame = bars.copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
        frame["symbol"] = frame["symbol"].astype(str)
        frame = frame.sort_values(["trade_date", "symbol"])
        scored = predictions.copy()
        scored["trade_date"] = pd.to_datetime(scored["trade_date"]).dt.normalize()
        scored["symbol"] = scored["symbol"].astype(str)
        calendar = pd.DatetimeIndex(sorted(frame["trade_date"].unique()))
        by_day = {
            pd.Timestamp(day): group.set_index("symbol")
            for day, group in frame.groupby("trade_date", sort=True)
        }
        predictions_by_day = {
            pd.Timestamp(day): group.sort_values(
                ["model_score", "symbol"],
                ascending=[False, True],
            )
            for day, group in scored.groupby("trade_date", sort=True)
        }
        account = Account(float(self.config["daily_open_simulation"]["initial_cash"]))
        last_close: dict[str, float] = {}
        prior_volume: dict[str, int] = {}
        pending_plans = []
        candidate_pool: set[str] | None = None
        risk_scale_cache: dict[str, float] = {}
        sector_by_symbol: dict[str, str] = {}
        nav_records: list[dict[str, Any]] = []
        holding_records: list[dict[str, Any]] = []
        trade_records: list[dict[str, Any]] = []
        order_records: list[dict[str, Any]] = []

        for calendar_index, trading_timestamp in enumerate(calendar):
            timestamp = pd.Timestamp(trading_timestamp)
            trading_date = timestamp.date()
            day = by_day[timestamp]
            open_prices = {
                str(symbol): float(row["open"])
                for symbol, row in day.iterrows()
                if pd.notna(row["open"]) and float(row["open"]) > 0
            }
            mark_at_open = {
                symbol: open_prices.get(symbol, last_close.get(symbol, 0.0))
                for symbol in set(account.lots) | set(open_prices)
            }
            portfolio_value, _ = self._marked_value(account, mark_at_open)
            observed_at = datetime.combine(
                trading_date,
                time(9, 30),
                tzinfo=ZoneInfo("Asia/Shanghai"),
            )
            daily_new_positions = 0
            add_counts: dict[str, int] = {}
            daily_turnover = 0.0
            for original_plan in sorted(
                pending_plans,
                key=lambda item: item.execution_priority,
            ):
                current_value = (
                    account.shares(original_plan.symbol)
                    * mark_at_open.get(original_plan.symbol, original_plan.reference_close)
                )
                current_weight = (
                    current_value / portfolio_value if portfolio_value > 0 else 0.0
                )
                plan = replace(
                    original_plan,
                    current_position_weight=current_weight,
                )
                decision = evaluate_daily_open(
                    plan,
                    open_prices.get(plan.symbol),
                    observed_at,
                    self.config,
                    self.variant,
                )
                gross_value = sum(
                    account.shares(symbol) * mark_at_open.get(symbol, 0.0)
                    for symbol in account.lots
                )
                sector = plan.sector_name or "DATA_UNAVAILABLE"
                sector_value = sum(
                    account.shares(symbol) * mark_at_open.get(symbol, 0.0)
                    for symbol in account.lots
                    if sector_by_symbol.get(symbol, "DATA_UNAVAILABLE") == sector
                )
                row = day.loc[plan.symbol] if plan.symbol in day.index else None
                order = simulate_order(
                    account,
                    plan,
                    decision,
                    open_prices.get(plan.symbol),
                    portfolio_value,
                    trading_date,
                    observed_at,
                    self.config,
                    is_trading=bool(row["is_trading"]) if row is not None else False,
                    is_st=bool(row["is_st"]) if row is not None else False,
                    previous_close=plan.reference_close,
                    available_liquidity_shares=prior_volume.get(plan.symbol),
                    quote_is_fresh=True,
                    execution_price_confirmed=plan.symbol in open_prices,
                    daily_new_positions=daily_new_positions,
                    daily_add_count=add_counts.get(plan.symbol, 0),
                    daily_turnover=daily_turnover,
                    current_gross_weight=(
                        gross_value / portfolio_value if portfolio_value > 0 else 0.0
                    ),
                    current_sector_weight=(
                        sector_value / portfolio_value if portfolio_value > 0 else 0.0
                    ),
                    current_position_count=sum(
                        account.shares(symbol) > 0 for symbol in account.lots
                    ),
                )
                record = order.to_dict()
                record.update(
                    {
                        "trade_date": trading_date,
                        "variant": self.variant,
                        "trigger_rule": decision.trigger_rule,
                    }
                )
                order_records.append(record)
                if order.fill_quantity:
                    side = (
                        "BUY"
                        if order.action in {"OPEN", "ADD"}
                        else "SELL"
                    )
                    trade_records.append(
                        {
                            "trade_date": trading_date,
                            "symbol": order.symbol,
                            "side": side,
                            "shares": order.fill_quantity,
                            "price": order.fill_price,
                            "gross": float(order.fill_price) * order.fill_quantity,
                            "fees": order.fees,
                            "slippage": order.slippage,
                            "action": order.action,
                            "variant": self.variant,
                        }
                    )
                    daily_turnover += float(order.planned_weight)
                    if order.action == "OPEN":
                        daily_new_positions += 1
                    elif order.action == "ADD":
                        add_counts[plan.symbol] = add_counts.get(plan.symbol, 0) + 1
            pending_plans = []

            for symbol, row in day.iterrows():
                if (
                    pd.notna(row["close"])
                    and float(row["close"]) > 0
                    and bool(row["is_trading"])
                ):
                    last_close[str(symbol)] = float(row["close"])
                if pd.notna(row["volume"]) and float(row["volume"]) >= 0:
                    prior_volume[str(symbol)] = int(float(row["volume"]))
            nav, market_value = self._marked_value(account, last_close)
            nav_records.append(
                {
                    "trade_date": timestamp,
                    "cash": account.cash,
                    "market_value": market_value,
                    "gross_exposure": market_value / nav if nav > 0 else 0.0,
                    "nav": nav,
                    "variant": self.variant,
                }
            )
            for symbol in sorted(account.lots):
                shares = account.shares(symbol)
                if shares <= 0:
                    continue
                close = last_close.get(symbol, np.nan)
                holding_records.append(
                    {
                        "trade_date": timestamp,
                        "symbol": symbol,
                        "shares": shares,
                        "sellable_shares": account.sellable_shares(symbol, trading_date),
                        "close": close,
                        "market_value": shares * close,
                        "sector_name": sector_by_symbol.get(symbol),
                        "variant": self.variant,
                    }
                )

            if timestamp not in predictions_by_day or calendar_index + 1 >= len(calendar):
                continue
            daily_scores = predictions_by_day[timestamp]
            valid = daily_scores[daily_scores["model_score"].notna()].copy()
            valid["signal_rank"] = np.arange(1, len(valid) + 1)
            next_timestamp = pd.Timestamp(calendar[calendar_index + 1])
            current_iso = timestamp.date().isocalendar()
            next_iso = next_timestamp.date().isocalendar()
            weekly_refresh = (current_iso.year, current_iso.week) != (
                next_iso.year,
                next_iso.week,
            )
            eligible_symbols = set(valid["symbol"].astype(str))
            pool_refreshed = candidate_pool is None or weekly_refresh
            if pool_refreshed:
                candidate_pool = set(
                    valid.head(int(self.policy["candidate_count"]))["symbol"].astype(str)
                )
            else:
                candidate_pool &= eligible_symbols
            selected_symbols = candidate_pool
            current_symbols = {
                symbol for symbol in account.lots if account.shares(symbol) > 0
            }
            plan_symbols = selected_symbols | current_symbols
            score_by_symbol = valid.set_index("symbol")
            signal_records = []
            for symbol in sorted(plan_symbols):
                if symbol in score_by_symbol.index:
                    row = score_by_symbol.loc[symbol]
                    rank = int(row["signal_rank"])
                    market_state = str(row.get("market_state", "DATA_UNAVAILABLE"))
                    sector_state = str(row.get("sector_state", "DATA_UNAVAILABLE"))
                    sector_name = row.get("sector_name")
                    score = float(row["model_score"])
                    probability = float(row.get("outperform_probability", 0.5))
                    predicted = float(row.get("predicted_excess_return", 0.0))
                    name = str(row.get("stock_name", symbol))
                    fold = str(row.get("fold", "UNKNOWN"))
                else:
                    rank = len(valid) + 1
                    market_state = "DATA_UNAVAILABLE"
                    sector_state = "DATA_UNAVAILABLE"
                    sector_name = sector_by_symbol.get(symbol)
                    score = -1.0
                    probability = 0.0
                    predicted = 0.0
                    name = symbol
                    fold = "OUT_OF_UNIVERSE"
                current_value = account.shares(symbol) * last_close.get(symbol, 0.0)
                current_weight = current_value / nav if nav > 0 else 0.0
                retained_by_rank = (
                    symbol in current_symbols
                    and symbol in score_by_symbol.index
                    and rank <= self.exit_rank
                )
                target = 0.0
                if symbol in selected_symbols or retained_by_rank:
                    target = float(self.policy["target_position_weight"])
                    target *= self._target_risk_scale(
                        symbol,
                        market_state,
                        sector_state,
                        pool_refreshed=pool_refreshed,
                        cache=risk_scale_cache,
                    )
                sector_by_symbol[symbol] = (
                    "DATA_UNAVAILABLE"
                    if sector_name is None or pd.isna(sector_name)
                    else str(sector_name)
                )
                signal_records.append(
                    {
                        "trade_date": trading_date.isoformat(),
                        "symbol": symbol,
                        "name": name,
                        "model_version": f"walk-forward-{fold}",
                        "model_signal": execution_signal(
                            rank,
                            current_weight,
                            target,
                            len(selected_symbols),
                        ),
                        "model_score": score,
                        "outperform_probability": probability,
                        "predicted_excess_return_5d": predicted,
                        "signal_rank": rank,
                        "current_position_weight": current_weight,
                        "target_position_weight": target,
                        "market_state": market_state,
                        "sector_state": sector_state,
                        "sector_name": sector_by_symbol[symbol],
                        "signal_valid_until": next_timestamp.date().isoformat(),
                        "data_quality_status": "PASS",
                    }
                )
            trailing = frame[
                frame["symbol"].isin(plan_symbols)
                & frame["trade_date"].between(
                    timestamp - pd.Timedelta(days=120),
                    timestamp,
                )
            ]
            contexts = price_contexts_from_bars(trailing)
            calibration = self.calibration_cache.get(timestamp)
            if calibration is None:
                lookback_start = timestamp - pd.DateOffset(
                    years=int(self.config["calibration"]["lookback_years"])
                )
                calibration_bars = frame[
                    frame["trade_date"].between(
                        lookback_start - pd.Timedelta(days=40),
                        timestamp,
                    )
                ]
                calibration_predictions = scored[
                    scored["trade_date"].between(lookback_start, timestamp)
                ]
                calibration = calibrate_from_oos_signals(
                    calibration_bars,
                    calibration_predictions,
                    timestamp,
                    self.config,
                )
                self.calibration_cache[timestamp] = calibration
            contexts = apply_calibration(contexts, calibration)
            available_signals = pd.DataFrame(signal_records)
            available_signals = available_signals[
                available_signals["symbol"].isin(contexts)
            ]
            pending_plans = build_execution_plans(
                available_signals,
                contexts,
                next_timestamp.date(),
                self.config,
                apply_risk_gate=self.apply_risk_gate,
            )

        return BacktestResult(
            pd.DataFrame(nav_records),
            pd.DataFrame(holding_records),
            pd.DataFrame(trade_records),
            pd.DataFrame(order_records),
        )

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from quant.backtest.rules import is_order_blocked_by_limit, stamp_duty_rate


@dataclass
class PositionLot:
    shares: int
    acquired_on: date


@dataclass
class Account:
    cash: float
    lots: dict[str, list[PositionLot]] = field(default_factory=dict)

    def shares(self, symbol: str) -> int:
        return sum(lot.shares for lot in self.lots.get(symbol, []))

    def sellable_shares(self, symbol: str, trade_date: date) -> int:
        return sum(lot.shares for lot in self.lots.get(symbol, []) if lot.acquired_on < trade_date)

    def add(self, symbol: str, shares: int, acquired_on: date) -> None:
        self.lots.setdefault(symbol, []).append(PositionLot(shares, acquired_on))

    def remove(self, symbol: str, shares: int, trade_date: date) -> None:
        remaining = shares
        lots = self.lots.get(symbol, [])
        for lot in lots:
            if lot.acquired_on >= trade_date or remaining <= 0:
                continue
            taken = min(lot.shares, remaining)
            lot.shares -= taken
            remaining -= taken
        self.lots[symbol] = [lot for lot in lots if lot.shares > 0]
        if remaining:
            raise ValueError("Attempted to sell non-sellable shares")


@dataclass
class BacktestResult:
    nav: pd.DataFrame
    holdings: pd.DataFrame
    trades: pd.DataFrame
    orders: pd.DataFrame


class AShareSimulator:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    def _fees(self, side: str, gross: float, trade_date: date) -> float:
        commission = max(
            float(self.config["minimum_commission"]), gross * float(self.config["commission_rate"])
        )
        transfer = gross * float(self.config["transfer_fee_rate"])
        stamp = gross * stamp_duty_rate(trade_date, self.config["stamp_duty"]) if side == "SELL" else 0.0
        return commission + transfer + stamp

    @staticmethod
    def _weekly_signal_dates(calendar: pd.DatetimeIndex) -> set[pd.Timestamp]:
        result: set[pd.Timestamp] = set()
        for index, current in enumerate(calendar):
            if (
                index == len(calendar) - 1
                or calendar[index + 1].isocalendar().week != current.isocalendar().week
            ):
                result.add(pd.Timestamp(current))
        return result

    def run(self, bars: pd.DataFrame, predictions: pd.DataFrame) -> BacktestResult:
        required = {"symbol", "trade_date", "open", "close", "is_trading", "is_st"}
        if missing := required - set(bars.columns):
            raise ValueError(f"Backtest bars missing: {sorted(missing)}")
        frame = bars.copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
        prediction_frame = predictions.copy()
        prediction_frame["trade_date"] = pd.to_datetime(prediction_frame["trade_date"]).dt.normalize()
        calendar = pd.DatetimeIndex(sorted(frame["trade_date"].unique()))
        signal_dates = self._weekly_signal_dates(calendar)
        bars_by_day = {
            pd.Timestamp(day): group.set_index("symbol")
            for day, group in frame.groupby("trade_date", sort=True)
        }
        predictions_by_day = {
            pd.Timestamp(day): group.sort_values("model_score", ascending=False)
            for day, group in prediction_frame.groupby("trade_date", sort=True)
        }
        previous_close: dict[str, float] = {}
        last_close: dict[str, float] = {}
        account = Account(float(self.config["initial_cash"]))
        pending_targets: list[str] | None = None
        pending_blocked_increases: set[str] = set()
        nav_records: list[dict[str, Any]] = []
        holding_records: list[dict[str, Any]] = []
        trade_records: list[dict[str, Any]] = []
        order_records: list[dict[str, Any]] = []

        for trading_timestamp in calendar:
            trading_date = pd.Timestamp(trading_timestamp).date()
            day = bars_by_day[pd.Timestamp(trading_timestamp)]
            if pending_targets is not None:
                self._rebalance(
                    account,
                    pending_targets,
                    pending_blocked_increases,
                    day,
                    previous_close,
                    trading_date,
                    trade_records,
                    order_records,
                )
                pending_targets = None
                pending_blocked_increases = set()
            for symbol, row in day.iterrows():
                if pd.notna(row["close"]) and bool(row["is_trading"]):
                    last_close[symbol] = float(row["close"])
            market_value = sum(
                account.shares(symbol) * last_close.get(symbol, 0.0) for symbol in account.lots
            )
            nav_records.append(
                {
                    "trade_date": trading_timestamp,
                    "cash": account.cash,
                    "market_value": market_value,
                    "nav": account.cash + market_value,
                }
            )
            for symbol in sorted(account.lots):
                shares = account.shares(symbol)
                if shares:
                    holding_records.append(
                        {
                            "trade_date": trading_timestamp,
                            "symbol": symbol,
                            "shares": shares,
                            "sellable_shares": account.sellable_shares(symbol, trading_date),
                            "close": last_close.get(symbol, np.nan),
                            "market_value": shares * last_close.get(symbol, 0.0),
                        }
                    )
            if (
                pd.Timestamp(trading_timestamp) in signal_dates
                and pd.Timestamp(trading_timestamp) in predictions_by_day
            ):
                candidates = predictions_by_day[pd.Timestamp(trading_timestamp)]
                valid = candidates[candidates["model_score"].notna()]
                selected = valid.head(int(self.config["top_k"]))
                pending_targets = selected["symbol"].astype(str).tolist()
                market_blocked = (
                    "market_state" in selected
                    and selected["market_state"].eq("HIGH_RISK").any()
                )
                if market_blocked:
                    pending_blocked_increases = set(pending_targets)
                elif "sector_state" in selected:
                    pending_blocked_increases = set(
                        selected.loc[
                            selected["sector_state"].isin(
                                {"HIGH_RISK", "DATA_UNAVAILABLE"}
                            ),
                            "symbol",
                        ].astype(str)
                    )
            for symbol, close in last_close.items():
                previous_close[symbol] = close

        return BacktestResult(
            pd.DataFrame(nav_records),
            pd.DataFrame(holding_records),
            pd.DataFrame(trade_records),
            pd.DataFrame(order_records),
        )

    def _rebalance(
        self,
        account: Account,
        targets: list[str],
        blocked_increases: set[str],
        day: pd.DataFrame,
        previous_close: dict[str, float],
        trade_date: date,
        trades: list[dict[str, Any]],
        orders: list[dict[str, Any]],
    ) -> None:
        marked_value = account.cash
        for symbol in account.lots:
            price = (
                float(day.loc[symbol, "open"])
                if symbol in day.index and pd.notna(day.loc[symbol, "open"])
                else previous_close.get(symbol, 0.0)
            )
            marked_value += account.shares(symbol) * price
        desired_weight = min(float(self.config["max_weight"]), 1.0 / len(targets) if targets else 0.0)
        target_values = {symbol: marked_value * desired_weight for symbol in targets}
        lot_size = int(self.config["lot_size"])
        slippage = float(self.config["slippage_rate"])

        desired_shares_by_symbol: dict[str, int] = {}
        for symbol in targets:
            if symbol not in day.index:
                continue
            open_price = (
                float(day.loc[symbol, "open"])
                if pd.notna(day.loc[symbol, "open"])
                else np.nan
            )
            raw_desired = (
                int(target_values[symbol] / open_price // lot_size * lot_size)
                if np.isfinite(open_price) and open_price > 0
                else 0
            )
            current_shares = account.shares(symbol)
            if symbol in blocked_increases and raw_desired > current_shares:
                orders.append(
                    self._rejection(
                        trade_date,
                        symbol,
                        "BUY",
                        "RISK_FILTER",
                        raw_desired - current_shares,
                    )
                )
                raw_desired = current_shares
            desired_shares_by_symbol[symbol] = raw_desired

        all_symbols = set(account.lots) | set(targets)
        for symbol in sorted(all_symbols):
            if symbol not in day.index:
                orders.append(self._rejection(trade_date, symbol, "SELL", "MISSING_BAR"))
                continue
            row = day.loc[symbol]
            open_price = float(row["open"]) if pd.notna(row["open"]) else np.nan
            current_shares = account.shares(symbol)
            desired_shares = desired_shares_by_symbol.get(symbol, 0)
            sell_shares = max(0, current_shares - desired_shares)
            if sell_shares:
                self._execute_sell(
                    account,
                    symbol,
                    sell_shares,
                    row,
                    previous_close.get(symbol),
                    trade_date,
                    slippage,
                    trades,
                    orders,
                )

        for symbol in targets:
            if symbol not in day.index:
                orders.append(self._rejection(trade_date, symbol, "BUY", "MISSING_BAR"))
                continue
            row = day.loc[symbol]
            open_price = float(row["open"]) if pd.notna(row["open"]) else np.nan
            if not np.isfinite(open_price) or open_price <= 0:
                orders.append(self._rejection(trade_date, symbol, "BUY", "INVALID_OPEN"))
                continue
            desired_shares = desired_shares_by_symbol.get(symbol, 0)
            buy_shares = max(0, desired_shares - account.shares(symbol))
            if buy_shares:
                self._execute_buy(
                    account,
                    symbol,
                    buy_shares,
                    row,
                    previous_close.get(symbol),
                    trade_date,
                    slippage,
                    lot_size,
                    trades,
                    orders,
                )

    def _execute_sell(
        self,
        account: Account,
        symbol: str,
        requested: int,
        row: pd.Series,
        previous_close: float | None,
        trade_date: date,
        slippage: float,
        trades: list[dict[str, Any]],
        orders: list[dict[str, Any]],
    ) -> None:
        if not bool(row["is_trading"]):
            orders.append(self._rejection(trade_date, symbol, "SELL", "SUSPENDED", requested))
            return
        sellable = account.sellable_shares(symbol, trade_date)
        if sellable <= 0:
            orders.append(self._rejection(trade_date, symbol, "SELL", "T_PLUS_ONE", requested))
            return
        shares = min(requested, sellable)
        open_price = float(row["open"])
        if previous_close and is_order_blocked_by_limit(
            "SELL", open_price, previous_close, symbol, trade_date, bool(row["is_st"])
        ):
            orders.append(self._rejection(trade_date, symbol, "SELL", "LIMIT_DOWN", shares))
            return
        price = open_price * (1.0 - slippage)
        gross = price * shares
        fees = self._fees("SELL", gross, trade_date)
        account.remove(symbol, shares, trade_date)
        account.cash += gross - fees
        trades.append(self._trade(trade_date, symbol, "SELL", shares, price, fees))
        orders.append(self._filled(trade_date, symbol, "SELL", shares))

    def _execute_buy(
        self,
        account: Account,
        symbol: str,
        requested: int,
        row: pd.Series,
        previous_close: float | None,
        trade_date: date,
        slippage: float,
        lot_size: int,
        trades: list[dict[str, Any]],
        orders: list[dict[str, Any]],
    ) -> None:
        if not bool(row["is_trading"]):
            orders.append(self._rejection(trade_date, symbol, "BUY", "SUSPENDED", requested))
            return
        open_price = float(row["open"])
        if previous_close and is_order_blocked_by_limit(
            "BUY", open_price, previous_close, symbol, trade_date, bool(row["is_st"])
        ):
            orders.append(self._rejection(trade_date, symbol, "BUY", "LIMIT_UP", requested))
            return
        price = open_price * (1.0 + slippage)
        shares = requested
        while shares >= lot_size:
            gross = price * shares
            fees = self._fees("BUY", gross, trade_date)
            if gross + fees <= account.cash + 1e-9:
                break
            shares -= lot_size
        if shares < lot_size:
            orders.append(self._rejection(trade_date, symbol, "BUY", "INSUFFICIENT_CASH", requested))
            return
        gross = price * shares
        fees = self._fees("BUY", gross, trade_date)
        account.cash -= gross + fees
        account.add(symbol, shares, trade_date)
        trades.append(self._trade(trade_date, symbol, "BUY", shares, price, fees))
        orders.append(self._filled(trade_date, symbol, "BUY", shares))

    @staticmethod
    def _trade(
        trade_date: date, symbol: str, side: str, shares: int, price: float, fees: float
    ) -> dict[str, Any]:
        return {
            "trade_date": trade_date,
            "symbol": symbol,
            "side": side,
            "shares": shares,
            "price": price,
            "gross": shares * price,
            "fees": fees,
        }

    @staticmethod
    def _filled(trade_date: date, symbol: str, side: str, shares: int) -> dict[str, Any]:
        return {
            "trade_date": trade_date,
            "symbol": symbol,
            "side": side,
            "requested_shares": shares,
            "status": "FILLED",
            "reason": "",
        }

    @staticmethod
    def _rejection(trade_date: date, symbol: str, side: str, reason: str, shares: int = 0) -> dict[str, Any]:
        return {
            "trade_date": trade_date,
            "symbol": symbol,
            "side": side,
            "requested_shares": shares,
            "status": "REJECTED",
            "reason": reason,
        }

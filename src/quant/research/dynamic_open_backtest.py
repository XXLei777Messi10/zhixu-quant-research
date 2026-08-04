from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.backtest.engine import BacktestResult
from quant.backtest.metrics import performance_metrics
from quant.backtest.rules import is_order_blocked_by_limit
from quant.config import ProjectPaths, load_config
from quant.execution.dynamic_policy import (
    allocate_dynamic_portfolio,
    quantize_open_scenario_gaps,
)
from quant.execution.simulator import execution_fees
from quant.signals.archive import immutable_write_json


@dataclass
class AdjustedLot:
    units: float
    acquired_on: date


@dataclass
class AdjustedAccount:
    cash: float
    lots: dict[str, list[AdjustedLot]] = field(default_factory=dict)

    def units(self, symbol: str) -> float:
        return sum(lot.units for lot in self.lots.get(symbol, []))

    def raw_shares(self, symbol: str, adjust_factor: float) -> int:
        return max(0, int(round(self.units(symbol) * adjust_factor)))

    def sellable_raw_shares(
        self,
        symbol: str,
        adjust_factor: float,
        trading_date: date,
    ) -> int:
        units = sum(
            lot.units
            for lot in self.lots.get(symbol, [])
            if lot.acquired_on < trading_date
        )
        # Never round sellable shares upward: doing so can request more adjusted
        # units than the account owns after a corporate-action factor change.
        return max(0, int(math.floor(units * adjust_factor + 1e-9)))

    def sellable_units(self, symbol: str, trading_date: date) -> float:
        return sum(
            lot.units
            for lot in self.lots.get(symbol, [])
            if lot.acquired_on < trading_date
        )

    def buy(
        self,
        symbol: str,
        raw_shares: int,
        adjust_factor: float,
        trading_date: date,
    ) -> None:
        self.lots.setdefault(symbol, []).append(
            AdjustedLot(raw_shares / adjust_factor, trading_date)
        )

    def sell(
        self,
        symbol: str,
        raw_shares: int,
        adjust_factor: float,
        trading_date: date,
    ) -> None:
        remaining_units = raw_shares / adjust_factor
        for lot in self.lots.get(symbol, []):
            if lot.acquired_on >= trading_date or remaining_units <= 1e-12:
                continue
            taken = min(lot.units, remaining_units)
            lot.units -= taken
            remaining_units -= taken
        self.lots[symbol] = [
            lot for lot in self.lots.get(symbol, []) if lot.units > 1e-10
        ]
        if remaining_units > 1e-6:
            raise ValueError("Attempted to sell non-sellable adjusted units")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _market_context(bars: pd.DataFrame) -> pd.DataFrame:
    frame = bars.sort_values(["symbol", "trade_date"]).copy()
    grouped = frame.groupby("symbol", sort=False)
    previous_close = grouped["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["atr_fraction"] = (
        true_range.groupby(frame["symbol"], sort=False)
        .rolling(20, min_periods=10)
        .mean()
        .reset_index(level=0, drop=True)
        / frame["close"]
    )
    adjusted_return = grouped["hfq_close"].pct_change(fill_method=None)
    frame["volatility_20"] = (
        adjusted_return.groupby(frame["symbol"], sort=False)
        .rolling(20, min_periods=10)
        .std()
        .reset_index(level=0, drop=True)
    )
    return frame[
        [
            "symbol",
            "trade_date",
            "close",
            "atr_fraction",
            "volatility_20",
        ]
    ].rename(columns={"close": "reference_close"})


def _risk_adjust_predictions(
    frame: pd.DataFrame,
    variant: dict[str, Any],
    horizons: list[int],
) -> pd.DataFrame:
    adjusted = frame.copy()
    market = adjusted.get(
        "market_state",
        pd.Series("DATA_UNAVAILABLE", index=adjusted.index),
    ).fillna("DATA_UNAVAILABLE")
    sector = adjusted.get(
        "sector_state",
        pd.Series("DATA_UNAVAILABLE", index=adjusted.index),
    ).fillna("DATA_UNAVAILABLE")
    market_scale = market.map(variant["market_state_multipliers"]).fillna(
        float(variant["market_state_multipliers"]["DATA_UNAVAILABLE"])
    )
    sector_scale = sector.map(variant["sector_state_multipliers"]).fillna(
        float(variant["sector_state_multipliers"]["DATA_UNAVAILABLE"])
    )
    adjusted["risk_state_multiplier"] = market_scale * sector_scale
    for horizon in horizons:
        prediction = f"predicted_excess_return_{horizon}d"
        probability = f"outperform_probability_{horizon}d"
        adjusted[prediction] = (
            pd.to_numeric(adjusted[prediction], errors="coerce")
            * adjusted["risk_state_multiplier"]
        )
        raw_probability = pd.to_numeric(adjusted[probability], errors="coerce")
        adjusted[probability] = (
            0.5 + (raw_probability - 0.5) * adjusted["risk_state_multiplier"]
        ).clip(0.0, 1.0)
    return adjusted


class DynamicOpenSimulator:
    def __init__(
        self,
        dynamic_config: dict[str, Any],
        execution_config: dict[str, Any],
        variant_name: str,
    ) -> None:
        self.dynamic = dynamic_config
        self.execution = execution_config
        self.variant_name = variant_name
        self.variant = dynamic_config["development_backtest"]["variants"][
            variant_name
        ]

    @staticmethod
    def _value(
        account: AdjustedAccount,
        day: pd.DataFrame,
        previous_hfq: dict[str, float],
        price_column: str,
    ) -> tuple[float, float]:
        market_value = 0.0
        for symbol in account.lots:
            if symbol in day.index and pd.notna(day.loc[symbol, price_column]):
                price = float(day.loc[symbol, price_column])
            else:
                price = previous_hfq.get(symbol, 0.0)
            market_value += account.units(symbol) * price
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
            "hfq_close",
        }
        if missing := required.difference(bars.columns):
            raise ValueError(f"Dynamic-open bars missing: {sorted(missing)}")
        frame = bars.copy()
        frame["symbol"] = frame["symbol"].astype(str)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
        frame = frame.sort_values(["trade_date", "symbol"])
        scored = predictions.copy()
        scored["symbol"] = scored["symbol"].astype(str)
        scored["trade_date"] = pd.to_datetime(scored["trade_date"]).dt.normalize()
        horizons = [int(value) for value in self.dynamic["horizons"]]
        scored = _risk_adjust_predictions(scored, self.variant, horizons)
        context = _market_context(frame)
        scored = scored.drop(
            columns=["reference_close", "atr_fraction", "volatility_20"],
            errors="ignore",
        )
        scored = scored.merge(
            context,
            on=["symbol", "trade_date"],
            how="left",
            validate="one_to_one",
        )
        prediction_start = pd.Timestamp(scored["trade_date"].min())
        prediction_end = pd.Timestamp(scored["trade_date"].max())
        calendar = pd.DatetimeIndex(
            sorted(
                frame.loc[
                    frame["trade_date"].between(prediction_start, prediction_end),
                    "trade_date",
                ].unique()
            )
        )
        by_day = {
            pd.Timestamp(day): group.set_index("symbol")
            for day, group in frame.groupby("trade_date", sort=True)
        }
        predictions_by_day = {
            pd.Timestamp(day): group
            for day, group in scored.groupby("trade_date", sort=True)
        }
        initial_cash = float(
            self.dynamic["development_backtest"]["initial_cash"]
        )
        account = AdjustedAccount(initial_cash)
        previous_raw_close: dict[str, float] = {}
        previous_hfq_close: dict[str, float] = {}
        previous_volume: dict[str, float] = {}
        pending_candidates: pd.DataFrame | None = None
        position_metadata: dict[str, dict[str, Any]] = {}
        nav_records: list[dict[str, Any]] = []
        holding_records: list[dict[str, Any]] = []
        trade_records: list[dict[str, Any]] = []
        order_records: list[dict[str, Any]] = []

        for calendar_index, timestamp in enumerate(calendar):
            timestamp = pd.Timestamp(timestamp)
            trading_date = timestamp.date()
            day = by_day[timestamp]
            open_value, _ = self._value(
                account,
                day,
                previous_hfq_close,
                "hfq_open",
            )
            if (
                pending_candidates is not None
                and open_value > 0
            ):
                benchmark_gap = 0.0
                raw_gaps: dict[str, float] = {}
                for symbol in pending_candidates["symbol"].astype(str):
                    if (
                        symbol in day.index
                        and symbol in previous_raw_close
                        and pd.notna(day.loc[symbol, "open"])
                        and previous_raw_close[symbol] > 0
                    ):
                        raw_gaps[symbol] = (
                            float(day.loc[symbol, "open"])
                            / previous_raw_close[symbol]
                            - 1.0
                        )
                if "SH000300" in day.index and "SH000300" in previous_raw_close:
                    benchmark_gap = (
                        float(day.loc["SH000300", "open"])
                        / previous_raw_close["SH000300"]
                        - 1.0
                    )
                relative_gaps = {
                    symbol: gap - benchmark_gap for symbol, gap in raw_gaps.items()
                }
                branch_gaps = quantize_open_scenario_gaps(
                    pending_candidates,
                    relative_gaps,
                    self.dynamic,
                )
                current_weights = {}
                for symbol in account.lots:
                    price = (
                        float(day.loc[symbol, "hfq_open"])
                        if symbol in day.index
                        and pd.notna(day.loc[symbol, "hfq_open"])
                        else previous_hfq_close.get(symbol, 0.0)
                    )
                    current_weights[symbol] = (
                        account.units(symbol) * price / open_value
                    )
                targets = allocate_dynamic_portfolio(
                    pending_candidates,
                    current_weights,
                    branch_gaps,
                    self.dynamic,
                )
                self._execute_targets(
                    account,
                    targets,
                    day,
                    previous_raw_close,
                    previous_volume,
                    open_value,
                    trading_date,
                    calendar_index,
                    position_metadata,
                    trade_records,
                    order_records,
                )
            pending_candidates = None

            close_value, market_value = self._value(
                account,
                day,
                previous_hfq_close,
                "hfq_close",
            )
            nav_records.append(
                {
                    "trade_date": timestamp,
                    "cash": account.cash,
                    "market_value": market_value,
                    "nav": close_value,
                    "gross_exposure": (
                        market_value / close_value if close_value > 0 else 0.0
                    ),
                    "variant": self.variant_name,
                }
            )
            for symbol in sorted(account.lots):
                if account.units(symbol) <= 1e-10:
                    continue
                factor = (
                    float(day.loc[symbol, "adjust_factor"])
                    if symbol in day.index
                    and pd.notna(day.loc[symbol, "adjust_factor"])
                    else 1.0
                )
                price = (
                    float(day.loc[symbol, "hfq_close"])
                    if symbol in day.index
                    and pd.notna(day.loc[symbol, "hfq_close"])
                    else previous_hfq_close.get(symbol, 0.0)
                )
                raw_shares = account.raw_shares(symbol, factor)
                if raw_shares <= 0:
                    continue
                holding_records.append(
                    {
                        "trade_date": timestamp,
                        "symbol": symbol,
                        "adjusted_units": account.units(symbol),
                        "raw_shares": raw_shares,
                        "market_value": account.units(symbol) * price,
                        "variant": self.variant_name,
                    }
                )

            if timestamp in predictions_by_day:
                all_candidates = predictions_by_day[timestamp].sort_values(
                    ["adaptive_model_score", "symbol"],
                    ascending=[False, True],
                )
                preselection = int(
                    self.dynamic["optimizer"]["candidate_preselection"]
                )
                candidates = all_candidates.head(preselection).copy()
                if self.dynamic["holding_persistence"][
                    "include_all_current_positions_in_candidate_set"
                ]:
                    current_symbols = {
                        symbol
                        for symbol in account.lots
                        if account.units(symbol) > 1e-10
                    }
                    incumbents = all_candidates[
                        all_candidates["symbol"].astype(str).isin(current_symbols)
                    ]
                    candidates = (
                        pd.concat([candidates, incumbents], ignore_index=True)
                        .drop_duplicates("symbol", keep="first")
                        .copy()
                    )
                candidates["holding_trading_days"] = candidates["symbol"].map(
                    {
                        symbol: calendar_index - int(metadata["entry_index"])
                        for symbol, metadata in position_metadata.items()
                    }
                )
                candidates["incumbent_horizon"] = candidates["symbol"].map(
                    {
                        symbol: int(metadata["selected_horizon"])
                        for symbol, metadata in position_metadata.items()
                    }
                )
                pending_candidates = candidates

            for symbol, row in day.iterrows():
                if pd.notna(row["close"]) and float(row["close"]) > 0:
                    previous_raw_close[str(symbol)] = float(row["close"])
                if pd.notna(row["hfq_close"]) and float(row["hfq_close"]) > 0:
                    previous_hfq_close[str(symbol)] = float(row["hfq_close"])
                if pd.notna(row["volume"]) and float(row["volume"]) >= 0:
                    previous_volume[str(symbol)] = float(row["volume"])

        return BacktestResult(
            pd.DataFrame(nav_records),
            pd.DataFrame(holding_records),
            pd.DataFrame(trade_records),
            pd.DataFrame(order_records),
        )

    def _execute_targets(
        self,
        account: AdjustedAccount,
        targets: pd.DataFrame,
        day: pd.DataFrame,
        previous_close: dict[str, float],
        previous_volume: dict[str, float],
        portfolio_value: float,
        trading_date: date,
        calendar_index: int,
        position_metadata: dict[str, dict[str, Any]],
        trades: list[dict[str, Any]],
        orders: list[dict[str, Any]],
    ) -> None:
        lot_size = int(self.dynamic["execution"]["lot_size"])
        slippage = float(self.execution["slippage_rate"])
        participation = float(self.execution["max_liquidity_participation"])
        target_by_symbol = targets.set_index("symbol")

        # Sales are evaluated first. A rejected sale never creates buying power.
        for symbol in sorted(account.lots):
            target_weight = (
                float(target_by_symbol.loc[symbol, "target_weight"])
                if symbol in target_by_symbol.index
                else 0.0
            )
            if symbol not in day.index:
                orders.append(
                    self._order(trading_date, symbol, "SELL", "REJECTED", "MISSING_BAR")
                )
                continue
            row = day.loc[symbol]
            factor = float(row["adjust_factor"])
            open_price = float(row["open"])
            current_shares = account.raw_shares(symbol, factor)
            target_shares = int(
                portfolio_value * target_weight / open_price // lot_size * lot_size
            )
            requested = max(0, current_shares - target_shares)
            if requested <= 0:
                self._settle_adjusted_residual(
                    account,
                    symbol,
                    target_weight,
                    row,
                    trading_date,
                    position_metadata,
                    trades,
                    orders,
                )
                continue
            sellable = account.sellable_raw_shares(
                symbol,
                factor,
                trading_date,
            )
            quantity = min(requested, sellable)
            if quantity < requested:
                orders.append(
                    self._order(
                        trading_date,
                        symbol,
                        "SELL",
                        "PARTIALLY_FILLED" if quantity else "REJECTED",
                        "T_PLUS_ONE",
                        requested,
                        quantity,
                    )
                )
            reason = self._blocked_reason(
                "SELL",
                symbol,
                row,
                previous_close.get(symbol),
                trading_date,
            )
            if reason or quantity <= 0:
                if reason:
                    orders.append(
                        self._order(
                            trading_date,
                            symbol,
                            "SELL",
                            "REJECTED",
                            reason,
                            requested,
                        )
                    )
                continue
            fill = open_price * (1.0 - slippage)
            gross = fill * quantity
            fees = execution_fees("SELL", gross, trading_date, self.execution)
            account.sell(symbol, quantity, factor, trading_date)
            account.cash += gross - fees
            trades.append(
                self._trade(
                    trading_date,
                    symbol,
                    "SELL",
                    quantity,
                    fill,
                    fees,
                )
            )
            self._settle_adjusted_residual(
                account,
                symbol,
                target_weight,
                row,
                trading_date,
                position_metadata,
                trades,
                orders,
            )
            if account.units(symbol) <= 1e-10:
                position_metadata.pop(symbol, None)

        # Purchases compete by marginal utility and use only cash actually available.
        buys = targets[targets["delta_weight"].gt(0)].sort_values(
            ["utility", "symbol"],
            ascending=[False, True],
        )
        for _, target in buys.iterrows():
            symbol = str(target["symbol"])
            if symbol not in day.index:
                orders.append(
                    self._order(trading_date, symbol, "BUY", "REJECTED", "MISSING_BAR")
                )
                continue
            row = day.loc[symbol]
            factor = float(row["adjust_factor"])
            open_price = float(row["open"])
            reason = self._blocked_reason(
                "BUY",
                symbol,
                row,
                previous_close.get(symbol),
                trading_date,
            )
            if reason:
                orders.append(
                    self._order(trading_date, symbol, "BUY", "REJECTED", reason)
                )
                continue
            current_shares = account.raw_shares(symbol, factor)
            target_shares = int(
                portfolio_value
                * float(target["target_weight"])
                / open_price
                // lot_size
                * lot_size
            )
            requested = max(0, target_shares - current_shares)
            liquidity = int(
                previous_volume.get(symbol, 0.0)
                * participation
                // lot_size
                * lot_size
            )
            quantity = min(requested, liquidity)
            fill = open_price * (1.0 + slippage)
            while quantity >= lot_size:
                gross = fill * quantity
                fees = execution_fees("BUY", gross, trading_date, self.execution)
                if gross + fees <= account.cash + 1e-9:
                    break
                quantity -= lot_size
            if quantity < lot_size:
                orders.append(
                    self._order(
                        trading_date,
                        symbol,
                        "BUY",
                        "REJECTED",
                        "CASH_OR_LIQUIDITY",
                        requested,
                    )
                )
                continue
            gross = fill * quantity
            fees = execution_fees("BUY", gross, trading_date, self.execution)
            account.cash -= gross + fees
            if account.cash < -1e-7:
                raise AssertionError("Dynamic simulation created negative cash")
            account.buy(symbol, quantity, factor, trading_date)
            if current_shares <= 0 or symbol not in position_metadata:
                position_metadata[symbol] = {
                    "entry_index": calendar_index,
                    "selected_horizon": int(target["selected_horizon"]),
                }
            trades.append(
                self._trade(
                    trading_date,
                    symbol,
                    "BUY",
                    quantity,
                    fill,
                    fees,
                )
            )
            status = "FILLED" if quantity == requested else "PARTIALLY_FILLED"
            orders.append(
                self._order(
                    trading_date,
                    symbol,
                    "BUY",
                    status,
                    "DYNAMIC_TARGET",
                    requested,
                    quantity,
                )
            )

    def _settle_adjusted_residual(
        self,
        account: AdjustedAccount,
        symbol: str,
        target_weight: float,
        row: pd.Series,
        trading_date: date,
        position_metadata: dict[str, dict[str, Any]],
        trades: list[dict[str, Any]],
        orders: list[dict[str, Any]],
    ) -> None:
        if (
            not self.dynamic["holding_persistence"][
                "sweep_subshare_adjusted_residual_to_cash"
            ]
            or target_weight > 1e-12
            or account.units(symbol) <= 1e-10
        ):
            return
        factor = float(row["adjust_factor"])
        if account.raw_shares(symbol, factor) >= 1:
            return
        if (
            account.sellable_units(symbol, trading_date)
            + 1e-10
            < account.units(symbol)
        ):
            return
        reason = self._blocked_reason(
            "SELL",
            symbol,
            row,
            float(row["close"]) if pd.notna(row["close"]) else None,
            trading_date,
        )
        if reason in {"SUSPENDED", "INVALID_OPEN"}:
            return
        value = account.units(symbol) * float(row["hfq_open"])
        if value <= 0:
            return
        account.cash += value
        account.lots.pop(symbol, None)
        position_metadata.pop(symbol, None)
        trades.append(
            {
                "trade_date": trading_date,
                "symbol": symbol,
                "side": "RESIDUAL_CASH_SETTLEMENT",
                "shares": 0,
                "price": float(row["open"]),
                "gross": value,
                "fees": 0.0,
                "variant": self.variant_name,
            }
        )
        orders.append(
            self._order(
                trading_date,
                symbol,
                "SELL",
                "FILLED",
                "ADJUSTED_SUBSHARE_CASH_SETTLEMENT",
            )
        )

    @staticmethod
    def _blocked_reason(
        side: str,
        symbol: str,
        row: pd.Series,
        previous_close: float | None,
        trading_date: date,
    ) -> str | None:
        if not bool(row["is_trading"]):
            return "SUSPENDED"
        open_price = float(row["open"])
        if not np.isfinite(open_price) or open_price <= 0:
            return "INVALID_OPEN"
        if previous_close is None or previous_close <= 0:
            return "MISSING_PREVIOUS_CLOSE"
        if is_order_blocked_by_limit(
            side,
            open_price,
            previous_close,
            symbol,
            trading_date,
            bool(row["is_st"]),
        ):
            return "LIMIT_UP" if side == "BUY" else "LIMIT_DOWN"
        return None

    def _trade(
        self,
        trading_date: date,
        symbol: str,
        side: str,
        shares: int,
        price: float,
        fees: float,
    ) -> dict[str, Any]:
        return {
            "trade_date": trading_date,
            "symbol": symbol,
            "side": side,
            "shares": shares,
            "price": price,
            "gross": price * shares,
            "fees": fees,
            "variant": self.variant_name,
        }

    def _order(
        self,
        trading_date: date,
        symbol: str,
        side: str,
        status: str,
        reason: str,
        requested: int = 0,
        filled: int = 0,
    ) -> dict[str, Any]:
        return {
            "trade_date": trading_date,
            "symbol": symbol,
            "side": side,
            "status": status,
            "reason": reason,
            "requested_shares": requested,
            "filled_shares": filled,
            "variant": self.variant_name,
        }


def _diagnostics(result: BacktestResult) -> dict[str, Any]:
    nav = result.nav
    counts = (
        result.holdings.groupby("trade_date")["symbol"].nunique()
        if not result.holdings.empty
        else pd.Series(dtype=float)
    )
    return {
        "average_cash_ratio": float((nav["cash"] / nav["nav"]).mean()),
        "minimum_cash_ratio": float((nav["cash"] / nav["nav"]).min()),
        "average_gross_exposure": float(nav["gross_exposure"].mean()),
        "maximum_gross_exposure": float(nav["gross_exposure"].max()),
        "average_holdings": float(counts.reindex(nav["trade_date"]).fillna(0).mean()),
        "maximum_holdings": int(counts.max()) if len(counts) else 0,
        "ending_holdings": int(counts.iloc[-1]) if len(counts) else 0,
    }


def run_dynamic_open_research(
    paths: ProjectPaths,
    config_name: str = "dynamic_decision_research",
) -> dict[str, Any]:
    dynamic = load_config(paths, config_name)
    execution = load_config(paths, "execution")
    data = load_config(paths, "data")
    research = dynamic["development_backtest"]
    start = pd.Timestamp(research["start"])
    end = pd.Timestamp(research["end"])
    predictions_path = paths.reports / "research" / str(
        research["prediction_file"]
    )
    bars_path = paths.curated / "bars.parquet"
    predictions = pd.read_parquet(
        predictions_path,
        filters=[("trade_date", ">=", start), ("trade_date", "<=", end)],
    )
    bars = pd.read_parquet(
        bars_path,
        columns=[
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
            "hfq_close",
        ],
        filters=[
            ("trade_date", ">=", start - pd.Timedelta(days=60)),
            ("trade_date", "<=", end),
        ],
    )
    benchmark_symbol = str(data["benchmark_symbol"])
    benchmark = bars[bars["symbol"].astype(str).eq(benchmark_symbol)][
        ["trade_date", "close"]
    ].copy()
    results: dict[str, Any] = {}
    artifact_root = (
        paths.reports / "research" / str(research["artifact_directory"])
    )
    if artifact_root.exists():
        raise FileExistsError(f"{artifact_root} already exists; use a revision")
    for variant in research["variants"]:
        simulator = DynamicOpenSimulator(dynamic, execution, str(variant))
        result = simulator.run(bars, predictions)
        variant_root = artifact_root / str(variant)
        variant_root.mkdir(parents=True, exist_ok=False)
        result.nav.to_parquet(variant_root / "nav.parquet", index=False)
        result.holdings.to_parquet(variant_root / "holdings.parquet", index=False)
        result.trades.to_parquet(variant_root / "trades.parquet", index=False)
        result.orders.to_parquet(variant_root / "orders.parquet", index=False)
        metrics = performance_metrics(result.nav, benchmark, result.trades)
        metrics["diagnostics"] = _diagnostics(result)
        results[str(variant)] = metrics
    report = {
        "status": "DYNAMIC_OPEN_DEVELOPMENT_COMPLETE",
        "created_at": datetime.now(UTC).isoformat(),
        "development_period": [start.date().isoformat(), end.date().isoformat()],
        "portfolio_constraints": {
            "single_name_hard_cap": None,
            "sector_hard_cap": None,
            "minimum_cash_reserve": None,
            "position_count_cap": None,
            "maximum_gross_exposure": 1.0,
            "reason": "No leverage and no negative cash",
        },
        "decision_timing": "T close plan -> T+1 official open branch -> deterministic allocation",
        "results": results,
        "locked_period_read": False,
        "input_versions": {
            "predictions": str(predictions_path),
            "predictions_sha256": _sha256(predictions_path),
            "bars": str(bars_path),
            "bars_sha256": _sha256(bars_path),
        },
        "simulation_only": True,
    }
    report_path = immutable_write_json(
        paths.reports / "research" / str(research["report_file"]),
        report,
    )
    return {**report, "report": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", default="dynamic_decision_research")
    args = parser.parse_args()
    result = run_dynamic_open_research(
        ProjectPaths(args.root.resolve()),
        str(args.config),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

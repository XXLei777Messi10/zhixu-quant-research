from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.backtest.engine import BacktestResult
from quant.backtest.metrics import performance_metrics
from quant.backtest.rules import is_order_blocked_by_limit
from quant.config import ProjectPaths, load_config
from quant.execution.backtest import ProductionMirrorSimulator
from quant.execution.simulator import execution_fees
from quant.signals.archive import immutable_write_json

BAR_COLUMNS = [
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
]


@dataclass
class CohortPosition:
    shares: float
    exit_index: int
    sector_name: str


@dataclass(frozen=True)
class PendingCohortOrder:
    due_index: int
    symbol: str
    action: str
    fraction: float
    exit_index: int
    sector_name: str
    signal_date: pd.Timestamp


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class HorizonCohortSimulator:
    """Daily signal cohorts aligned to the T+1-open through T+6-open label."""

    def __init__(
        self,
        research_config: dict[str, Any],
        execution_config: dict[str, Any],
        *,
        staged_entry: bool,
        trading_constraints: bool,
        risk_gate: bool,
        variant_name: str,
    ) -> None:
        self.research = research_config
        self.execution = execution_config
        self.staged_entry = staged_entry
        self.trading_constraints = trading_constraints
        self.risk_gate = risk_gate
        self.variant = variant_name

    def _entry_fractions(self) -> list[float]:
        if not self.staged_entry:
            return [1.0]
        fractions = [
            float(value) for value in self.research["staged_entry_fractions"]
        ]
        if not np.isclose(sum(fractions), 1.0):
            raise ValueError("Staged entry fractions must total one")
        return fractions

    @staticmethod
    def _risk_blocked(row: pd.Series) -> bool:
        return str(row.get("market_state", "DATA_UNAVAILABLE")) == "HIGH_RISK" or str(
            row.get("sector_state", "DATA_UNAVAILABLE")
        ) in {"HIGH_RISK", "DATA_UNAVAILABLE"}

    @staticmethod
    def _marked_value(
        cash: float,
        positions: dict[str, CohortPosition],
        prices: dict[str, float],
    ) -> tuple[float, float]:
        market_value = sum(
            position.shares * float(prices.get(symbol, 0.0))
            for symbol, position in positions.items()
        )
        return cash + market_value, market_value

    def run(self, bars: pd.DataFrame, predictions: pd.DataFrame) -> BacktestResult:
        required = {
            "symbol",
            "trade_date",
            "open",
            "close",
            "volume",
            "is_trading",
            "is_st",
        }
        if missing := required - set(bars.columns):
            raise ValueError(f"Horizon cohort bars missing: {sorted(missing)}")
        frame = bars.copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
        frame["symbol"] = frame["symbol"].astype(str)
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

        cash = float(self.execution["daily_open_simulation"]["initial_cash"])
        positions: dict[str, CohortPosition] = {}
        pending: list[PendingCohortOrder] = []
        last_close: dict[str, float] = {}
        previous_close: dict[str, float] = {}
        prior_volume: dict[str, int] = {}
        nav_records: list[dict[str, Any]] = []
        holding_records: list[dict[str, Any]] = []
        trade_records: list[dict[str, Any]] = []
        order_records: list[dict[str, Any]] = []

        for calendar_index, timestamp in enumerate(calendar):
            timestamp = pd.Timestamp(timestamp)
            trading_date = timestamp.date()
            day = by_day[timestamp]
            open_prices = {
                str(symbol): float(row["open"])
                for symbol, row in day.iterrows()
                if pd.notna(row["open"]) and float(row["open"]) > 0
            }
            mark_at_open = {
                symbol: open_prices.get(symbol, last_close.get(symbol, 0.0))
                for symbol in set(positions) | set(open_prices)
            }

            for symbol in sorted(
                [
                    value
                    for value, position in positions.items()
                    if position.exit_index <= calendar_index
                ]
            ):
                row = day.loc[symbol] if symbol in day.index else None
                open_price = open_prices.get(symbol)
                position = positions[symbol]
                blocked_reason = self._blocked_reason(
                    "SELL",
                    symbol,
                    row,
                    open_price,
                    previous_close.get(symbol),
                    trading_date,
                )
                if blocked_reason:
                    position.exit_index = calendar_index + 1
                    order_records.append(
                        self._order_record(
                            trading_date,
                            symbol,
                            "EXIT",
                            "REJECTED",
                            blocked_reason,
                        )
                    )
                    continue
                slippage_rate = (
                    float(self.execution["slippage_rate"])
                    if self.trading_constraints
                    else 0.0
                )
                fill_price = float(open_price) * (1.0 - slippage_rate)
                gross = fill_price * position.shares
                fees = (
                    execution_fees("SELL", gross, trading_date, self.execution)
                    if self.trading_constraints
                    else 0.0
                )
                cash += gross - fees
                trade_records.append(
                    self._trade_record(
                        trading_date,
                        symbol,
                        "SELL",
                        position.shares,
                        fill_price,
                        fees,
                        abs(fill_price - float(open_price)) * position.shares,
                        "EXIT",
                    )
                )
                order_records.append(
                    self._order_record(
                        trading_date,
                        symbol,
                        "EXIT",
                        "FILLED",
                        "",
                    )
                )
                del positions[symbol]

            due = sorted(
                [order for order in pending if order.due_index == calendar_index],
                key=lambda order: (order.action != "OPEN", order.symbol),
            )
            pending = [order for order in pending if order.due_index > calendar_index]
            daily_new_positions = 0
            for order in due:
                if order.action == "ADD" and order.symbol not in positions:
                    order_records.append(
                        self._order_record(
                            trading_date,
                            order.symbol,
                            "ADD",
                            "REJECTED",
                            "INITIAL_ENTRY_NOT_FILLED",
                        )
                    )
                    continue
                if order.action == "OPEN" and order.symbol in positions:
                    order_records.append(
                        self._order_record(
                            trading_date,
                            order.symbol,
                            "OPEN",
                            "CANCELLED",
                            "POSITION_ALREADY_EXISTS",
                        )
                    )
                    continue
                row = day.loc[order.symbol] if order.symbol in day.index else None
                open_price = open_prices.get(order.symbol)
                blocked_reason = self._blocked_reason(
                    "BUY",
                    order.symbol,
                    row,
                    open_price,
                    previous_close.get(order.symbol),
                    trading_date,
                )
                if blocked_reason:
                    order_records.append(
                        self._order_record(
                            trading_date,
                            order.symbol,
                            order.action,
                            "REJECTED",
                            blocked_reason,
                        )
                    )
                    continue
                if (
                    self.trading_constraints
                    and order.action == "OPEN"
                    and daily_new_positions
                    >= int(self.execution["max_daily_new_positions"])
                ):
                    order_records.append(
                        self._order_record(
                            trading_date,
                            order.symbol,
                            order.action,
                            "REJECTED",
                            "MAX_DAILY_NEW_POSITIONS",
                        )
                    )
                    continue
                portfolio_value, market_value = self._marked_value(
                    cash,
                    positions,
                    mark_at_open,
                )
                current_position = positions.get(order.symbol)
                current_value = (
                    current_position.shares * float(open_price)
                    if current_position is not None
                    else 0.0
                )
                target_weight = float(self.research["target_position_weight"])
                desired_value = min(
                    portfolio_value * target_weight * order.fraction,
                    max(0.0, portfolio_value * target_weight - current_value),
                )
                sector_value = sum(
                    position.shares * mark_at_open.get(symbol, 0.0)
                    for symbol, position in positions.items()
                    if position.sector_name == order.sector_name
                )
                shares, fill_price, reason = self._buy_quantity(
                    desired_value,
                    cash,
                    portfolio_value,
                    market_value,
                    sector_value,
                    order.symbol,
                    float(open_price),
                    prior_volume.get(order.symbol),
                    trading_date,
                )
                if shares <= 0:
                    order_records.append(
                        self._order_record(
                            trading_date,
                            order.symbol,
                            order.action,
                            "REJECTED",
                            reason,
                        )
                    )
                    continue
                gross = fill_price * shares
                fees = (
                    execution_fees("BUY", gross, trading_date, self.execution)
                    if self.trading_constraints
                    else 0.0
                )
                cash -= gross + fees
                if current_position is None:
                    positions[order.symbol] = CohortPosition(
                        shares=shares,
                        exit_index=order.exit_index,
                        sector_name=order.sector_name,
                    )
                    daily_new_positions += 1
                else:
                    current_position.shares += shares
                trade_records.append(
                    self._trade_record(
                        trading_date,
                        order.symbol,
                        "BUY",
                        shares,
                        fill_price,
                        fees,
                        abs(fill_price - float(open_price)) * shares,
                        order.action,
                    )
                )
                order_records.append(
                    self._order_record(
                        trading_date,
                        order.symbol,
                        order.action,
                        "FILLED",
                        "",
                    )
                )

            for symbol, row in day.iterrows():
                if (
                    pd.notna(row["close"])
                    and float(row["close"]) > 0
                    and bool(row["is_trading"])
                ):
                    last_close[str(symbol)] = float(row["close"])
                if pd.notna(row["volume"]) and float(row["volume"]) >= 0:
                    prior_volume[str(symbol)] = int(float(row["volume"]))
            nav, market_value = self._marked_value(cash, positions, last_close)
            nav_records.append(
                {
                    "trade_date": timestamp,
                    "cash": cash,
                    "market_value": market_value,
                    "gross_exposure": market_value / nav if nav > 0 else 0.0,
                    "nav": nav,
                    "variant": self.variant,
                }
            )
            for symbol, position in sorted(positions.items()):
                close = last_close.get(symbol, np.nan)
                holding_records.append(
                    {
                        "trade_date": timestamp,
                        "symbol": symbol,
                        "shares": position.shares,
                        "close": close,
                        "market_value": position.shares * close,
                        "sector_name": position.sector_name,
                        "exit_index": position.exit_index,
                        "variant": self.variant,
                    }
                )

            if calendar_index + 1 >= len(calendar) or timestamp not in predictions_by_day:
                previous_close.update(last_close)
                continue
            valid = predictions_by_day[timestamp]
            valid = valid[valid["model_score"].notna()].head(
                int(self.research["top_pool_size"])
            )
            unavailable = set(positions) | {
                order.symbol for order in pending if order.due_index > calendar_index
            }
            available_slots = max(
                0,
                int(self.research["cohort_position_cap"]) - len(unavailable),
            )
            selected = valid[~valid["symbol"].isin(unavailable)].head(
                min(
                    int(self.research["new_positions_per_day"]),
                    available_slots,
                )
            )
            fractions = self._entry_fractions()
            for _, row in selected.iterrows():
                symbol = str(row["symbol"])
                if self.risk_gate and self._risk_blocked(row):
                    order_records.append(
                        self._order_record(
                            calendar[calendar_index + 1].date(),
                            symbol,
                            "OPEN",
                            "CANCELLED",
                            "RISK_GATE",
                        )
                    )
                    continue
                sector = (
                    "DATA_UNAVAILABLE"
                    if pd.isna(row.get("sector_name"))
                    else str(row.get("sector_name"))
                )
                entry_index = calendar_index + 1
                exit_index = entry_index + int(
                    self.research["holding_trading_days"]
                )
                for offset, fraction in enumerate(fractions):
                    pending.append(
                        PendingCohortOrder(
                            due_index=entry_index + offset,
                            symbol=symbol,
                            action="OPEN" if offset == 0 else "ADD",
                            fraction=fraction,
                            exit_index=exit_index,
                            sector_name=sector,
                            signal_date=timestamp,
                        )
                    )
            previous_close.update(last_close)

        return BacktestResult(
            pd.DataFrame(nav_records),
            pd.DataFrame(holding_records),
            pd.DataFrame(trade_records),
            pd.DataFrame(order_records),
        )

    def _blocked_reason(
        self,
        side: str,
        symbol: str,
        row: pd.Series | None,
        open_price: float | None,
        prior_close: float | None,
        trading_date: date,
    ) -> str:
        if row is None or open_price is None or open_price <= 0:
            return "MISSING_OR_INVALID_OPEN"
        if not bool(row["is_trading"]):
            return "SUSPENDED"
        if (
            self.trading_constraints
            and prior_close is not None
            and prior_close > 0
            and is_order_blocked_by_limit(
                side,
                float(open_price),
                float(prior_close),
                symbol,
                trading_date,
                bool(row["is_st"]),
            )
        ):
            return "LIMIT_UP" if side == "BUY" else "LIMIT_DOWN"
        return ""

    def _buy_quantity(
        self,
        desired_value: float,
        cash: float,
        portfolio_value: float,
        market_value: float,
        sector_value: float,
        symbol: str,
        open_price: float,
        prior_volume: int | None,
        trading_date: date,
    ) -> tuple[float, float, str]:
        del symbol
        slippage_rate = (
            float(self.execution["slippage_rate"])
            if self.trading_constraints
            else 0.0
        )
        fill_price = open_price * (1.0 + slippage_rate)
        capacity = min(desired_value, cash)
        if self.trading_constraints:
            policy = self.execution["portfolio_policy"]
            capacity = min(
                capacity,
                max(
                    0.0,
                    portfolio_value * float(policy["max_gross_exposure"])
                    - market_value,
                ),
                max(
                    0.0,
                    cash
                    - portfolio_value * float(policy["minimum_cash_reserve"]),
                ),
                max(
                    0.0,
                    portfolio_value * float(policy["max_sector_exposure"])
                    - sector_value,
                ),
            )
            lot = int(self.execution["lot_size"])
            shares = int(capacity / fill_price // lot * lot)
            if prior_volume is not None:
                liquid = int(
                    prior_volume
                    * float(self.execution["max_liquidity_participation"])
                    // lot
                    * lot
                )
                shares = min(shares, liquid)
            while shares >= lot:
                gross = fill_price * shares
                fees = execution_fees(
                    "BUY",
                    gross,
                    trading_date,
                    self.execution,
                )
                if gross + fees <= cash + 1e-9:
                    break
                shares -= lot
            return (
                float(shares),
                fill_price,
                "" if shares >= lot else "CASH_EXPOSURE_SECTOR_OR_LIQUIDITY_CAP",
            )
        shares = capacity / fill_price if fill_price > 0 else 0.0
        return shares, fill_price, "" if shares > 0 else "INSUFFICIENT_CASH"

    def _order_record(
        self,
        trade_date: date,
        symbol: str,
        action: str,
        status: str,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "trade_date": trade_date,
            "symbol": symbol,
            "action": action,
            "status": status,
            "reason": reason,
            "variant": self.variant,
        }

    def _trade_record(
        self,
        trade_date: date,
        symbol: str,
        side: str,
        shares: float,
        price: float,
        fees: float,
        slippage: float,
        action: str,
    ) -> dict[str, Any]:
        return {
            "trade_date": trade_date,
            "symbol": symbol,
            "side": side,
            "shares": shares,
            "price": price,
            "gross": shares * price,
            "fees": fees,
            "slippage": slippage,
            "action": action,
            "variant": self.variant,
        }


def _diagnostics(result: BacktestResult) -> dict[str, Any]:
    nav = result.nav.copy()
    nav_value = pd.to_numeric(nav["nav"], errors="coerce")
    cash = pd.to_numeric(nav["cash"], errors="coerce")
    exposure = pd.to_numeric(nav["gross_exposure"], errors="coerce")
    holdings = (
        result.holdings.groupby("trade_date")["symbol"].nunique()
        if not result.holdings.empty
        else pd.Series(dtype=float)
    )
    reasons = (
        result.orders["reason"].fillna("").value_counts().to_dict()
        if not result.orders.empty
        else {}
    )
    return {
        "average_cash_ratio": float((cash / nav_value).mean()),
        "average_gross_exposure": float(exposure.mean()),
        "maximum_gross_exposure": float(exposure.max()),
        "average_holdings": float(holdings.mean()) if not holdings.empty else 0.0,
        "maximum_holdings": int(holdings.max()) if not holdings.empty else 0,
        "total_orders": int(len(result.orders)),
        "total_trades": int(len(result.trades)),
        "order_reason_counts": {str(key): int(value) for key, value in reasons.items()},
    }


def run_decision_attribution(
    paths: ProjectPaths,
    config_name: str = "decision_attribution_research",
) -> dict[str, Any]:
    research = load_config(paths, config_name)
    execution = load_config(paths, "execution")
    data = load_config(paths, "data")
    prediction_path = paths.reports / "research" / str(research["prediction_file"])
    bars_path = paths.curated / "bars.parquet"
    predictions = pd.read_parquet(prediction_path)
    predictions["trade_date"] = pd.to_datetime(predictions["trade_date"]).dt.normalize()
    bars = pd.read_parquet(bars_path, columns=BAR_COLUMNS)
    bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.normalize()
    benchmark_symbol = str(data["benchmark_symbol"])
    benchmark = bars[bars["symbol"].astype(str).eq(benchmark_symbol)][
        ["trade_date", "close"]
    ].copy()
    bars = bars[~bars["symbol"].astype(str).eq(benchmark_symbol)].copy()
    start = pd.Timestamp(research["development_start"])
    end = pd.Timestamp(research["development_end"])
    bars = bars[bars["trade_date"].between(start, end)].copy()
    predictions = predictions[predictions["trade_date"].between(start, end)].copy()
    benchmark = benchmark[benchmark["trade_date"].between(start, end)].copy()

    stage_results: dict[str, dict[str, Any]] = {}
    run_id = str(research["run_id"])
    if Path(run_id).name != run_id:
        raise ValueError("run_id must be a single directory name")
    output_root = paths.reports / "research" / "decision-attribution" / run_id
    for stage in research["stages"]:
        name = str(stage["name"])
        result = HorizonCohortSimulator(
            research,
            execution,
            staged_entry=bool(stage["staged_entry"]),
            trading_constraints=bool(stage["trading_constraints"]),
            risk_gate=bool(stage["risk_gate"]),
            variant_name=name,
        ).run(bars, predictions)
        metrics = performance_metrics(result.nav, benchmark, result.trades)
        metrics["diagnostics"] = _diagnostics(result)
        if (
            metrics["diagnostics"]["maximum_holdings"]
            > int(research["cohort_position_cap"])
        ):
            raise RuntimeError(f"{name} exceeded the cohort position cap")
        stage_results[name] = metrics
        stage_root = output_root / name
        stage_root.mkdir(parents=True, exist_ok=True)
        result.nav.to_parquet(stage_root / "nav.parquet", index=False)
        result.holdings.to_parquet(stage_root / "holdings.parquet", index=False)
        result.trades.to_parquet(stage_root / "trades.parquet", index=False)
        result.orders.to_parquet(stage_root / "orders.parquet", index=False)

    baseline_name = str(research["production_baseline_name"])
    production = ProductionMirrorSimulator(
        execution,
        apply_risk_gate=True,
        variant_name=baseline_name,
    ).run(bars, predictions)
    production_metrics = performance_metrics(
        production.nav,
        benchmark,
        production.trades,
    )
    production_metrics["diagnostics"] = _diagnostics(production)
    stage_results[baseline_name] = production_metrics
    baseline_root = output_root / baseline_name
    baseline_root.mkdir(parents=True, exist_ok=True)
    production.nav.to_parquet(baseline_root / "nav.parquet", index=False)
    production.holdings.to_parquet(
        baseline_root / "holdings.parquet",
        index=False,
    )
    production.trades.to_parquet(baseline_root / "trades.parquet", index=False)
    production.orders.to_parquet(baseline_root / "orders.parquet", index=False)

    candidate_name = "D_horizon_staged_tradable_gated"
    candidate = stage_results[candidate_name]
    baseline = stage_results[baseline_name]
    rules = research["acceptance"]
    checks = {
        "annualized_return_improvement": (
            float(candidate["annualized_return"])
            - float(baseline["annualized_return"])
            >= float(rules["minimum_annualized_return_improvement"])
        ),
        "sharpe_improvement": (
            float(candidate["sharpe_ratio"]) - float(baseline["sharpe_ratio"])
            >= float(rules["minimum_sharpe_improvement"])
        ),
        "drawdown_not_materially_worse": (
            float(candidate["max_drawdown"])
            >= float(baseline["max_drawdown"])
            - float(rules["maximum_drawdown_deterioration"])
        ),
    }
    attribution = {}
    stage_names = [str(stage["name"]) for stage in research["stages"]] + [
        baseline_name
    ]
    for left, right in zip(stage_names, stage_names[1:], strict=False):
        attribution[f"{left}_to_{right}"] = {
            "annualized_return_delta": float(
                stage_results[right]["annualized_return"]
                - stage_results[left]["annualized_return"]
            ),
            "sharpe_delta": float(
                stage_results[right]["sharpe_ratio"]
                - stage_results[left]["sharpe_ratio"]
            ),
            "max_drawdown_delta": float(
                stage_results[right]["max_drawdown"]
                - stage_results[left]["max_drawdown"]
            ),
        }
    report = {
        "status": (
            "DECISION_ALIGNMENT_DEVELOPMENT_ACCEPTED"
            if all(checks.values())
            else "DECISION_ALIGNMENT_DEVELOPMENT_REJECTED"
        ),
        "created_at": datetime.now(UTC).isoformat(),
        "development_period": [start.date().isoformat(), end.date().isoformat()],
        "run_id": run_id,
        "model_predictions_fixed": True,
        "model_training_changed": False,
        "stage_definitions": research["stages"],
        "stage_results": stage_results,
        "incremental_attribution": attribution,
        "candidate_stage": candidate_name,
        "production_baseline_stage": baseline_name,
        "acceptance": {"passed": all(checks.values()), "checks": checks, "rules": rules},
        "locked_period_read": False,
        "input_versions": {
            "predictions": str(prediction_path),
            "predictions_sha256": _sha256(prediction_path),
            "curated_bars": str(bars_path),
            "curated_bars_sha256": _sha256(bars_path),
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
    parser.add_argument("--config", default="decision_attribution_research")
    args = parser.parse_args()
    result = run_decision_attribution(
        ProjectPaths(args.root.resolve()),
        config_name=str(args.config),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

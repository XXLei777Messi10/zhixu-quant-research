from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from quant.config import ProjectPaths, load_config
from quant.execution.account import account_payload, load_simulation_account
from quant.execution.archive import load_execution_plans
from quant.execution.models import Action, Decision, ExecutionPlan, ExecutionState
from quant.execution.simulator import simulate_order
from quant.signals.archive import immutable_write_json


def _decision_id(
    plan: ExecutionPlan,
    action: Action,
    trigger_rule: str,
    variant: str,
) -> str:
    identity = {
        "plan_id": plan.plan_id,
        "action": action.value,
        "trigger_rule": trigger_rule,
        "variant": variant,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode(),
    ).hexdigest()[:24]
    return f"decision-{digest}"


def evaluate_daily_open(
    plan: ExecutionPlan,
    open_price: float | None,
    observed_at: datetime,
    config: dict[str, Any],
    variant: str,
) -> Decision:
    state = ExecutionState(plan.initial_state)
    action = Action.SKIP
    next_state = state
    planned_weight = 0.0
    trigger_rule = "NO_DAILY_OPEN_ACTION"
    reason = "Plan does not require an opening transaction"

    if open_price is None or open_price <= 0:
        next_state = ExecutionState.DATA_ERROR
        trigger_rule = "INVALID_DAILY_OPEN"
        reason = "Confirmed daily open is missing or non-positive"
    elif state is ExecutionState.CANCELLED:
        action = Action.CANCEL
        trigger_rule = "PLAN_PREVIOUSLY_CANCELLED"
        reason = "Risk or calibration gate cancelled the planned increase"
    elif state in {ExecutionState.READY_TO_OPEN, ExecutionState.READY_TO_ADD}:
        proposed = Action.OPEN if state is ExecutionState.READY_TO_OPEN else Action.ADD
        base_weight = (
            plan.initial_order_weight
            if proposed is Action.OPEN
            else plan.add_order_weight
        )
        lower_bound = (
            plan.entry_price_low
            if proposed is Action.OPEN
            else plan.add_price
        )
        if open_price < plan.invalidation_price:
            action = Action.CANCEL
            next_state = (
                ExecutionState.CANCELLED
                if proposed is Action.OPEN
                else ExecutionState.HOLDING
            )
            trigger_rule = "OPEN_BELOW_INVALIDATION"
            reason = "Opening price invalidated the buy/add signal"
        elif open_price < lower_bound:
            trigger_rule = (
                "OPEN_BELOW_ENTRY_ZONE"
                if proposed is Action.OPEN
                else "OPEN_BELOW_ADD_ZONE"
            )
            reason = (
                "Low open is observed but is not treated as an automatic dip buy"
                if proposed is Action.OPEN
                else "Opening price is below the calibrated add zone; no averaging down"
            )
        elif open_price <= plan.entry_price_high:
            action = proposed
            next_state = (
                ExecutionState.OPENED
                if proposed is Action.OPEN
                else ExecutionState.HOLDING
            )
            planned_weight = base_weight
            trigger_rule = "OPEN_IN_ENTRY_ZONE"
            reason = "Confirmed daily open is inside the calibrated entry zone"
        elif open_price <= plan.chase_limit_price:
            action = proposed
            next_state = (
                ExecutionState.OPENED
                if proposed is Action.OPEN
                else ExecutionState.HOLDING
            )
            planned_weight = base_weight * float(
                config["calibration"]["reduced_entry_multiplier"]
            )
            trigger_rule = "OPEN_IN_REDUCED_ENTRY_ZONE"
            reason = "Confirmed open is above the entry zone but below chase limit"
        else:
            action = Action.CANCEL
            next_state = (
                ExecutionState.CANCELLED
                if proposed is Action.OPEN
                else ExecutionState.HOLDING
            )
            trigger_rule = "OPEN_ABOVE_CHASE_LIMIT"
            reason = "Opening price exceeded the calibrated chase limit"
    elif state is ExecutionState.READY_TO_EXIT:
        action = Action.EXIT
        next_state = ExecutionState.NO_SIGNAL
        planned_weight = plan.current_position_weight
        trigger_rule = "MODEL_EXIT_AT_DAILY_OPEN"
        reason = "Model target is zero; exit at the confirmed next open"
    elif state is ExecutionState.READY_TO_REDUCE:
        action = Action.REDUCE
        next_state = ExecutionState.HOLDING
        planned_weight = max(
            0.0,
            plan.current_position_weight - plan.target_position_weight,
        )
        trigger_rule = "MODEL_REDUCE_AT_DAILY_OPEN"
        reason = "Model target weight decreased"
    elif (
        state is ExecutionState.HOLDING
        and open_price <= plan.hard_exit_price
        and plan.current_position_weight > 0
    ):
        action = Action.EXIT
        next_state = ExecutionState.NO_SIGNAL
        planned_weight = plan.current_position_weight
        trigger_rule = "HARD_EXIT_AT_DAILY_OPEN"
        reason = "Confirmed opening price crossed the calibrated hard-exit boundary"

    return Decision(
        decision_id=_decision_id(plan, action, trigger_rule, variant),
        plan_id=plan.plan_id,
        symbol=plan.symbol,
        decided_at=observed_at.isoformat(),
        previous_state=state.value,
        next_state=next_state.value,
        action=action.value,
        planned_weight=planned_weight,
        trigger_price=open_price,
        trigger_rule=trigger_rule,
        reason=reason,
        data_level="DAILY_OPEN",
        data_source="CURATED_DAILY_BAR",
        model_version=plan.model_version,
        rule_version=plan.rule_version,
        final=True,
    )


def _revision_number(path: Path) -> int:
    marker = "__r"
    if marker not in path.stem:
        return 1
    try:
        return int(path.stem.rsplit(marker, maxsplit=1)[1])
    except ValueError:
        return 0


def _latest_account_file(directory: Path, before: date | None = None) -> Path | None:
    candidates = []
    for path in directory.glob("*.json"):
        day_text = path.stem.split("__r", maxsplit=1)[0]
        try:
            day = date.fromisoformat(day_text)
        except ValueError:
            continue
        if before is None or day < before:
            candidates.append((day, _revision_number(path), path))
    return max(candidates, default=(None, None, None))[-1]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temp, path)


def _plan_path(paths: ProjectPaths, directory: str, execution_date: date) -> Path:
    return paths.reports / directory / f"{execution_date.isoformat()}.json"


def settle_daily_account(
    paths: ProjectPaths,
    execution_date: date,
    *,
    variant: str,
    plan_directory: str,
) -> dict[str, Any]:
    report_root = paths.reports / "simulation" / variant
    accounts_dir = report_root / "accounts"
    existing = _latest_account_file(accounts_dir)
    if existing is not None:
        existing_day = date.fromisoformat(existing.stem.split("__r", maxsplit=1)[0])
        if existing_day == execution_date:
            return {
                "variant": variant,
                "status": "ALREADY_SETTLED",
                "account": str(existing),
            }

    execution_config = load_config(paths, "execution")
    daily_config = execution_config["daily_open_simulation"]
    previous_account = _latest_account_file(accounts_dir, before=execution_date)
    account = load_simulation_account(
        previous_account or accounts_dir / "missing.json",
        float(daily_config["initial_cash"]),
    )
    bars_path = paths.curated / "bars.parquet"
    recent = pd.read_parquet(
        bars_path,
        columns=[
            "symbol",
            "trade_date",
            "open",
            "close",
            "volume",
            "is_trading",
            "is_st",
        ],
        filters=[
            (
                "trade_date",
                ">=",
                pd.Timestamp(execution_date - timedelta(days=15)),
            ),
            ("trade_date", "<=", pd.Timestamp(execution_date)),
        ],
    )
    recent["trade_date"] = pd.to_datetime(recent["trade_date"]).dt.normalize()
    recent = recent.sort_values(["symbol", "trade_date"])
    day = recent[recent["trade_date"].eq(pd.Timestamp(execution_date))].set_index(
        "symbol",
    )
    if day.empty:
        raise RuntimeError(
            f"Curated daily bars are missing for account settlement {execution_date}"
        )
    prior = (
        recent[recent["trade_date"].lt(pd.Timestamp(execution_date))]
        .groupby("symbol", observed=True)
        .tail(1)
        .set_index("symbol")
    )
    plan_path = _plan_path(paths, plan_directory, execution_date)
    try:
        plans = load_execution_plans(plan_path)
    except FileNotFoundError:
        plans = []

    previous_prices: dict[str, float] = {}
    if previous_account is not None:
        previous_payload = json.loads(previous_account.read_text(encoding="utf-8"))
        previous_prices = {
            str(item["symbol"]): float(item.get("mark_price", 0.0))
            for item in previous_payload.get("positions", [])
        }
    close_prices = dict(previous_prices)
    close_prices.update(
        {
            str(symbol): float(row["close"])
            for symbol, row in day.iterrows()
            if pd.notna(row["close"]) and float(row["close"]) > 0
        }
    )
    open_prices = {
        str(symbol): float(row["open"])
        for symbol, row in day.iterrows()
        if pd.notna(row["open"]) and float(row["open"]) > 0
    }
    portfolio_value = account.cash + sum(
        account.shares(symbol) * open_prices.get(
            symbol,
            previous_prices.get(symbol, 0.0),
        )
        for symbol in account.lots
    )
    observed_at = datetime.combine(
        execution_date,
        time(9, 30),
        tzinfo=ZoneInfo("Asia/Shanghai"),
    )
    decisions: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []
    new_positions = 0
    add_counts: dict[str, int] = {}
    daily_turnover = 0.0
    sector_by_symbol = {
        item.symbol: item.sector_name or "DATA_UNAVAILABLE"
        for item in plans
    }
    for original_plan in sorted(plans, key=lambda item: item.execution_priority):
        plan = original_plan
        current_value = account.shares(plan.symbol) * open_prices.get(
            plan.symbol,
            previous_prices.get(plan.symbol, plan.reference_close),
        )
        current_weight = current_value / portfolio_value if portfolio_value > 0 else 0.0
        plan = replace(plan, current_position_weight=current_weight)
        open_price = open_prices.get(plan.symbol)
        decision = evaluate_daily_open(
            plan,
            open_price,
            observed_at,
            execution_config,
            variant,
        )
        decisions.append(decision.to_dict())
        if decision.action not in {
            Action.OPEN.value,
            Action.ADD.value,
            Action.REDUCE.value,
            Action.EXIT.value,
        }:
            continue
        row = day.loc[plan.symbol] if plan.symbol in day.index else None
        prior_row = prior.loc[plan.symbol] if plan.symbol in prior.index else None
        available = (
            int(prior_row["volume"])
            if prior_row is not None
            and pd.notna(prior_row["volume"])
            and float(prior_row["volume"]) >= 0
            else None
        )
        gross_value = sum(
            account.shares(symbol)
            * open_prices.get(symbol, previous_prices.get(symbol, 0.0))
            for symbol in account.lots
        )
        plan_sector = plan.sector_name or "DATA_UNAVAILABLE"
        sector_value = sum(
            account.shares(symbol)
            * open_prices.get(symbol, previous_prices.get(symbol, 0.0))
            for symbol in account.lots
            if sector_by_symbol.get(symbol, "DATA_UNAVAILABLE") == plan_sector
        )
        order = simulate_order(
            account,
            plan,
            decision,
            open_price,
            portfolio_value,
            execution_date,
            observed_at,
            execution_config,
            is_trading=bool(row["is_trading"]) if row is not None else False,
            is_st=bool(row["is_st"]) if row is not None else False,
            previous_close=plan.reference_close,
            available_liquidity_shares=available,
            quote_is_fresh=True,
            execution_price_confirmed=open_price is not None,
            daily_new_positions=new_positions,
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
        orders.append(record)
        if order.fill_quantity:
            daily_turnover += float(order.planned_weight)
            if order.action == Action.OPEN.value:
                new_positions += 1
            elif order.action == Action.ADD.value:
                add_counts[plan.symbol] = add_counts.get(plan.symbol, 0) + 1

    model_version = plans[0].model_version if plans else "NO_PLAN"
    account_state = account_payload(
        account,
        close_prices,
        datetime.combine(
            execution_date,
            time(15, 0),
            tzinfo=ZoneInfo("Asia/Shanghai"),
        ).isoformat(),
        model_version,
        str(execution_config["rule_version"]),
    )
    account_state.update(
        {
            "execution_date": execution_date.isoformat(),
            "variant": variant,
            "risk_gate_mode": (
                "ENABLED" if variant == "gated" else "DISABLED_SHADOW"
            ),
            "plan_directory": plan_directory,
            "plan_count": len(plans),
            "decision_count": len(decisions),
            "order_count": len(orders),
            "fill_count": sum(bool(item["fill_quantity"]) for item in orders),
            "fees_today": sum(float(item["fees"]) for item in orders),
            "slippage_today": sum(float(item["slippage"]) for item in orders),
            "simulation_only": True,
            "portfolio_policy": execution_config["portfolio_policy"],
        }
    )
    account_path = immutable_write_json(
        accounts_dir / f"{execution_date.isoformat()}.json",
        account_state,
    )
    orders_path = immutable_write_json(
        report_root / "orders" / f"{execution_date.isoformat()}.json",
        {
            "execution_date": execution_date.isoformat(),
            "variant": variant,
            "simulation_only": True,
            "decisions": decisions,
            "orders": orders,
        },
    )
    fills_path = immutable_write_json(
        report_root / "fills" / f"{execution_date.isoformat()}.json",
        {
            "execution_date": execution_date.isoformat(),
            "variant": variant,
            "simulation_only": True,
            "fills": [item for item in orders if item["fill_quantity"]],
        },
    )
    _atomic_json(paths.state / "simulation" / variant / "current.json", account_state)
    return {
        "variant": variant,
        "status": "SETTLED" if plans else "INITIALIZED_NO_PLAN",
        "account": str(account_path),
        "orders": str(orders_path),
        "fills": str(fills_path),
        "nav": float(account_state["nav"]),
        "fill_count": int(account_state["fill_count"]),
    }


def settle_daily_accounts(
    paths: ProjectPaths,
    execution_date: date,
) -> dict[str, Any]:
    config = load_config(paths, "execution")["daily_open_simulation"]
    if not bool(config["enabled"]):
        return {"status": "DISABLED"}
    primary = settle_daily_account(
        paths,
        execution_date,
        variant=str(config["primary_variant"]),
        plan_directory=str(config["primary_plan_directory"]),
    )
    shadow = settle_daily_account(
        paths,
        execution_date,
        variant=str(config["shadow_variant"]),
        plan_directory=str(config["shadow_plan_directory"]),
    )
    return {
        "status": "SUCCESS",
        "execution_date": execution_date.isoformat(),
        "primary": primary,
        "shadow": shadow,
        "simulation_only": True,
    }

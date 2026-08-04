from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from typing import Any

from quant.backtest.engine import Account
from quant.backtest.rules import is_order_blocked_by_limit, stamp_duty_rate
from quant.execution.models import (
    Action,
    Decision,
    ExecutionPlan,
    OrderStatus,
    SimulatedOrder,
)


def _order_id(decision: Decision) -> str:
    payload = {
        "plan": decision.plan_id,
        "action": decision.action,
        "trigger_rule": decision.trigger_rule,
        "rule": decision.rule_version,
    }
    return "order-" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def order_id_for_decision(decision: Decision) -> str:
    return _order_id(decision)


def execution_fees(
    side: str,
    gross: float,
    trade_date: date,
    config: dict[str, Any],
) -> float:
    commission = max(float(config["minimum_commission"]), gross * float(config["commission_rate"]))
    transfer = gross * float(config["transfer_fee_rate"])
    tax = gross * stamp_duty_rate(trade_date, config["sell_tax_rate"]) if side == "SELL" else 0.0
    return commission + transfer + tax


def _order(
    decision: Decision,
    plan: ExecutionPlan,
    quantity: int,
    status: OrderStatus,
    reason: str,
    created_at: datetime,
    *,
    fill_price: float | None = None,
    fill_quantity: int = 0,
    fees: float = 0.0,
    slippage: float = 0.0,
) -> SimulatedOrder:
    return SimulatedOrder(
        order_id=_order_id(decision),
        plan_id=plan.plan_id,
        symbol=plan.symbol,
        action=decision.action,
        planned_weight=decision.planned_weight,
        planned_quantity=quantity,
        trigger_price=decision.trigger_price,
        fill_price=fill_price,
        fill_quantity=fill_quantity,
        fees=fees,
        slippage=slippage,
        created_at=created_at.isoformat(),
        filled_at=created_at.isoformat() if fill_quantity else None,
        status=status.value,
        reason=reason,
        data_source=decision.data_source,
        model_version=plan.model_version,
        rule_version=plan.rule_version,
    )


def simulate_order(
    account: Account,
    plan: ExecutionPlan,
    decision: Decision,
    official_price: float | None,
    portfolio_value: float,
    trade_date: date,
    created_at: datetime,
    config: dict[str, Any],
    *,
    is_trading: bool = True,
    is_st: bool = False,
    previous_close: float | None = None,
    available_liquidity_shares: int | None = None,
    quote_is_fresh: bool = True,
    execution_price_confirmed: bool = False,
    fallback_quote: bool = False,
    daily_new_positions: int = 0,
    daily_add_count: int = 0,
    daily_turnover: float = 0.0,
    current_gross_weight: float = 0.0,
    current_sector_weight: float = 0.0,
    current_position_count: int = 0,
) -> SimulatedOrder:
    action = Action(decision.action)
    if not decision.final:
        return _order(decision, plan, 0, OrderStatus.REJECTED, "OBSERVATION_ONLY_NO_ORDER", created_at)
    if action not in {Action.OPEN, Action.ADD, Action.REDUCE, Action.EXIT}:
        return _order(
            decision, plan, 0, OrderStatus.CANCELLED, f"NO_EXECUTION_FOR_{action.value}", created_at
        )
    if action is Action.OPEN and daily_new_positions >= int(config["max_daily_new_positions"]):
        return _order(decision, plan, 0, OrderStatus.REJECTED, "MAX_DAILY_NEW_POSITIONS", created_at)
    if (
        action is Action.OPEN
        and current_position_count >= int(config["portfolio_policy"]["max_positions"])
    ):
        return _order(
            decision,
            plan,
            0,
            OrderStatus.REJECTED,
            "MAX_PORTFOLIO_POSITIONS",
            created_at,
        )
    if action is Action.ADD and daily_add_count >= int(config["max_daily_add_count"]):
        return _order(decision, plan, 0, OrderStatus.REJECTED, "MAX_DAILY_ADD_COUNT", created_at)
    if action in {
        Action.OPEN,
        Action.ADD,
        Action.REDUCE,
    } and daily_turnover + decision.planned_weight > float(config["max_portfolio_turnover"]):
        return _order(decision, plan, 0, OrderStatus.REJECTED, "MAX_PORTFOLIO_TURNOVER", created_at)
    if not quote_is_fresh or not execution_price_confirmed or official_price is None or official_price <= 0:
        return _order(
            decision,
            plan,
            0,
            OrderStatus.REJECTED,
            "UNCONFIRMED_INVALID_OR_STALE_EXECUTION_PRICE",
            created_at,
        )
    if not is_trading:
        return _order(decision, plan, 0, OrderStatus.REJECTED, "SUSPENDED", created_at)
    if available_liquidity_shares is None or available_liquidity_shares < 0:
        return _order(decision, plan, 0, OrderStatus.REJECTED, "LIQUIDITY_CAP_UNAVAILABLE", created_at)

    lot = int(config["lot_size"])
    side = "BUY" if action in {Action.OPEN, Action.ADD} else "SELL"
    if previous_close and is_order_blocked_by_limit(
        side, official_price, previous_close, plan.symbol, trade_date, is_st
    ):
        reason = "LIMIT_UP" if side == "BUY" else "LIMIT_DOWN"
        return _order(decision, plan, 0, OrderStatus.REJECTED, reason, created_at)

    requested = int((portfolio_value * decision.planned_weight) / official_price // lot * lot)
    liquidity_cap = int(
        available_liquidity_shares * float(config["max_liquidity_participation"]) // lot * lot
    )
    quantity = min(requested, liquidity_cap)
    if side == "SELL":
        quantity = min(quantity, account.sellable_shares(plan.symbol, trade_date))
    if quantity < lot:
        reason = "T_PLUS_ONE_OR_NO_POSITION" if side == "SELL" else "BELOW_LOT_OR_LIQUIDITY_CAP"
        return _order(decision, plan, requested, OrderStatus.REJECTED, reason, created_at)

    slippage_rate = float(config["fallback_slippage_rate"] if fallback_quote else config["slippage_rate"])
    fill_price = official_price * (1.0 + slippage_rate if side == "BUY" else 1.0 - slippage_rate)
    if side == "BUY":
        max_position_value = portfolio_value * min(plan.target_position_weight, plan.max_position_weight)
        existing_value = account.shares(plan.symbol) * official_price
        cap_quantity = int(max(0.0, max_position_value - existing_value) / fill_price // lot * lot)
        policy = config["portfolio_policy"]
        gross_capacity = max(
            0.0,
            portfolio_value
            * (float(policy["max_gross_exposure"]) - current_gross_weight),
        )
        cash_reserve_capacity = max(
            0.0,
            account.cash
            - portfolio_value * float(policy["minimum_cash_reserve"]),
        )
        sector_capacity = max(
            0.0,
            portfolio_value
            * (float(policy["max_sector_exposure"]) - current_sector_weight),
        )
        portfolio_cap_quantity = int(
            min(gross_capacity, cash_reserve_capacity, sector_capacity)
            / fill_price
            // lot
            * lot
        )
        quantity = min(quantity, cap_quantity, portfolio_cap_quantity)
        while quantity >= lot:
            gross = fill_price * quantity
            fee = execution_fees(side, gross, trade_date, config)
            if gross + fee <= account.cash + 1e-9:
                break
            quantity -= lot
        if quantity < lot:
            return _order(
                decision,
                plan,
                requested,
                OrderStatus.REJECTED,
                "CASH_RESERVE_GROSS_SECTOR_OR_POSITION_CAP",
                created_at,
            )
        gross = fill_price * quantity
        fee = execution_fees(side, gross, trade_date, config)
        account.cash -= gross + fee
        account.add(plan.symbol, quantity, trade_date)
    else:
        gross = fill_price * quantity
        fee = execution_fees(side, gross, trade_date, config)
        account.remove(plan.symbol, quantity, trade_date)
        account.cash += gross - fee

    status = OrderStatus.FILLED if quantity == requested else OrderStatus.PARTIALLY_FILLED
    return _order(
        decision,
        plan,
        requested,
        status,
        "" if status is OrderStatus.FILLED else "CAPPED_BY_CASH_POSITION_OR_LIQUIDITY",
        created_at,
        fill_price=fill_price,
        fill_quantity=quantity,
        fees=fee,
        slippage=abs(fill_price - official_price) * quantity,
    )

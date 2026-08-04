from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from quant.execution.auction import _decision
from quant.execution.models import (
    Action,
    AuctionDataLevel,
    Decision,
    ExecutionPlan,
    ExecutionState,
    ModelSignal,
)


def evaluate_continuous_quote(
    plan: ExecutionPlan,
    price: float | None,
    observed_at: datetime,
    source_timestamp: datetime | None,
    source: str,
    max_data_age_seconds: int,
    current_state: ExecutionState | None = None,
) -> Decision:
    state = current_state or ExecutionState(plan.initial_state)
    local = observed_at.astimezone(ZoneInfo("Asia/Shanghai"))
    if local.time().replace(tzinfo=None) < datetime.strptime("09:30:00", "%H:%M:%S").time():
        return _decision(
            plan,
            observed_at,
            state,
            state,
            Action.SKIP,
            0.0,
            price,
            "CONTINUOUS_MARKET_GATE",
            "Continuous execution is prohibited before 09:30",
            AuctionDataLevel.C,
            source,
            False,
        )
    if observed_at > datetime.fromisoformat(plan.valid_until):
        return _decision(
            plan,
            observed_at,
            state,
            ExecutionState.EXPIRED,
            Action.CANCEL,
            0.0,
            price,
            "PLAN_EXPIRED",
            "Execution plan is past valid_until",
            AuctionDataLevel.C,
            source,
            True,
        )
    age = (observed_at - source_timestamp).total_seconds() if source_timestamp is not None else float("inf")
    if price is None or price <= 0 or age < -1 or age > max_data_age_seconds:
        return _decision(
            plan,
            observed_at,
            state,
            ExecutionState.DATA_ERROR,
            Action.SKIP,
            0.0,
            price,
            "STALE_OR_INVALID_QUOTE",
            "Fresh valid quote is required for every simulated trigger",
            AuctionDataLevel.C,
            source,
            True,
        )
    if plan.model_signal == ModelSignal.EXIT.value or plan.target_position_weight <= 0:
        return _decision(
            plan,
            observed_at,
            state,
            ExecutionState.READY_TO_EXIT,
            Action.EXIT,
            plan.current_position_weight,
            price,
            "MODEL_EXIT",
            "Model exit remains authoritative intraday",
            AuctionDataLevel.C,
            source,
            True,
        )
    if plan.model_signal == ModelSignal.REDUCE.value or (
        plan.target_position_weight < plan.current_position_weight
    ):
        return _decision(
            plan,
            observed_at,
            state,
            ExecutionState.READY_TO_REDUCE,
            Action.REDUCE,
            plan.current_position_weight - plan.target_position_weight,
            price,
            "MODEL_REDUCE",
            "Model target weight has decreased",
            AuctionDataLevel.C,
            source,
            True,
        )
    if price <= plan.hard_exit_price and plan.current_position_weight > 0:
        return _decision(
            plan,
            observed_at,
            state,
            ExecutionState.READY_TO_EXIT,
            Action.EXIT,
            plan.current_position_weight,
            price,
            "HARD_EXIT_PRICE",
            "Fresh quote crossed calibrated hard exit",
            AuctionDataLevel.C,
            source,
            True,
        )
    if price < plan.invalidation_price:
        return _decision(
            plan,
            observed_at,
            state,
            ExecutionState.CANCELLED,
            Action.CANCEL,
            0.0,
            price,
            "SIGNAL_INVALIDATION",
            "Signal invalidated; no averaging down is permitted",
            AuctionDataLevel.C,
            source,
            True,
        )
    if state is ExecutionState.READY_TO_OPEN:
        if price > plan.chase_limit_price:
            action, next_state, weight, rule = (
                Action.CANCEL,
                ExecutionState.CANCELLED,
                0.0,
                "CHASE_LIMIT",
            )
        elif plan.entry_price_low <= price <= plan.entry_price_high:
            action, next_state, weight, rule = (
                Action.OPEN,
                ExecutionState.READY_TO_OPEN,
                plan.initial_order_weight,
                "ENTRY_ZONE",
            )
        else:
            action, next_state, weight, rule = Action.SKIP, ExecutionState.WATCH, 0.0, "NO_TRIGGER"
    elif state is ExecutionState.READY_TO_ADD:
        if price <= plan.add_price:
            action, next_state, weight, rule = (
                Action.ADD,
                ExecutionState.READY_TO_ADD,
                plan.add_order_weight,
                "ADD_ZONE",
            )
        else:
            action, next_state, weight, rule = Action.HOLD, ExecutionState.HOLDING, 0.0, "NO_ADD"
    elif state is ExecutionState.READY_TO_REDUCE and price >= plan.reduce_price:
        action, next_state, weight, rule = (
            Action.REDUCE,
            ExecutionState.READY_TO_REDUCE,
            max(0.0, plan.current_position_weight - plan.target_position_weight),
            "REDUCE_PRICE",
        )
    else:
        action, next_state, weight, rule = Action.HOLD, ExecutionState.HOLDING, 0.0, "HOLD"
    return _decision(
        plan,
        observed_at,
        state,
        next_state,
        action,
        weight,
        price,
        rule,
        rule.replace("_", " ").title(),
        AuctionDataLevel.C,
        source,
        True,
    )

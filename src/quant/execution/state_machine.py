from __future__ import annotations

import hashlib
import json
from datetime import datetime

from quant.execution.journal import AppendOnlyJournal
from quant.execution.models import (
    Action,
    Decision,
    ExecutionState,
    OrderStatus,
    SimulatedOrder,
)

ALLOWED_TRANSITIONS: dict[ExecutionState, set[ExecutionState]] = {
    ExecutionState.NO_SIGNAL: {ExecutionState.NO_SIGNAL, ExecutionState.WATCH},
    ExecutionState.WATCH: {
        ExecutionState.WATCH,
        ExecutionState.READY_TO_OPEN,
        ExecutionState.CANCELLED,
        ExecutionState.EXPIRED,
        ExecutionState.DATA_ERROR,
    },
    ExecutionState.READY_TO_OPEN: {
        ExecutionState.READY_TO_OPEN,
        ExecutionState.WATCH,
        ExecutionState.OPENED,
        ExecutionState.CANCELLED,
        ExecutionState.EXPIRED,
        ExecutionState.DATA_ERROR,
    },
    ExecutionState.OPENED: {
        ExecutionState.OPENED,
        ExecutionState.READY_TO_ADD,
        ExecutionState.HOLDING,
        ExecutionState.READY_TO_REDUCE,
        ExecutionState.READY_TO_EXIT,
    },
    ExecutionState.READY_TO_ADD: {
        ExecutionState.READY_TO_ADD,
        ExecutionState.HOLDING,
        ExecutionState.READY_TO_REDUCE,
        ExecutionState.READY_TO_EXIT,
        ExecutionState.CANCELLED,
        ExecutionState.DATA_ERROR,
    },
    ExecutionState.HOLDING: {
        ExecutionState.HOLDING,
        ExecutionState.READY_TO_ADD,
        ExecutionState.READY_TO_REDUCE,
        ExecutionState.READY_TO_EXIT,
        ExecutionState.DATA_ERROR,
    },
    ExecutionState.READY_TO_REDUCE: {
        ExecutionState.READY_TO_REDUCE,
        ExecutionState.HOLDING,
        ExecutionState.READY_TO_EXIT,
        ExecutionState.DATA_ERROR,
    },
    ExecutionState.READY_TO_EXIT: {
        ExecutionState.READY_TO_EXIT,
        ExecutionState.CANCELLED,
        ExecutionState.DATA_ERROR,
    },
    ExecutionState.CANCELLED: {ExecutionState.CANCELLED},
    ExecutionState.EXPIRED: {ExecutionState.EXPIRED},
    ExecutionState.DATA_ERROR: {
        ExecutionState.DATA_ERROR,
        ExecutionState.WATCH,
        ExecutionState.READY_TO_EXIT,
    },
}


def validate_transition(decision: Decision) -> None:
    previous = ExecutionState(decision.previous_state)
    next_state = ExecutionState(decision.next_state)
    if next_state not in ALLOWED_TRANSITIONS[previous]:
        raise ValueError(f"Illegal execution transition: {previous.value} -> {next_state.value}")


def append_decision(journal: AppendOnlyJournal, decision: Decision) -> bool:
    validate_transition(decision)
    return journal.append(decision.to_dict())


def fill_transition(
    decision: Decision,
    order: SimulatedOrder,
    occurred_at: datetime,
) -> Decision | None:
    if order.status not in {OrderStatus.FILLED.value, OrderStatus.PARTIALLY_FILLED.value}:
        return None
    action = Action(order.action)
    previous = ExecutionState(decision.next_state)
    next_state = {
        Action.OPEN: ExecutionState.OPENED,
        Action.ADD: ExecutionState.HOLDING,
        Action.REDUCE: ExecutionState.HOLDING,
        Action.EXIT: ExecutionState.CANCELLED,
    }[action]
    basis = json.dumps({"order_id": order.order_id, "next_state": next_state.value}, sort_keys=True)
    return Decision(
        decision_id="decision-" + hashlib.sha256(basis.encode()).hexdigest()[:24],
        plan_id=decision.plan_id,
        symbol=decision.symbol,
        decided_at=occurred_at.isoformat(),
        previous_state=previous.value,
        next_state=next_state.value,
        action=decision.action,
        planned_weight=decision.planned_weight,
        trigger_price=order.fill_price,
        trigger_rule=f"{decision.trigger_rule}_FILLED",
        reason=f"Simulated order {order.order_id} {order.status}",
        data_level=decision.data_level,
        data_source=decision.data_source,
        model_version=decision.model_version,
        rule_version=decision.rule_version,
        final=True,
    )

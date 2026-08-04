from __future__ import annotations

from pathlib import Path
from typing import Any

from quant.execution.archive import load_execution_plans
from quant.execution.journal import replay_journal
from quant.execution.models import Decision, ExecutionState
from quant.execution.state_machine import validate_transition


def replay_execution_day(
    plans_path: Path,
    decisions_path: Path,
    orders_path: Path,
) -> dict[str, Any]:
    plans = load_execution_plans(plans_path)
    plan_ids = {plan.plan_id for plan in plans}
    decisions = replay_journal(decisions_path, "decision_id")
    orders = replay_journal(orders_path, "order_id")
    states = {plan.plan_id: ExecutionState(plan.initial_state) for plan in plans}
    for payload in decisions:
        if payload["plan_id"] not in plan_ids:
            raise ValueError("Decision references an unknown plan")
        decision = Decision(**payload)
        expected = states[decision.plan_id]
        if decision.previous_state != expected.value:
            raise ValueError("Decision replay state does not match prior event")
        validate_transition(decision)
        states[decision.plan_id] = ExecutionState(decision.next_state)
    decision_ids = {item["decision_id"] for item in decisions}
    for order in orders:
        if order["plan_id"] not in plan_ids:
            raise ValueError("Order references an unknown plan")
    return {
        "plan_count": len(plans),
        "decision_count": len(decisions),
        "order_count": len(orders),
        "decision_ids": sorted(decision_ids),
        "final_states": {key: value.value for key, value in states.items()},
    }

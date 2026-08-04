from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from quant.execution.auction import evaluate_auction
from quant.execution.models import AuctionSnapshot, ExecutionPlan


def evaluate_daily_open_only(
    plan: ExecutionPlan,
    bar: dict[str, Any],
    execution_config: dict[str, Any],
    auction_config: dict[str, Any],
) -> dict[str, Any]:
    """Conservative daily-data evaluation: only the confirmed next open may execute."""
    if execution_config["historical_daily_conflict_policy"] != "OPEN_ONLY":
        raise ValueError("Only OPEN_ONLY is implemented; favorable intraday ordering is prohibited")
    execution_day = datetime.fromisoformat(plan.execution_date).date()
    observed_at = datetime.combine(
        execution_day,
        datetime.strptime("09:25:00", "%H:%M:%S").time(),
        tzinfo=ZoneInfo("Asia/Shanghai"),
    )
    snapshot = AuctionSnapshot(
        symbol=plan.symbol,
        observed_at=observed_at,
        source_timestamp=observed_at,
        auction_price=float(bar["open"]) if bar.get("open") is not None else None,
        previous_close=plan.reference_close,
        source="HISTORICAL_DAILY_OPEN",
        final_open_confirmed=True,
    )
    decision, features = evaluate_auction(plan, [snapshot], observed_at, auction_config)
    high = float(bar["high"])
    low = float(bar["low"])
    add_or_exit_conflict = high >= plan.reduce_price and low <= min(plan.add_price, plan.invalidation_price)
    return {
        "decision": decision.to_dict(),
        "auction_features": features.__dict__,
        "intraday_trigger_order_uncertain": add_or_exit_conflict,
        "intraday_actions_simulated": False,
        "policy": "OPEN_ONLY",
        "note": (
            "High/low crossed multiple boundaries; ignored because daily bars cannot establish order"
            if add_or_exit_conflict
            else "Only confirmed open was evaluated"
        ),
    }

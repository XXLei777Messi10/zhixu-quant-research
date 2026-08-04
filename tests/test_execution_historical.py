from datetime import date

from quant.execution.historical import evaluate_daily_open_only
from quant.execution.planner import build_execution_plan
from tests.test_auction_execution import AUCTION_CONFIG
from tests.test_execution_planner import EXECUTION_CONFIG, context, signal


def test_daily_multiple_trigger_conflict_never_uses_favorable_order() -> None:
    plan = build_execution_plan(signal(), context(), date(2026, 1, 6), EXECUTION_CONFIG)
    config = {**EXECUTION_CONFIG, "historical_daily_conflict_policy": "OPEN_ONLY"}
    report = evaluate_daily_open_only(
        plan,
        {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.2},
        config,
        AUCTION_CONFIG,
    )
    assert report["intraday_trigger_order_uncertain"]
    assert not report["intraday_actions_simulated"]
    assert report["policy"] == "OPEN_ONLY"

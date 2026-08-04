from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from quant.backtest.engine import Account
from quant.execution.archive import archive_execution_plans
from quant.execution.auction import classify_data_level, compute_auction_features, evaluate_auction
from quant.execution.continuous import evaluate_continuous_quote
from quant.execution.journal import AppendOnlyJournal
from quant.execution.models import AuctionDataLevel, AuctionSnapshot, ExecutionState
from quant.execution.planner import build_execution_plan
from quant.execution.replay import replay_execution_day
from quant.execution.simulator import simulate_order
from quant.execution.state_machine import append_decision
from tests.test_execution_planner import EXECUTION_CONFIG, context, signal

TZ = ZoneInfo("Asia/Shanghai")
AUCTION_CONFIG = {
    "rule_version": "auction-v1",
    "timezone": "Asia/Shanghai",
    "collection_start": "09:15:00",
    "observation_only_end": "09:20:00",
    "decision_start": "09:24:30",
    "auction_end": "09:25:00",
    "continuous_market_start": "09:30:00",
    "max_data_age_seconds": 20,
    "minimum_snapshot_count": 3,
    "reduced_entry_multiplier": 0.5,
    "price_stability": {"window_start": "09:20:00"},
}
FILL_CONFIG = {
    **EXECUTION_CONFIG,
    "commission_rate": 0.0003,
    "minimum_commission": 5.0,
    "transfer_fee_rate": 0.00002,
    "sell_tax_rate": [{"start": "1900-01-01", "end": "2099-12-31", "rate": 0.001}],
    "slippage_rate": 0.0005,
    "fallback_slippage_rate": 0.001,
    "lot_size": 100,
    "max_liquidity_participation": 0.01,
    "max_daily_new_positions": 5,
    "max_daily_add_count": 1,
    "max_portfolio_turnover": 0.30,
    "portfolio_policy": {
        "max_gross_exposure": 0.80,
        "minimum_cash_reserve": 0.20,
        "max_sector_exposure": 0.20,
        "max_positions": 16,
    },
}


def plan(**signal_changes):
    return build_execution_plan(signal(**signal_changes), context(), date(2026, 1, 6), EXECUTION_CONFIG)


def snapshots(price: float = 10.0, full: bool = True):
    result = []
    for second, delta in ((0, -0.01), (20, 0.01), (40, 0.0)):
        observed = datetime(2026, 1, 6, 9, 24, second, tzinfo=TZ)
        result.append(
            AuctionSnapshot(
                symbol="SH600000",
                observed_at=observed,
                source_timestamp=observed,
                auction_price=price + delta,
                previous_close=10.0,
                matched_volume=100_000 if full else None,
                matched_amount=1_000_000 if full else None,
                unmatched_buy_volume=20_000 if full else None,
                unmatched_sell_volume=10_000 if full else None,
                market_auction_return=0.0,
                sector_auction_return=0.0,
                source="TEST",
                final_open_confirmed=False,
            )
        )
    return result


def test_915_to_920_never_produces_final_trade() -> None:
    now = datetime(2026, 1, 6, 9, 18, tzinfo=TZ)
    snap = AuctionSnapshot("SH600000", now, 10.0, 10.0, source_timestamp=now, source="TEST")
    decision, _ = evaluate_auction(plan(), [snap], now, AUCTION_CONFIG)
    assert decision.action == "SKIP"
    assert not decision.final


def test_level_a_b_c_and_stability() -> None:
    now = datetime(2026, 1, 6, 9, 24, 45, tzinfo=TZ)
    full = snapshots()
    assert classify_data_level(full, now, AUCTION_CONFIG)[0] is AuctionDataLevel.A
    assert classify_data_level(snapshots(full=False), now, AUCTION_CONFIG)[0] is AuctionDataLevel.B
    stale = [replace(full[-1], source_timestamp=datetime(2026, 1, 6, 9, 20, tzinfo=TZ))]
    assert classify_data_level(stale, now, AUCTION_CONFIG)[0] is AuctionDataLevel.C
    features = compute_auction_features(full)
    assert features.auction_price_stability["snapshot_count"] == 3
    assert features.imbalance_ratio is not None
    missing_timestamp = [replace(full[-1], source_timestamp=None)]
    assert classify_data_level(missing_timestamp, now, AUCTION_CONFIG)[0] is AuctionDataLevel.C


def test_normal_open_high_gap_and_invalidation() -> None:
    now = datetime(2026, 1, 6, 9, 24, 45, tzinfo=TZ)
    normal, _ = evaluate_auction(plan(), snapshots(10.0), now, AUCTION_CONFIG)
    high, _ = evaluate_auction(plan(), snapshots(10.5), now, AUCTION_CONFIG)
    low, _ = evaluate_auction(plan(), snapshots(9.4), now, AUCTION_CONFIG)
    assert normal.action == "OPEN"
    assert high.action == "CANCEL" and high.trigger_rule == "CHASE_LIMIT"
    assert low.action == "CANCEL" and low.trigger_rule == "SIGNAL_INVALIDATION"


def test_held_add_is_blocked_after_invalidation_or_target() -> None:
    now = datetime(2026, 1, 6, 9, 24, 45, tzinfo=TZ)
    held = plan(current_position_weight=0.04)
    invalid, _ = evaluate_auction(held, snapshots(9.4), now, AUCTION_CONFIG)
    target = plan(current_position_weight=0.08)
    no_add, _ = evaluate_auction(target, snapshots(9.8), now, AUCTION_CONFIG)
    assert invalid.action == "CANCEL"
    assert no_add.action == "HOLD"


def test_market_high_risk_blocks_new_position() -> None:
    now = datetime(2026, 1, 6, 9, 24, 45, tzinfo=TZ)
    decision, _ = evaluate_auction(plan(market_state="HIGH_RISK"), snapshots(10.0), now, AUCTION_CONFIG)
    assert decision.action == "CANCEL"


def test_model_reduce_and_exit_are_authoritative() -> None:
    now = datetime(2026, 1, 6, 9, 24, 45, tzinfo=TZ)
    reduce, _ = evaluate_auction(
        plan(
            model_signal="REDUCE",
            current_position_weight=0.08,
            target_position_weight=0.04,
        ),
        snapshots(10.0),
        now,
        AUCTION_CONFIG,
    )
    exit_decision, _ = evaluate_auction(
        plan(
            model_signal="EXIT",
            current_position_weight=0.08,
            target_position_weight=0.0,
        ),
        snapshots(10.0),
        now,
        AUCTION_CONFIG,
    )
    assert reduce.action == "REDUCE"
    assert exit_decision.action == "EXIT"


def test_930_fallback_requires_first_fresh_quote() -> None:
    now = datetime(2026, 1, 6, 9, 30, 1, tzinfo=TZ)
    stale = evaluate_continuous_quote(
        plan(),
        10.0,
        now,
        datetime(2026, 1, 6, 9, 29, tzinfo=TZ),
        "TEST",
        20,
    )
    fresh = evaluate_continuous_quote(plan(), 10.0, now, now, "TEST", 20)
    assert stale.next_state == ExecutionState.DATA_ERROR
    assert fresh.action == "OPEN"


def test_expired_plan_cannot_trigger_continuous_order() -> None:
    now = datetime(2026, 1, 6, 15, 1, tzinfo=TZ)
    decision = evaluate_continuous_quote(plan(), 10.0, now, now, "TEST", 20)
    assert decision.next_state == ExecutionState.EXPIRED
    assert decision.action == "CANCEL"


def test_confirmed_open_simulates_lots_costs_and_is_idempotent(tmp_path: Path) -> None:
    now = datetime(2026, 1, 6, 9, 25, tzinfo=TZ)
    current_plan = plan()
    decision, _ = evaluate_auction(current_plan, snapshots(10.0), now, AUCTION_CONFIG)
    account = Account(1_000_000)
    order = simulate_order(
        account,
        current_plan,
        decision,
        10.0,
        1_000_000,
        date(2026, 1, 6),
        now,
        FILL_CONFIG,
        previous_close=10.0,
        available_liquidity_shares=1_000_000,
        execution_price_confirmed=True,
    )
    assert order.status in {"FILLED", "PARTIALLY_FILLED"}
    assert order.fill_quantity % 100 == 0
    assert order.fees > 0 and order.slippage > 0
    journal = AppendOnlyJournal(tmp_path / "orders.jsonl", "order_id")
    assert journal.append(order.to_dict())
    assert not journal.append(order.to_dict())


def test_925_to_930_cannot_fill_unconfirmed_price() -> None:
    now = datetime(2026, 1, 6, 9, 26, tzinfo=TZ)
    current_plan = plan()
    fresh = AuctionSnapshot(
        "SH600000",
        now,
        10.0,
        10.0,
        source_timestamp=now,
        source="TEST",
        final_open_confirmed=False,
    )
    decision, _ = evaluate_auction(current_plan, [fresh], now, AUCTION_CONFIG)
    order = simulate_order(
        Account(1_000_000),
        current_plan,
        decision,
        10.0,
        1_000_000,
        date(2026, 1, 6),
        now,
        FILL_CONFIG,
        previous_close=10.0,
        available_liquidity_shares=1_000_000,
        execution_price_confirmed=False,
    )
    assert order.status == "REJECTED"
    assert "UNCONFIRMED" in order.reason


@pytest.mark.parametrize(
    ("is_trading", "official_price", "cash", "reason"),
    [
        (False, 10.0, 1_000_000, "SUSPENDED"),
        (True, 11.0, 1_000_000, "LIMIT_UP"),
        (True, 10.0, 0.0, "CASH_RESERVE_GROSS_SECTOR_OR_POSITION_CAP"),
    ],
)
def test_execution_rejections(is_trading: bool, official_price: float, cash: float, reason: str) -> None:
    now = datetime(2026, 1, 6, 9, 25, tzinfo=TZ)
    current_plan = plan()
    decision, _ = evaluate_auction(current_plan, snapshots(10.0), now, AUCTION_CONFIG)
    order = simulate_order(
        Account(cash),
        current_plan,
        decision,
        official_price,
        1_000_000,
        date(2026, 1, 6),
        now,
        FILL_CONFIG,
        is_trading=is_trading,
        previous_close=10.0,
        available_liquidity_shares=1_000_000,
        execution_price_confirmed=True,
    )
    assert order.status == "REJECTED"
    assert order.reason == reason


def test_daily_new_position_and_turnover_caps() -> None:
    now = datetime(2026, 1, 6, 9, 25, tzinfo=TZ)
    current_plan = plan()
    decision, _ = evaluate_auction(current_plan, snapshots(10.0), now, AUCTION_CONFIG)
    order = simulate_order(
        Account(1_000_000),
        current_plan,
        decision,
        10.0,
        1_000_000,
        date(2026, 1, 6),
        now,
        FILL_CONFIG,
        previous_close=10.0,
        available_liquidity_shares=1_000_000,
        execution_price_confirmed=True,
        daily_new_positions=5,
    )
    assert order.reason == "MAX_DAILY_NEW_POSITIONS"
    turnover = simulate_order(
        Account(1_000_000),
        current_plan,
        decision,
        10.0,
        1_000_000,
        date(2026, 1, 6),
        now,
        FILL_CONFIG,
        previous_close=10.0,
        available_liquidity_shares=1_000_000,
        execution_price_confirmed=True,
        daily_turnover=0.29,
    )
    assert turnover.reason == "MAX_PORTFOLIO_TURNOVER"


def test_t_plus_one_and_position_cap_reject_orders() -> None:
    now = datetime(2026, 1, 6, 9, 25, tzinfo=TZ)
    reduce_plan = plan(
        model_signal="REDUCE",
        current_position_weight=0.08,
        target_position_weight=0.04,
    )
    decision, _ = evaluate_auction(reduce_plan, snapshots(10.0), now, AUCTION_CONFIG)
    account = Account(100_000)
    account.add("SH600000", 8_000, date(2026, 1, 6))
    order = simulate_order(
        account,
        reduce_plan,
        decision,
        10.0,
        1_000_000,
        date(2026, 1, 6),
        now,
        FILL_CONFIG,
        previous_close=10.0,
        available_liquidity_shares=1_000_000,
        execution_price_confirmed=True,
    )
    assert order.status == "REJECTED"
    assert order.reason == "T_PLUS_ONE_OR_NO_POSITION"


def test_replay_reconstructs_state(tmp_path: Path) -> None:
    now = datetime(2026, 1, 6, 9, 24, 45, tzinfo=TZ)
    current_plan = plan()
    decision, _ = evaluate_auction(current_plan, snapshots(10.0), now, AUCTION_CONFIG)
    plans_path, _ = archive_execution_plans([current_plan], tmp_path, "2026-01-06")
    decisions_path = tmp_path / "decisions.jsonl"
    orders_path = tmp_path / "orders.jsonl"
    decision_journal = AppendOnlyJournal(decisions_path, "decision_id")
    assert append_decision(decision_journal, decision)
    AppendOnlyJournal(orders_path, "order_id")
    replay = replay_execution_day(plans_path, decisions_path, orders_path)
    assert replay["plan_count"] == 1
    assert replay["decision_count"] == 1

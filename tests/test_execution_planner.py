from datetime import date

import pandas as pd
import pytest

from quant.execution.boundaries import calculate_boundaries
from quant.execution.models import ExecutionState
from quant.execution.planner import (
    build_execution_plan,
    price_contexts_from_bars,
    price_range_fields,
)

EXECUTION_CONFIG = {
    "rule_version": "execution-v1",
    "initial_entry_fraction": 0.5,
    "add_fraction": 0.25,
    "max_single_position": 0.10,
    "price_rounding": 0.01,
    "formal_mode": False,
    "calibration": {
        "source": "engineering_fallback_requires_walk_forward_calibration",
        "entry_low_atr": 0.4,
        "entry_high_atr": 0.25,
        "add_atr": 0.55,
        "chase_atr": 0.65,
        "reduce_atr": 1.0,
        "invalidation_atr": 1.1,
        "hard_exit_atr": 1.75,
    },
}


def signal(**changes):
    base = {
        "trade_date": "2026-01-05",
        "symbol": "SH600000",
        "name": "测试股份",
        "model_version": "m1",
        "model_signal": "BUY",
        "model_score": 0.9,
        "outperform_probability": 0.64,
        "predicted_excess_return_5d": 0.03,
        "signal_rank": 1,
        "current_position_weight": 0.0,
        "target_position_weight": 0.08,
        "market_state": "NORMAL",
        "sector_state": "NORMAL",
        "signal_valid_until": "2026-01-06",
        "data_quality_status": "PASS",
    }
    base.update(changes)
    return base


def context():
    return {
        "reference_close": 10.0,
        "atr20": 0.4,
        "support_20": 9.4,
        "resistance_20": 10.8,
        "volatility_20": 0.02,
    }


def test_boundaries_are_ordered_and_explained() -> None:
    result = calculate_boundaries(context(), EXECUTION_CONFIG)
    assert result.hard_exit_price < result.invalidation_price < result.entry_price_low
    assert result.entry_price_low < result.entry_price_high < result.chase_limit_price
    assert result.explanations["invalidation_price"]


def test_unheld_and_held_plans_are_distinct() -> None:
    unheld = build_execution_plan(signal(), context(), date(2026, 1, 6), EXECUTION_CONFIG)
    held = build_execution_plan(
        signal(current_position_weight=0.04), context(), date(2026, 1, 6), EXECUTION_CONFIG
    )
    assert unheld.initial_state == ExecutionState.READY_TO_OPEN
    assert unheld.initial_order_weight == pytest.approx(0.04)
    assert held.initial_state == ExecutionState.READY_TO_ADD
    assert held.add_order_weight == pytest.approx(0.02)
    assert unheld.plan_id != held.plan_id


def test_failed_quality_cannot_create_plan() -> None:
    for status in ("FAIL", "UNKNOWN"):
        with pytest.raises(ValueError, match="failed/unknown"):
            build_execution_plan(
                signal(data_quality_status=status),
                context(),
                date(2026, 1, 6),
                EXECUTION_CONFIG,
            )


def test_audited_warning_quality_can_create_simulation_plan() -> None:
    plan = build_execution_plan(
        signal(data_quality_status="WARN"),
        context(),
        date(2026, 1, 6),
        EXECUTION_CONFIG,
    )
    assert plan.data_quality_status == "WARN"


def test_formal_mode_requires_walk_forward_calibration() -> None:
    config = {**EXECUTION_CONFIG, "formal_mode": True}
    with pytest.raises(ValueError, match="walk-forward"):
        calculate_boundaries(context(), config)


def test_missing_calibration_cancels_new_position() -> None:
    gated_config = {**EXECUTION_CONFIG, "require_walk_forward_calibration": True}
    calibration_plan = build_execution_plan(
        signal(),
        context(),
        date(2026, 1, 6),
        gated_config,
    )
    assert calibration_plan.initial_state == ExecutionState.CANCELLED
    assert calibration_plan.add_order_weight == 0


def test_passed_calibration_allows_configured_plan() -> None:
    gated_config = {**EXECUTION_CONFIG, "require_walk_forward_calibration": True}
    calibrated = {**context(), "calibration_status": "WALK_FORWARD_PASS"}
    plan = build_execution_plan(signal(), calibrated, date(2026, 1, 6), gated_config)
    assert plan.initial_state == ExecutionState.READY_TO_OPEN


@pytest.mark.parametrize(
    ("market_state", "sector_state"),
    [("HIGH_RISK", "NORMAL"), ("NORMAL", "HIGH_RISK"), ("NORMAL", "DATA_UNAVAILABLE")],
)
def test_risk_filters_block_new_position_even_after_calibration(
    market_state: str,
    sector_state: str,
) -> None:
    calibrated = {**context(), "calibration_status": "WALK_FORWARD_PASS"}
    plan = build_execution_plan(
        signal(market_state=market_state, sector_state=sector_state),
        calibrated,
        date(2026, 1, 6),
        {**EXECUTION_CONFIG, "require_walk_forward_calibration": True},
    )

    assert plan.initial_state == ExecutionState.CANCELLED
    assert plan.initial_order_weight == 0
    assert "风险过滤" in plan.explanations["execution_gate"]


def test_context_uses_only_trailing_bars() -> None:
    dates = pd.bdate_range("2025-11-01", periods=30)
    bars = pd.DataFrame(
        {
            "trade_date": dates,
            "symbol": "SH600000",
            "high": [10 + i * 0.01 for i in range(30)],
            "low": [9.8 + i * 0.01 for i in range(30)],
            "close": [9.9 + i * 0.01 for i in range(30)],
        }
    )
    original = price_contexts_from_bars(bars)["SH600000"]
    future = pd.concat(
        [
            bars,
            pd.DataFrame(
                [
                    {
                        "trade_date": pd.Timestamp("2030-01-01"),
                        "symbol": "SH600000",
                        "high": 100,
                        "low": 1,
                        "close": 50,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    cutoff = future[future.trade_date <= dates[-1]]
    assert price_contexts_from_bars(cutoff)["SH600000"] == original


def test_daily_signal_price_ranges_are_ordered_and_auditable() -> None:
    result = price_range_fields(context(), EXECUTION_CONFIG)

    assert result["buy_price_low"] < result["buy_price_high"]
    assert result["sell_price_low"] <= result["sell_price_high"]
    assert result["signal_invalid_below"] < result["buy_price_low"]
    assert result["price_range_status"] == "RESEARCH_ONLY_UNCALIBRATED"
    assert "ATR20" in result["price_range_explanation"]


def test_daily_signal_price_ranges_mark_passed_calibration() -> None:
    result = price_range_fields(
        {
            **context(),
            "calibration_status": "WALK_FORWARD_PASS",
            "peer_mfe_q70": 0.05,
            "peer_mfe_q90": 0.08,
        },
        EXECUTION_CONFIG,
    )

    assert result["price_range_status"] == "WALK_FORWARD_CALIBRATED"
    assert result["sell_price_high"] == pytest.approx(10.8)


def test_price_context_continuously_scales_a_split_without_future_prices() -> None:
    dates = pd.bdate_range("2025-11-01", periods=30)
    raw_close = [10.0] * 15 + [5.0] * 15
    factors = [1.0] * 15 + [2.0] * 15
    bars = pd.DataFrame(
        {
            "trade_date": dates,
            "symbol": "SH600000",
            "high": [value + (0.1 if index >= 15 else 0.2) for index, value in enumerate(raw_close)],
            "low": [value - (0.1 if index >= 15 else 0.2) for index, value in enumerate(raw_close)],
            "close": raw_close,
            "adjust_factor": factors,
        }
    )

    adjusted = price_contexts_from_bars(bars)["SH600000"]
    unadjusted = price_contexts_from_bars(bars.drop(columns="adjust_factor"))["SH600000"]

    assert adjusted["reference_close"] == 5.0
    assert adjusted["atr20"] < 0.25
    assert unadjusted["atr20"] > 0.25
    assert adjusted["resistance_20"] < 5.2

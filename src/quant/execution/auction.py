from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from quant.execution.models import (
    Action,
    AuctionDataLevel,
    AuctionSnapshot,
    Decision,
    ExecutionPlan,
    ExecutionState,
    ModelSignal,
)


@dataclass(frozen=True)
class AuctionFeatures:
    auction_gap: float | None
    relative_auction_gap: float | None
    auction_price_stability: dict[str, float | int | None]
    matched_volume: float | None
    matched_amount: float | None
    matched_amount_ratio: float | None
    unmatched_buy_volume: float | None
    unmatched_sell_volume: float | None
    imbalance_ratio: float | None
    market_auction_return: float | None
    sector_auction_return: float | None


def _clock(value: str) -> time:
    return time.fromisoformat(value)


def auction_phase(moment: datetime, config: dict[str, Any]) -> str:
    local = moment.astimezone(ZoneInfo(config["timezone"])).time().replace(tzinfo=None)
    if local < _clock(config["collection_start"]):
        return "PRECHECK"
    if local < _clock(config["observation_only_end"]):
        return "OBSERVATION_ONLY"
    if local < _clock(config["decision_start"]):
        return "NON_CANCEL_OBSERVATION"
    if local < _clock(config["auction_end"]):
        return "FINAL_FILTER"
    if local < _clock(config["continuous_market_start"]):
        return "OPEN_CONFIRMATION"
    return "CONTINUOUS"


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return float(numerator / denominator)


def compute_auction_features(
    snapshots: Iterable[AuctionSnapshot],
    average_daily_amount: float | None = None,
    stability_start: str = "09:20:00",
) -> AuctionFeatures:
    ordered = sorted(snapshots, key=lambda item: item.observed_at)
    if not ordered:
        return AuctionFeatures(None, None, {}, None, None, None, None, None, None, None, None)
    latest = ordered[-1]
    usable = [
        item
        for item in ordered
        if item.auction_price is not None
        and item.auction_price > 0
        and item.observed_at.time().replace(tzinfo=None) >= _clock(stability_start)
    ]
    prices = np.asarray([float(item.auction_price) for item in usable], dtype=float)
    stability: dict[str, float | int | None]
    if prices.size:
        diffs = np.diff(prices)
        signs = np.sign(diffs[diffs != 0])
        reversals = int(np.sum(signs[1:] != signs[:-1])) if signs.size > 1 else 0
        last_time = usable[-1].observed_at
        last_60 = [
            float(item.auction_price)
            for item in usable
            if (last_time - item.observed_at).total_seconds() <= 60
        ]
        stability = {
            "snapshot_count": int(prices.size),
            "std_ratio": float(np.std(prices) / np.mean(prices)) if np.mean(prices) else None,
            "range_ratio": float((np.max(prices) - np.min(prices)) / prices[-1]),
            "last_60s_change": (float(last_60[-1] / last_60[0] - 1.0) if len(last_60) >= 2 else None),
            "direction_reversals": reversals,
        }
    else:
        stability = {
            "snapshot_count": 0,
            "std_ratio": None,
            "range_ratio": None,
            "last_60s_change": None,
            "direction_reversals": 0,
        }

    price = latest.auction_price if latest.auction_price and latest.auction_price > 0 else None
    gap = _safe_ratio(price, latest.previous_close)
    gap = gap - 1.0 if gap is not None else None
    relative = (
        gap - latest.market_auction_return
        if gap is not None and latest.market_auction_return is not None
        else None
    )
    denominator_values = (
        latest.matched_volume,
        latest.unmatched_buy_volume,
        latest.unmatched_sell_volume,
    )
    imbalance = None
    if all(value is not None and value >= 0 for value in denominator_values):
        total = sum(float(value) for value in denominator_values)
        imbalance = (
            (float(latest.unmatched_buy_volume) - float(latest.unmatched_sell_volume)) / total
            if total > 0
            else None
        )
    return AuctionFeatures(
        auction_gap=gap,
        relative_auction_gap=relative,
        auction_price_stability=stability,
        matched_volume=latest.matched_volume,
        matched_amount=latest.matched_amount,
        matched_amount_ratio=_safe_ratio(latest.matched_amount, average_daily_amount),
        unmatched_buy_volume=latest.unmatched_buy_volume,
        unmatched_sell_volume=latest.unmatched_sell_volume,
        imbalance_ratio=imbalance,
        market_auction_return=latest.market_auction_return,
        sector_auction_return=latest.sector_auction_return,
    )


def classify_data_level(
    snapshots: list[AuctionSnapshot],
    now: datetime,
    config: dict[str, Any],
) -> tuple[AuctionDataLevel, str]:
    if not snapshots:
        return AuctionDataLevel.C, "NO_AUCTION_SNAPSHOTS"
    latest = max(snapshots, key=lambda item: item.observed_at)
    if latest.source_timestamp is None:
        return AuctionDataLevel.C, "SOURCE_TIMESTAMP_MISSING"
    source_time = latest.source_timestamp
    age = (now - source_time).total_seconds()
    if age < -1 or age > float(config["max_data_age_seconds"]):
        return AuctionDataLevel.C, "STALE_AUCTION_DATA"
    if latest.auction_price is None or latest.auction_price <= 0:
        return AuctionDataLevel.C, "FINAL_PRICE_MISSING"
    full_fields = (
        latest.matched_volume,
        latest.matched_amount,
        latest.unmatched_buy_volume,
        latest.unmatched_sell_volume,
    )
    post_920 = [
        item for item in snapshots if item.observed_at.time().replace(tzinfo=None) >= _clock("09:20:00")
    ]
    if all(value is not None for value in full_fields) and len(post_920) >= int(
        config["minimum_snapshot_count"]
    ):
        return AuctionDataLevel.A, "FULL_AUCTION_DATA"
    return AuctionDataLevel.B, "FINAL_PRICE_ONLY"


def _decision_id(plan: ExecutionPlan, now: datetime, phase: str, action: Action) -> str:
    payload = {
        "plan": plan.plan_id,
        "time": now.isoformat(),
        "phase": phase,
        "action": action.value,
        "rule": plan.rule_version,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    return f"decision-{digest}"


def _decision(
    plan: ExecutionPlan,
    now: datetime,
    previous: ExecutionState,
    next_state: ExecutionState,
    action: Action,
    weight: float,
    price: float | None,
    rule: str,
    reason: str,
    level: AuctionDataLevel,
    source: str,
    final: bool,
) -> Decision:
    return Decision(
        decision_id=_decision_id(
            plan,
            now,
            auction_phase(
                now,
                {
                    "timezone": "Asia/Shanghai",
                    "collection_start": "09:15:00",
                    "observation_only_end": "09:20:00",
                    "decision_start": "09:24:30",
                    "auction_end": "09:25:00",
                    "continuous_market_start": "09:30:00",
                },
            ),
            action,
        ),
        plan_id=plan.plan_id,
        symbol=plan.symbol,
        decided_at=now.isoformat(),
        previous_state=previous.value,
        next_state=next_state.value,
        action=action.value,
        planned_weight=max(0.0, weight),
        trigger_price=price,
        trigger_rule=rule,
        reason=reason,
        data_level=level.value,
        data_source=source,
        model_version=plan.model_version,
        rule_version=plan.rule_version,
        final=final,
    )


def evaluate_auction(
    plan: ExecutionPlan,
    snapshots: list[AuctionSnapshot],
    now: datetime,
    config: dict[str, Any],
    current_state: ExecutionState | None = None,
) -> tuple[Decision, AuctionFeatures]:
    state = current_state or ExecutionState(plan.initial_state)
    phase = auction_phase(now, config)
    features = compute_auction_features(snapshots, stability_start=config["price_stability"]["window_start"])
    level, level_reason = classify_data_level(snapshots, now, config)
    latest = max(snapshots, key=lambda item: item.observed_at) if snapshots else None
    source = latest.source if latest else "NONE"
    price = latest.auction_price if latest else None

    if phase in {"PRECHECK", "OBSERVATION_ONLY", "NON_CANCEL_OBSERVATION"}:
        return (
            _decision(
                plan,
                now,
                state,
                state,
                Action.SKIP,
                0.0,
                price,
                "AUCTION_OBSERVATION_GATE",
                f"{phase}: snapshots recorded; final trading decision prohibited",
                level,
                source,
                False,
            ),
            features,
        )
    if level is AuctionDataLevel.C:
        return (
            _decision(
                plan,
                now,
                state,
                ExecutionState.WATCH if state not in {ExecutionState.READY_TO_EXIT} else state,
                Action.SKIP,
                0.0,
                None,
                "AUCTION_FALLBACK_C",
                f"{level_reason}; wait for first valid quote after 09:30",
                level,
                source,
                True,
            ),
            features,
        )

    if plan.model_signal == ModelSignal.EXIT.value or plan.target_position_weight <= 0:
        return (
            _decision(
                plan,
                now,
                state,
                ExecutionState.READY_TO_EXIT,
                Action.EXIT,
                plan.current_position_weight,
                price,
                "MODEL_EXIT",
                "Model signal is EXIT or target position is zero",
                level,
                source,
                True,
            ),
            features,
        )
    if plan.model_signal == ModelSignal.REDUCE.value or (
        plan.target_position_weight < plan.current_position_weight
    ):
        return (
            _decision(
                plan,
                now,
                state,
                ExecutionState.READY_TO_REDUCE,
                Action.REDUCE,
                plan.current_position_weight - plan.target_position_weight,
                price,
                "MODEL_REDUCE",
                "Model target weight has decreased",
                level,
                source,
                True,
            ),
            features,
        )

    risk_off = plan.market_state in {"HIGH_RISK", "RISK_OFF", "DATA_UNAVAILABLE"} or plan.sector_state in {
        "HIGH_RISK",
        "RISK_OFF",
        "DATA_UNAVAILABLE",
    }
    if risk_off and plan.target_position_weight > plan.current_position_weight:
        return (
            _decision(
                plan,
                now,
                state,
                ExecutionState.CANCELLED,
                Action.CANCEL,
                0.0,
                price,
                "RISK_FILTER",
                "Market or sector risk filter blocked a new position increase",
                level,
                source,
                True,
            ),
            features,
        )
    if price is None:
        raise AssertionError("B/A auction data must contain a price")
    if price <= plan.hard_exit_price and plan.current_position_weight > 0:
        action = Action.EXIT
        next_state = ExecutionState.READY_TO_EXIT
        weight = plan.current_position_weight
        rule = "HARD_EXIT_PRICE"
        reason = "Confirmed price is at or below the calibrated hard exit"
    elif price < plan.invalidation_price:
        action = Action.CANCEL
        next_state = ExecutionState.CANCELLED
        weight = 0.0
        rule = "SIGNAL_INVALIDATION"
        reason = "Confirmed price is below the signal invalidation line; averaging down is prohibited"
    elif state is ExecutionState.READY_TO_OPEN:
        if price > plan.chase_limit_price:
            action, next_state, weight = Action.CANCEL, ExecutionState.CANCELLED, 0.0
            rule, reason = "CHASE_LIMIT", "Confirmed price exceeded chase limit"
        elif plan.entry_price_low <= price <= plan.entry_price_high:
            action, next_state, weight = Action.OPEN, ExecutionState.READY_TO_OPEN, plan.initial_order_weight
            rule, reason = "ENTRY_ZONE", "Confirmed price is inside the calibrated entry zone"
        elif plan.entry_price_high < price <= plan.chase_limit_price:
            multiplier = float(config.get("reduced_entry_multiplier", 0.5))
            action, next_state = Action.OPEN, ExecutionState.READY_TO_OPEN
            weight = plan.initial_order_weight * multiplier
            rule, reason = "REDUCED_ENTRY_ZONE", "Gap is acceptable but entry size is reduced"
        else:
            action, next_state, weight = Action.SKIP, ExecutionState.WATCH, 0.0
            rule, reason = (
                "LOW_OPEN_CONFIRMATION",
                "Price is above invalidation but below entry zone; wait for 09:30",
            )
    elif state is ExecutionState.READY_TO_ADD:
        if price <= plan.add_price and price > plan.invalidation_price:
            action, next_state, weight = Action.ADD, ExecutionState.READY_TO_ADD, plan.add_order_weight
            rule, reason = "ADD_ZONE", "Price reached calibrated add line while signal remains valid"
        else:
            action, next_state, weight = Action.HOLD, ExecutionState.HOLDING, 0.0
            rule, reason = "ADD_CONDITIONS_NOT_MET", "No valid add trigger"
    else:
        action, next_state, weight = Action.HOLD, ExecutionState.HOLDING, 0.0
        rule, reason = "HOLD", "No opening, adding, reducing, or exit condition"

    return (
        _decision(
            plan,
            now,
            state,
            next_state,
            action,
            weight,
            price,
            rule,
            reason,
            level,
            source,
            True,
        ),
        features,
    )

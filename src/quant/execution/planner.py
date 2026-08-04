from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from quant.execution.boundaries import calculate_boundaries, round_to_tick
from quant.execution.models import ExecutionPlan, ExecutionState, ModelSignal

BUY_SIGNALS = {ModelSignal.STRONG_BUY.value, ModelSignal.BUY.value}


def price_range_fields(
    price_context: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Convert a trailing daily-bar context into auditable research price ranges."""
    boundaries = calculate_boundaries(price_context, config)
    resistance = float(price_context.get("resistance_20", boundaries.reduce_price))
    peer_mfe_high = price_context.get("peer_mfe_q90")
    if peer_mfe_high is not None and float(peer_mfe_high) > 0:
        calibrated_sell_high = float(price_context["reference_close"]) * (
            1.0 + float(peer_mfe_high)
        )
        resistance = min(resistance, calibrated_sell_high)
    sell_high = round_to_tick(
        max(boundaries.reduce_price, resistance),
        float(config["price_rounding"]),
    )
    calibrated = price_context.get("calibration_status") == "WALK_FORWARD_PASS"
    return {
        "reference_close": float(price_context["reference_close"]),
        "buy_price_low": boundaries.entry_price_low,
        "buy_price_high": boundaries.entry_price_high,
        "sell_price_low": boundaries.reduce_price,
        "sell_price_high": sell_high,
        "avoid_chasing_above": boundaries.chase_limit_price,
        "signal_invalid_below": boundaries.invalidation_price,
        "hard_exit_below": boundaries.hard_exit_price,
        "price_range_status": (
            "WALK_FORWARD_CALIBRATED" if calibrated else "RESEARCH_ONLY_UNCALIBRATED"
        ),
        "price_range_basis": boundaries.parameter_source,
        "price_range_explanation": json.dumps(
            {
                "buy_price_low": boundaries.explanations["entry_price_low"],
                "buy_price_high": boundaries.explanations["entry_price_high"],
                "sell_price_low": boundaries.explanations["reduce_price"],
                "sell_price_high": (
                    "历史OOS同类信号未来最大有利波动90%分位，"
                    "并受20日滚动压力位封顶"
                    if calibrated
                    else "20日滚动压力位，且不低于卖出区间下沿"
                ),
                "signal_invalid_below": boundaries.explanations["invalidation_price"],
                "hard_exit_below": boundaries.explanations["hard_exit_price"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def _plan_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode()
    return "plan-" + hashlib.sha256(encoded).hexdigest()[:24]


def _initial_state(signal: str, current: float, target: float) -> ExecutionState:
    if signal in BUY_SIGNALS and target > current:
        return ExecutionState.READY_TO_OPEN if current <= 0 else ExecutionState.READY_TO_ADD
    if signal == ModelSignal.EXIT.value or target <= 0 < current:
        return ExecutionState.READY_TO_EXIT
    if signal == ModelSignal.REDUCE.value or target < current:
        return ExecutionState.READY_TO_REDUCE
    if current > 0:
        return ExecutionState.HOLDING
    if signal == ModelSignal.AVOID.value:
        return ExecutionState.CANCELLED
    return ExecutionState.WATCH


def build_execution_plan(
    signal: dict[str, Any],
    price_context: dict[str, Any],
    execution_date: date,
    config: dict[str, Any],
    *,
    apply_risk_gate: bool = True,
) -> ExecutionPlan:
    required = {
        "trade_date",
        "symbol",
        "name",
        "model_version",
        "model_signal",
        "model_score",
        "outperform_probability",
        "predicted_excess_return_5d",
        "signal_rank",
        "current_position_weight",
        "target_position_weight",
        "market_state",
        "sector_state",
        "signal_valid_until",
        "data_quality_status",
    }
    if missing := required - set(signal):
        raise ValueError(f"Signal missing execution fields: {sorted(missing)}")
    if signal["data_quality_status"] not in {"PASS", "WARN"}:
        raise ValueError("Execution plan cannot be created from failed/unknown data")

    boundaries = calculate_boundaries(price_context, config)
    current = max(0.0, float(signal["current_position_weight"]))
    target = min(
        max(0.0, float(signal["target_position_weight"])),
        float(config["max_single_position"]),
    )
    gap = max(0.0, target - current)
    initial_weight = min(gap, target * float(config["initial_entry_fraction"]))
    add_weight = min(gap, target * float(config["add_fraction"]))
    state = _initial_state(str(signal["model_signal"]), current, target)
    entry_or_add = state in {ExecutionState.READY_TO_OPEN, ExecutionState.READY_TO_ADD}
    calibration_blocked = bool(config.get("require_walk_forward_calibration")) and (
        price_context.get("calibration_status") != "WALK_FORWARD_PASS"
    )
    risk_blocked = apply_risk_gate and (
        signal["market_state"] == "HIGH_RISK"
        or signal["sector_state"] in {"HIGH_RISK", "DATA_UNAVAILABLE"}
    )
    if entry_or_add and (calibration_blocked or risk_blocked):
        state = ExecutionState.CANCELLED if current <= 0 else ExecutionState.HOLDING
        initial_weight = 0.0
        add_weight = 0.0
    tz = ZoneInfo("Asia/Shanghai")
    valid_from = datetime.combine(execution_date, time(9, 15), tzinfo=tz)
    signal_until = pd.Timestamp(signal["signal_valid_until"]).date()
    valid_until = datetime.combine(signal_until, time(15, 0), tzinfo=tz)

    identity = {
        "execution_date": execution_date.isoformat(),
        "symbol": signal["symbol"],
        "model_version": signal["model_version"],
        "rule_version": config["rule_version"],
        "signal": signal["model_signal"],
        "current": current,
        "target": target,
        "initial_state": state.value,
        "boundaries": boundaries.__dict__,
        "calibration_blocked": calibration_blocked,
        "risk_blocked": risk_blocked,
        "risk_gate_mode": "ENABLED" if apply_risk_gate else "DISABLED_SHADOW",
    }
    explanations = dict(boundaries.explanations)
    if calibration_blocked:
        explanations["execution_gate"] = "未通过滚动样本外价格边界校准，禁止新增或加仓"
    elif risk_blocked:
        explanations["execution_gate"] = "市场或行业风险过滤未通过，禁止新增或加仓"
    return ExecutionPlan(
        plan_id=_plan_id(identity),
        trade_date=str(signal["trade_date"]),
        execution_date=execution_date.isoformat(),
        symbol=str(signal["symbol"]),
        name=str(signal["name"]),
        model_version=str(signal["model_version"]),
        model_signal=str(signal["model_signal"]),
        model_score=float(signal["model_score"]),
        outperform_probability=float(signal["outperform_probability"]),
        predicted_excess_return_5d=float(signal["predicted_excess_return_5d"]),
        signal_rank=int(signal["signal_rank"]),
        current_position_weight=current,
        target_position_weight=target,
        market_state=str(signal["market_state"]),
        sector_state=str(signal["sector_state"]),
        data_quality_status=str(signal["data_quality_status"]),
        reference_close=float(price_context["reference_close"]),
        entry_price_low=boundaries.entry_price_low,
        entry_price_high=boundaries.entry_price_high,
        add_price=boundaries.add_price,
        chase_limit_price=boundaries.chase_limit_price,
        reduce_price=boundaries.reduce_price,
        invalidation_price=boundaries.invalidation_price,
        hard_exit_price=boundaries.hard_exit_price,
        initial_order_weight=initial_weight,
        add_order_weight=add_weight,
        max_position_weight=float(config["max_single_position"]),
        valid_from=valid_from.isoformat(),
        valid_until=valid_until.isoformat(),
        execution_priority=int(signal["signal_rank"]),
        initial_state=state.value,
        parameter_source=boundaries.parameter_source,
        rule_version=str(config["rule_version"]),
        explanations=explanations,
        sector_name=(
            None
            if pd.isna(signal.get("sector_name"))
            else str(signal.get("sector_name"))
        ),
    )


def build_execution_plans(
    signals: pd.DataFrame,
    price_contexts: dict[str, dict[str, Any]],
    execution_date: date,
    config: dict[str, Any],
    *,
    apply_risk_gate: bool = True,
) -> list[ExecutionPlan]:
    plans: list[ExecutionPlan] = []
    for record in signals.sort_values(["signal_rank", "symbol"]).to_dict("records"):
        symbol = str(record["symbol"])
        if symbol not in price_contexts:
            continue
        plans.append(
            build_execution_plan(
                record,
                price_contexts[symbol],
                execution_date,
                config,
                apply_risk_gate=apply_risk_gate,
            )
        )
    return plans


def price_contexts_from_bars(bars: pd.DataFrame) -> dict[str, dict[str, Any]]:
    required = {"symbol", "trade_date", "high", "low", "close"}
    if missing := required - set(bars):
        raise ValueError(f"Bars missing execution context fields: {sorted(missing)}")
    frame = bars.sort_values(["symbol", "trade_date"]).copy()
    price_columns = ["high", "low", "close"]
    if "adjust_factor" in frame:
        factor = pd.to_numeric(frame["adjust_factor"], errors="coerce")
        latest_factor = factor.groupby(frame["symbol"]).transform("last")
        scale = factor / latest_factor
        if scale.isna().any() or scale.le(0).any():
            raise ValueError("Adjustment factors must be positive and complete")
        for column in price_columns:
            frame[f"_continuous_{column}"] = frame[column] * scale
    else:
        for column in price_columns:
            frame[f"_continuous_{column}"] = frame[column]

    previous = frame.groupby("symbol")["_continuous_close"].shift(1)
    true_range = pd.concat(
        [
            frame["_continuous_high"] - frame["_continuous_low"],
            (frame["_continuous_high"] - previous).abs(),
            (frame["_continuous_low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["atr20"] = (
        true_range.groupby(frame["symbol"]).rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    )
    frame["support_20"] = (
        frame["_continuous_low"]
        .groupby(frame["symbol"])
        .rolling(20, min_periods=20)
        .min()
        .reset_index(level=0, drop=True)
    )
    frame["resistance_20"] = (
        frame["_continuous_high"]
        .groupby(frame["symbol"])
        .rolling(20, min_periods=20)
        .max()
        .reset_index(level=0, drop=True)
    )
    frame["return_1"] = frame.groupby("symbol")["_continuous_close"].pct_change(fill_method=None)
    frame["volatility_20"] = (
        frame["return_1"]
        .groupby(frame["symbol"])
        .rolling(20, min_periods=20)
        .std()
        .reset_index(level=0, drop=True)
    )
    latest = frame.groupby("symbol", sort=True).tail(1)
    contexts: dict[str, dict[str, Any]] = {}
    for row in latest.to_dict("records"):
        values = {
            "reference_close": row["close"],
            "atr20": row["atr20"],
            "support_20": row["support_20"],
            "resistance_20": row["resistance_20"],
            "volatility_20": row["volatility_20"],
        }
        if all(pd.notna(value) for value in values.values()):
            contexts[str(row["symbol"])] = {key: float(value) for key, value in values.items()}
    return contexts

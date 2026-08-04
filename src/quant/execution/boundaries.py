from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any


@dataclass(frozen=True)
class PriceBoundaries:
    entry_price_low: float
    entry_price_high: float
    add_price: float
    chase_limit_price: float
    reduce_price: float
    invalidation_price: float
    hard_exit_price: float
    explanations: dict[str, str]
    parameter_source: str


def round_to_tick(value: float, tick: float = 0.01) -> float:
    if value <= 0 or tick <= 0:
        raise ValueError("Price and tick must be positive")
    units = (Decimal(str(value)) / Decimal(str(tick))).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return float(units * Decimal(str(tick)))


def _positive(context: dict[str, Any], key: str) -> float | None:
    value = context.get(key)
    if value is None:
        return None
    number = float(value)
    return number if number > 0 else None


def calculate_boundaries(
    context: dict[str, Any],
    config: dict[str, Any],
) -> PriceBoundaries:
    close = float(context["reference_close"])
    atr = _positive(context, "atr20")
    if atr is None:
        volatility = max(
            float(context.get("volatility_20", 0.0)),
            float(context.get("volatility_10", 0.0)),
            float(context.get("volatility_5", 0.0)),
        )
        if volatility <= 0:
            raise ValueError("ATR20 or a positive trailing volatility is required")
        atr = close * volatility

    calibration = config["calibration"]
    parameters = {
        key: float(context.get(key, calibration[key]))
        for key in (
            "entry_low_atr",
            "entry_high_atr",
            "add_atr",
            "chase_atr",
            "reduce_atr",
            "invalidation_atr",
            "hard_exit_atr",
        )
    }
    source = str(
        context.get("calibration_source")
        or calibration.get("source", "engineering_fallback_requires_walk_forward_calibration")
    )
    if bool(config.get("formal_mode")) and context.get("calibration_status") != "WALK_FORWARD_PASS":
        raise ValueError("Formal execution requires a passed walk-forward boundary calibration")

    tick = float(config["price_rounding"])
    support = _positive(context, "support_20")
    resistance = _positive(context, "resistance_20")
    peer_mfe = _positive(context, "peer_mfe_q70")
    peer_mae = _positive(context, "peer_mae_q80")

    raw_invalidation = (
        close * (1.0 - peer_mae)
        if peer_mae is not None and peer_mae < 1
        else close - parameters["invalidation_atr"] * atr
    )
    if support is not None and support < close:
        raw_invalidation = min(close - tick, max(raw_invalidation, support - 0.10 * atr))
    raw_hard_exit = min(
        raw_invalidation - tick,
        close - parameters["hard_exit_atr"] * atr,
    )

    raw_entry_low = close - parameters["entry_low_atr"] * atr
    raw_entry_high = close + parameters["entry_high_atr"] * atr
    raw_add = close - parameters["add_atr"] * atr
    raw_chase = close + parameters["chase_atr"] * atr
    raw_reduce = (
        close * (1.0 + peer_mfe)
        if peer_mfe is not None
        else close + parameters["reduce_atr"] * atr
    )
    if resistance is not None and resistance > close:
        raw_reduce = min(raw_reduce, resistance)

    invalidation = round_to_tick(raw_invalidation, tick)
    hard_exit = round_to_tick(raw_hard_exit, tick)
    entry_low = round_to_tick(max(raw_entry_low, raw_invalidation + tick), tick)
    entry_high = round_to_tick(max(raw_entry_high, raw_entry_low + tick), tick)
    add_price = round_to_tick(min(raw_add, raw_entry_low), tick)
    chase = round_to_tick(max(raw_chase, raw_entry_high + tick), tick)
    reduce = round_to_tick(max(raw_reduce, raw_entry_high + tick), tick)

    explanations = {
        "entry_price_low": (
            f"昨日收盘价 - {parameters['entry_low_atr']:.4f} × ATR20；且必须高于信号失效线"
        ),
        "entry_price_high": f"昨日收盘价 + {parameters['entry_high_atr']:.4f} × ATR20",
        "add_price": (
            f"昨日收盘价 - {parameters['add_atr']:.4f} × ATR20；仅在模型仍为BUY且未跌破失效线时有效"
        ),
        "chase_limit_price": f"昨日收盘价 + {parameters['chase_atr']:.4f} × ATR20",
        "reduce_price": (
            "历史同类信号未来最大有利波动70%分位"
            if peer_mfe is not None
            else f"工程验证回退：昨日收盘价 + {parameters['reduce_atr']:.4f} × ATR20"
        ),
        "invalidation_price": (
            "历史同类信号未来最大不利波动80%分位"
            if peer_mae is not None
            else f"昨日收盘价 - {parameters['invalidation_atr']:.4f} × ATR20"
        ),
        "hard_exit_price": f"昨日收盘价 - {parameters['hard_exit_atr']:.4f} × ATR20",
    }
    return PriceBoundaries(
        entry_low,
        entry_high,
        add_price,
        chase,
        reduce,
        invalidation,
        hard_exit,
        explanations,
        source,
    )

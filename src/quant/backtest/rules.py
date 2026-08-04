from __future__ import annotations

from datetime import date
from decimal import ROUND_HALF_UP, Decimal


def price_limit_ratio(symbol: str, trade_date: date, is_st: bool = False) -> float:
    if is_st:
        return 0.05
    if symbol.startswith("BJ"):
        return 0.30
    code = symbol[2:]
    if symbol.startswith("SH") and code.startswith("68"):
        return 0.20
    if symbol.startswith("SZ") and code.startswith("30"):
        return 0.20 if trade_date >= date(2020, 8, 24) else 0.10
    return 0.10


def rounded_limit_price(previous_close: float, ratio: float, direction: str) -> float:
    multiplier = Decimal("1") + Decimal(str(ratio)) * (Decimal("1") if direction == "up" else Decimal("-1"))
    return float(
        (Decimal(str(previous_close)) * multiplier).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    )


def is_order_blocked_by_limit(
    side: str,
    open_price: float,
    previous_close: float,
    symbol: str,
    trade_date: date,
    is_st: bool = False,
) -> bool:
    ratio = price_limit_ratio(symbol, trade_date, is_st)
    if side == "BUY":
        return open_price >= rounded_limit_price(previous_close, ratio, "up") - 1e-9
    if side == "SELL":
        return open_price <= rounded_limit_price(previous_close, ratio, "down") + 1e-9
    raise ValueError(f"Unsupported side: {side}")


def stamp_duty_rate(trade_date: date, schedule: list[dict]) -> float:
    for period in schedule:
        if date.fromisoformat(period["start"]) <= trade_date <= date.fromisoformat(period["end"]):
            return float(period["rate"])
    raise ValueError(f"No stamp-duty rate configured for {trade_date}")

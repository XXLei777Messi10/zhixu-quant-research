from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class ModelSignal(StrEnum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    EXIT = "EXIT"
    AVOID = "AVOID"


class ExecutionState(StrEnum):
    NO_SIGNAL = "NO_SIGNAL"
    WATCH = "WATCH"
    READY_TO_OPEN = "READY_TO_OPEN"
    OPENED = "OPENED"
    READY_TO_ADD = "READY_TO_ADD"
    HOLDING = "HOLDING"
    READY_TO_REDUCE = "READY_TO_REDUCE"
    READY_TO_EXIT = "READY_TO_EXIT"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    DATA_ERROR = "DATA_ERROR"


class Action(StrEnum):
    OPEN = "OPEN"
    ADD = "ADD"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    EXIT = "EXIT"
    CANCEL = "CANCEL"
    SKIP = "SKIP"


class OrderStatus(StrEnum):
    PLANNED = "PLANNED"
    TRIGGERED = "TRIGGERED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class AuctionDataLevel(StrEnum):
    A = "A"
    B = "B"
    C = "C"


@dataclass(frozen=True)
class ExecutionPlan:
    plan_id: str
    trade_date: str
    execution_date: str
    symbol: str
    name: str
    model_version: str
    model_signal: str
    model_score: float
    outperform_probability: float
    predicted_excess_return_5d: float
    signal_rank: int
    current_position_weight: float
    target_position_weight: float
    market_state: str
    sector_state: str
    data_quality_status: str
    reference_close: float
    entry_price_low: float
    entry_price_high: float
    add_price: float
    chase_limit_price: float
    reduce_price: float
    invalidation_price: float
    hard_exit_price: float
    initial_order_weight: float
    add_order_weight: float
    max_position_weight: float
    valid_from: str
    valid_until: str
    execution_priority: int
    initial_state: str
    parameter_source: str
    rule_version: str
    explanations: dict[str, str] = field(default_factory=dict)
    sector_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AuctionSnapshot:
    symbol: str
    observed_at: datetime
    auction_price: float | None
    previous_close: float
    matched_volume: float | None = None
    matched_amount: float | None = None
    unmatched_buy_volume: float | None = None
    unmatched_sell_volume: float | None = None
    market_auction_return: float | None = None
    sector_auction_return: float | None = None
    source: str = "UNKNOWN"
    source_timestamp: datetime | None = None
    final_open_confirmed: bool = False


@dataclass(frozen=True)
class Decision:
    decision_id: str
    plan_id: str
    symbol: str
    decided_at: str
    previous_state: str
    next_state: str
    action: str
    planned_weight: float
    trigger_price: float | None
    trigger_rule: str
    reason: str
    data_level: str
    data_source: str
    model_version: str
    rule_version: str
    final: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SimulatedOrder:
    order_id: str
    plan_id: str
    symbol: str
    action: str
    planned_weight: float
    planned_quantity: int
    trigger_price: float | None
    fill_price: float | None
    fill_quantity: int
    fees: float
    slippage: float
    created_at: str
    filled_at: str | None
    status: str
    reason: str
    data_source: str
    model_version: str
    rule_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

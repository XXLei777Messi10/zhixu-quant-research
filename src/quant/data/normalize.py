from __future__ import annotations

from datetime import UTC, datetime
from typing import Final

import numpy as np
import pandas as pd

STANDARD_COLUMNS: Final[list[str]] = [
    "symbol",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turnover",
    "adjust_factor",
    "is_trading",
    "source",
    "retrieved_at",
    "adjustment",
    "quality_status",
    "request_id",
    "is_st",
]

AKSHARE_REQUIRED = {"日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额"}
AKSHARE_TX_REQUIRED = {"date", "open", "close", "high", "low", "volume", "amount"}
BAOSTOCK_REQUIRED = {
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "tradestatus",
}


def canonical_symbol(value: str) -> str:
    text = value.strip().upper()
    if text.startswith(("SH", "SZ", "BJ")) and "." not in text:
        return text
    if "." in text:
        market, code = text.split(".", maxsplit=1)
        return f"{market}{code}"
    if len(text) != 6 or not text.isdigit():
        raise ValueError(f"Unsupported A-share symbol: {value!r}")
    if text.startswith(("6", "9")):
        return f"SH{text}"
    if text.startswith(("4", "8")):
        return f"BJ{text}"
    return f"SZ{text}"


def provider_symbol(symbol: str, source: str) -> str:
    canonical = canonical_symbol(symbol)
    market, code = canonical[:2], canonical[2:]
    if source == "akshare":
        return code
    if source == "akshare_tx":
        return f"{market.lower()}{code}"
    if source == "baostock":
        return f"{market.lower()}.{code}"
    raise ValueError(f"Unknown provider: {source}")


def _coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _finish(
    frame: pd.DataFrame,
    symbol: str,
    source: str,
    retrieved_at: datetime,
    adjustment: str,
    request_id: str,
) -> pd.DataFrame:
    frame["symbol"] = canonical_symbol(symbol)
    frame["source"] = source
    frame["retrieved_at"] = pd.Timestamp(retrieved_at.astimezone(UTC))
    frame["adjustment"] = adjustment
    frame["quality_status"] = "UNCHECKED"
    frame["request_id"] = request_id
    for column in STANDARD_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["is_trading"] = frame["is_trading"].astype("boolean")
    frame["is_st"] = frame["is_st"].astype("boolean")
    return frame[STANDARD_COLUMNS].sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def normalize_akshare_bars(
    raw: pd.DataFrame,
    symbol: str,
    retrieved_at: datetime,
    adjustment: str,
    request_id: str,
    volume_in_lots: bool = True,
) -> pd.DataFrame:
    missing = AKSHARE_REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(f"AKShare schema changed, missing columns: {sorted(missing)}")
    frame = raw.rename(
        columns={
            "日期": "trade_date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover",
        }
    ).copy()
    _coerce_numeric(frame, ["open", "high", "low", "close", "volume", "amount", "turnover"])
    if volume_in_lots:
        frame["volume"] *= 100.0
    if "turnover" in frame:
        frame["turnover"] /= 100.0
    frame["adjust_factor"] = np.nan
    frame["is_trading"] = frame[["open", "high", "low", "close"]].notna().all(axis=1)
    frame["is_st"] = False
    return _finish(frame, symbol, "akshare", retrieved_at, adjustment, request_id)


def normalize_akshare_tx_bars(
    raw: pd.DataFrame,
    symbol: str,
    retrieved_at: datetime,
    adjustment: str,
    request_id: str,
    volume_multiplier: float | None = None,
) -> pd.DataFrame:
    missing = AKSHARE_TX_REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(f"AKShare Tencent schema changed, missing columns: {sorted(missing)}")
    frame = raw.rename(columns={"date": "trade_date"}).copy()
    _coerce_numeric(frame, ["open", "high", "low", "close", "volume", "amount", "turnover"])
    if volume_multiplier is None:
        volume_multiplier = infer_akshare_tx_volume_multiplier(frame)
    frame["volume"] *= volume_multiplier
    frame["adjust_factor"] = np.nan
    frame["is_trading"] = frame[["open", "high", "low", "close"]].notna().all(axis=1)
    frame["is_st"] = False
    return _finish(frame, symbol, "akshare", retrieved_at, adjustment, request_id)


def infer_akshare_tx_volume_multiplier(raw: pd.DataFrame) -> float:
    frame = raw.copy()
    _coerce_numeric(frame, ["close", "volume", "amount"])
    unit_sample = frame.loc[
        frame["close"].gt(0) & frame["volume"].gt(0) & frame["amount"].gt(0),
        ["close", "volume", "amount"],
    ]
    if unit_sample.empty:
        raise ValueError("AKShare Tencent volume unit cannot be inferred from empty notional data")
    amount_to_notional = (unit_sample["amount"] / (unit_sample["close"] * unit_sample["volume"])).median()
    if 20.0 <= amount_to_notional <= 200.0:
        return 100.0
    if 0.2 <= amount_to_notional <= 5.0:
        return 1.0
    raise ValueError(
        f"AKShare Tencent volume unit is ambiguous: median amount/notional ratio={amount_to_notional:.6f}"
    )


def normalize_baostock_bars(
    raw: pd.DataFrame,
    symbol: str,
    retrieved_at: datetime,
    adjustment: str,
    request_id: str,
) -> pd.DataFrame:
    missing = BAOSTOCK_REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(f"Baostock schema changed, missing columns: {sorted(missing)}")
    frame = raw.rename(columns={"date": "trade_date", "turn": "turnover"}).copy()
    _coerce_numeric(
        frame,
        ["open", "high", "low", "close", "volume", "amount", "turnover", "tradestatus", "isST"],
    )
    if "turnover" in frame:
        frame["turnover"] /= 100.0
    frame["is_trading"] = frame["tradestatus"].eq(1)
    frame["is_st"] = frame.get("isST", pd.Series(False, index=frame.index)).fillna(0).eq(1)
    frame["adjust_factor"] = np.nan
    return _finish(frame, symbol, "baostock", retrieved_at, adjustment, request_id)


def add_adjustment_factor(raw: pd.DataFrame, hfq: pd.DataFrame) -> pd.DataFrame:
    keys = ["symbol", "trade_date"]
    adjusted = hfq[keys + ["open", "high", "low", "close"]].rename(
        columns={column: f"hfq_{column}" for column in ["open", "high", "low", "close"]}
    )
    result = raw.merge(adjusted, on=keys, how="left", validate="one_to_one")
    result["adjust_factor"] = result["hfq_close"] / result["close"]
    return result

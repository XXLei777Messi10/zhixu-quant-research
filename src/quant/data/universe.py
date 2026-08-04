from __future__ import annotations

import os
import uuid
from pathlib import Path

import pandas as pd

from quant.data.normalize import canonical_symbol


def normalize_constituents(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"trade_date", "symbol"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Constituent data requires {sorted(required)}")
    result = frame[["trade_date", "symbol"]].copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.normalize()
    result["symbol"] = result["symbol"].map(canonical_symbol)
    return result.drop_duplicates().sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def write_constituents(frame: pd.DataFrame, path: Path) -> None:
    normalized = normalize_constituents(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = pd.read_parquet(path)
        normalized = normalize_constituents(pd.concat([previous, normalized], ignore_index=True))
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    normalized.to_parquet(temp, index=False, compression="zstd")
    os.replace(temp, path)


def dynamic_universe(
    bars: pd.DataFrame,
    min_history_days: int = 250,
    liquidity_window: int = 60,
    min_median_amount: float = 20_000_000,
    max_symbols: int = 300,
) -> pd.DataFrame:
    ordered = bars.sort_values(["symbol", "trade_date"]).copy()
    if isinstance(ordered["symbol"].dtype, pd.CategoricalDtype):
        ordered["symbol"] = ordered["symbol"].cat.remove_unused_categories()
    grouped = ordered.groupby("symbol", group_keys=False)
    ordered["history_days"] = grouped.cumcount() + 1
    ordered["median_amount"] = (
        grouped["amount"].rolling(liquidity_window).median().reset_index(level=0, drop=True)
    )
    eligible = ordered[
        ordered["history_days"].ge(min_history_days)
        & ordered["median_amount"].ge(min_median_amount)
        & ordered["is_trading"].fillna(False)
    ].copy()
    eligible["liquidity_rank"] = eligible.groupby("trade_date")["median_amount"].rank(
        ascending=False, method="first"
    )
    selected = eligible.loc[
        eligible["liquidity_rank"].le(max_symbols),
        ["trade_date", "symbol", "median_amount", "liquidity_rank"],
    ]
    return selected.sort_values(["trade_date", "liquidity_rank"]).reset_index(drop=True)

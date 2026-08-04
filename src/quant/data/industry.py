from __future__ import annotations

import os
import uuid
from pathlib import Path

import pandas as pd

INDUSTRY_COLUMNS = [
    "as_of_date",
    "symbol",
    "stock_name",
    "industry_name",
    "industry_classification",
    "mapping_update_date",
    "source",
    "retrieved_at",
    "request_id",
]
INDUSTRY_KEY = ["as_of_date", "symbol", "industry_classification"]


def _partition_path(root: Path, as_of: pd.Timestamp) -> Path:
    return root / f"as_of_date={pd.Timestamp(as_of).date().isoformat()}" / "membership.parquet"


def write_industry_snapshot(frame: pd.DataFrame, root: Path) -> int:
    if missing := set(INDUSTRY_COLUMNS) - set(frame):
        raise ValueError(f"Industry snapshot missing: {sorted(missing)}")
    incoming = frame[INDUSTRY_COLUMNS].copy()
    for column in ("as_of_date", "mapping_update_date", "retrieved_at"):
        incoming[column] = pd.to_datetime(incoming[column], errors="coerce")
    if incoming["as_of_date"].isna().any() or incoming["symbol"].isna().any():
        raise ValueError("Industry snapshot contains an invalid date or symbol")
    inserted = 0
    for as_of, part in incoming.groupby("as_of_date", sort=True):
        path = _partition_path(root, pd.Timestamp(as_of))
        previous = (
            pd.read_parquet(path)
            if path.exists()
            else pd.DataFrame(columns=INDUSTRY_COLUMNS)
        )
        before = len(previous)
        merged = (
            part.copy()
            if previous.empty
            else pd.concat([previous, part], ignore_index=True)
        )
        merged = (
            merged.sort_values("retrieved_at")
            .drop_duplicates(INDUSTRY_KEY, keep="last")
            .sort_values(["as_of_date", "symbol"])
            .reset_index(drop=True)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        merged.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
        inserted += max(0, len(merged) - before)
    return inserted


def load_industry_snapshot(
    root: Path,
    as_of: pd.Timestamp,
    max_age_days: int,
) -> pd.DataFrame:
    candidates: list[tuple[pd.Timestamp, Path]] = []
    for path in root.glob("as_of_date=*/membership.parquet"):
        try:
            snapshot_date = pd.Timestamp(path.parent.name.split("=", maxsplit=1)[1])
        except (IndexError, ValueError):
            continue
        if snapshot_date <= pd.Timestamp(as_of).normalize():
            candidates.append((snapshot_date, path))
    if not candidates:
        return pd.DataFrame(columns=INDUSTRY_COLUMNS)
    snapshot_date, path = max(candidates, key=lambda item: item[0])
    frame = pd.read_parquet(path)
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"]).dt.normalize()
    as_of = pd.Timestamp(as_of).normalize()
    if (as_of - snapshot_date).days > max_age_days:
        return pd.DataFrame(columns=INDUSTRY_COLUMNS)
    return (
        frame[frame["as_of_date"].eq(snapshot_date)]
        .sort_values(["symbol", "retrieved_at"])
        .drop_duplicates("symbol", keep="last")
        .reset_index(drop=True)
    )


def synthetic_industry_snapshot(
    frame: pd.DataFrame,
    as_of_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    bars = frame[frame["symbol"].ne("SH000300")]
    symbols = sorted(bars["symbol"].astype(str).unique())
    if as_of_dates is None:
        as_of_dates = pd.DatetimeIndex([pd.Timestamp(bars["trade_date"].max()).normalize()])
    retrieved_at = pd.Timestamp(bars["retrieved_at"].max())
    snapshots = []
    for as_of in as_of_dates:
        snapshots.append(
            pd.DataFrame(
                {
                    "as_of_date": pd.Timestamp(as_of).normalize(),
                    "symbol": symbols,
                    "stock_name": symbols,
                    "industry_name": [
                        f"SYNTHETIC_SECTOR_{index % 4}" for index in range(len(symbols))
                    ],
                    "industry_classification": "SYNTHETIC_TEST_ONLY",
                    "mapping_update_date": pd.Timestamp(as_of).normalize(),
                    "source": "synthetic",
                    "retrieved_at": retrieved_at,
                    "request_id": "synthetic-industry",
                }
            )
        )
    return pd.concat(snapshots, ignore_index=True)[INDUSTRY_COLUMNS]

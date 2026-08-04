from __future__ import annotations

import os
import uuid
from pathlib import Path

import pandas as pd

from quant.data.normalize import STANDARD_COLUMNS, add_adjustment_factor

PRIMARY_KEY = ["source", "adjustment", "symbol", "trade_date"]


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_parquet(temp, index=False, compression="zstd")
    os.replace(temp, path)


class ParquetStore:
    def __init__(self, root: Path):
        self.root = root

    def path_for(self, source: str, adjustment: str, symbol: str) -> Path:
        return (
            self.root / f"source={source}" / f"adjustment={adjustment}" / f"symbol={symbol}" / "bars.parquet"
        )

    def upsert(self, frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0
        inserted = 0
        for (source, adjustment, symbol), part in frame.groupby(
            ["source", "adjustment", "symbol"], sort=False, observed=True
        ):
            path = self.path_for(str(source), str(adjustment), str(symbol))
            previous = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=STANDARD_COLUMNS)
            before = len(previous)
            merged = (
                part.copy().reset_index(drop=True)
                if previous.empty
                else pd.concat([previous, part], ignore_index=True)
            )
            merged = merged.sort_values("retrieved_at").drop_duplicates(PRIMARY_KEY, keep="last")
            merged = merged.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
            _atomic_parquet(merged, path)
            inserted += max(0, len(merged) - before)
        return inserted

    def read(
        self,
        source: str | None = None,
        adjustment: str | None = None,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        paths = list(self.root.glob("source=*/adjustment=*/symbol=*/bars.parquet"))
        selected: list[Path] = []
        for path in paths:
            parts = {part.split("=", 1)[0]: part.split("=", 1)[1] for part in path.parts if "=" in part}
            if source is not None and parts.get("source") != source:
                continue
            if adjustment is not None and parts.get("adjustment") != adjustment:
                continue
            if symbols is not None and parts.get("symbol") not in symbols:
                continue
            selected.append(path)
        if not selected:
            return pd.DataFrame(columns=STANDARD_COLUMNS)
        return pd.concat([pd.read_parquet(path) for path in selected], ignore_index=True)

    def build_curated(
        self,
        output: Path,
        accepted_symbols: list[str] | None = None,
        excluded_records: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        raw = self.read("akshare", "raw", accepted_symbols)
        hfq = self.read("akshare", "hfq", accepted_symbols)
        if excluded_records is not None and not excluded_records.empty:
            exclusions = excluded_records[
                excluded_records["source"].eq("akshare") & excluded_records["adjustment"].eq("raw")
            ][["symbol", "trade_date"]].drop_duplicates()
            if not exclusions.empty:
                raw = raw.merge(
                    exclusions.assign(_excluded=True),
                    on=["symbol", "trade_date"],
                    how="left",
                )
                raw = raw[raw["_excluded"].isna()].drop(columns="_excluded")
        if raw.empty or hfq.empty:
            raise ValueError("Both AKShare raw and hfq data are required to build curated data")
        curated = add_adjustment_factor(raw, hfq)
        curated["quality_status"] = "PASS"
        _atomic_parquet(curated, output)
        return curated

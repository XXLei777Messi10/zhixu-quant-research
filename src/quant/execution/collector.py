from __future__ import annotations

import hashlib
import json
import time as time_module
from dataclasses import asdict
from datetime import datetime
from typing import Protocol
from zoneinfo import ZoneInfo

from quant.execution.journal import AppendOnlyJournal
from quant.execution.models import AuctionSnapshot


class AuctionProvider(Protocol):
    def fetch(self, symbols: list[str], observed_at: datetime) -> list[AuctionSnapshot]: ...


class AkshareCandidateQuoteProvider:
    """Candidate-only AKShare quote adapter.

    The endpoint does not expose a trustworthy exchange timestamp or complete
    unmatched auction book, so snapshots are intentionally classified as C
    until a reliable timestamped provider is configured. They remain useful
    as an immutable operational record.
    """

    def __init__(self, minimum_interval_seconds: float = 2.0) -> None:
        import akshare as ak

        self.ak = ak
        self.minimum_interval_seconds = max(0.0, float(minimum_interval_seconds))

    @staticmethod
    def _number(value: object) -> float | None:
        try:
            number = float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None

    def fetch(self, symbols: list[str], observed_at: datetime) -> list[AuctionSnapshot]:
        output: list[AuctionSnapshot] = []
        for index, symbol in enumerate(symbols):
            if index:
                time_module.sleep(self.minimum_interval_seconds)
            raw = self.ak.stock_bid_ask_em(symbol=symbol[2:])
            if not {"item", "value"}.issubset(raw.columns):
                raise RuntimeError("AKShare stock_bid_ask_em schema changed")
            values = {str(row["item"]).strip(): row["value"] for row in raw.to_dict("records")}
            previous = self._number(values.get("昨收"))
            current = self._number(values.get("最新"))
            official_open = self._number(values.get("今开"))
            if previous is None:
                raise RuntimeError(f"Previous close missing for {symbol}")
            is_after_925 = (
                observed_at.astimezone(ZoneInfo("Asia/Shanghai")).time()
                >= datetime.strptime("09:25:00", "%H:%M:%S").time()
            )
            output.append(
                AuctionSnapshot(
                    symbol=symbol,
                    observed_at=observed_at,
                    auction_price=official_open if is_after_925 and official_open else current,
                    previous_close=previous,
                    source="AKSHARE_STOCK_BID_ASK_EM",
                    source_timestamp=None,
                    final_open_confirmed=bool(is_after_925 and official_open),
                )
            )
        return output


def snapshot_id(snapshot: AuctionSnapshot) -> str:
    payload = asdict(snapshot)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:24]
    return f"auction-{digest}"


def collect_once(
    provider: AuctionProvider,
    symbols: list[str],
    observed_at: datetime,
    journal: AppendOnlyJournal,
    request_config: dict,
) -> tuple[list[AuctionSnapshot], list[str]]:
    errors: list[str] = []
    snapshots: list[AuctionSnapshot] = []
    retries = int(request_config["retries"])
    for attempt in range(retries + 1):
        try:
            snapshots = provider.fetch(symbols, observed_at)
            break
        except Exception as error:
            errors.append(f"attempt={attempt + 1}: {error}")
            if attempt < retries:
                time_module.sleep(float(request_config["backoff_seconds"]) * (2**attempt))
    for snapshot in snapshots:
        payload = {"snapshot_id": snapshot_id(snapshot), **asdict(snapshot)}
        journal.append(payload)
    return snapshots, errors

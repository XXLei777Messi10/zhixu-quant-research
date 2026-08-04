from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class QualityStatus(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class QualityIssue:
    code: str
    severity: QualityStatus
    message: str
    symbol: str | None = None
    trade_date: date | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    as_of: date
    status: QualityStatus
    issues: list[QualityIssue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.status == QualityStatus.FAIL


@dataclass(frozen=True)
class RawSnapshot:
    source: str
    endpoint: str
    retrieved_at: datetime
    request_id: str
    content_hash: str
    data_path: Path
    metadata_path: Path
    row_count: int

from __future__ import annotations

import gzip
import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant.types import RawSnapshot


class RawArchive:
    """Content-addressed, append-only provider response archive."""

    def __init__(self, root: Path):
        self.root = root

    def save(
        self,
        source: str,
        endpoint: str,
        frame: pd.DataFrame,
        params: dict[str, Any],
        provider_version: str,
        retrieved_at: datetime | None = None,
    ) -> RawSnapshot:
        retrieved_at = (retrieved_at or datetime.now(UTC)).astimezone(UTC)
        request_id = uuid.uuid4().hex
        payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")
        digest = hashlib.sha256(payload).hexdigest()
        day_dir = self.root / source / retrieved_at.strftime("%Y/%m/%d")
        content_dir = self.root / source / "content"
        day_dir.mkdir(parents=True, exist_ok=True)
        content_dir.mkdir(parents=True, exist_ok=True)
        data_path = content_dir / f"{digest}.csv.gz"
        if not data_path.exists():
            compressed = gzip.compress(payload, compresslevel=9, mtime=0)
            try:
                with data_path.open("xb") as handle:
                    handle.write(compressed)
            except FileExistsError:
                pass

        metadata = {
            "source": source,
            "endpoint": endpoint,
            "provider_version": provider_version,
            "params": params,
            "retrieved_at": retrieved_at.isoformat(),
            "request_id": request_id,
            "content_hash": digest,
            "row_count": int(len(frame)),
            "columns": [str(column) for column in frame.columns],
            "content_path": str(data_path.relative_to(self.root)),
        }
        metadata_path = day_dir / f"{retrieved_at.strftime('%H%M%S.%f')}_{request_id}.json"
        with metadata_path.open("x", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, sort_keys=True, indent=2)
        manifest = self.root / source / "manifest.jsonl"
        with manifest.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n")
        return RawSnapshot(
            source=source,
            endpoint=endpoint,
            retrieved_at=retrieved_at,
            request_id=request_id,
            content_hash=digest,
            data_path=data_path,
            metadata_path=metadata_path,
            row_count=len(frame),
        )

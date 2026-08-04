from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class AppendOnlyJournal:
    """JSONL journal with deterministic-event idempotency and fsync."""

    def __init__(self, path: Path, id_field: str):
        self.path = path
        self.id_field = id_field
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._known = self._load_ids()

    def _load_ids(self) -> set[str]:
        if not self.path.exists():
            return set()
        identifiers: set[str] = set()
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                identifiers.add(str(payload[self.id_field]))
        return identifiers

    def append(self, payload: dict[str, Any]) -> bool:
        identifier = str(payload[self.id_field])
        if identifier in self._known:
            return False
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._known.add(identifier)
        return True

    def contains(self, identifier: str) -> bool:
        return str(identifier) in self._known

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


def replay_journal(path: Path, id_field: str) -> list[dict[str, Any]]:
    journal = AppendOnlyJournal(path, id_field)
    records = journal.records()
    identifiers = [str(item[id_field]) for item in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Journal contains duplicate {id_field}")
    return records

from __future__ import annotations

import os
import socket
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

import duckdb


class ProcessLock(AbstractContextManager["ProcessLock"]):
    def __init__(self, path: Path):
        self.path = path
        self.fd: int | None = None

    def __enter__(self) -> ProcessLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RuntimeError(f"Another quant process holds {self.path}") from error
        os.write(self.fd, f"{os.getpid()}@{socket.gethostname()}\n".encode())
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        self.path.unlink(missing_ok=True)


class RunLedger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(str(path))
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS runs(
                run_id VARCHAR PRIMARY KEY,
                command VARCHAR NOT NULL,
                started_at TIMESTAMPTZ NOT NULL,
                finished_at TIMESTAMPTZ,
                status VARCHAR NOT NULL,
                detail VARCHAR
            )
            """
        )

    def start(self, run_id: str, command: str) -> None:
        self.connection.execute(
            "INSERT INTO runs VALUES (?, ?, ?, NULL, 'RUNNING', NULL)",
            [run_id, command, datetime.now(UTC)],
        )

    def finish(self, run_id: str, status: str, detail: str = "") -> None:
        self.connection.execute(
            "UPDATE runs SET finished_at=?, status=?, detail=? WHERE run_id=?",
            [datetime.now(UTC), status, detail, run_id],
        )

    def close(self) -> None:
        self.connection.close()

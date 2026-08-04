from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from quant.config import ProjectPaths
from quant.ops.verify_daily import verify_daily_run
from quant.signals.archive import immutable_write_json


def _write_ledger(paths: ProjectPaths, day: date, detail: dict) -> None:
    paths.state.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(paths.state / "runs.duckdb"))
    connection.execute(
        """
        CREATE TABLE runs(
            run_id VARCHAR PRIMARY KEY,
            command VARCHAR NOT NULL,
            started_at TIMESTAMPTZ NOT NULL,
            finished_at TIMESTAMPTZ,
            status VARCHAR NOT NULL,
            detail VARCHAR
        )
        """
    )
    started = datetime(day.year, day.month, day.day, 9, tzinfo=UTC)
    connection.execute(
        "INSERT INTO runs VALUES (?, 'run-daily', ?, ?, 'SUCCESS', ?)",
        ["run-1", started, started, json.dumps(detail)],
    )
    connection.close()


def test_verify_daily_run_checks_both_simulation_variants(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    paths.ensure_runtime_dirs()
    day = date(2026, 7, 28)
    execution_date = "2026-07-29"
    _write_ledger(paths, day, {"status": "SUCCESS"})
    immutable_write_json(
        paths.reports / "data-quality" / f"{day}.json",
        {"status": "WARN"},
    )
    signals = pd.DataFrame(
        [
            {
                "symbol": "SH600000",
                "data_cutoff": day.isoformat(),
                "data_quality_status": "WARN",
                "signal_valid_until": execution_date,
            }
        ]
    )
    signals.to_csv(
        paths.reports / "signals" / f"{day}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (paths.reports / "signals" / f"{day}.html").write_text(
        "<html></html>",
        encoding="utf-8",
    )
    for variant, plan_directory in (
        ("gated", "execution-plans"),
        ("ungated", "shadow-execution-plans"),
    ):
        immutable_write_json(
            paths.reports / "simulation" / variant / "accounts" / f"{day}.json",
            {
                "variant": variant,
                "execution_date": day.isoformat(),
                "simulation_only": True,
            },
        )
        immutable_write_json(
            paths.reports / plan_directory / f"{execution_date}.json",
            {
                "execution_date": execution_date,
                "simulation_only": True,
                "plans": [],
            },
        )
    immutable_write_json(
        paths.reports / "simulation" / "manual" / "accounts" / f"{day}.json",
        {
            "variant": "manual",
            "execution_date": day.isoformat(),
            "automatic_execution": False,
            "simulation_only": True,
        },
    )
    immutable_write_json(
        paths.reports / "daily-strategy" / f"{day}.json",
        {
            "account_roles": {
                "gated": "PRIMARY_MODEL_ACCOUNT_AUTOMATIC",
                "ungated": "SHADOW_MODEL_ACCOUNT_AUTOMATIC",
                "manual": "USER_REPORTED_COMPARISON_NO_AUTOMATIC_ORDERS",
            },
            "simulation_only": True,
        },
    )

    result = verify_daily_run(paths, day)

    assert result["status"] == "PASS"
    assert result["quality_status"] == "WARN"
    assert result["execution_date"] == execution_date
    assert Path(result["output"]).exists()


def test_verify_daily_run_accepts_non_trading_day_skip(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    paths.ensure_runtime_dirs()
    day = date(2026, 7, 26)
    _write_ledger(
        paths,
        day,
        {"status": "SKIPPED_NON_TRADING_DAY", "date": day.isoformat()},
    )

    result = verify_daily_run(paths, day)

    assert result["status"] == "VERIFIED_NON_TRADING_DAY_SKIP"


def test_verify_daily_run_reports_plain_text_failure_detail(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    paths.ensure_runtime_dirs()
    day = date(2026, 7, 28)
    connection = duckdb.connect(str(paths.state / "runs.duckdb"))
    connection.execute(
        """
        CREATE TABLE runs(
            run_id VARCHAR PRIMARY KEY,
            command VARCHAR NOT NULL,
            started_at TIMESTAMPTZ NOT NULL,
            finished_at TIMESTAMPTZ,
            status VARCHAR NOT NULL,
            detail VARCHAR
        )
        """
    )
    started = datetime(day.year, day.month, day.day, 9, tzinfo=UTC)
    connection.execute(
        "INSERT INTO runs VALUES (?, 'run-daily', ?, ?, 'FAILED', ?)",
        ["run-failed", started, started, "AKShare returned no rows"],
    )
    connection.close()

    with pytest.raises(RuntimeError, match="AKShare returned no rows"):
        verify_daily_run(paths, day)

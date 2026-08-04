from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd

from quant.config import ProjectPaths
from quant.signals.archive import immutable_write_json


def _revision_number(path: Path) -> int:
    if "__r" not in path.stem:
        return 1
    try:
        return int(path.stem.rsplit("__r", maxsplit=1)[1])
    except ValueError:
        return 0


def _latest_revision(directory: Path, stem: str, suffix: str) -> Path:
    candidates = list(directory.glob(f"{stem}*{suffix}"))
    if not candidates:
        raise FileNotFoundError(directory / f"{stem}{suffix}")
    return max(candidates, key=_revision_number)


def _latest_daily_run(paths: ProjectPaths) -> dict[str, Any]:
    ledger_path = paths.state / "runs.duckdb"
    if not ledger_path.exists():
        raise FileNotFoundError("Daily run ledger is absent")
    connection = duckdb.connect(str(ledger_path), read_only=True)
    try:
        row = connection.execute(
            """
            SELECT run_id, started_at, finished_at, status, detail
            FROM runs
            WHERE command = 'run-daily'
            ORDER BY started_at DESC
            LIMIT 1
            """
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise RuntimeError("Daily run ledger has no run-daily record")
    detail: dict[str, Any] = {}
    if row[4]:
        try:
            decoded = json.loads(row[4])
        except (json.JSONDecodeError, TypeError):
            detail = {
                "message": str(row[4]),
                "serialization": "plain_text",
            }
        else:
            detail = (
                decoded
                if isinstance(decoded, dict)
                else {"value": decoded, "serialization": "json_non_object"}
            )
    return {
        "run_id": row[0],
        "started_at": row[1],
        "finished_at": row[2],
        "ledger_status": row[3],
        "detail": detail,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_daily_run(
    paths: ProjectPaths,
    verification_date: date | None = None,
) -> dict[str, Any]:
    day = verification_date or pd.Timestamp.now(
        tz=ZoneInfo("Asia/Shanghai"),
    ).date()
    run = _latest_daily_run(paths)
    started_day = pd.Timestamp(run["started_at"]).tz_convert(
        "Asia/Shanghai",
    ).date()
    if started_day != day:
        raise RuntimeError(
            f"Latest run-daily belongs to {started_day}, expected {day}"
        )
    if run["ledger_status"] != "SUCCESS":
        raise RuntimeError(
            f"Daily run ledger status is {run['ledger_status']}: "
            f"{run['detail']}"
        )
    result_status = run["detail"].get("status")
    if result_status == "SKIPPED_NON_TRADING_DAY":
        payload = {
            "verification_date": day.isoformat(),
            "status": "VERIFIED_NON_TRADING_DAY_SKIP",
            "run": run,
            "simulation_only": True,
        }
        output = immutable_write_json(
            paths.reports
            / "operations"
            / "daily-verification"
            / f"{day.isoformat()}.json",
            payload,
        )
        return {**payload, "output": str(output)}
    if result_status != "SUCCESS":
        raise RuntimeError(f"Daily command result is not SUCCESS: {run['detail']}")

    day_text = day.isoformat()
    quality_path = _latest_revision(
        paths.reports / "data-quality",
        day_text,
        ".json",
    )
    quality = _read_json(quality_path)
    if quality.get("status") not in {"PASS", "WARN"}:
        raise RuntimeError(
            f"Daily data quality is not usable: {quality.get('status')}"
        )
    signal_path = _latest_revision(
        paths.reports / "signals",
        day_text,
        ".csv",
    )
    signal_html = _latest_revision(
        paths.reports / "signals",
        day_text,
        ".html",
    )
    signals = pd.read_csv(signal_path, encoding="utf-8-sig")
    if set(signals["data_cutoff"].astype(str)) != {day_text}:
        raise RuntimeError("Signal data cutoff does not match verification date")
    if not signals["data_quality_status"].isin({"PASS", "WARN"}).all():
        raise RuntimeError("Signals contain failed or unknown data quality")
    execution_dates = set(signals["signal_valid_until"].astype(str))
    if len(execution_dates) != 1:
        raise RuntimeError("Signals do not have one deterministic execution date")
    execution_date = execution_dates.pop()

    artifacts: dict[str, str] = {
        "quality": str(quality_path),
        "signals_csv": str(signal_path),
        "signals_html": str(signal_html),
    }
    for variant, plan_directory in (
        ("gated", "execution-plans"),
        ("ungated", "shadow-execution-plans"),
    ):
        account_path = _latest_revision(
            paths.reports / "simulation" / variant / "accounts",
            day_text,
            ".json",
        )
        account = _read_json(account_path)
        if account.get("variant") != variant:
            raise RuntimeError(f"{variant} account variant metadata is invalid")
        if account.get("execution_date") != day_text:
            raise RuntimeError(f"{variant} account execution date is invalid")
        if not account.get("simulation_only"):
            raise RuntimeError(f"{variant} account is not marked simulation-only")
        plan_path = _latest_revision(
            paths.reports / plan_directory,
            execution_date,
            ".json",
        )
        plan = _read_json(plan_path)
        if plan.get("execution_date") != execution_date:
            raise RuntimeError(f"{variant} plan execution date is invalid")
        if not plan.get("simulation_only"):
            raise RuntimeError(f"{variant} plan is not marked simulation-only")
        artifacts[f"{variant}_account"] = str(account_path)
        artifacts[f"{variant}_plan"] = str(plan_path)
    manual_account_path = _latest_revision(
        paths.reports / "simulation" / "manual" / "accounts",
        day_text,
        ".json",
    )
    manual_account = _read_json(manual_account_path)
    if manual_account.get("variant") != "manual":
        raise RuntimeError("Manual comparison account metadata is invalid")
    if manual_account.get("automatic_execution") is not False:
        raise RuntimeError("Manual comparison account must never auto-execute")
    strategy_path = _latest_revision(
        paths.reports / "daily-strategy",
        day_text,
        ".json",
    )
    strategy = _read_json(strategy_path)
    expected_roles = {
        "gated": "PRIMARY_MODEL_ACCOUNT_AUTOMATIC",
        "ungated": "SHADOW_MODEL_ACCOUNT_AUTOMATIC",
        "manual": "USER_REPORTED_COMPARISON_NO_AUTOMATIC_ORDERS",
    }
    if strategy.get("account_roles") != expected_roles:
        raise RuntimeError("Daily strategy account roles are inconsistent")
    artifacts["manual_account"] = str(manual_account_path)
    artifacts["daily_strategy"] = str(strategy_path)

    payload = {
        "verification_date": day_text,
        "status": "PASS",
        "execution_date": execution_date,
        "quality_status": quality["status"],
        "signal_rows": len(signals),
        "run": run,
        "artifacts": artifacts,
        "simulation_only": True,
    }
    output = immutable_write_json(
        paths.reports
        / "operations"
        / "daily-verification"
        / f"{day_text}.json",
        payload,
    )
    return {**payload, "output": str(output)}

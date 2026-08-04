from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from quant.config import ProjectPaths, load_config
from quant.execution.account import account_payload, load_simulation_account
from quant.execution.daily import _atomic_json, _latest_account_file
from quant.execution.simulator import execution_fees
from quant.signals.archive import immutable_write_json


def _canonical_symbol(value: str) -> str:
    text = value.strip().upper().replace(".", "")
    if text.startswith(("SH", "SZ", "BJ")):
        return text
    if len(text) != 6 or not text.isdigit():
        raise ValueError(f"Invalid A-share symbol: {value}")
    if text.startswith(("6", "5", "9")):
        return f"SH{text}"
    if text.startswith(("0", "1", "2", "3")):
        return f"SZ{text}"
    return f"BJ{text}"


def _trade_id(trade_date: date, trade: dict[str, Any]) -> str:
    identity = {
        "trade_date": trade_date.isoformat(),
        "symbol": _canonical_symbol(str(trade["symbol"])),
        "side": str(trade["side"]).upper(),
        "quantity": int(trade["quantity"]),
        "price": float(trade["price"]),
        "reported_fill_time": trade.get("reported_fill_time"),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    return f"manual-{digest}"


def _latest_revision(directory: Path, stem: str) -> Path | None:
    candidates = list(directory.glob(f"{stem}*.json"))
    if not candidates:
        return None

    def revision(path: Path) -> int:
        if "__r" not in path.stem:
            return 1
        try:
            return int(path.stem.rsplit("__r", maxsplit=1)[1])
        except ValueError:
            return 0

    return max(candidates, key=revision)


def _load_day_trades(paths: ProjectPaths, trade_date: date) -> list[dict[str, Any]]:
    path = _latest_revision(
        paths.reports / "simulation" / "manual" / "trades",
        trade_date.isoformat(),
    )
    if path is None:
        return []
    return list(json.loads(path.read_text(encoding="utf-8")).get("trades", []))


def _latest_marks(
    paths: ProjectPaths,
    trade_date: date,
    symbols: set[str],
) -> dict[str, float]:
    bars_path = paths.curated / "bars.parquet"
    if not bars_path.exists() or not symbols:
        return {}
    frame = pd.read_parquet(
        bars_path,
        columns=["symbol", "trade_date", "close"],
        filters=[
            ("symbol", "in", sorted(symbols)),
            ("trade_date", "<=", pd.Timestamp(trade_date)),
        ],
    ).sort_values(["symbol", "trade_date"])
    latest = frame.groupby("symbol", observed=True).tail(1)
    return {
        str(row["symbol"]): float(row["close"])
        for row in latest.to_dict("records")
        if pd.notna(row["close"]) and float(row["close"]) > 0
    }


def settle_manual_account(
    paths: ProjectPaths,
    trade_date: date,
    new_trades: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    execution = load_config(paths, "execution")
    initial_cash = float(execution["daily_open_simulation"]["initial_cash"])
    report_root = paths.reports / "simulation" / "manual"
    previous_path = _latest_account_file(report_root / "accounts", before=trade_date)
    account = load_simulation_account(
        previous_path or report_root / "accounts" / "missing.json",
        initial_cash,
    )

    combined = {
        str(item["trade_id"]): item
        for item in _load_day_trades(paths, trade_date)
        if item.get("trade_id")
    }
    for raw in new_trades or []:
        side = str(raw["side"]).upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError("Manual trade side must be BUY or SELL")
        quantity = int(raw["quantity"])
        if quantity <= 0:
            raise ValueError("Manual trade quantity must be positive")
        price = float(raw["price"])
        if price <= 0:
            raise ValueError("Manual trade price must be positive")
        trade_id = _trade_id(trade_date, raw)
        combined[trade_id] = {
            "trade_id": trade_id,
            "trade_date": trade_date.isoformat(),
            "symbol": _canonical_symbol(str(raw["symbol"])),
            "name": str(raw.get("name") or raw["symbol"]),
            "side": side,
            "quantity": quantity,
            "price": price,
            "reported_fill_time": raw.get("reported_fill_time"),
            "reported_by": str(raw.get("reported_by", "USER")),
            "fee": (
                float(raw["fee"])
                if raw.get("fee") is not None
                else None
            ),
            "fee_source": (
                "USER_REPORTED"
                if raw.get("fee") is not None
                else "CONFIG_ESTIMATE"
            ),
        }

    trades = sorted(combined.values(), key=lambda item: item["trade_id"])
    applied: list[dict[str, Any]] = []
    for item in trades:
        symbol = str(item["symbol"])
        quantity = int(item["quantity"])
        price = float(item["price"])
        gross = price * quantity
        fee = (
            float(item["fee"])
            if item.get("fee") is not None
            else execution_fees(str(item["side"]), gross, trade_date, execution)
        )
        if item["side"] == "BUY":
            if gross + fee > account.cash + 1e-9:
                raise ValueError(f"Manual account has insufficient cash for {item['trade_id']}")
            account.cash -= gross + fee
            account.add(symbol, quantity, trade_date)
        else:
            if quantity > account.sellable_shares(symbol, trade_date):
                raise ValueError(
                    f"Manual sell exceeds T+1 sellable shares for {item['trade_id']}"
                )
            account.remove(symbol, quantity, trade_date)
            account.cash += gross - fee
        applied.append({**item, "gross": gross, "fee": fee})

    trade_payload = {
        "trade_date": trade_date.isoformat(),
        "account_role": "user_reported_manual_comparison",
        "automatic_execution": False,
        "trades": applied,
        "simulation_only": True,
    }
    trades_path = immutable_write_json(
        report_root / "trades" / f"{trade_date.isoformat()}.json",
        trade_payload,
    )
    marks = _latest_marks(
        paths,
        trade_date,
        set(account.lots) | {str(item["symbol"]) for item in applied},
    )
    for item in applied:
        marks.setdefault(str(item["symbol"]), float(item["price"]))
    as_of = datetime.combine(
        trade_date,
        time(15, 30),
        tzinfo=ZoneInfo("Asia/Shanghai"),
    ).isoformat()
    state = account_payload(
        account,
        marks,
        as_of,
        "USER_DECIDED",
        str(execution["rule_version"]),
    )
    state.update(
        {
            "execution_date": trade_date.isoformat(),
            "variant": "manual",
            "account_role": "user_reported_manual_comparison",
            "automatic_execution": False,
            "trade_count_today": len(applied),
            "fees_today": sum(float(item["fee"]) for item in applied),
            "portfolio_policy_reference": execution["portfolio_policy"],
        }
    )
    account_path = immutable_write_json(
        report_root / "accounts" / f"{trade_date.isoformat()}.json",
        state,
    )
    _atomic_json(paths.state / "simulation" / "manual" / "current.json", state)
    return {
        "status": "RECORDED" if applied else "MARKED_NO_REPORTED_TRADES",
        "trade_date": trade_date.isoformat(),
        "trades": str(trades_path),
        "account": str(account_path),
        "trade_count": len(applied),
        "cash": float(state["cash"]),
        "nav": float(state["nav"]),
        "simulation_only": True,
    }


def parse_manual_trade(value: str) -> dict[str, Any]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) not in {4, 5}:
        raise ValueError(
            "Manual trade format is SIDE,SYMBOL,QUANTITY,PRICE[,NAME]"
        )
    return {
        "side": parts[0],
        "symbol": parts[1],
        "quantity": int(parts[2]),
        "price": float(parts[3]),
        "name": parts[4] if len(parts) == 5 else parts[1],
    }

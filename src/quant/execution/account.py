from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from quant.backtest.engine import Account, PositionLot


def load_simulation_account(path: Path, initial_cash: float) -> Account:
    if not path.exists():
        return Account(float(initial_cash))
    payload = json.loads(path.read_text(encoding="utf-8"))
    account = Account(float(payload["cash"]))
    for item in payload.get("lots", []):
        account.lots.setdefault(str(item["symbol"]), []).append(
            PositionLot(int(item["shares"]), date.fromisoformat(item["acquired_on"]))
        )
    return account


def account_payload(
    account: Account,
    prices: dict[str, float],
    as_of: str,
    model_version: str,
    rule_version: str,
) -> dict[str, Any]:
    positions = []
    lots = []
    market_value = 0.0
    for symbol in sorted(account.lots):
        shares = account.shares(symbol)
        if shares <= 0:
            continue
        price = float(prices.get(symbol, 0.0))
        value = shares * price
        market_value += value
        positions.append(
            {
                "symbol": symbol,
                "shares": shares,
                "mark_price": price,
                "market_value": value,
            }
        )
        for lot in account.lots[symbol]:
            lots.append(
                {
                    "symbol": symbol,
                    "shares": lot.shares,
                    "acquired_on": lot.acquired_on.isoformat(),
                }
            )
    nav = account.cash + market_value
    for item in positions:
        item["current_position_weight"] = item["market_value"] / nav if nav > 0 else 0.0
    return {
        "as_of": as_of,
        "simulation_only": True,
        "cash": account.cash,
        "market_value": market_value,
        "nav": nav,
        "positions": positions,
        "lots": lots,
        "model_version": model_version,
        "rule_version": rule_version,
    }

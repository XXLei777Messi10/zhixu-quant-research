from __future__ import annotations

import json
import os
import time as time_module
import uuid
from dataclasses import asdict
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from quant.config import ProjectPaths, load_config
from quant.execution.account import account_payload, load_simulation_account
from quant.execution.archive import archive_execution_day, load_execution_plans
from quant.execution.auction import auction_phase, compute_auction_features, evaluate_auction
from quant.execution.collector import AkshareCandidateQuoteProvider, AuctionProvider, collect_once
from quant.execution.journal import AppendOnlyJournal
from quant.execution.models import AuctionSnapshot, ExecutionState
from quant.execution.simulator import order_id_for_decision, simulate_order
from quant.execution.state_machine import append_decision, fill_transition

TERMINAL_STATES = {ExecutionState.CANCELLED, ExecutionState.EXPIRED}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temp, path)


def _snapshot_from_record(payload: dict[str, Any]) -> AuctionSnapshot:
    values = {key: value for key, value in payload.items() if key != "snapshot_id"}
    values["observed_at"] = datetime.fromisoformat(values["observed_at"])
    if values.get("source_timestamp"):
        values["source_timestamp"] = datetime.fromisoformat(values["source_timestamp"])
    return AuctionSnapshot(**values)


def _state_by_plan(plans: list, decision_records: list[dict[str, Any]]) -> dict[str, ExecutionState]:
    states = {plan.plan_id: ExecutionState(plan.initial_state) for plan in plans}
    for record in decision_records:
        states[record["plan_id"]] = ExecutionState(record["next_state"])
    return states


def run_auction_cycle(
    paths: ProjectPaths,
    observed_at: datetime,
    provider: AuctionProvider,
) -> dict[str, Any]:
    day = observed_at.astimezone(ZoneInfo("Asia/Shanghai")).date().isoformat()
    plans_path = paths.reports / "execution-plans" / f"{day}.json"
    plans = load_execution_plans(plans_path)
    state_root = paths.state / "execution" / day
    snapshots_journal = AppendOnlyJournal(state_root / "auction_snapshots.jsonl", "snapshot_id")
    decisions_journal = AppendOnlyJournal(state_root / "decisions.jsonl", "decision_id")
    orders_journal = AppendOnlyJournal(state_root / "orders.jsonl", "order_id")
    fills_journal = AppendOnlyJournal(state_root / "fills.jsonl", "order_id")
    auction_config = load_config(paths, "auction")
    execution_config = load_config(paths, "execution")
    snapshots, errors = collect_once(
        provider,
        sorted({plan.symbol for plan in plans}),
        observed_at,
        snapshots_journal,
        auction_config["request"],
    )
    all_snapshots = [_snapshot_from_record(item) for item in snapshots_journal.records()]
    grouped = {
        symbol: [item for item in all_snapshots if item.symbol == symbol]
        for symbol in {plan.symbol for plan in plans}
    }
    states = _state_by_plan(plans, decisions_journal.records())
    account_path = state_root / "account-current.json"
    prior_accounts = sorted(
        path for path in (paths.state / "execution").glob("*/account-current.json") if path.parent.name < day
    )
    account_source = (
        account_path if account_path.exists() else (prior_accounts[-1] if prior_accounts else account_path)
    )
    account = load_simulation_account(account_source, 1_000_000)
    decisions = []
    new_orders = []
    latest_prices = {
        item.symbol: float(item.auction_price)
        for item in snapshots
        if item.auction_price is not None and item.auction_price > 0
    }
    portfolio_value = account.cash + sum(
        account.shares(symbol) * latest_prices.get(symbol, 0.0) for symbol in account.lots
    )
    phase = auction_phase(observed_at, auction_config)
    existing_orders = orders_journal.records()

    for plan in plans:
        state = states[plan.plan_id]
        if state in TERMINAL_STATES:
            continue
        decision, _ = evaluate_auction(plan, grouped.get(plan.symbol, []), observed_at, auction_config, state)
        if append_decision(decisions_journal, decision):
            decisions.append(decision.to_dict())
        if phase not in {"OPEN_CONFIRMATION", "CONTINUOUS"}:
            continue
        if decision.action not in {"OPEN", "ADD", "REDUCE", "EXIT"}:
            continue
        proposed_id = order_id_for_decision(decision)
        if orders_journal.contains(proposed_id):
            continue
        filled_orders = [item for item in existing_orders if item["status"] in {"FILLED", "PARTIALLY_FILLED"}]
        daily_new_positions = sum(item["action"] == "OPEN" for item in filled_orders)
        daily_add_count = sum(
            item["action"] == "ADD" and item["symbol"] == plan.symbol for item in filled_orders
        )
        daily_turnover = sum(float(item["planned_weight"]) for item in filled_orders)
        latest = grouped.get(plan.symbol, [])[-1] if grouped.get(plan.symbol) else None
        available = int(latest.matched_volume) if latest and latest.matched_volume is not None else None
        confirmed = bool(latest and latest.final_open_confirmed)
        order = simulate_order(
            account,
            plan,
            decision,
            latest.auction_price if latest else None,
            portfolio_value,
            observed_at.date(),
            observed_at,
            execution_config,
            previous_close=plan.reference_close,
            available_liquidity_shares=available,
            quote_is_fresh=decision.data_level != "C",
            execution_price_confirmed=confirmed,
            daily_new_positions=daily_new_positions,
            daily_add_count=daily_add_count,
            daily_turnover=daily_turnover,
        )
        orders_journal.append(order.to_dict())
        new_orders.append(order.to_dict())
        existing_orders.append(order.to_dict())
        if order.fill_quantity:
            fills_journal.append(order.to_dict())
            transition = fill_transition(decision, order, observed_at)
            if transition is not None:
                append_decision(decisions_journal, transition)
    latest_model = plans[0].model_version if plans else "NONE"
    payload = account_payload(
        account, latest_prices, observed_at.isoformat(), latest_model, execution_config["rule_version"]
    )
    _atomic_json(account_path, payload)
    return {
        "execution_date": day,
        "phase": phase,
        "snapshot_count": len(snapshots),
        "new_decisions": len(decisions),
        "new_orders": len(new_orders),
        "errors": errors,
        "data_safe": not errors,
    }


def finalize_auction_day(paths: ProjectPaths, execution_day: date) -> dict[str, str]:
    day = execution_day.isoformat()
    state_root = paths.state / "execution" / day
    plans = load_execution_plans(paths.reports / "execution-plans" / f"{day}.json")
    snapshots = AppendOnlyJournal(state_root / "auction_snapshots.jsonl", "snapshot_id").records()
    decisions = AppendOnlyJournal(state_root / "decisions.jsonl", "decision_id").records()
    orders = AppendOnlyJournal(state_root / "orders.jsonl", "order_id").records()
    fills = AppendOnlyJournal(state_root / "fills.jsonl", "order_id").records()
    account_path = state_root / "account-current.json"
    positions = (
        json.loads(account_path.read_text(encoding="utf-8"))
        if account_path.exists()
        else {"as_of": day, "simulation_only": True, "positions": []}
    )
    snapshot_objects = [_snapshot_from_record(item) for item in snapshots]
    auction_report = {
        "execution_date": day,
        "simulation_only": True,
        "original_model_signals": [asdict(plan) for plan in plans],
        "snapshots": snapshots,
        "features": {
            plan.symbol: asdict(
                compute_auction_features([item for item in snapshot_objects if item.symbol == plan.symbol])
            )
            for plan in plans
        },
        "decisions": decisions,
        "degradation": (
            "C"
            if not snapshots
            else max(
                (item.get("data_level", "C") for item in decisions),
                key=lambda level: {"A": 0, "B": 1, "C": 2}[level],
                default="C",
            )
        ),
    }
    return archive_execution_day(
        paths.reports,
        day,
        auction=auction_report,
        orders=orders,
        fills=fills,
        positions=positions,
    )


def monitor_auction_session(paths: ProjectPaths) -> dict[str, Any]:
    config = load_config(paths, "auction")
    tz = ZoneInfo(config["timezone"])
    today = datetime.now(tz).date()
    plans_path = paths.reports / "execution-plans" / f"{today.isoformat()}.json"
    if today.weekday() >= 5 or not plans_path.exists():
        return {
            "status": "SKIPPED_NO_VALID_PLAN_OR_NON_TRADING_DAY",
            "date": today.isoformat(),
            "simulation_only": True,
        }
    provider = AkshareCandidateQuoteProvider(config["request"]["minimum_interval_seconds"])
    results = []
    continuous_end = time.fromisoformat(config["continuous_fallback_end"])
    while True:
        now = datetime.now(tz)
        phase = auction_phase(now, config)
        if phase == "PRECHECK":
            time_module.sleep(min(30, max(1, int(config["snapshot_interval_seconds"]))))
            continue
        if phase == "CONTINUOUS" and now.time().replace(tzinfo=None) >= continuous_end:
            break
        results.append(run_auction_cycle(paths, now, provider))
        time_module.sleep(max(1, int(config["snapshot_interval_seconds"])))
    outputs = finalize_auction_day(paths, datetime.now(tz).date())
    return {"cycles": len(results), "outputs": outputs, "simulation_only": True}

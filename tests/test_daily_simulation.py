from __future__ import annotations

import json
import shutil
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from quant.config import ProjectPaths, load_config
from quant.execution.archive import archive_execution_plans
from quant.execution.daily import evaluate_daily_open, settle_daily_accounts
from quant.execution.models import Action, ExecutionState
from quant.execution.planner import build_execution_plan


def _signal(**changes):
    payload = {
        "trade_date": "2026-01-05",
        "symbol": "SH600000",
        "name": "测试股份",
        "model_version": "m1",
        "model_signal": "BUY",
        "model_score": 0.9,
        "outperform_probability": 0.64,
        "predicted_excess_return_5d": 0.03,
        "signal_rank": 1,
        "current_position_weight": 0.0,
        "target_position_weight": 0.08,
        "market_state": "NORMAL",
        "sector_state": "NORMAL",
        "signal_valid_until": "2026-01-06",
        "data_quality_status": "PASS",
    }
    payload.update(changes)
    return payload


def _context():
    return {
        "reference_close": 10.0,
        "atr20": 0.4,
        "support_20": 9.4,
        "resistance_20": 10.8,
        "volatility_20": 0.02,
        "calibration_status": "WALK_FORWARD_PASS",
    }


def _project(tmp_path: Path) -> ProjectPaths:
    project = Path(__file__).resolve().parents[1]
    shutil.copytree(project / "configs", tmp_path / "configs")
    paths = ProjectPaths(tmp_path)
    paths.ensure_runtime_dirs()
    return paths


def test_daily_open_uses_entry_and_invalidation_boundaries(tmp_path: Path) -> None:
    paths = _project(tmp_path)
    config = load_config(paths, "execution")
    plan = build_execution_plan(
        _signal(),
        _context(),
        date(2026, 1, 6),
        config,
    )
    now = datetime(2026, 1, 6, 9, 30, tzinfo=ZoneInfo("Asia/Shanghai"))

    accepted = evaluate_daily_open(
        plan,
        (plan.entry_price_low + plan.entry_price_high) / 2,
        now,
        config,
        "gated",
    )
    invalidated = evaluate_daily_open(
        plan,
        plan.invalidation_price - 0.01,
        now,
        config,
        "gated",
    )
    chased = evaluate_daily_open(
        plan,
        plan.chase_limit_price + 0.01,
        now,
        config,
        "gated",
    )

    assert accepted.action == Action.OPEN
    assert accepted.final
    assert invalidated.action == Action.CANCEL
    assert invalidated.trigger_rule == "OPEN_BELOW_INVALIDATION"
    assert chased.action == Action.CANCEL
    assert chased.trigger_rule == "OPEN_ABOVE_CHASE_LIMIT"


def test_gated_and_ungated_accounts_are_separate_and_idempotent(
    tmp_path: Path,
) -> None:
    paths = _project(tmp_path)
    config = load_config(paths, "execution")
    execution_date = date(2026, 1, 6)
    gated = build_execution_plan(
        _signal(market_state="HIGH_RISK"),
        _context(),
        execution_date,
        config,
    )
    ungated = build_execution_plan(
        _signal(market_state="HIGH_RISK"),
        _context(),
        execution_date,
        config,
        apply_risk_gate=False,
    )
    assert gated.initial_state == ExecutionState.CANCELLED
    assert ungated.initial_state == ExecutionState.READY_TO_OPEN
    archive_execution_plans(
        [gated],
        paths.reports,
        execution_date.isoformat(),
    )
    archive_execution_plans(
        [ungated],
        paths.reports,
        execution_date.isoformat(),
        directory="shadow-execution-plans",
    )
    open_price = (ungated.entry_price_low + ungated.entry_price_high) / 2
    pd.DataFrame(
        [
            {
                "symbol": "SH600000",
                "trade_date": pd.Timestamp("2026-01-05"),
                "open": 10.0,
                "close": 10.0,
                "volume": 1_000_000,
                "is_trading": True,
                "is_st": False,
            },
            {
                "symbol": "SH600000",
                "trade_date": pd.Timestamp(execution_date),
                "open": open_price,
                "close": open_price,
                "volume": 1_000_000,
                "is_trading": True,
                "is_st": False,
            },
        ]
    ).to_parquet(paths.curated / "bars.parquet", index=False)

    first = settle_daily_accounts(paths, execution_date)
    second = settle_daily_accounts(paths, execution_date)

    assert first["primary"]["fill_count"] == 0
    assert first["shadow"]["fill_count"] == 1
    assert second["primary"]["status"] == "ALREADY_SETTLED"
    assert second["shadow"]["status"] == "ALREADY_SETTLED"
    gated_payload = json.loads(
        Path(first["primary"]["account"]).read_text(encoding="utf-8")
    )
    ungated_payload = json.loads(
        Path(first["shadow"]["account"]).read_text(encoding="utf-8")
    )
    assert gated_payload["cash"] == 1_000_000
    assert not gated_payload["positions"]
    assert ungated_payload["cash"] < 1_000_000
    assert ungated_payload["positions"][0]["symbol"] == "SH600000"
    assert len(list((paths.reports / "simulation" / "gated" / "accounts").glob("*.json"))) == 1
    assert len(list((paths.reports / "simulation" / "ungated" / "accounts").glob("*.json"))) == 1

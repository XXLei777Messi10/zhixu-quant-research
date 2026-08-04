from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from quant.config import ProjectPaths, load_config
from quant.execution.backtest import ProductionMirrorSimulator
from quant.execution.manual import settle_manual_account
from quant.execution.policy import policy_snapshot, portfolio_policy
from quant.signals.archive import immutable_write_json


def _project(tmp_path: Path) -> ProjectPaths:
    project = Path(__file__).resolve().parents[1]
    shutil.copytree(project / "configs", tmp_path / "configs")
    paths = ProjectPaths(tmp_path)
    paths.ensure_runtime_dirs()
    return paths


def test_frozen_production_policy_matches_daily_reporting_config(
    tmp_path: Path,
) -> None:
    paths = _project(tmp_path)
    config = load_config(paths, "execution")
    policy = portfolio_policy(config)
    snapshot = policy_snapshot(config)

    assert config["daily_open_simulation"]["primary_variant"] == "gated"
    assert config["daily_open_simulation"]["shadow_variant"] == "ungated"
    assert policy["max_gross_exposure"] == pytest.approx(0.80)
    assert policy["minimum_cash_reserve"] == pytest.approx(0.20)
    assert policy["target_position_weight"] == pytest.approx(0.05)
    assert policy["max_positions"] == 16
    assert snapshot["initial_order_weight"] == pytest.approx(0.025)
    assert snapshot["add_order_weight"] == pytest.approx(0.0125)


def test_manual_comparison_account_records_user_fills_once(
    tmp_path: Path,
) -> None:
    paths = _project(tmp_path)
    pd.DataFrame(
        [
            {
                "symbol": "SZ002050",
                "trade_date": pd.Timestamp("2026-07-28"),
                "close": 35.621,
            },
            {
                "symbol": "SH688472",
                "trade_date": pd.Timestamp("2026-07-28"),
                "close": 9.543,
            },
        ]
    ).to_parquet(paths.curated / "bars.parquet", index=False)
    trades = [
        {
            "side": "BUY",
            "symbol": "002050",
            "quantity": 700,
            "price": 35.621,
            "name": "三花智控",
        },
        {
            "side": "BUY",
            "symbol": "688472",
            "quantity": 2620,
            "price": 9.543,
            "name": "阿特斯",
        },
    ]
    first = settle_manual_account(paths, date(2026, 7, 28), trades)
    second = settle_manual_account(paths, date(2026, 7, 28), trades)
    account = json.loads(Path(first["account"]).read_text(encoding="utf-8"))

    assert first["account"] == second["account"]
    assert first["trades"] == second["trades"]
    assert account["variant"] == "manual"
    assert account["automatic_execution"] is False
    assert {item["symbol"] for item in account["positions"]} == {
        "SZ002050",
        "SH688472",
    }
    assert account["cash"] < 950_063
    assert len(list((paths.reports / "simulation" / "manual" / "accounts").glob("*.json"))) == 1


def test_immutable_archive_reuses_matching_noncanonical_revision(
    tmp_path: Path,
) -> None:
    base = tmp_path / "report.json"
    first = immutable_write_json(base, {"version": 1})
    second = immutable_write_json(base, {"version": 2})
    repeated = immutable_write_json(base, {"version": 2})

    assert first == base
    assert second.name == "report__r2.json"
    assert repeated == second
    assert len(list(tmp_path.glob("report*.json"))) == 2


def test_production_mirror_enforces_staged_entries_and_cash_reserve(
    tmp_path: Path,
) -> None:
    paths = _project(tmp_path)
    config = load_config(paths, "execution")
    config["require_walk_forward_calibration"] = False
    config["calibration"]["minimum_peer_samples"] = 10_000
    dates = pd.bdate_range("2026-01-01", periods=45)
    symbols = [f"SH6000{index:02d}" for index in range(20)]
    bars = []
    predictions = []
    for day_index, day in enumerate(dates):
        for symbol_index, symbol in enumerate(symbols):
            close = 10.0 + symbol_index * 0.05 + day_index * 0.001
            bars.append(
                {
                    "trade_date": day,
                    "symbol": symbol,
                    "open": close,
                    "high": close + 0.10,
                    "low": close - 0.10,
                    "close": close,
                    "hfq_open": close,
                    "hfq_high": close + 0.10,
                    "hfq_low": close - 0.10,
                    "hfq_close": close,
                    "volume": 10_000_000,
                    "adjust_factor": 1.0,
                    "is_trading": True,
                    "is_st": False,
                }
            )
            predictions.append(
                {
                    "trade_date": day,
                    "symbol": symbol,
                    "model_score": 1.0 - symbol_index / 100,
                    "outperform_probability": 0.60,
                    "predicted_excess_return": 0.02,
                    "market_state": "NORMAL",
                    "sector_state": "NORMAL",
                    "sector_name": f"sector-{symbol_index // 4}",
                    "stock_name": symbol,
                    "fold": 1,
                }
            )
    result = ProductionMirrorSimulator(
        config,
        apply_risk_gate=False,
    ).run(pd.DataFrame(bars), pd.DataFrame(predictions))

    buys = result.trades[result.trades["side"].eq("BUY")]
    assert not buys.empty
    assert buys["gross"].max() < 26_000
    assert result.nav["gross_exposure"].max() <= 0.805
    assert result.nav["cash"].min() > 195_000
    filled_new = result.orders[
        result.orders["action"].eq("OPEN")
        & result.orders["status"].isin({"FILLED", "PARTIALLY_FILLED"})
    ]
    assert filled_new.groupby("trade_date").size().max() <= 5
    latest_day = result.holdings["trade_date"].max()
    assert len(result.holdings[result.holdings["trade_date"].eq(latest_day)]) <= 16


def test_production_mirror_retries_weekly_pool_on_each_daily_open(
    tmp_path: Path,
) -> None:
    paths = _project(tmp_path)
    config = load_config(paths, "execution")
    config["require_walk_forward_calibration"] = False
    dates = pd.bdate_range("2026-01-05", periods=30)
    symbols = [f"SH6000{index:02d}" for index in range(20)]
    bars = []
    predictions = []
    for day in dates:
        for symbol_index, symbol in enumerate(symbols):
            price = 10.0 + symbol_index * 0.01
            bars.append(
                {
                    "trade_date": day,
                    "symbol": symbol,
                    "open": price,
                    "high": price + 0.10,
                    "low": price - 0.10,
                    "close": price,
                    "hfq_open": price,
                    "hfq_high": price + 0.10,
                    "hfq_low": price - 0.10,
                    "hfq_close": price,
                    "volume": 10_000_000,
                    "adjust_factor": 1.0,
                    "is_trading": True,
                    "is_st": False,
                }
            )
            predictions.append(
                {
                    "trade_date": day,
                    "symbol": symbol,
                    "model_score": 1.0 - symbol_index / 100,
                    "outperform_probability": 0.60,
                    "predicted_excess_return": 0.02,
                    "market_state": "NORMAL",
                    "sector_state": "NORMAL",
                    "sector_name": f"sector-{symbol_index // 4}",
                    "stock_name": symbol,
                    "fold": 1,
                }
            )
    result = ProductionMirrorSimulator(
        config,
        apply_risk_gate=False,
    ).run(pd.DataFrame(bars), pd.DataFrame(predictions))

    first_symbol = symbols[0]
    symbol_buys = result.trades[
        result.trades["symbol"].eq(first_symbol)
        & result.trades["side"].eq("BUY")
    ].sort_values("trade_date")
    assert symbol_buys["action"].head(3).tolist() == ["OPEN", "ADD", "ADD"]
    buy_dates = pd.to_datetime(symbol_buys["trade_date"])
    weekly_counts = symbol_buys.groupby(
        [buy_dates.dt.isocalendar().year, buy_dates.dt.isocalendar().week]
    ).size()
    assert weekly_counts.max() >= 3


def test_production_mirror_research_overlay_validates_exit_rank(
    tmp_path: Path,
) -> None:
    paths = _project(tmp_path)
    config = load_config(paths, "execution")

    with pytest.raises(ValueError, match="Exit rank"):
        ProductionMirrorSimulator(
            config,
            apply_risk_gate=False,
            exit_rank=19,
        )


def test_production_mirror_research_overlay_scales_risk_weight(
    tmp_path: Path,
) -> None:
    paths = _project(tmp_path)
    config = load_config(paths, "execution")
    simulator = ProductionMirrorSimulator(
        config,
        apply_risk_gate=False,
        exit_rank=40,
        risk_weight_scales={
            "market": {
                "NORMAL": 1.0,
                "CAUTIOUS": 0.75,
                "HIGH_RISK": 0.50,
            },
            "sector": {
                "NORMAL": 1.0,
                "CAUTIOUS": 0.75,
                "HIGH_RISK": 0.50,
            },
        },
        variant_name="research-soft-gate",
    )

    assert simulator.exit_rank == 40
    assert simulator.variant == "research-soft-gate"
    assert simulator._risk_weight_scale("NORMAL", "CAUTIOUS") == pytest.approx(0.75)
    assert simulator._risk_weight_scale("HIGH_RISK", "NORMAL") == pytest.approx(0.50)
    assert simulator._risk_weight_scale("DATA_UNAVAILABLE", "NORMAL") == 0.0


def test_weekly_risk_scale_is_stable_until_pool_refresh(
    tmp_path: Path,
) -> None:
    paths = _project(tmp_path)
    config = load_config(paths, "execution")
    simulator = ProductionMirrorSimulator(
        config,
        apply_risk_gate=False,
        risk_weight_scales={
            "market": {"NORMAL": 1.0, "HIGH_RISK": 0.5},
            "sector": {"NORMAL": 1.0, "HIGH_RISK": 0.5},
        },
        risk_scale_refresh="weekly",
    )
    cache: dict[str, float] = {}

    first = simulator._target_risk_scale(
        "SH600000",
        "NORMAL",
        "NORMAL",
        pool_refreshed=True,
        cache=cache,
    )
    midweek = simulator._target_risk_scale(
        "SH600000",
        "HIGH_RISK",
        "HIGH_RISK",
        pool_refreshed=False,
        cache=cache,
    )
    next_refresh = simulator._target_risk_scale(
        "SH600000",
        "HIGH_RISK",
        "HIGH_RISK",
        pool_refreshed=True,
        cache=cache,
    )

    assert first == 1.0
    assert midweek == 1.0
    assert next_refresh == 0.5

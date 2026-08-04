from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant.config import ProjectPaths, load_config
from quant.research.dynamic_open_backtest import AdjustedAccount, AdjustedLot
from quant.research.rank_buffer_strategy import (
    OpenCalibrationBook,
    RankBufferSimulator,
    _candidate_for_market_state,
    _convert_position,
    _exposure,
    _is_preexisting_position_cap_breach,
    _material_position_count,
    _material_positions,
    _position_availability,
    _purge_nonmaterial_residuals,
    _target_symbols,
    _target_weights,
)


def research_config() -> dict:
    return load_config(ProjectPaths(Path(".").resolve()), "rank_buffer_research")


def execution_config() -> dict:
    return load_config(ProjectPaths(Path(".").resolve()), "execution")


def candidate(name: str = "weekly_top20_drop2_exit30_equal") -> dict:
    return next(value for value in research_config()["candidates"] if value["name"] == name)


def test_data_repair_evaluation_preserves_frozen_strategy() -> None:
    paths = ProjectPaths(Path(".").resolve())
    original = load_config(paths, "rank_buffer_final")
    frozen_keys = [
        "prediction_file",
        "history_start",
        "locked_period",
        "initial_cash",
        "benchmark_symbol",
        "rebalance_frequency",
        "execution_price",
        "lot_size",
        "maximum_material_positions",
        "candidate_pool_size",
        "locked_acceptance",
        "candidate",
        "open_filter",
    ]
    frozen = {key: original[key] for key in frozen_keys}
    for config_name in (
        "rank_buffer_final_data_repair",
        "rank_buffer_final_accounting_repair",
    ):
        repaired = load_config(paths, config_name)
        assert frozen == {key: repaired[key] for key in frozen_keys}


def signals(count: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [f"SH60{index:04d}" for index in range(count)],
            "model_score": [1.0 - index / max(count, 1) for index in range(count)],
            "outperform_probability": [0.70 - index / max(count, 1) * 0.10 for index in range(count)],
            "annualized_volatility_20": [0.15 + index / max(count, 1) * 0.20 for index in range(count)],
            "sector_name": [f"行业{index % 4}" for index in range(count)],
        }
    )


def test_rank_buffer_drops_only_configured_worst_holdings() -> None:
    frame = signals()
    held = set(frame.iloc[:18]["symbol"]) | set(frame.iloc[30:32]["symbol"])
    targets, exits, buys = _target_symbols(frame, held, candidate())
    assert len(exits) == 2
    assert set(exits) == set(frame.iloc[30:32]["symbol"])
    assert len(buys) == 2
    assert len(targets) == 20


def test_rank_buffer_does_not_trade_inside_exit_buffer() -> None:
    frame = signals()
    held = set(frame.iloc[5:25]["symbol"])
    targets, exits, buys = _target_symbols(frame, held, candidate())
    assert not exits
    assert not buys
    assert set(targets) == held


def test_positions_absent_from_current_signal_exit_without_rank_buffer() -> None:
    frame = signals()
    missing = {"SH699998", "SH699999"}
    constrained = {**candidate(), "max_drop": 1}
    targets, exits, _ = _target_symbols(
        frame,
        set(frame.iloc[:18]["symbol"]) | missing,
        constrained,
    )
    assert missing.issubset(exits)
    assert not missing.intersection(targets)


def test_risk_overlay_scales_exposure_without_leverage() -> None:
    risk_candidate = {
        "exposure_mode": "benchmark_trend_volatility",
        "target_annualized_volatility": 0.18,
        "below_ma120_exposure_multiplier": 0.50,
        "minimum_exposure": 0.30,
    }
    normal = pd.Series(
        {
            "annualized_volatility_20": 0.12,
            "hfq_close": 110.0,
            "ma120": 100.0,
        }
    )
    stressed = pd.Series(
        {
            "annualized_volatility_20": 0.36,
            "hfq_close": 90.0,
            "ma120": 100.0,
        }
    )
    assert _exposure(risk_candidate, normal) == 1.0
    assert _exposure(risk_candidate, stressed) == 0.30


def test_risk_weights_are_capped_and_preserve_dynamic_cash() -> None:
    frame = signals()
    risk_candidate = candidate("weekly_top20_drop2_exit30_risk_weighted")
    targets = list(frame.iloc[:20]["symbol"])
    weights = _target_weights(frame, targets, 1.0, risk_candidate)
    assert 0.0 < sum(weights.values()) <= 1.0 + 1e-12
    assert max(weights.values()) <= 0.10 + 1e-12
    sectors = frame.set_index("symbol")["sector_name"].to_dict()
    sector_weights: dict[str, float] = {}
    for symbol, weight in weights.items():
        sector = sectors[symbol]
        sector_weights[sector] = sector_weights.get(sector, 0.0) + weight
    assert max(sector_weights.values()) <= 0.25 + 1e-12


def test_calibrated_edge_controls_dynamic_breadth_and_alpha_weights() -> None:
    frame = signals(8)
    frame["calibrated_lower_bound"] = [
        0.010,
        0.008,
        0.006,
        0.001,
        0.000,
        -0.001,
        -0.002,
        -0.003,
    ]
    frame["horizon_agreement"] = [0.9, 0.8, 0.7, 0.9, 0.9, 0.9, 0.9, 0.9]
    dynamic = {
        "topk": 5,
        "exit_rank": 8,
        "max_drop": 5,
        "ranking_column": "model_score",
        "eligibility_column": "calibrated_lower_bound",
        "minimum_eligibility_value": 0.002,
        "minimum_horizon_agreement": 0.6,
        "weight_mode": "calibrated_alpha_budget",
        "weight_alpha_column": "calibrated_lower_bound",
        "alpha_cost_hurdle": 0.002,
        "alpha_full_size": 0.008,
        "risk_volatility_target": 0.20,
        "alpha_power": 1.0,
        "volatility_power": 1.0,
        "volatility_floor": 0.12,
        "max_single_weight": 0.20,
        "max_sector_weight": 1.0,
    }
    targets, exits, buys = _target_symbols(frame, set(), dynamic)
    assert not exits
    assert len(targets) == 3
    assert set(targets) == set(frame.iloc[:3]["symbol"])
    assert set(buys) == set(targets)

    weights = _target_weights(frame, targets, 1.0, dynamic)
    assert 0.0 < sum(weights.values()) <= 0.60 + 1e-12
    assert weights[frame.iloc[0]["symbol"]] > weights[frame.iloc[2]["symbol"]]


def test_open_calibration_uses_only_matured_prior_outcomes() -> None:
    open_config = dict(research_config()["open_filter"])
    open_config.update(
        {
            "minimum_exact_samples": 2,
            "minimum_market_samples": 2,
            "minimum_global_samples": 2,
        }
    )
    available = pd.Timestamp("2025-01-10")
    schedule = {
        available: [
            {
                "market_state": "NORMAL",
                "sector_state": "NORMAL",
                "gap_bucket": 0,
                "realized_excess_return": 0.02,
            },
            {
                "market_state": "NORMAL",
                "sector_state": "NORMAL",
                "gap_bucket": 0,
                "realized_excess_return": 0.03,
            },
            {
                "market_state": "NORMAL",
                "sector_state": "NORMAL",
                "gap_bucket": 4,
                "realized_excess_return": -0.02,
            },
            {
                "market_state": "NORMAL",
                "sector_state": "NORMAL",
                "gap_bucket": 4,
                "realized_excess_return": -0.01,
            },
        ]
    }
    book = OpenCalibrationBook(schedule, open_config)
    book.mature(available)
    unavailable = book.decision("NORMAL", "NORMAL", -2.0)
    assert unavailable["reason"] == "INSUFFICIENT_WALK_FORWARD_OPEN_HISTORY"
    book.mature(available + pd.Timedelta(days=1))
    favorable = book.decision("NORMAL", "NORMAL", -2.0)
    adverse = book.decision("NORMAL", "NORMAL", 2.0)
    assert favorable["scale"] == 1.0
    assert favorable["sample_count"] == 2
    assert adverse["scale"] == 0.0
    assert adverse["reason"] == "OPEN_BUCKET_NET_EDGE_REJECTED"


def test_mild_open_calibration_only_rejects_strong_negative_edge() -> None:
    open_config = dict(load_config(ProjectPaths(Path(".").resolve()), "rank_buffer_final")["open_filter"])
    open_config.update(
        {
            "minimum_exact_samples": 2,
            "minimum_market_samples": 2,
            "minimum_global_samples": 2,
        }
    )
    available = pd.Timestamp("2025-01-10")
    schedule = {
        available: [
            {
                "market_state": "NORMAL",
                "sector_state": "NORMAL",
                "gap_bucket": 0,
                "realized_excess_return": -0.030,
            },
            {
                "market_state": "NORMAL",
                "sector_state": "NORMAL",
                "gap_bucket": 0,
                "realized_excess_return": -0.025,
            },
            {
                "market_state": "NORMAL",
                "sector_state": "NORMAL",
                "gap_bucket": 2,
                "realized_excess_return": -0.004,
            },
            {
                "market_state": "NORMAL",
                "sector_state": "NORMAL",
                "gap_bucket": 2,
                "realized_excess_return": -0.003,
            },
            {
                "market_state": "NORMAL",
                "sector_state": "NORMAL",
                "gap_bucket": 2,
                "realized_excess_return": -0.002,
            },
            {
                "market_state": "NORMAL",
                "sector_state": "NORMAL",
                "gap_bucket": 2,
                "realized_excess_return": 0.006,
            },
            {
                "market_state": "NORMAL",
                "sector_state": "NORMAL",
                "gap_bucket": 4,
                "realized_excess_return": 0.015,
            },
            {
                "market_state": "NORMAL",
                "sector_state": "NORMAL",
                "gap_bucket": 4,
                "realized_excess_return": 0.020,
            },
        ]
    }
    book = OpenCalibrationBook(schedule, open_config)
    book.mature(available + pd.Timedelta(days=1))
    rejected = book.decision("NORMAL", "NORMAL", -2.0)
    reduced = book.decision("NORMAL", "NORMAL", 0.0)
    allowed = book.decision("NORMAL", "NORMAL", 2.0)
    assert rejected["scale"] == 0.0
    assert rejected["reason"] == "OPEN_BUCKET_STRONG_NEGATIVE_REJECTED"
    assert reduced["scale"] == 0.50
    assert reduced["reason"] == "OPEN_BUCKET_MILD_REDUCTION"
    assert allowed["scale"] == 1.0
    assert allowed["reason"] == "OPEN_BUCKET_NO_MATERIAL_ADVERSE_EDGE"


def test_subshare_adjusted_residual_does_not_consume_position_slot() -> None:
    account = AdjustedAccount(
        cash=1000.0,
        lots={
            "SH600000": [AdjustedLot(units=0.1, acquired_on=pd.Timestamp("2025-01-01").date())],
            "SH600001": [AdjustedLot(units=100.0, acquired_on=pd.Timestamp("2025-01-01").date())],
            "SH600002": [AdjustedLot(units=100.0, acquired_on=pd.Timestamp("2025-01-01").date())],
        },
    )
    day = pd.DataFrame(
        {
            "symbol": ["SH600000", "SH600001"],
            "adjust_factor": [1.0, 1.0],
        }
    ).set_index("symbol")
    # SH600000 rounds to zero raw shares, SH600001 is material, and the
    # temporarily missing SH600002 still reserves a slot.
    assert _material_position_count(account, day) == 2
    assert _material_positions(account, day) == {"SH600001", "SH600002"}


def test_position_availability_separates_suspension_from_missing_data() -> None:
    trading_date = pd.Timestamp("2025-04-22").date()
    account = AdjustedAccount(
        cash=0.0,
        lots={
            symbol: [
                AdjustedLot(
                    units=100.0,
                    acquired_on=pd.Timestamp("2025-01-01").date(),
                )
            ]
            for symbol in ("SH600001", "SH600002", "SH600003")
        },
    )
    day = pd.DataFrame(
        {
            "symbol": ["SH600002"],
            "adjust_factor": [1.0],
            "is_trading": [False],
        }
    ).set_index("symbol")
    suspended, missing = _position_availability(
        account,
        day,
        trading_date,
        {("SH600001", trading_date)},
    )
    assert suspended == {"SH600001", "SH600002"}
    assert missing == {"SH600003"}


def test_regime_controls_change_exposure_and_topk() -> None:
    configured = candidate("weekly_top20_drop2_exit30_equal")
    configured["market_state_exposure_multipliers"] = {
        "NORMAL": 1.0,
        "CAUTIOUS": 0.70,
        "HIGH_RISK": 0.35,
        "DATA_UNAVAILABLE": 0.50,
    }
    configured["topk_by_market_state"] = {
        "NORMAL": 20,
        "CAUTIOUS": 15,
        "HIGH_RISK": 10,
        "DATA_UNAVAILABLE": 10,
    }
    assert _exposure(configured, None, "HIGH_RISK") == pytest.approx(0.35)
    effective = _candidate_for_market_state(configured, "CAUTIOUS")
    assert effective["topk"] == 15
    assert configured["topk"] == 20


def test_position_cap_allows_only_preexisting_unresolved_breach() -> None:
    assert _is_preexisting_position_cap_breach(21, 21, 20)
    assert not _is_preexisting_position_cap_breach(20, 20, 20)
    try:
        _is_preexisting_position_cap_breach(20, 21, 20)
    except RuntimeError as error:
        assert "Trading logic created" in str(error)
    else:
        raise AssertionError("A newly created position-cap breach must fail")


def test_nonmaterial_adjusted_residual_is_permanently_purged() -> None:
    acquired = pd.Timestamp("2025-01-01").date()
    trading_date = pd.Timestamp("2025-01-03").date()
    account = AdjustedAccount(
        cash=1000.0,
        lots={"SZ300502": [AdjustedLot(units=0.4, acquired_on=acquired)]},
    )
    day = pd.DataFrame(
        {
            "symbol": ["SZ300502"],
            "adjust_factor": [1.0],
        }
    ).set_index("symbol")
    assert _purge_nonmaterial_residuals(account, day, trading_date) == ["SZ300502"]
    assert "SZ300502" not in account.lots


def test_corporate_action_converts_position_and_compensates_fraction() -> None:
    acquired = pd.Timestamp("2016-12-01").date()
    account = AdjustedAccount(
        cash=1000.0,
        lots={"SH600005": [AdjustedLot(units=701.0, acquired_on=acquired)]},
    )
    result = _convert_position(
        account=account,
        source_symbol="SH600005",
        target_symbol="SH600019",
        share_ratio=0.56,
        source_adjust_factor=1.0,
        target_adjust_factor=2.0,
        trading_date=pd.Timestamp("2017-02-27").date(),
        fractional_share_price=8.0,
    )
    assert result["source_shares"] == 701
    assert result["target_shares"] == 392
    assert result["fractional_target_shares"] == pytest.approx(0.56)
    assert result["cash_compensation"] == pytest.approx(4.48)
    assert "SH600005" not in account.lots
    assert account.raw_shares("SH600019", 2.0) == 392
    assert account.sellable_raw_shares(
        "SH600019",
        2.0,
        pd.Timestamp("2017-02-27").date(),
    ) == 392


def test_corporate_action_is_idempotent_and_audited() -> None:
    account = AdjustedAccount(
        cash=0.0,
        lots={
            "SH600005": [
                AdjustedLot(
                    units=1000.0,
                    acquired_on=pd.Timestamp("2016-01-01").date(),
                )
            ]
        },
    )
    day = pd.DataFrame(
        {
            "symbol": ["SH600019"],
            "open": [8.0],
            "adjust_factor": [1.0],
        }
    ).set_index("symbol")
    action = {
        "action_id": "MERGER",
        "effective_date": "2017-02-27",
        "source_symbol": "SH600005",
        "target_symbol": "SH600019",
        "share_ratio": 0.56,
        "reference_urls": ["https://example.invalid/official"],
    }
    applied: set[str] = set()
    orders: list[dict[str, object]] = []
    for _ in range(2):
        RankBufferSimulator._apply_corporate_actions(
            account,
            day,
            {"SH600005": 1.0},
            pd.Timestamp("2017-02-27").date(),
            [action],
            applied,
            orders,
        )
    assert account.raw_shares("SH600019", 1.0) == 560
    assert applied == {"MERGER"}
    assert len(orders) == 1
    assert orders[0]["side"] == "CONVERT"
    assert orders[0]["status"] == "FILLED"


def test_terminal_event_writes_off_position_once() -> None:
    account = AdjustedAccount(
        cash=1000.0,
        lots={
            "SZ002411": [
                AdjustedLot(
                    units=1000.0,
                    acquired_on=pd.Timestamp("2023-01-01").date(),
                )
            ]
        },
    )
    event = {
        "event_id": "DELIST",
        "effective_date": "2023-07-13",
        "symbol": "SZ002411",
        "recovery_rate": 0.0,
        "reason": "CONSERVATIVE_ZERO_RECOVERY_AFTER_EXCHANGE_TERMINATION",
    }
    applied: set[str] = set()
    orders: list[dict[str, object]] = []
    for _ in range(2):
        RankBufferSimulator._apply_terminal_events(
            account,
            {"SZ002411": 0.69},
            {"SZ002411": 1.0},
            pd.Timestamp("2023-07-13").date(),
            [event],
            applied,
            orders,
        )
    assert "SZ002411" not in account.lots
    assert account.cash == pytest.approx(1000.0)
    assert applied == {"DELIST"}
    assert len(orders) == 1
    assert orders[0]["side"] == "WRITE_OFF"


def test_rank_buffer_simulator_executes_after_signal_and_caps_positions() -> None:
    dates = pd.bdate_range("2025-01-02", periods=80)
    rows = []
    symbols = ["SH000300", *[f"SH60{index:04d}" for index in range(25)]]
    for symbol_index, symbol in enumerate(symbols):
        initial = 100.0 if symbol == "SH000300" else 10.0 + symbol_index * 0.1
        for day_index, trading_day in enumerate(dates):
            price = initial * 1.0005**day_index
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trading_day,
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 20_000_000,
                    "is_trading": True,
                    "is_st": False,
                    "adjust_factor": 1.0,
                    "hfq_open": price,
                    "hfq_close": price,
                }
            )
    prediction_rows = []
    for trading_day in dates:
        for index, symbol in enumerate(symbols[1:]):
            prediction_rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trading_day,
                    "model_score": 1.0 - index / 25.0,
                    "outperform_probability": 0.65,
                    "sector_name": f"行业{index % 4}",
                    "market_state": "NORMAL",
                    "sector_state": "NORMAL",
                }
            )
    for candidate_name in (
        "weekly_top20_drop2_exit30_equal",
        "weekly_top20_drop2_exit30_risk_open",
    ):
        simulation_start = dates[20]
        simulator = RankBufferSimulator(
            research_config(),
            execution_config(),
            candidate(candidate_name),
        )
        result = simulator.run(
            pd.DataFrame(rows),
            pd.DataFrame(prediction_rows),
            simulation_start=simulation_start,
        )
        assert not result.trades.empty
        assert pd.to_datetime(result.nav["trade_date"]).min() >= simulation_start
        first_trade = pd.to_datetime(result.trades["trade_date"]).min()
        assert first_trade > simulation_start
        counts = result.holdings.groupby("trade_date")["symbol"].nunique()
        assert counts.max() <= 20
        assert result.nav["cash"].min() >= 0

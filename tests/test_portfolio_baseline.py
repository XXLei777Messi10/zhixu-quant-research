from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant.config import ProjectPaths
from quant.research.portfolio_baseline import (
    _eligible,
    _load_inherited_config,
    topk_and_calibration_diagnostics,
)


def test_topk_diagnostics_preserve_monotone_score_buckets() -> None:
    predictions: list[dict] = []
    labels: list[dict] = []
    for day in pd.to_datetime(["2024-01-05", "2024-01-12"]):
        for index in range(20):
            symbol = f"SH60{index:04d}"
            score = (index + 1) / 20
            predictions.append(
                {
                    "symbol": symbol,
                    "trade_date": day,
                    "model_score": score,
                    "calibration_bin": min(9, int(score * 10)),
                    "calibrated_excess_return": score * 0.02 - 0.01,
                    "calibrated_outperform_probability": score,
                }
            )
            labels.append(
                {
                    "symbol": symbol,
                    "trade_date": day,
                    "label_regression": score * 0.02 - 0.01,
                }
            )
    result = topk_and_calibration_diagnostics(
        pd.DataFrame(predictions),
        pd.DataFrame(labels),
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-31"),
        [5, 10],
    )
    assert result["dates"] == 2
    assert set(result["topk"]) == {"5", "10"}
    decile_returns = [
        row["mean_excess_return"]
        for row in result["score_deciles"]
    ]
    assert decile_returns == sorted(decile_returns)
    assert result["calibration_bins"]


def test_candidate_gate_requires_both_development_subperiods() -> None:
    rules = {
        "minimum_excess_cumulative_return": 0.0,
        "maximum_drawdown_floor": -0.35,
        "maximum_annual_turnover_gross": 10.0,
    }
    passing = {
        "excess_cumulative_return": 0.10,
        "max_drawdown": -0.20,
        "annual_turnover_gross": 5.0,
        "maximum_missing_data_positions": 0,
    }
    failing = {**passing, "max_drawdown": -0.40}
    assert _eligible(passing, passing, rules)
    assert not _eligible(passing, failing, rules)


def test_accounting_config_recursively_inherits_data_controls() -> None:
    paths = ProjectPaths(Path(".").resolve())
    config = _load_inherited_config(
        paths,
        "portfolio_baseline_accounting",
    )
    assert config["history_start"] == "2010-01-01"
    assert config["report_file"].endswith("baseline-accounting.json")
    assert any(
        event["symbol"] == "SH600485"
        for event in config["terminal_events"]
    )
    assert any(
        event["symbol"] == "SH600074"
        and event["effective_date"] == "2020-05-26"
        for event in config["terminal_events"]
    )


def test_terminal_repair_has_distinct_immutable_outputs() -> None:
    paths = ProjectPaths(Path(".").resolve())
    config = _load_inherited_config(
        paths,
        "portfolio_baseline_final",
    )
    assert config["run_id"].endswith("portfolio-baseline-final")
    assert config["report_file"].endswith("portfolio-baseline-final.json")
    assert config["artifact_directory"].endswith("portfolio-baseline-final")

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant.backtest.metrics import performance_metrics
from quant.config import ProjectPaths, load_config
from quant.execution.backtest import ProductionMirrorSimulator
from quant.signals.archive import immutable_write_json

BAR_COLUMNS = [
    "symbol",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "is_trading",
    "is_st",
    "adjust_factor",
    "hfq_open",
    "hfq_high",
    "hfq_low",
    "hfq_close",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _diagnostics(result) -> dict[str, Any]:
    nav = result.nav.copy()
    nav["trade_date"] = pd.to_datetime(nav["trade_date"]).dt.normalize()
    nav_value = pd.to_numeric(nav["nav"], errors="coerce")
    cash = pd.to_numeric(nav["cash"], errors="coerce")
    exposure = pd.to_numeric(nav["gross_exposure"], errors="coerce")
    counts = pd.Series(0, index=nav["trade_date"])
    if not result.holdings.empty:
        holding_counts = (
            result.holdings.assign(
                trade_date=pd.to_datetime(result.holdings["trade_date"]).dt.normalize()
            )
            .groupby("trade_date")["symbol"]
            .nunique()
        )
        counts = nav["trade_date"].map(holding_counts).fillna(0)
    reasons = (
        result.orders["reason"].fillna("").value_counts().to_dict()
        if not result.orders.empty
        else {}
    )
    return {
        "average_cash_ratio": float((cash / nav_value).mean()),
        "average_gross_exposure": float(exposure.mean()),
        "median_gross_exposure": float(exposure.median()),
        "fraction_above_50pct_exposure": float(exposure.gt(0.50).mean()),
        "fraction_zero_exposure": float(exposure.lt(1e-9).mean()),
        "average_holdings": float(counts.mean()),
        "maximum_holdings": int(counts.max()),
        "ending_holdings": int(counts.iloc[-1]),
        "order_reason_counts": {str(key): int(value) for key, value in reasons.items()},
    }


def _period_inputs(
    bars: pd.DataFrame,
    predictions: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    period_bars = bars[bars["trade_date"].between(start, end)].copy()
    period_predictions = predictions[
        predictions["trade_date"].between(start, end)
    ].copy()
    return period_bars, period_predictions


def _candidate_config(
    execution_config: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    config = copy.deepcopy(execution_config)
    config["initial_entry_fraction"] = float(candidate["initial_entry_fraction"])
    config["add_fraction"] = float(candidate["add_fraction"])
    config["rule_version"] = f"policy-research-{candidate['name']}"
    return config


def _run_candidate(
    bars: pd.DataFrame,
    predictions: pd.DataFrame,
    benchmark: pd.DataFrame,
    execution_config: dict[str, Any],
    candidate: dict[str, Any],
    calibration_cache: dict[pd.Timestamp, dict[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    config = _candidate_config(execution_config, candidate)
    result = ProductionMirrorSimulator(
        config,
        apply_risk_gate=bool(candidate.get("apply_risk_gate", False)),
        exit_rank=int(candidate["exit_rank"]),
        risk_weight_scales=candidate.get("risk_weight_scales"),
        risk_scale_refresh=str(candidate.get("risk_scale_refresh", "daily")),
        variant_name=str(candidate["name"]),
        calibration_cache=calibration_cache,
    ).run(bars, predictions)
    metrics = performance_metrics(result.nav, benchmark, result.trades)
    metrics["diagnostics"] = _diagnostics(result)
    return result, metrics


def _selection_key(metrics: dict[str, Any], config: dict[str, Any]) -> tuple[float, ...]:
    primary = str(config["selection"]["primary_metric"])
    tie_breakers = [str(value) for value in config["selection"]["tie_breakers"]]
    return tuple(float(metrics.get(key, float("-inf"))) for key in [primary, *tie_breakers])


def _locked_acceptance(metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    rules = config["locked_acceptance"]
    checks = {
        "annualized_return": float(metrics["annualized_return"])
        >= float(rules["minimum_annualized_return"]),
        "sharpe_ratio": float(metrics["sharpe_ratio"])
        >= float(rules["minimum_sharpe_ratio"]),
        "max_drawdown": float(metrics["max_drawdown"])
        >= float(rules["maximum_drawdown_floor"]),
        "excess_cumulative_return": float(metrics["excess_cumulative_return"])
        >= float(rules["minimum_excess_cumulative_return"]),
    }
    return {"passed": all(checks.values()), "checks": checks, "rules": rules}


def run_policy_research(
    paths: ProjectPaths,
    config_name: str = "policy_research",
) -> dict[str, Any]:
    research_config = load_config(paths, config_name)
    execution_config = load_config(paths, "execution")
    data_config = load_config(paths, "data")
    prediction_name = str(
        research_config.get(
            "prediction_file",
            "rolling_predictions.parquet",
        )
    )
    if Path(prediction_name).name != prediction_name:
        raise ValueError("prediction_file must be a file name inside reports/research")
    prediction_path = paths.reports / "research" / prediction_name
    bars_path = paths.curated / "bars.parquet"
    if not prediction_path.exists() or not bars_path.exists():
        raise FileNotFoundError(
            "Existing rolling predictions and curated bars are required"
        )
    predictions = pd.read_parquet(prediction_path)
    predictions["trade_date"] = pd.to_datetime(
        predictions["trade_date"]
    ).dt.normalize()
    bars = pd.read_parquet(bars_path, columns=BAR_COLUMNS)
    bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.normalize()
    benchmark_symbol = str(data_config["benchmark_symbol"])
    benchmark = bars[bars["symbol"].astype(str).eq(benchmark_symbol)][
        ["trade_date", "close"]
    ].copy()
    bars = bars[~bars["symbol"].astype(str).eq(benchmark_symbol)].copy()

    development_start = pd.Timestamp(research_config["development_start"])
    development_end = pd.Timestamp(research_config["development_end"])
    development_bars, development_predictions = _period_inputs(
        bars,
        predictions,
        development_start,
        development_end,
    )
    development_benchmark = benchmark[
        benchmark["trade_date"].between(development_start, development_end)
    ]
    calibration_cache: dict[pd.Timestamp, dict[str, Any]] = {}
    development_results: dict[str, dict[str, Any]] = {}
    eligible: list[tuple[str, dict[str, Any]]] = []
    drawdown_floor = float(
        research_config["selection"]["maximum_drawdown_floor"]
    )
    for candidate in research_config["candidates"]:
        _, metrics = _run_candidate(
            development_bars,
            development_predictions,
            development_benchmark,
            execution_config,
            candidate,
            calibration_cache,
        )
        name = str(candidate["name"])
        development_results[name] = metrics
        if float(metrics["max_drawdown"]) >= drawdown_floor:
            eligible.append((name, metrics))
    if not eligible:
        rejection = {
            "status": "DEVELOPMENT_REJECTED",
            "created_at": datetime.now(UTC).isoformat(),
            "development_period": [
                development_start.date().isoformat(),
                development_end.date().isoformat(),
            ],
            "selection_rule": research_config["selection"],
            "candidates": research_config["candidates"],
            "development_results": development_results,
            "selected_candidate": None,
            "locked_period_read": False,
            "rejection_reason": (
                "No pre-registered candidate passed the development-period "
                f"maximum drawdown floor {drawdown_floor:.2%}"
            ),
            "input_versions": {
                "rolling_predictions": str(prediction_path),
                "rolling_predictions_sha256": _sha256(prediction_path),
                "curated_bars": str(bars_path),
                "curated_bars_sha256": _sha256(bars_path),
            },
            "simulation_only": True,
        }
        rejection_path = immutable_write_json(
            paths.reports
            / "research"
            / f"{config_name.replace('_', '-')}-development-rejected.json",
            rejection,
        )
        return {**rejection, "report": str(rejection_path)}
    selected_name, _ = max(
        eligible,
        key=lambda item: _selection_key(item[1], research_config),
    )
    selected = next(
        item
        for item in research_config["candidates"]
        if str(item["name"]) == selected_name
    )

    full_start = development_start
    full_end = predictions["trade_date"].max()
    full_bars, full_predictions = _period_inputs(
        bars,
        predictions,
        full_start,
        full_end,
    )
    full_benchmark = benchmark[
        benchmark["trade_date"].between(full_start, full_end)
    ]
    selected_result, full_metrics = _run_candidate(
        full_bars,
        full_predictions,
        full_benchmark,
        execution_config,
        selected,
        calibration_cache,
    )
    locked_start = pd.Timestamp(research_config["locked_start"])
    locked_nav = selected_result.nav[
        pd.to_datetime(selected_result.nav["trade_date"]).between(
            locked_start,
            full_end,
        )
    ]
    locked_trades = selected_result.trades[
        pd.to_datetime(selected_result.trades["trade_date"]).between(
            locked_start,
            full_end,
        )
    ]
    locked_benchmark = benchmark[
        benchmark["trade_date"].between(locked_start, full_end)
    ]
    locked_metrics = performance_metrics(
        locked_nav,
        locked_benchmark,
        locked_trades,
    )
    locked_metrics["diagnostics"] = _diagnostics(
        type(selected_result)(
            locked_nav,
            selected_result.holdings[
                pd.to_datetime(selected_result.holdings["trade_date"]).between(
                    locked_start,
                    full_end,
                )
            ],
            locked_trades,
            selected_result.orders[
                pd.to_datetime(selected_result.orders["trade_date"]).between(
                    locked_start,
                    full_end,
                )
            ],
        )
    )
    acceptance = _locked_acceptance(locked_metrics, research_config)

    output_root = paths.reports / "research" / "policy-improvement" / selected_name
    output_root.mkdir(parents=True, exist_ok=True)
    selected_result.nav.to_parquet(output_root / "nav.parquet", index=False)
    selected_result.holdings.to_parquet(output_root / "holdings.parquet", index=False)
    selected_result.trades.to_parquet(output_root / "trades.parquet", index=False)
    selected_result.orders.to_parquet(output_root / "orders.parquet", index=False)
    report = {
        "status": (
            "PORTFOLIO_POLICY_ACCEPTED"
            if acceptance["passed"]
            else "MODEL_RESEARCH_REQUIRED"
        ),
        "created_at": datetime.now(UTC).isoformat(),
        "development_period": [
            development_start.date().isoformat(),
            development_end.date().isoformat(),
        ],
        "locked_period": [
            locked_start.date().isoformat(),
            full_end.date().isoformat(),
        ],
        "selection_rule": research_config["selection"],
        "candidates": research_config["candidates"],
        "development_results": development_results,
        "selected_candidate": selected_name,
        "selected_full_period": full_metrics,
        "selected_locked_period": locked_metrics,
        "locked_acceptance": acceptance,
        "input_versions": {
            "rolling_predictions": str(prediction_path),
            "rolling_predictions_sha256": _sha256(prediction_path),
            "curated_bars": str(bars_path),
            "curated_bars_sha256": _sha256(bars_path),
        },
        "simulation_only": True,
    }
    report_path = immutable_write_json(
        paths.reports / "research" / f"{config_name.replace('_', '-')}.json",
        report,
    )
    return {**report, "report": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", default="policy_research")
    args = parser.parse_args()
    result = run_policy_research(
        ProjectPaths(args.root.resolve()),
        config_name=str(args.config),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

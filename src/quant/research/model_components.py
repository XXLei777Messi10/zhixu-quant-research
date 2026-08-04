from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.config import ProjectPaths, load_config
from quant.signals.archive import immutable_write_json


def build_forward_labels(
    bars: pd.DataFrame,
    benchmark_symbol: str,
    horizon: int = 5,
    entry_offset: int = 1,
) -> pd.DataFrame:
    """Build forward open-to-open excess-return labels without future features."""
    if horizon <= 0:
        raise ValueError("Label horizon must be positive")
    if entry_offset <= 0:
        raise ValueError("Label entry offset must be positive")
    exit_offset = entry_offset + horizon
    required = {"symbol", "trade_date", "hfq_open"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"Bars missing label columns: {sorted(missing)}")
    frame = bars[list(required)].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    frame["hfq_open"] = pd.to_numeric(frame["hfq_open"], errors="coerce")
    frame = frame.sort_values(["symbol", "trade_date"])

    benchmark = frame[frame["symbol"].astype(str).eq(benchmark_symbol)].copy()
    if benchmark.empty:
        raise ValueError(f"Benchmark {benchmark_symbol} is missing from curated bars")
    benchmark = benchmark.drop_duplicates("trade_date").sort_values("trade_date")
    benchmark["benchmark_label_return"] = (
        benchmark["hfq_open"].shift(-exit_offset)
        / benchmark["hfq_open"].shift(-entry_offset)
        - 1.0
    )

    stocks = frame[~frame["symbol"].astype(str).eq(benchmark_symbol)].copy()
    grouped_open = stocks.groupby("symbol", sort=False)["hfq_open"]
    stocks["stock_label_return"] = (
        grouped_open.shift(-exit_offset) / grouped_open.shift(-entry_offset) - 1.0
    )
    stocks = stocks.merge(
        benchmark[["trade_date", "benchmark_label_return"]],
        on="trade_date",
        how="left",
        validate="many_to_one",
    )
    stocks["label_regression"] = (
        stocks["stock_label_return"] - stocks["benchmark_label_return"]
    )
    stocks["label_classification"] = stocks["label_regression"].gt(0).astype("Int8")
    stocks.loc[stocks["label_regression"].isna(), "label_classification"] = pd.NA
    return stocks[
        ["symbol", "trade_date", "label_regression", "label_classification"]
    ]


def weighted_score(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    if not weights:
        raise ValueError("Candidate weights cannot be empty")
    missing = set(weights).difference(frame.columns)
    if missing:
        raise ValueError(f"Predictions missing score columns: {sorted(missing)}")
    total = float(sum(weights.values()))
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Candidate weights must total 1.0, got {total}")
    if any(float(value) < 0 for value in weights.values()):
        raise ValueError("Candidate weights cannot be negative")
    score = pd.Series(0.0, index=frame.index, dtype="float64")
    for column, weight in weights.items():
        score = score + pd.to_numeric(frame[column], errors="coerce") * float(weight)
    return score


def _daily_metrics(group: pd.DataFrame, top_k: int) -> pd.Series:
    valid = group.dropna(subset=["candidate_score", "label_regression"]).copy()
    if len(valid) < max(top_k * 2, 5):
        return pd.Series(dtype="float64")
    rank_ic = valid["candidate_score"].corr(
        valid["label_regression"], method="spearman"
    )
    ordered = valid.sort_values(
        ["candidate_score", "symbol"], ascending=[False, True]
    )
    top = ordered.head(top_k)
    bottom = ordered.tail(top_k)
    return pd.Series(
        {
            "rank_ic": rank_ic,
            "top_k_mean_excess_return": top["label_regression"].mean(),
            "top_k_hit_rate": top["label_regression"].gt(0).mean(),
            "top_bottom_spread": (
                top["label_regression"].mean() - bottom["label_regression"].mean()
            ),
        }
    )


def candidate_metrics(
    merged: pd.DataFrame,
    weights: dict[str, float],
    top_k: int,
) -> dict[str, Any]:
    frame = merged.copy()
    frame["candidate_score"] = weighted_score(frame, weights)
    daily = frame.groupby("trade_date", sort=True).apply(
        _daily_metrics,
        top_k=top_k,
        include_groups=False,
    )
    if isinstance(daily, pd.Series):
        if not isinstance(daily.index, pd.MultiIndex):
            raise ValueError("No valid daily cross-sections for candidate")
        daily = daily.unstack(level=-1)
    daily = daily.dropna(subset=["rank_ic"])
    if daily.empty:
        raise ValueError("No valid daily cross-sections for candidate")
    daily_std = float(daily["rank_ic"].std(ddof=1))
    annual = daily.assign(year=daily.index.year).groupby("year").agg(
        mean_daily_rank_ic=("rank_ic", "mean"),
        top_k_mean_excess_return=("top_k_mean_excess_return", "mean"),
        top_k_hit_rate=("top_k_hit_rate", "mean"),
        top_bottom_spread=("top_bottom_spread", "mean"),
        dates=("rank_ic", "size"),
    )
    positive_years = int(annual["mean_daily_rank_ic"].gt(0).sum())
    return {
        "rows": int(len(frame)),
        "dates": int(len(daily)),
        "mean_daily_rank_ic": float(daily["rank_ic"].mean()),
        "rank_ic_information_ratio": (
            float(daily["rank_ic"].mean() / daily_std * math.sqrt(252.0))
            if daily_std > 0
            else np.nan
        ),
        "positive_rank_ic_fraction": float(daily["rank_ic"].gt(0).mean()),
        "top_k_mean_excess_return": float(
            daily["top_k_mean_excess_return"].mean()
        ),
        "top_k_hit_rate": float(daily["top_k_hit_rate"].mean()),
        "top_bottom_spread": float(daily["top_bottom_spread"].mean()),
        "positive_years": positive_years,
        "negative_years": int(len(annual) - positive_years),
        "annual": {
            str(int(year)): {
                key: (int(value) if key == "dates" else float(value))
                for key, value in row.items()
            }
            for year, row in annual.to_dict(orient="index").items()
        },
    }


def _selection_key(
    metrics: dict[str, Any],
    selection: dict[str, Any],
) -> tuple[float, ...]:
    names = [
        str(selection["primary_metric"]),
        *[str(value) for value in selection.get("tie_breakers", [])],
    ]
    return tuple(float(metrics[name]) for name in names)


def run_component_research(
    paths: ProjectPaths,
    config_name: str = "model_component_research",
) -> dict[str, Any]:
    config = load_config(paths, config_name)
    data_config = load_config(paths, "data")
    prediction_path = paths.reports / "research" / "rolling_predictions.parquet"
    bars_path = paths.curated / "bars.parquet"
    predictions = pd.read_parquet(prediction_path)
    bars = pd.read_parquet(
        bars_path,
        columns=["symbol", "trade_date", "hfq_open"],
    )
    predictions["trade_date"] = pd.to_datetime(
        predictions["trade_date"]
    ).dt.normalize()
    start = pd.Timestamp(config["development_start"])
    end = pd.Timestamp(config["development_end"])
    predictions = predictions[predictions["trade_date"].between(start, end)].copy()
    labels = build_forward_labels(bars, str(data_config["benchmark_symbol"]))
    merged = predictions.merge(
        labels,
        on=["symbol", "trade_date"],
        how="inner",
        validate="one_to_one",
    ).dropna(subset=["label_regression"])

    results: dict[str, dict[str, Any]] = {}
    eligible: list[tuple[str, dict[str, Any]]] = []
    selection = config["selection"]
    minimum_positive_years = int(selection["minimum_positive_years"])
    for candidate in config["candidates"]:
        name = str(candidate["name"])
        metrics = candidate_metrics(
            merged,
            {str(key): float(value) for key, value in candidate["weights"].items()},
            int(config["top_k"]),
        )
        results[name] = metrics
        if metrics["positive_years"] >= minimum_positive_years:
            eligible.append((name, metrics))
    if not eligible:
        raise RuntimeError(
            "No pre-registered score blend passed the positive-year requirement"
        )
    selected_name, _ = max(
        eligible,
        key=lambda item: _selection_key(item[1], selection),
    )
    selected = next(
        item for item in config["candidates"] if str(item["name"]) == selected_name
    )
    report = {
        "status": "DEVELOPMENT_SELECTION_COMPLETE",
        "created_at": datetime.now(UTC).isoformat(),
        "development_period": [start.date().isoformat(), end.date().isoformat()],
        "top_k": int(config["top_k"]),
        "selection": selection,
        "candidates": config["candidates"],
        "results": results,
        "selected_candidate": selected_name,
        "selected_weights": selected["weights"],
        "locked_period_read": False,
        "warning": (
            "The 2024+ period was not read by this command. Earlier project rounds "
            "have already inspected it, so it is not a pristine holdout."
        ),
        "simulation_only": True,
    }
    report_path = immutable_write_json(
        paths.reports / "research" / "model-component-development.json",
        report,
    )
    return {**report, "report": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", default="model_component_research")
    args = parser.parse_args()
    report = run_component_research(
        ProjectPaths(args.root.resolve()),
        config_name=str(args.config),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

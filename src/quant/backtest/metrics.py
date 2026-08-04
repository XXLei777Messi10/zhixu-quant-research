from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss


def performance_metrics(
    nav: pd.DataFrame,
    benchmark: pd.DataFrame | None = None,
    trades: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if nav.empty:
        raise ValueError("NAV is empty")
    values = nav.set_index("trade_date")["nav"].astype(float)
    returns = values.pct_change(fill_method=None).dropna()
    cumulative = values.iloc[-1] / values.iloc[0] - 1.0
    annualized = (1.0 + cumulative) ** (252.0 / max(len(returns), 1)) - 1.0
    volatility = returns.std(ddof=1) * math.sqrt(252.0)
    drawdown = values / values.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    sharpe = annualized / volatility if volatility > 0 else np.nan
    calmar = annualized / abs(max_drawdown) if max_drawdown < 0 else np.nan
    monthly = values.resample("ME").last().pct_change(fill_method=None).dropna()
    yearly = values.resample("YE").last().pct_change(fill_method=None).dropna()
    output: dict[str, Any] = {
        "cumulative_return": float(cumulative),
        "annualized_return": float(annualized),
        "annualized_volatility": float(volatility),
        "max_drawdown": max_drawdown,
        "sharpe_ratio": float(sharpe),
        "calmar_ratio": float(calmar),
        "monthly_win_rate": float(monthly.gt(0).mean()) if len(monthly) else np.nan,
        "yearly_returns": {str(index.year): float(value) for index, value in yearly.items()},
        "worst_months": {str(index.date()): float(value) for index, value in monthly.nsmallest(5).items()},
    }
    if trades is not None and not trades.empty:
        output["total_transaction_cost"] = float(trades["fees"].sum())
        output["turnover_gross"] = float(trades["gross"].sum() / values.mean())
        contribution = trades.groupby("symbol")["gross"].sum()
        weights = contribution / contribution.sum()
        output["trade_concentration_hhi"] = float((weights**2).sum())
    if benchmark is not None and not benchmark.empty:
        benchmark_values = benchmark.set_index("trade_date")["close"].reindex(values.index).ffill()
        benchmark_returns = benchmark_values.pct_change(fill_method=None)
        aligned = pd.concat(
            [returns.rename("portfolio"), benchmark_returns.rename("benchmark")], axis=1
        ).dropna()
        excess = aligned["portfolio"] - aligned["benchmark"]
        output["benchmark_cumulative_return"] = float(
            benchmark_values.iloc[-1] / benchmark_values.iloc[0] - 1.0
        )
        output["excess_cumulative_return"] = float(cumulative - output["benchmark_cumulative_return"])
        output["information_ratio"] = (
            float(excess.mean() / excess.std(ddof=1) * math.sqrt(252.0)) if excess.std(ddof=1) > 0 else np.nan
        )
    return output


def signal_metrics(predictions: pd.DataFrame, labels: pd.DataFrame) -> dict[str, Any]:
    merged = predictions.merge(
        labels[["symbol", "trade_date", "label_regression", "label_classification"]],
        on=["symbol", "trade_date"],
        how="inner",
    ).dropna(subset=["label_regression", "label_classification"])
    if merged.empty:
        return {}
    daily_ic = merged.groupby("trade_date").apply(
        lambda group: group["predicted_excess_return"].corr(group["label_regression"]),
        include_groups=False,
    )
    daily_rank_ic = merged.groupby("trade_date").apply(
        lambda group: group["model_score"].corr(group["label_regression"], method="spearman"),
        include_groups=False,
    )
    probability = merged["outperform_probability"].clip(1e-6, 1 - 1e-6)
    truth = merged["label_classification"].astype(int)
    fraction, mean_probability = calibration_curve(truth, probability, n_bins=10, strategy="quantile")
    merged["group"] = merged.groupby("trade_date")["model_score"].transform(
        lambda series: pd.qcut(series.rank(method="first"), 5, labels=False, duplicates="drop")
    )
    group_returns = merged.groupby("group", observed=True)["label_regression"].mean()
    return {
        "ic": float(daily_ic.mean()),
        "rank_ic": float(daily_rank_ic.mean()),
        "brier": float(brier_score_loss(truth, probability)),
        "log_loss": float(log_loss(truth, probability, labels=[0, 1])),
        "calibration": [
            {"mean_probability": float(x), "fraction_positive": float(y)}
            for x, y in zip(mean_probability, fraction, strict=True)
        ],
        "group_mean_excess_return": {str(int(index)): float(value) for index, value in group_returns.items()},
    }

from __future__ import annotations

from typing import Any

import pandas as pd


def compute_sector_states(
    bars: pd.DataFrame,
    membership: pd.DataFrame,
    as_of: pd.Timestamp,
    config: dict[str, Any],
) -> pd.DataFrame:
    required_bars = {"symbol", "trade_date", "hfq_close", "is_trading"}
    required_membership = {
        "symbol",
        "stock_name",
        "industry_name",
        "industry_classification",
        "as_of_date",
    }
    if missing := required_bars - set(bars):
        raise ValueError(f"Sector bars missing: {sorted(missing)}")
    if missing := required_membership - set(membership):
        raise ValueError(f"Sector membership missing: {sorted(missing)}")
    if membership.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "stock_name",
                "sector_name",
                "sector_state",
                "sector_return_20",
                "sector_volatility_20",
                "sector_member_count",
                "sector_mapping_as_of",
                "sector_classification",
            ]
        )

    as_of = pd.Timestamp(as_of).normalize()
    frame = bars.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    frame = frame[
        frame["trade_date"].le(as_of)
        & frame["is_trading"].fillna(False)
        & frame["hfq_close"].gt(0)
    ].sort_values(["symbol", "trade_date"])
    frame["stock_return_1d"] = frame.groupby("symbol", observed=True)[
        "hfq_close"
    ].pct_change(fill_method=None)
    mapping = membership[
        [
            "symbol",
            "stock_name",
            "industry_name",
            "industry_classification",
            "as_of_date",
        ]
    ].copy()
    valid_mapping = mapping.dropna(subset=["industry_name"])
    merged = frame.merge(
        valid_mapping[["symbol", "industry_name"]],
        on="symbol",
        how="inner",
        validate="many_to_one",
    )
    daily = (
        merged.dropna(subset=["stock_return_1d"])
        .groupby(["industry_name", "trade_date"], observed=True)["stock_return_1d"]
        .mean()
        .rename("sector_return_1d")
        .reset_index()
    )
    lookback = int(config["lookback_trading_days"])
    latest_rows: list[dict[str, Any]] = []
    member_counts = valid_mapping.groupby("industry_name", observed=True)["symbol"].nunique()
    for industry_name, group in daily.groupby("industry_name", observed=True):
        trailing = group.sort_values("trade_date").tail(lookback)
        if len(trailing) < lookback:
            continue
        latest_rows.append(
            {
                "industry_name": industry_name,
                "sector_return_20": float(
                    (1.0 + trailing["sector_return_1d"]).prod() - 1.0
                ),
                "sector_volatility_20": float(trailing["sector_return_1d"].std()),
                "sector_member_count": int(member_counts.get(industry_name, 0)),
            }
        )
    metrics = pd.DataFrame(
        latest_rows,
        columns=[
            "industry_name",
            "sector_return_20",
            "sector_volatility_20",
            "sector_member_count",
        ],
    )
    minimum_members = int(config["minimum_members"])
    valid = metrics[metrics["sector_member_count"].ge(minimum_members)].copy()
    if valid.empty:
        states = pd.DataFrame(columns=["industry_name", "sector_state"])
    else:
        weak_return = valid["sector_return_20"].quantile(
            float(config["weak_return_quantile"])
        )
        high_volatility = valid["sector_volatility_20"].quantile(
            float(config["high_volatility_quantile"])
        )
        valid["sector_state"] = "NORMAL"
        cautious = valid["sector_return_20"].le(weak_return) | valid[
            "sector_volatility_20"
        ].ge(high_volatility)
        high_risk = valid["sector_return_20"].le(weak_return) & valid[
            "sector_volatility_20"
        ].ge(high_volatility)
        valid.loc[cautious, "sector_state"] = "CAUTIOUS"
        valid.loc[high_risk, "sector_state"] = "HIGH_RISK"
        states = valid

    output = mapping.merge(states, on="industry_name", how="left")
    output["sector_state"] = output["sector_state"].fillna("DATA_UNAVAILABLE")
    output = output.rename(
        columns={
            "industry_name": "sector_name",
            "industry_classification": "sector_classification",
            "as_of_date": "sector_mapping_as_of",
        }
    )
    return output[
        [
            "symbol",
            "stock_name",
            "sector_name",
            "sector_state",
            "sector_return_20",
            "sector_volatility_20",
            "sector_member_count",
            "sector_mapping_as_of",
            "sector_classification",
        ]
    ].sort_values("symbol", ignore_index=True)

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_20",
    "ret_60",
    "excess_ret_1",
    "excess_ret_5",
    "excess_ret_10",
    "excess_ret_20",
    "excess_ret_60",
    "vol_5",
    "vol_10",
    "vol_20",
    "vol_60",
    "ma_pos_5",
    "ma_pos_10",
    "ma_pos_20",
    "ma_pos_60",
    "ma_slope_5",
    "ma_slope_20",
    "drawdown_20",
    "drawdown_60",
    "range_5",
    "range_20",
    "range_60",
    "overnight_gap",
    "intraday_return",
    "close_location",
    "volume_chg_1",
    "volume_ratio_5",
    "volume_ratio_20",
    "volume_ratio_60",
    "amount_chg_1",
    "amount_ratio_5",
    "amount_ratio_20",
    "amount_ratio_60",
    "price_volume_corr_5",
    "price_volume_corr_20",
    "price_volume_corr_60",
    "up_ratio_5",
    "up_ratio_20",
    "turnover_mean_5",
    "turnover_mean_20",
    "market_ret_1",
    "market_ret_20",
    "market_ret_60",
    "market_vol_20",
    "market_vol_60",
    "cs_rank_ret_5",
    "cs_rank_ret_20",
    "cs_rank_vol_20",
    "cs_rank_amount_20",
]


def _group_rolling(series: pd.Series, groups: pd.Series, window: int, operation: str) -> pd.Series:
    rolling = series.groupby(groups).rolling(window, min_periods=window)
    result = getattr(rolling, operation)()
    return result.reset_index(level=0, drop=True)


def _group_pct_change(series: pd.Series, groups: pd.Series, periods: int) -> pd.Series:
    return series.groupby(groups).pct_change(periods=periods, fill_method=None)


def build_features(
    curated: pd.DataFrame,
    benchmark_symbol: str = "SH000300",
    universe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required = {
        "symbol",
        "trade_date",
        "hfq_open",
        "hfq_high",
        "hfq_low",
        "hfq_close",
        "volume",
        "amount",
        "turnover",
    }
    missing = required - set(curated.columns)
    if missing:
        raise ValueError(f"Curated data missing feature fields: {sorted(missing)}")
    frame = curated.sort_values(["symbol", "trade_date"]).copy()
    frame = frame.rename(
        columns={
            "hfq_open": "feature_open",
            "hfq_high": "feature_high",
            "hfq_low": "feature_low",
            "hfq_close": "feature_close",
        }
    )
    stock = frame[frame["symbol"].ne(benchmark_symbol)].copy()
    if isinstance(stock["symbol"].dtype, pd.CategoricalDtype):
        stock["symbol"] = stock["symbol"].cat.remove_unused_categories()
    benchmark = (
        frame[frame["symbol"].eq(benchmark_symbol)]
        .drop_duplicates("trade_date", keep="last")
        .set_index("trade_date")["feature_close"]
        .sort_index()
    )
    if benchmark.empty:
        raise ValueError(f"Benchmark {benchmark_symbol} is absent")

    symbol = stock["symbol"]
    close = stock["feature_close"]
    high = stock["feature_high"]
    low = stock["feature_low"]
    open_ = stock["feature_open"]
    volume = stock["volume"].replace(0, np.nan)
    amount = stock["amount"].replace(0, np.nan)
    turnover = stock["turnover"]

    for window in (1, 5, 10, 20, 60):
        stock[f"ret_{window}"] = _group_pct_change(close, symbol, window)
    stock["log_volume"] = np.log1p(volume)
    for window in (5, 10, 20, 60):
        stock[f"vol_{window}"] = _group_rolling(stock["ret_1"], symbol, window, "std")
        ma = _group_rolling(close, symbol, window, "mean")
        stock[f"ma_{window}"] = ma
        stock[f"ma_pos_{window}"] = close / ma - 1.0
    stock["ma_slope_5"] = _group_pct_change(stock["ma_5"], symbol, 5)
    stock["ma_slope_20"] = _group_pct_change(stock["ma_20"], symbol, 5)
    for window in (20, 60):
        rolling_high = _group_rolling(close, symbol, window, "max")
        stock[f"drawdown_{window}"] = close / rolling_high - 1.0
    for window in (5, 20, 60):
        highest = _group_rolling(high, symbol, window, "max")
        lowest = _group_rolling(low, symbol, window, "min")
        stock[f"range_{window}"] = (highest - lowest) / close
    previous_close = close.groupby(symbol).shift(1)
    stock["overnight_gap"] = open_ / previous_close - 1.0
    stock["intraday_return"] = close / open_ - 1.0
    stock["close_location"] = (close - low) / (high - low).replace(0, np.nan)
    stock["volume_chg_1"] = _group_pct_change(volume, symbol, 1)
    stock["amount_chg_1"] = _group_pct_change(amount, symbol, 1)
    for window in (5, 20, 60):
        stock[f"volume_ratio_{window}"] = volume / _group_rolling(volume, symbol, window, "mean")
        stock[f"amount_ratio_{window}"] = amount / _group_rolling(amount, symbol, window, "mean")
        correlation = stock.groupby("symbol", group_keys=False).apply(
            lambda group, window=window: group["ret_1"]
            .rolling(window, min_periods=window)
            .corr(group["log_volume"]),
            include_groups=False,
        )
        stock[f"price_volume_corr_{window}"] = correlation.reset_index(level=0, drop=True).reindex(
            stock.index
        )
    for window in (5, 20):
        positive = stock["ret_1"].gt(0).astype(float)
        stock[f"up_ratio_{window}"] = _group_rolling(positive, symbol, window, "mean")
        stock[f"turnover_mean_{window}"] = _group_rolling(turnover, symbol, window, "mean")

    market = pd.DataFrame({"benchmark_close": benchmark})
    market["market_ret_1"] = market["benchmark_close"].pct_change(fill_method=None)
    market["market_ret_20"] = market["benchmark_close"].pct_change(20, fill_method=None)
    market["market_ret_60"] = market["benchmark_close"].pct_change(60, fill_method=None)
    market["market_vol_20"] = market["market_ret_1"].rolling(20, min_periods=20).std()
    market["market_vol_60"] = market["market_ret_1"].rolling(60, min_periods=60).std()
    stock = stock.merge(market.reset_index(), on="trade_date", how="left", validate="many_to_one")
    for window in (1, 5, 10, 20, 60):
        market_return = benchmark.pct_change(window, fill_method=None).rename(f"benchmark_ret_{window}")
        stock = stock.merge(market_return.reset_index(), on="trade_date", how="left", validate="many_to_one")
        stock[f"excess_ret_{window}"] = stock[f"ret_{window}"] - stock[f"benchmark_ret_{window}"]

    universe_mask = pd.Series(True, index=stock.index)
    if universe is not None:
        membership = universe[["trade_date", "symbol"]].drop_duplicates().copy()
        membership["trade_date"] = pd.to_datetime(membership["trade_date"]).dt.normalize()
        membership_index = pd.MultiIndex.from_frame(membership[["trade_date", "symbol"]])
        stock_index = pd.MultiIndex.from_frame(stock[["trade_date", "symbol"]])
        universe_mask = pd.Series(stock_index.isin(membership_index), index=stock.index)
    stock["in_universe"] = universe_mask.astype(bool)

    for source, target in (
        ("ret_5", "cs_rank_ret_5"),
        ("ret_20", "cs_rank_ret_20"),
        ("vol_20", "cs_rank_vol_20"),
        ("amount_ratio_20", "cs_rank_amount_20"),
    ):
        stock[target] = np.nan
        stock.loc[universe_mask, target] = (
            stock.loc[universe_mask].groupby("trade_date")[source].rank(pct=True, method="average")
        )

    output_columns = [
        "symbol",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "feature_open",
        "feature_close",
        "volume",
        "amount",
        "turnover",
        "is_trading",
        "is_st",
        "in_universe",
        *FEATURE_COLUMNS,
    ]
    result = stock[output_columns].copy()
    result[FEATURE_COLUMNS] = result[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).astype("float32")
    return result.sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def add_labels(
    features: pd.DataFrame,
    benchmark: pd.DataFrame,
    horizon: int = 5,
    entry_offset: int = 1,
) -> pd.DataFrame:
    """Add configurable forward open-to-open arithmetic excess-return labels."""
    if horizon <= 0 or entry_offset <= 0:
        raise ValueError("Label horizon and entry offset must be positive")
    exit_offset = entry_offset + horizon
    frame = features.sort_values(["symbol", "trade_date"]).copy()
    group = frame.groupby("symbol")["feature_open"]
    stock_entry = group.shift(-entry_offset)
    stock_exit = group.shift(-exit_offset)
    stock_return = stock_exit / stock_entry - 1.0
    bench = benchmark.sort_values("trade_date").drop_duplicates("trade_date").set_index("trade_date")
    bench_open = bench["hfq_open"]
    bench_return = (
        bench_open.shift(-exit_offset) / bench_open.shift(-entry_offset) - 1.0
    )
    frame = frame.merge(
        bench_return.rename("benchmark_label_return").reset_index(),
        on="trade_date",
        how="left",
        validate="many_to_one",
    )
    frame["label_regression"] = stock_return.to_numpy() - frame["benchmark_label_return"]
    frame["label_classification"] = frame["label_regression"].gt(0).astype("Int8")
    frame.loc[frame["label_regression"].isna(), "label_classification"] = pd.NA
    return frame

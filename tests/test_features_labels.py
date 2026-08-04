import pandas as pd
import pytest

from quant.data.normalize import add_adjustment_factor
from quant.qlib_workflow.features import add_labels, build_features
from quant.synthetic import synthetic_normalized


def _curated(periods: int = 100):
    frame = synthetic_normalized(symbols=2, periods=periods)
    raw = frame[(frame.source == "akshare") & (frame.adjustment == "raw")]
    hfq = frame[(frame.source == "akshare") & (frame.adjustment == "hfq")]
    return add_adjustment_factor(raw, hfq)


def test_features_do_not_change_when_future_changes() -> None:
    curated = _curated(100)
    cutoff = sorted(curated.trade_date.unique())[79]
    prefix = curated[curated.trade_date <= cutoff]
    first = build_features(prefix)
    changed = curated.copy()
    changed.loc[changed.trade_date > cutoff, ["hfq_open", "hfq_high", "hfq_low", "hfq_close"]] *= 100
    second = build_features(changed)
    columns = [column for column in first if column.startswith(("ret_", "vol_", "ma_", "cs_"))]
    pd.testing.assert_frame_equal(
        first[["symbol", "trade_date", *columns]].reset_index(drop=True),
        second.loc[second.trade_date <= cutoff, ["symbol", "trade_date", *columns]].reset_index(drop=True),
    )


def test_label_is_t1_open_to_t6_open_excess() -> None:
    curated = _curated(80)
    features = build_features(curated)
    benchmark = curated[curated.symbol == "SH000300"]
    labeled = add_labels(features, benchmark)
    symbol = "SH600000"
    rows = curated[curated.symbol == symbol].sort_values("trade_date").reset_index(drop=True)
    bench = benchmark.sort_values("trade_date").reset_index(drop=True)
    expected = rows.loc[6, "hfq_open"] / rows.loc[1, "hfq_open"] - 1
    expected -= bench.loc[6, "hfq_open"] / bench.loc[1, "hfq_open"] - 1
    actual = labeled[labeled.symbol == symbol].sort_values("trade_date").iloc[0]["label_regression"]
    assert actual == pytest.approx(expected)


def test_cross_sectional_ranks_only_use_point_in_time_members() -> None:
    curated = _curated(100)
    dates = sorted(curated["trade_date"].unique())
    universe = pd.DataFrame(
        {
            "trade_date": dates,
            "symbol": "SH600000",
        }
    )

    features = build_features(curated, universe=universe)
    latest = features[features["trade_date"].eq(dates[-1])].set_index("symbol")

    assert latest.loc["SH600000", "cs_rank_ret_20"] == pytest.approx(1.0)
    assert pd.isna(latest.loc["SH600001", "cs_rank_ret_20"])
    assert bool(latest.loc["SH600000", "in_universe"])
    assert not bool(latest.loc["SH600001", "in_universe"])

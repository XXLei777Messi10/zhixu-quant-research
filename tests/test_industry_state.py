from __future__ import annotations

import pandas as pd

from quant.data.industry import (
    load_industry_snapshot,
    write_industry_snapshot,
)
from quant.signals.sector import compute_sector_states


def membership(as_of: str = "2026-01-30") -> pd.DataFrame:
    rows = []
    for index in range(12):
        rows.append(
            {
                "as_of_date": pd.Timestamp(as_of),
                "symbol": f"SH{600000 + index:06d}",
                "stock_name": f"股票{index}",
                "industry_name": f"行业{index // 3}",
                "industry_classification": "证监会行业分类",
                "mapping_update_date": pd.Timestamp(as_of),
                "source": "baostock",
                "retrieved_at": pd.Timestamp(f"{as_of} 09:00:00", tz="UTC"),
                "request_id": "request-1",
            }
        )
    return pd.DataFrame(rows)


def sector_bars() -> pd.DataFrame:
    dates = pd.bdate_range("2025-12-01", periods=45)
    rows = []
    for index in range(12):
        sector = index // 3
        daily_return = (-0.006, -0.001, 0.002, 0.005)[sector]
        close = 10.0 + index
        for trade_date in dates:
            close *= 1.0 + daily_return
            rows.append(
                {
                    "symbol": f"SH{600000 + index:06d}",
                    "trade_date": trade_date,
                    "hfq_close": close,
                    "is_trading": True,
                }
            )
    return pd.DataFrame(rows)


CONFIG = {
    "lookback_trading_days": 20,
    "minimum_members": 3,
    "weak_return_quantile": 0.20,
    "high_volatility_quantile": 0.80,
}


def test_industry_snapshot_is_idempotent_and_point_in_time(tmp_path) -> None:
    path = tmp_path / "snapshots"
    frame = membership()

    assert write_industry_snapshot(frame, path) == len(frame)
    assert write_industry_snapshot(frame, path) == 0
    loaded = load_industry_snapshot(path, pd.Timestamp("2026-02-02"), max_age_days=7)
    stale = load_industry_snapshot(path, pd.Timestamp("2026-02-20"), max_age_days=7)

    assert len(loaded) == len(frame)
    assert stale.empty


def test_sector_state_uses_trailing_member_returns() -> None:
    bars = sector_bars()
    result = compute_sector_states(
        bars,
        membership(as_of=str(pd.Timestamp(bars["trade_date"].max()).date())),
        pd.Timestamp(bars["trade_date"].max()),
        CONFIG,
    )

    assert result["sector_state"].ne("DATA_UNAVAILABLE").all()
    sector_returns = result.groupby("sector_name")["sector_return_20"].first()
    assert sector_returns["行业0"] < sector_returns["行业3"]
    assert set(result["sector_classification"]) == {"证监会行业分类"}


def test_sector_state_ignores_future_bars() -> None:
    bars = sector_bars()
    as_of = pd.Timestamp(bars["trade_date"].max())
    expected = compute_sector_states(
        bars,
        membership(as_of=str(as_of.date())),
        as_of,
        CONFIG,
    )
    future = bars.iloc[[0]].copy()
    future["trade_date"] = pd.Timestamp("2030-01-01")
    future["hfq_close"] = 1_000_000.0
    actual = compute_sector_states(
        pd.concat([bars, future], ignore_index=True),
        membership(as_of=str(as_of.date())),
        as_of,
        CONFIG,
    )

    pd.testing.assert_frame_equal(actual, expected)

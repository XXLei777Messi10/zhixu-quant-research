from __future__ import annotations

import pandas as pd

from quant.cli import _feature_research_bars


def test_feature_research_bars_excludes_post_membership_execution_tail() -> None:
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    curated = pd.DataFrame(
        {
            "symbol": (
                ["SH000300"] * len(dates)
                + ["SH600000"] * len(dates)
                + ["SH600001"] * len(dates)
            ),
            "trade_date": list(dates) * 3,
            "close": list(range(1, 16)),
        }
    )
    universe = pd.DataFrame(
        {
            "symbol": ["SH600000", "SH600000"],
            "trade_date": [dates[1], dates[2]],
        }
    )
    scoped = _feature_research_bars(curated, universe, "SH000300")
    assert scoped.loc[scoped["symbol"].eq("SH000300"), "trade_date"].tolist() == list(
        dates
    )
    assert scoped.loc[scoped["symbol"].eq("SH600000"), "trade_date"].tolist() == list(
        dates[:3]
    )
    assert "SH600001" not in set(scoped["symbol"])

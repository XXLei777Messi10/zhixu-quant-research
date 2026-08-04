from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant.cli import _historical_sector_states
from quant.config import ProjectPaths
from quant.data.industry import write_industry_snapshot


def test_historical_sector_states_use_latest_past_weekly_mapping(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = ProjectPaths(tmp_path)
    paths.reports.joinpath("research").mkdir(parents=True)
    config = {
        "sector_state": {
            "lookback_trading_days": 20,
            "minimum_members": 3,
            "weak_return_quantile": 0.20,
            "high_volatility_quantile": 0.80,
            "max_mapping_age_days": 7,
        }
    }
    monkeypatch.setattr("quant.cli.load_config", lambda *_args: config)
    friday = pd.Timestamp("2026-01-30")
    monday = pd.Timestamp("2026-02-02")
    memberships = []
    bars = []
    dates = pd.bdate_range(end=monday, periods=30)
    for index in range(12):
        symbol = f"SH{600000 + index:06d}"
        memberships.append(
            {
                "as_of_date": friday,
                "symbol": symbol,
                "stock_name": symbol,
                "industry_name": f"sector-{index // 3}",
                "industry_classification": "test",
                "mapping_update_date": friday,
                "source": "synthetic",
                "retrieved_at": friday.tz_localize("UTC"),
                "request_id": "request-1",
            }
        )
        close = 10.0 + index
        for date in dates:
            close *= 1.0 + (index // 3 - 1.5) * 0.001
            bars.append(
                {
                    "symbol": symbol,
                    "trade_date": date,
                    "hfq_close": close,
                    "is_trading": True,
                }
            )
    write_industry_snapshot(
        pd.DataFrame(memberships),
        paths.normalized / "industry" / "snapshots",
    )
    predictions = pd.DataFrame(
        {
            "symbol": [item["symbol"] for item in memberships],
            "trade_date": monday,
        }
    )

    result = _historical_sector_states(
        paths,
        pd.DataFrame(bars),
        predictions,
        {monday},
    )

    assert result["sector_mapping_as_of"].eq(friday).all()
    assert result["sector_state"].ne("DATA_UNAVAILABLE").all()
    assert result["trade_date"].eq(monday).all()

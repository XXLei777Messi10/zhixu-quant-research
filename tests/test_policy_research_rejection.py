from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from quant.config import ProjectPaths
from quant.research import policy


def test_development_rejection_is_archived_without_reading_locked_period(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = ProjectPaths(tmp_path)
    paths.reports.joinpath("research").mkdir(parents=True)
    paths.curated.mkdir(parents=True)
    predictions = pd.DataFrame(
        {
            "symbol": ["SH600000", "SH600000"],
            "trade_date": pd.to_datetime(["2023-12-29", "2024-01-02"]),
            "model_score": [0.9, 0.8],
        }
    )
    predictions.to_parquet(
        paths.reports / "research" / "rolling_predictions.parquet",
        index=False,
    )
    bars = pd.DataFrame(
        {
            "symbol": ["SH000300", "SH600000"],
            "trade_date": pd.to_datetime(["2023-12-29", "2023-12-29"]),
            "open": [10.0, 10.0],
            "high": [10.1, 10.1],
            "low": [9.9, 9.9],
            "close": [10.0, 10.0],
            "volume": [1_000_000, 1_000_000],
            "is_trading": [True, True],
            "is_st": [False, False],
            "adjust_factor": [1.0, 1.0],
            "hfq_open": [10.0, 10.0],
            "hfq_high": [10.1, 10.1],
            "hfq_low": [9.9, 9.9],
            "hfq_close": [10.0, 10.0],
        }
    )
    bars.to_parquet(paths.curated / "bars.parquet", index=False)
    research_config = {
        "development_start": "2023-01-01",
        "development_end": "2023-12-31",
        "locked_start": "2024-01-01",
        "selection": {
            "maximum_drawdown_floor": -0.25,
            "primary_metric": "annualized_return",
            "tie_breakers": [],
        },
        "locked_acceptance": {},
        "candidates": [
            {
                "name": "rejected",
                "exit_rank": 20,
                "initial_entry_fraction": 1.0,
                "add_fraction": 0.25,
                "risk_weight_scales": None,
            }
        ],
    }

    def fake_config(_paths, name):
        if name == "policy_research":
            return research_config
        if name == "data":
            return {"benchmark_symbol": "SH000300"}
        return {
            "daily_open_simulation": {"initial_cash": 1_000_000},
            "portfolio_policy": {"candidate_count": 20},
        }

    rejected_metrics = {
        "annualized_return": 0.1,
        "sharpe_ratio": 0.5,
        "calmar_ratio": 0.2,
        "max_drawdown": -0.30,
    }
    monkeypatch.setattr(policy, "load_config", fake_config)
    monkeypatch.setattr(
        policy,
        "_run_candidate",
        lambda *_args, **_kwargs: (SimpleNamespace(), rejected_metrics),
    )

    result = policy.run_policy_research(paths)

    assert result["status"] == "DEVELOPMENT_REJECTED"
    assert result["locked_period_read"] is False
    assert result["selected_candidate"] is None
    archived = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    assert archived["development_results"]["rejected"]["max_drawdown"] == -0.30

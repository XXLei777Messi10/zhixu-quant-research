from __future__ import annotations

import pandas as pd
import pytest

from quant.research.model_components import (
    build_forward_labels,
    candidate_metrics,
    weighted_score,
)


def test_build_forward_labels_uses_next_to_sixth_open() -> None:
    dates = pd.date_range("2024-01-02", periods=8, freq="B")
    bars = pd.concat(
        [
            pd.DataFrame(
                {
                    "symbol": "SH000300",
                    "trade_date": dates,
                    "hfq_open": [100.0] * 8,
                }
            ),
            pd.DataFrame(
                {
                    "symbol": "SH600000",
                    "trade_date": dates,
                    "hfq_open": [10.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                }
            ),
        ],
        ignore_index=True,
    )
    labels = build_forward_labels(bars, "SH000300")
    first = labels.loc[labels["trade_date"].eq(dates[0])].iloc[0]
    assert first["label_regression"] == pytest.approx(0.5)
    assert first["label_classification"] == 1


def test_weighted_score_validates_and_combines() -> None:
    frame = pd.DataFrame({"a": [0.2, 0.8], "b": [0.6, 0.4]})
    score = weighted_score(frame, {"a": 0.75, "b": 0.25})
    assert score.tolist() == pytest.approx([0.3, 0.7])
    with pytest.raises(ValueError, match="total 1.0"):
        weighted_score(frame, {"a": 0.5, "b": 0.4})


def test_candidate_metrics_rewards_correct_cross_section() -> None:
    rows = []
    for date in pd.date_range("2024-01-02", periods=3, freq="B"):
        for index in range(40):
            rows.append(
                {
                    "trade_date": date,
                    "symbol": f"SH{index:06d}",
                    "signal_rank": index / 39,
                    "label_regression": index / 1000,
                }
            )
    metrics = candidate_metrics(
        pd.DataFrame(rows),
        {"signal_rank": 1.0},
        top_k=20,
    )
    assert metrics["mean_daily_rank_ic"] == pytest.approx(1.0)
    assert metrics["top_bottom_spread"] > 0
    assert metrics["positive_years"] == 1


def test_candidate_metrics_handles_unmatured_trailing_labels() -> None:
    rows = []
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    for date in dates:
        for index in range(40):
            rows.append(
                {
                    "trade_date": date,
                    "symbol": f"SH{index:06d}",
                    "signal_rank": index / 39,
                    "label_regression": (
                        index / 1000 if date != dates[-1] else float("nan")
                    ),
                }
            )
    metrics = candidate_metrics(
        pd.DataFrame(rows),
        {"signal_rank": 1.0},
        top_k=20,
    )
    assert metrics["dates"] == 2
    assert metrics["mean_daily_rank_ic"] == pytest.approx(1.0)

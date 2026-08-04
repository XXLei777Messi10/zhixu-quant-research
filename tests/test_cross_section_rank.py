from __future__ import annotations

import pandas as pd
import pytest

from quant.config import ProjectPaths
from quant.research.cross_section_rank import (
    _known_suspensions,
    _research_sample_dates,
    apply_walk_forward_score_calibration,
    candidate_score,
    candidate_score_components,
    fit_walk_forward_score_calibration,
    nested_year_splits,
    neutralize_score,
    relevance_grades,
    select_fold_candidate,
    select_inner_candidate,
)


def test_known_suspensions_reads_each_symbol_partition(tmp_path) -> None:
    paths = ProjectPaths(tmp_path)
    root = (
        paths.normalized
        / "bars"
        / "source=baostock"
        / "adjustment=raw"
    )
    for symbol, trading in (("SH600001", False), ("SZ000002", True)):
        partition = root / f"symbol={symbol}"
        partition.mkdir(parents=True)
        pd.DataFrame(
            {
                "symbol": [symbol],
                "trade_date": [pd.Timestamp("2025-04-22")],
                "is_trading": [trading],
            }
        ).to_parquet(partition / "bars.parquet", index=False)
    assert _known_suspensions(
        paths,
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-12-31"),
    ) == {("SH600001", pd.Timestamp("2025-04-22").date())}


def test_weekly_research_samples_use_last_trading_day_of_week() -> None:
    calendar = pd.bdate_range("2025-01-06", "2025-01-17")
    samples = _research_sample_dates(calendar, "weekly_last_trading_day")
    assert samples == {
        pd.Timestamp("2025-01-10"),
    }


def test_nested_year_split_has_double_embargo_and_outer_is_untouched() -> None:
    calendar = pd.bdate_range("2010-01-01", "2026-07-27")
    splits = nested_year_splits(
        calendar,
        pd.Timestamp("2016-01-01"),
        pd.Timestamp("2026-07-27"),
        selection_months=12,
        early_stopping_months=12,
        embargo_days=21,
        minimum_training_years=3,
    )
    assert splits
    assert splits[0].year == 2016
    for split in splits:
        train_end = calendar.get_loc(split.train_end)
        early_start = calendar.get_loc(split.early_stop_start)
        early_end = calendar.get_loc(split.early_stop_end)
        selection_start = calendar.get_loc(split.selection_start)
        selection_end = calendar.get_loc(split.selection_end)
        test_start = calendar.get_loc(split.test_start)
        assert early_start - train_end > 21
        assert selection_start - early_end > 21
        assert test_start - selection_end > 21


def test_relevance_grades_follow_cross_sectional_return_order() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2025-01-02")] * 10,
            "label": list(range(10)),
        }
    )
    grades = relevance_grades(frame, "label", 5, 10)
    assert grades.min() == 0
    assert grades.max() == 4
    assert grades.is_monotonic_increasing


def test_sector_liquidity_neutralization_removes_group_means() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2025-01-02")] * 20,
            "sector_name": ["A"] * 10 + ["B"] * 10,
            "amount": list(range(1, 21)),
        }
    )
    raw = pd.Series([10.0] * 10 + [20.0] * 10)
    residual = neutralize_score(frame, raw)
    sector_mean = residual.groupby(frame["sector_name"]).mean()
    assert sector_mean.abs().max() < 1e-9


def test_candidate_blend_is_daily_rank_and_weights_are_enforced() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2025-01-02")] * 4,
            "sector_name": ["A", "A", "B", "B"],
            "amount": [1.0, 2.0, 3.0, 4.0],
            "ranker_raw_5d": [1.0, 2.0, 3.0, 4.0],
            "ranker_raw_10d": [4.0, 3.0, 2.0, 1.0],
        }
    )
    candidate = {
        "horizon_weights": {5: 0.75, 10: 0.25},
        "neutralization": "global",
    }
    score = candidate_score(frame, candidate, [5, 10])
    assert score.between(0.0, 1.0).all()
    assert score.iloc[-1] > score.iloc[0]


def test_candidate_components_preserve_consensus_and_agreement() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-05"] * 3),
            "sector_name": ["A", "A", "B"],
            "amount": [1.0, 2.0, 3.0],
            "ranker_raw_5d": [1.0, 2.0, 3.0],
            "ranker_raw_10d": [1.0, 2.0, 3.0],
        }
    )
    candidate = {
        "horizon_weights": {5: 0.5, 10: 0.5},
        "neutralization": "global",
    }
    components = candidate_score_components(frame, candidate, [5, 10])
    assert components["model_score"].is_monotonic_increasing
    assert components["consensus_score"].is_monotonic_increasing
    assert components["horizon_agreement"].eq(1.0).all()


def test_walk_forward_calibration_is_monotone_and_reports_uncertainty() -> None:
    rows = []
    for day in pd.date_range("2022-01-07", periods=20, freq="7D"):
        for index in range(10):
            score = (index + 0.5) / 10
            rows.append(
                {
                    "trade_date": day,
                    "label": score * 0.04 - 0.02,
                    "score": score,
                }
            )
    frame = pd.DataFrame(rows)
    table = fit_walk_forward_score_calibration(
        frame,
        frame["score"],
        "label",
        5,
    )
    expected = [row["expected_excess_return"] for row in table]
    assert expected == sorted(expected)
    assert all(row["date_count"] == 20 for row in table)

    applied = apply_walk_forward_score_calibration(
        pd.Series([0.1, 0.9]),
        table,
        uncertainty_multiplier=1.0,
    )
    assert (
        applied["calibrated_lower_bound"]
        <= applied["calibrated_excess_return"]
    ).all()
    assert (
        applied["calibrated_excess_return"].iloc[1]
        > applied["calibrated_excess_return"].iloc[0]
    )


def test_inner_selection_never_uses_outer_metrics() -> None:
    candidates = [{"name": "a"}, {"name": "b"}]
    metrics = {
        "a": {
            "mean_daily_rank_ic": 0.02,
            "top_k_mean_excess_return": 0.001,
            "positive_rank_ic_fraction": 0.52,
            "rank_ic_information_ratio": 0.5,
        },
        "b": {
            "mean_daily_rank_ic": 0.01,
            "top_k_mean_excess_return": 0.002,
            "positive_rank_ic_fraction": 0.51,
            "rank_ic_information_ratio": 0.4,
        },
    }
    rules = {
        "minimum_mean_daily_rank_ic": 0.0,
        "minimum_top_k_mean_excess_return": 0.0,
        "minimum_positive_rank_ic_fraction": 0.48,
        "primary_metric": "mean_daily_rank_ic",
        "tie_breakers": ["top_k_mean_excess_return", "rank_ic_information_ratio"],
    }
    selected, fallback = select_inner_candidate(metrics, candidates, rules)
    assert selected == "a"
    assert not fallback


def test_fixed_fold_candidate_is_stable_and_validated() -> None:
    candidates = [{"name": "a"}, {"name": "balanced"}]
    selected, fallback = select_fold_candidate(
        {},
        candidates,
        {},
        mode="fixed",
        fixed_candidate_name="balanced",
    )
    assert selected == "balanced"
    assert not fallback

    with pytest.raises(ValueError, match="absent from the candidate budget"):
        select_fold_candidate(
            {},
            candidates,
            {},
            mode="fixed",
            fixed_candidate_name="missing",
        )

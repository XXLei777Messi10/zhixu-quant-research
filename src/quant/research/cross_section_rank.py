from __future__ import annotations

import gc
import hashlib
import math
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from quant.config import ProjectPaths, load_config
from quant.qlib_workflow.features import FEATURE_COLUMNS
from quant.research.model_components import build_forward_labels, candidate_metrics
from quant.research.rank_buffer_strategy import (
    RankBufferSimulator,
    _period_metrics,
    _weekly_signal_dates,
)
from quant.signals.archive import immutable_write_json


@dataclass(frozen=True)
class NestedYearSplit:
    year: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    early_stop_start: pd.Timestamp
    early_stop_end: pd.Timestamp
    selection_start: pd.Timestamp
    selection_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _known_suspensions(
    paths: ProjectPaths,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> set[tuple[str, date]]:
    root = paths.normalized / "bars" / "source=baostock" / "adjustment=raw"
    if not root.exists():
        return set()
    suspensions: set[tuple[str, date]] = set()
    for path in sorted(root.glob("symbol=*/bars.parquet")):
        status = pd.read_parquet(
            path,
            columns=["symbol", "trade_date", "is_trading"],
            filters=[
                ("trade_date", ">=", start),
                ("trade_date", "<=", end),
            ],
        )
        if status.empty:
            continue
        status["trade_date"] = pd.to_datetime(status["trade_date"]).dt.date
        for row in status[~status["is_trading"].fillna(False)].itertuples():
            suspensions.add((str(row.symbol), row.trade_date))
    return suspensions


def _research_sample_dates(
    calendar: pd.DatetimeIndex,
    frequency: str,
) -> set[pd.Timestamp]:
    if frequency == "daily":
        return {pd.Timestamp(value) for value in calendar}
    if frequency == "weekly_last_trading_day":
        return _weekly_signal_dates(calendar)
    raise ValueError(f"Unsupported research sample frequency: {frequency}")


def _previous_trading_day(
    calendar: pd.DatetimeIndex,
    value: pd.Timestamp,
    full_gap: int,
) -> pd.Timestamp:
    position = calendar.searchsorted(pd.Timestamp(value), side="left") - full_gap - 1
    if position < 0:
        raise ValueError("Not enough history for nested embargo")
    return pd.Timestamp(calendar[position])


def nested_year_splits(
    calendar: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    selection_months: int,
    early_stopping_months: int,
    embargo_days: int,
    minimum_training_years: int,
) -> list[NestedYearSplit]:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(calendar).normalize().unique()))
    output: list[NestedYearSplit] = []
    for year in range(int(start.year), int(end.year) + 1):
        test = dates[
            (dates >= max(start, pd.Timestamp(year=year, month=1, day=1)))
            & (dates <= min(end, pd.Timestamp(year=year, month=12, day=31)))
        ]
        if test.empty:
            continue
        test_start, test_end = pd.Timestamp(test.min()), pd.Timestamp(test.max())
        try:
            selection_end = _previous_trading_day(dates, test_start, embargo_days)
            selection_floor = selection_end - pd.DateOffset(months=selection_months)
            selection_dates = dates[
                (dates >= selection_floor) & (dates <= selection_end)
            ]
            selection_start = pd.Timestamp(selection_dates.min())
            early_stop_end = _previous_trading_day(
                dates,
                selection_start,
                embargo_days,
            )
            early_stop_floor = early_stop_end - pd.DateOffset(
                months=early_stopping_months
            )
            early_stop_dates = dates[
                (dates >= early_stop_floor) & (dates <= early_stop_end)
            ]
            early_stop_start = pd.Timestamp(early_stop_dates.min())
            train_end = _previous_trading_day(
                dates,
                early_stop_start,
                embargo_days,
            )
        except (ValueError, TypeError):
            continue
        train_dates = dates[dates <= train_end]
        if train_dates.empty:
            continue
        train_start = pd.Timestamp(train_dates.min())
        if train_end < train_start + pd.DateOffset(years=minimum_training_years):
            continue
        output.append(
            NestedYearSplit(
                year,
                train_start,
                train_end,
                early_stop_start,
                early_stop_end,
                selection_start,
                selection_end,
                test_start,
                test_end,
            )
        )
    return output


def relevance_grades(
    frame: pd.DataFrame,
    label_column: str,
    grade_count: int,
    minimum_cross_section: int,
) -> pd.Series:
    if grade_count < 2:
        raise ValueError("At least two relevance grades are required")
    labels = pd.to_numeric(frame[label_column], errors="coerce")
    counts = labels.notna().groupby(frame["trade_date"]).transform("sum")
    percentile = labels.groupby(frame["trade_date"]).rank(
        method="first",
        pct=True,
    )
    grades = np.floor(percentile * grade_count).clip(0, grade_count - 1)
    return grades.where(counts >= minimum_cross_section).astype("Int8")


def _ranking_dataset(
    frame: pd.DataFrame,
    features: list[str],
    label_column: str,
    grade_count: int,
    minimum_cross_section: int,
    reference: lgb.Dataset | None = None,
) -> tuple[lgb.Dataset, pd.DataFrame]:
    selected = frame[
        ["symbol", "trade_date", label_column, *features]
    ].copy()
    selected["relevance"] = relevance_grades(
        selected,
        label_column,
        grade_count,
        minimum_cross_section,
    )
    selected = (
        selected.dropna(subset=["relevance"])
        .sort_values(["trade_date", "symbol"])
        .reset_index(drop=True)
    )
    if selected.empty:
        raise ValueError("Ranking dataset has no eligible cross-sections")
    groups = selected.groupby("trade_date", sort=False).size().to_numpy()
    dataset = lgb.Dataset(
        selected[features].astype("float32"),
        label=selected["relevance"].astype(int),
        group=groups,
        feature_name=features,
        reference=reference,
        free_raw_data=False,
    )
    return dataset, selected


def _ranker_parameters(config: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
    params = dict(config)
    rounds = int(params.pop("num_boost_round"))
    early_stopping = int(params.pop("early_stopping_rounds"))
    eval_at = [int(value) for value in params.pop("eval_at")]
    seed = int(params.pop("seed", 42))
    params.update(
        {
            "ndcg_eval_at": eval_at,
            "verbosity": -1,
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "deterministic": True,
            "force_col_wise": True,
        }
    )
    return params, rounds, early_stopping


def _fit_horizon_ranker(
    frame: pd.DataFrame,
    split: NestedYearSplit,
    features: list[str],
    label_column: str,
    research: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, float]]:
    train = frame[frame["trade_date"].between(split.train_start, split.train_end)]
    early_stop = frame[
        frame["trade_date"].between(split.early_stop_start, split.early_stop_end)
    ]
    selection = frame[
        frame["trade_date"].between(split.selection_start, split.selection_end)
    ].sort_values(["trade_date", "symbol"])
    test = frame[
        frame["trade_date"].between(split.test_start, split.test_end)
    ].sort_values(["trade_date", "symbol"])
    train_set, train_rows = _ranking_dataset(
        train,
        features,
        label_column,
        int(research["relevance_grades"]),
        int(research["minimum_cross_section"]),
    )
    valid_set, valid_rows = _ranking_dataset(
        early_stop,
        features,
        label_column,
        int(research["relevance_grades"]),
        int(research["minimum_cross_section"]),
        train_set,
    )
    params, rounds, early_stopping = _ranker_parameters(
        {**research["ranker"], "seed": int(research["seed"])}
    )
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=rounds,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
    )
    best_iteration = max(1, int(booster.best_iteration or rounds))
    selection_score = booster.predict(
        selection[features].astype("float32"),
        num_iteration=best_iteration,
    )
    refit = frame[
        frame["trade_date"].between(split.train_start, split.selection_end)
    ]
    refit_set, refit_rows = _ranking_dataset(
        refit,
        features,
        label_column,
        int(research["relevance_grades"]),
        int(research["minimum_cross_section"]),
    )
    final_booster = lgb.train(
        params,
        refit_set,
        num_boost_round=best_iteration,
    )
    test_score = final_booster.predict(
        test[features].astype("float32"),
        num_iteration=best_iteration,
    )
    importance = {
        feature: float(value)
        for feature, value in zip(
            features,
            final_booster.feature_importance(importance_type="gain"),
            strict=True,
        )
    }
    manifest = {
        "best_iteration": best_iteration,
        "train_rows": len(train_rows),
        "early_stop_rows": len(valid_rows),
        "refit_rows": len(refit_rows),
        "selection_rows": len(selection),
        "test_rows": len(test),
    }
    return selection_score, test_score, manifest, importance


def _liquidity_bucket(frame: pd.DataFrame, buckets: int = 5) -> pd.Series:
    rank = pd.to_numeric(frame["amount"], errors="coerce").groupby(
        frame["trade_date"]
    ).rank(method="first", pct=True)
    return np.floor(rank * buckets).clip(0, buckets - 1).fillna(-1).astype(int)


def neutralize_score(frame: pd.DataFrame, raw_score: pd.Series) -> pd.Series:
    residual = pd.to_numeric(raw_score, errors="coerce").astype(float)
    sector = frame["sector_name"].fillna("UNKNOWN").astype(str)
    liquidity = _liquidity_bucket(frame)
    for _ in range(3):
        residual = residual - residual.groupby(
            [frame["trade_date"], sector]
        ).transform("mean")
        residual = residual - residual.groupby(
            [frame["trade_date"], liquidity]
        ).transform("mean")
    return residual


def candidate_score(
    frame: pd.DataFrame,
    candidate: dict[str, Any],
    horizons: list[int],
) -> pd.Series:
    return candidate_score_components(frame, candidate, horizons)[
        "model_score"
    ]


def candidate_score_components(
    frame: pd.DataFrame,
    candidate: dict[str, Any],
    horizons: list[int],
) -> pd.DataFrame:
    weights = {
        int(key): float(value)
        for key, value in candidate["horizon_weights"].items()
    }
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Horizon weights must total one")
    horizon_ranks: dict[int, pd.Series] = {}
    for horizon in horizons:
        raw = pd.to_numeric(frame[f"ranker_raw_{horizon}d"], errors="coerce")
        if str(candidate["neutralization"]) == "sector_liquidity":
            raw = neutralize_score(frame, raw)
        elif str(candidate["neutralization"]) != "global":
            raise ValueError("Unsupported score neutralization")
        horizon_ranks[horizon] = raw.groupby(frame["trade_date"]).rank(pct=True)
    consensus = pd.Series(0.0, index=frame.index, dtype="float64")
    for horizon, ranked in horizon_ranks.items():
        consensus = consensus + float(weights.get(horizon, 0.0)) * ranked
    variance = pd.Series(0.0, index=frame.index, dtype="float64")
    for horizon, ranked in horizon_ranks.items():
        variance = variance + float(weights.get(horizon, 0.0)) * (
            ranked - consensus
        ).pow(2)
    agreement = 1.0 - np.sqrt(variance).clip(0.0, 0.5) / 0.5
    return pd.DataFrame(
        {
            "model_score": consensus.groupby(frame["trade_date"]).rank(
                pct=True
            ),
            "consensus_score": consensus,
            "horizon_agreement": agreement.clip(0.0, 1.0),
        },
        index=frame.index,
    )


def fit_walk_forward_score_calibration(
    frame: pd.DataFrame,
    score: pd.Series,
    label_column: str,
    bin_count: int,
) -> list[dict[str, Any]]:
    if bin_count < 3:
        raise ValueError("Score calibration requires at least three bins")
    calibration = frame[["trade_date", label_column]].copy()
    calibration["score"] = pd.to_numeric(score, errors="coerce")
    calibration["label"] = pd.to_numeric(
        calibration[label_column],
        errors="coerce",
    )
    calibration = calibration.dropna(subset=["score", "label"])
    if calibration.empty:
        raise ValueError("Score calibration has no eligible observations")
    calibration["score_bin"] = np.floor(
        calibration["score"].clip(0.0, 1.0 - 1e-12) * bin_count
    ).astype(int)
    calibration["outperformed"] = calibration["label"].gt(0).astype(float)
    daily = (
        calibration.groupby(["trade_date", "score_bin"], as_index=False)
        .agg(
            mean_excess=("label", "mean"),
            outperform_rate=("outperformed", "mean"),
        )
    )
    grouped = daily.groupby("score_bin").agg(
        mean_excess=("mean_excess", "mean"),
        excess_std=("mean_excess", "std"),
        outperform_probability=("outperform_rate", "mean"),
        date_count=("trade_date", "nunique"),
    )
    global_daily = calibration.groupby("trade_date")["label"].mean()
    global_stderr = float(global_daily.std(ddof=1) / np.sqrt(len(global_daily)))
    if not np.isfinite(global_stderr) or global_stderr <= 0:
        global_stderr = float(calibration["label"].std(ddof=1))
    observed = grouped.dropna(subset=["mean_excess"])
    if len(observed) < 2:
        raise ValueError("Score calibration needs at least two populated bins")
    centers = (observed.index.to_numpy(dtype=float) + 0.5) / bin_count
    isotonic = IsotonicRegression(
        increasing=True,
        out_of_bounds="clip",
    )
    isotonic.fit(
        centers,
        observed["mean_excess"].to_numpy(dtype=float),
        sample_weight=observed["date_count"].to_numpy(dtype=float),
    )
    output: list[dict[str, Any]] = []
    for score_bin in range(bin_count):
        center = (score_bin + 0.5) / bin_count
        row = grouped.loc[score_bin] if score_bin in grouped.index else None
        date_count = int(row["date_count"]) if row is not None else 0
        standard_error = (
            float(row["excess_std"]) / math.sqrt(date_count)
            if row is not None
            and date_count > 1
            and np.isfinite(row["excess_std"])
            else global_stderr
        )
        probability = (
            float(row["outperform_probability"])
            if row is not None
            and np.isfinite(row["outperform_probability"])
            else 0.5
        )
        output.append(
            {
                "score_bin": score_bin,
                "score_low": score_bin / bin_count,
                "score_high": (score_bin + 1) / bin_count,
                "expected_excess_return": float(isotonic.predict([center])[0]),
                "standard_error": float(max(0.0, standard_error)),
                "outperform_probability": float(
                    np.clip(probability, 0.0, 1.0)
                ),
                "date_count": date_count,
            }
        )
    return output


def apply_walk_forward_score_calibration(
    score: pd.Series,
    calibration: list[dict[str, Any]],
    uncertainty_multiplier: float,
) -> pd.DataFrame:
    if not calibration:
        raise ValueError("Score calibration table is empty")
    bin_count = len(calibration)
    score_bin = np.floor(
        pd.to_numeric(score, errors="coerce").clip(
            0.0,
            1.0 - 1e-12,
        )
        * bin_count
    ).astype("Int64")
    table = {
        int(row["score_bin"]): row
        for row in calibration
    }
    expected = score_bin.map(
        {
            key: float(value["expected_excess_return"])
            for key, value in table.items()
        }
    )
    standard_error = score_bin.map(
        {
            key: float(value["standard_error"])
            for key, value in table.items()
        }
    )
    probability = score_bin.map(
        {
            key: float(value["outperform_probability"])
            for key, value in table.items()
        }
    )
    date_count = score_bin.map(
        {
            key: int(value["date_count"])
            for key, value in table.items()
        }
    )
    return pd.DataFrame(
        {
            "calibration_bin": score_bin,
            "calibrated_excess_return": expected,
            "calibration_standard_error": standard_error,
            "calibrated_lower_bound": (
                expected - float(uncertainty_multiplier) * standard_error
            ),
            "calibrated_outperform_probability": probability,
            "calibration_date_count": date_count,
        },
        index=score.index,
    )


def _score_metrics(
    frame: pd.DataFrame,
    score: pd.Series,
    label_column: str,
    top_k: int,
) -> dict[str, Any]:
    scored = frame[["symbol", "trade_date", label_column]].copy()
    scored["model_score"] = score
    scored = scored.rename(columns={label_column: "label_regression"})
    return candidate_metrics(scored, {"model_score": 1.0}, top_k)


def select_inner_candidate(
    metrics: dict[str, dict[str, Any]],
    candidates: list[dict[str, Any]],
    rules: dict[str, Any],
) -> tuple[str, bool]:
    eligible = [
        str(candidate["name"])
        for candidate in candidates
        if float(metrics[str(candidate["name"])]["mean_daily_rank_ic"])
        >= float(rules["minimum_mean_daily_rank_ic"])
        and float(metrics[str(candidate["name"])]["top_k_mean_excess_return"])
        >= float(rules["minimum_top_k_mean_excess_return"])
        and float(metrics[str(candidate["name"])]["positive_rank_ic_fraction"])
        >= float(rules["minimum_positive_rank_ic_fraction"])
    ]
    fallback = not eligible
    names = eligible or [str(candidate["name"]) for candidate in candidates]
    keys = [
        str(rules["primary_metric"]),
        *[str(value) for value in rules["tie_breakers"]],
    ]
    selected = max(
        names,
        key=lambda name: tuple(float(metrics[name][key]) for key in keys),
    )
    return selected, fallback


def select_fold_candidate(
    metrics: dict[str, dict[str, Any]],
    candidates: list[dict[str, Any]],
    rules: dict[str, Any],
    mode: str = "nested",
    fixed_candidate_name: str | None = None,
) -> tuple[str, bool]:
    if mode == "nested":
        return select_inner_candidate(metrics, candidates, rules)
    if mode != "fixed":
        raise ValueError(f"Unsupported candidate selection mode: {mode}")
    names = {str(candidate["name"]) for candidate in candidates}
    selected = str(fixed_candidate_name or "")
    if selected not in names:
        raise ValueError(
            f"Fixed candidate {selected!r} is absent from the candidate budget"
        )
    return selected, False


def _period_score_metrics(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    top_k: int,
) -> dict[str, Any]:
    merged = predictions[
        predictions["trade_date"].between(start, end)
    ].merge(
        labels[["symbol", "trade_date", "label_regression"]],
        on=["symbol", "trade_date"],
        how="inner",
        validate="one_to_one",
    )
    return candidate_metrics(merged, {"model_score": 1.0}, top_k)


def _yearly_positive_ic(metrics: dict[str, Any]) -> int:
    return sum(
        int(float(values["mean_daily_rank_ic"]) > 0)
        for values in metrics["annual"].values()
    )


def run_cross_section_rank_research(
    paths: ProjectPaths,
    config_name: str = "rank_research",
) -> dict[str, Any]:
    research = load_config(paths, config_name)
    base_config_name = research.get("base_config")
    if base_config_name:
        inherited = load_config(paths, str(base_config_name))
        inherited.update(research)
        research = inherited
    config_path = paths.configs / f"{config_name}.yaml"
    base_config_path = (
        paths.configs / f"{base_config_name}.yaml"
        if base_config_name
        else None
    )
    execution = load_config(paths, "execution")
    features_path = paths.curated / str(research["features_file"])
    bars_path = paths.curated / str(research["bars_file"])
    research_root = paths.reports / "research"
    metadata_path = research_root / str(research["sector_metadata_file"])
    baseline_path = research_root / str(research["baseline_prediction_file"])
    prediction_path = research_root / str(research["prediction_file"])
    artifact_root = research_root / str(research["artifact_directory"])
    report_target = research_root / str(research["report_file"])
    if prediction_path.exists() or artifact_root.exists() or report_target.exists():
        raise FileExistsError("Rank research outputs already exist; use a revision")

    start = pd.Timestamp(research["outer_start"])
    research_start = pd.Timestamp(research["research_start"])
    end = pd.Timestamp(research["outer_end"])
    development_end = pd.Timestamp(research["development_end"])
    stress_start = pd.Timestamp(research["known_stress_start"])
    horizons = [int(value) for value in research["horizons"]]
    features = pd.read_parquet(features_path)
    features["trade_date"] = pd.to_datetime(features["trade_date"]).dt.normalize()
    features["symbol"] = features["symbol"].astype(str)
    features = features[
        features["in_universe"].fillna(False)
        & features["trade_date"].between(
            pd.Timestamp(research["history_start"]),
            end,
        )
    ].copy()
    metadata = pd.read_parquet(
        metadata_path,
        columns=["symbol", "trade_date", "market_state", "sector_name", "sector_state"],
        filters=[("trade_date", ">=", pd.Timestamp(research["history_start"])), ("trade_date", "<=", end)],
    )
    metadata["trade_date"] = pd.to_datetime(metadata["trade_date"]).dt.normalize()
    metadata["symbol"] = metadata["symbol"].astype(str)
    metadata = metadata.drop_duplicates(["symbol", "trade_date"])
    features = features.merge(
        metadata,
        on=["symbol", "trade_date"],
        how="left",
        validate="one_to_one",
    )
    features["market_state"] = features["market_state"].fillna("DATA_UNAVAILABLE")
    features["sector_name"] = features["sector_name"].fillna("UNKNOWN")
    features["sector_state"] = features["sector_state"].fillna("DATA_UNAVAILABLE")
    bars = pd.read_parquet(
        bars_path,
        columns=[
            "symbol",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "is_trading",
            "is_st",
            "adjust_factor",
            "hfq_open",
            "hfq_close",
        ],
        filters=[
            ("trade_date", ">=", pd.Timestamp(research["history_start"]) - pd.Timedelta(days=200)),
            ("trade_date", "<=", end),
        ],
    )
    bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.normalize()
    bars["symbol"] = bars["symbol"].astype(str)
    labels: dict[int, pd.DataFrame] = {}
    for horizon in horizons:
        label = build_forward_labels(
            bars[["symbol", "trade_date", "hfq_open"]],
            str(research["benchmark_symbol"]),
            horizon=horizon,
            entry_offset=1,
        )
        label_column = f"label_{horizon}d"
        labels[horizon] = label.rename(
            columns={"label_regression": label_column}
        )[["symbol", "trade_date", label_column]]
        features = features.merge(
            labels[horizon],
            on=["symbol", "trade_date"],
            how="left",
            validate="one_to_one",
        )

    calendar = pd.DatetimeIndex(sorted(features["trade_date"].unique()))
    sample_frequency = str(research.get("sample_frequency", "daily"))
    sample_dates = _research_sample_dates(calendar, sample_frequency)
    features = features[features["trade_date"].isin(sample_dates)].copy()
    feature_columns = [column for column in FEATURE_COLUMNS if column in features]
    splits = nested_year_splits(
        calendar,
        start,
        end,
        int(research["inner_selection_months"]),
        int(research["early_stopping_months"]),
        int(research["embargo_trading_days"]),
        int(research["minimum_training_years"]),
    )
    if not splits:
        raise RuntimeError("No eligible nested annual folds")

    all_predictions: list[pd.DataFrame] = []
    fold_reports: list[dict[str, Any]] = []
    importance_total = pd.Series(0.0, index=feature_columns)
    for split in splits:
        selection = features[
            features["trade_date"].between(split.selection_start, split.selection_end)
        ].sort_values(["trade_date", "symbol"]).copy()
        test = features[
            features["trade_date"].between(split.test_start, split.test_end)
        ].sort_values(["trade_date", "symbol"]).copy()
        horizon_manifests: dict[str, Any] = {}
        for horizon in horizons:
            selection_score, test_score, manifest, importance = _fit_horizon_ranker(
                features,
                split,
                feature_columns,
                f"label_{horizon}d",
                research,
            )
            selection[f"ranker_raw_{horizon}d"] = selection_score
            test[f"ranker_raw_{horizon}d"] = test_score
            horizon_manifests[str(horizon)] = manifest
            importance_total = importance_total.add(
                pd.Series(importance),
                fill_value=0.0,
            )
            gc.collect()
        inner_metrics: dict[str, dict[str, Any]] = {}
        candidate_scores: dict[str, pd.Series] = {}
        for candidate in research["candidates"]:
            name = str(candidate["name"])
            score = candidate_score(selection, candidate, horizons)
            candidate_scores[name] = score
            inner_metrics[name] = _score_metrics(
                selection,
                score,
                f"label_{int(research['selection_label_horizon'])}d",
                int(research["top_k"]),
            )
        selection_mode = str(
            research.get("candidate_selection_mode", "nested")
        )
        selected_name, fallback = select_fold_candidate(
            inner_metrics,
            research["candidates"],
            research["inner_selection"],
            mode=selection_mode,
            fixed_candidate_name=research.get("fixed_candidate_name"),
        )
        selected_candidate = next(
            candidate
            for candidate in research["candidates"]
            if str(candidate["name"]) == selected_name
        )
        selection_components = candidate_score_components(
            selection,
            selected_candidate,
            horizons,
        )
        test_components = candidate_score_components(
            test,
            selected_candidate,
            horizons,
        )
        test_score = test_components["model_score"]
        mapping = research["score_mapping"]
        test_output = test[
            [
                "symbol",
                "trade_date",
                "market_state",
                "sector_name",
                "sector_state",
            ]
        ].copy()
        test_output["model_score"] = test_score
        test_output["consensus_score"] = test_components["consensus_score"]
        test_output["horizon_agreement"] = test_components[
            "horizon_agreement"
        ]
        calibration_report: list[dict[str, Any]] | None = None
        calibration_config = research.get("score_calibration", {})
        if bool(calibration_config.get("enabled", False)):
            calibration_report = fit_walk_forward_score_calibration(
                selection,
                selection_components["consensus_score"],
                f"label_{int(research['selection_label_horizon'])}d",
                int(calibration_config["bin_count"]),
            )
            calibrated = apply_walk_forward_score_calibration(
                test_components["consensus_score"],
                calibration_report,
                float(calibration_config["uncertainty_multiplier"]),
            )
            for column in calibrated:
                test_output[column] = calibrated[column]
            test_output["predicted_excess_return"] = test_output[
                "calibrated_excess_return"
            ]
            test_output["outperform_probability"] = test_output[
                "calibrated_outperform_probability"
            ]
        else:
            test_output["predicted_excess_return"] = (
                test_score - 0.5
            ) * float(mapping["excess_return_scale"])
            test_output["outperform_probability"] = (
                float(mapping["probability_floor"])
                + test_score
                * (
                    float(mapping["probability_ceiling"])
                    - float(mapping["probability_floor"])
                )
            )
        test_output["fold_year"] = split.year
        test_output["selected_candidate"] = selected_name
        all_predictions.append(test_output)
        fold_reports.append(
            {
                "split": {
                    key: (
                        value.date().isoformat()
                        if isinstance(value, pd.Timestamp)
                        else value
                    )
                    for key, value in asdict(split).items()
                },
                "selected_candidate": selected_name,
                "selection_fallback": fallback,
                "inner_metrics": inner_metrics,
                "score_calibration": calibration_report,
                "horizon_models": horizon_manifests,
            }
        )
        del selection, test, candidate_scores
        gc.collect()

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_parquet(prediction_path, index=False, compression="zstd")
    selection_horizon = int(research["selection_label_horizon"])
    selection_labels = labels[selection_horizon].rename(
        columns={f"label_{selection_horizon}d": "label_regression"}
    )
    development_score = _period_score_metrics(
        predictions,
        selection_labels,
        start,
        development_end,
        int(research["top_k"]),
    )
    stress_score = _period_score_metrics(
        predictions,
        selection_labels,
        stress_start,
        end,
        int(research["top_k"]),
    )

    simulator = RankBufferSimulator(
        research,
        execution,
        research["portfolio"],
    )
    suspensions = _known_suspensions(
        paths,
        pd.Timestamp(research["history_start"]),
        end,
    )
    rank_result = simulator.run(
        bars,
        predictions,
        simulation_start=start,
        known_suspensions=suspensions,
    )
    baseline = pd.read_parquet(
        baseline_path,
        filters=[("trade_date", ">=", start), ("trade_date", "<=", end)],
    )
    baseline["trade_date"] = pd.to_datetime(baseline["trade_date"]).dt.normalize()
    baseline["symbol"] = baseline["symbol"].astype(str)
    baseline_result = simulator.run(
        bars,
        baseline,
        simulation_start=start,
        known_suspensions=suspensions,
    )
    benchmark = bars[
        bars["symbol"].eq(str(research["benchmark_symbol"]))
    ][["trade_date", "close"]]
    portfolio_metrics = {
        "rank_development": _period_metrics(
            rank_result,
            benchmark,
            start,
            development_end,
        ),
        "baseline_development": _period_metrics(
            baseline_result,
            benchmark,
            start,
            development_end,
        ),
        "rank_known_stress": _period_metrics(
            rank_result,
            benchmark,
            stress_start,
            end,
        ),
        "baseline_known_stress": _period_metrics(
            baseline_result,
            benchmark,
            stress_start,
            end,
        ),
    }
    acceptance = research["research_acceptance"]
    checks = {
        "development_mean_rank_ic": float(development_score["mean_daily_rank_ic"])
        >= float(acceptance["minimum_development_mean_rank_ic"]),
        "development_positive_ic_years": _yearly_positive_ic(development_score)
        >= int(acceptance["minimum_development_positive_ic_years"]),
        "development_excess": float(
            portfolio_metrics["rank_development"]["excess_cumulative_return"]
        )
        >= float(acceptance["minimum_development_excess_cumulative_return"]),
        "development_drawdown": float(
            portfolio_metrics["rank_development"]["max_drawdown"]
        )
        >= float(acceptance["maximum_development_drawdown_floor"]),
        "stress_mean_rank_ic": float(stress_score["mean_daily_rank_ic"])
        >= float(acceptance["minimum_stress_mean_rank_ic"]),
        "stress_excess": float(
            portfolio_metrics["rank_known_stress"]["excess_cumulative_return"]
        )
        >= float(acceptance["minimum_stress_excess_cumulative_return"]),
        "stress_drawdown": float(
            portfolio_metrics["rank_known_stress"]["max_drawdown"]
        )
        >= float(acceptance["maximum_stress_drawdown_floor"]),
        "turnover": float(
            portfolio_metrics["rank_known_stress"]["annual_turnover_gross"]
        )
        <= float(acceptance["maximum_annual_turnover_gross"]),
        "development_data_availability": int(
            portfolio_metrics["rank_development"]["maximum_missing_data_positions"]
        )
        <= int(acceptance["maximum_missing_data_positions"]),
        "stress_data_availability": int(
            portfolio_metrics["rank_known_stress"]["maximum_missing_data_positions"]
        )
        <= int(acceptance["maximum_missing_data_positions"]),
    }
    artifact_root.mkdir(parents=True)
    rank_result.nav.to_parquet(artifact_root / "rank_nav.parquet", index=False)
    rank_result.holdings.to_parquet(artifact_root / "rank_holdings.parquet", index=False)
    rank_result.trades.to_parquet(artifact_root / "rank_trades.parquet", index=False)
    rank_result.orders.to_parquet(artifact_root / "rank_orders.parquet", index=False)
    baseline_result.nav.to_parquet(artifact_root / "baseline_nav.parquet", index=False)
    report = {
        "status": (
            "CROSS_SECTION_RANK_RESEARCH_ACCEPTED"
            if all(checks.values())
            else "CROSS_SECTION_RANK_RESEARCH_REJECTED"
        ),
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": str(research["run_id"]),
        "protocol": {
            "research_period": [str(research_start.date()), str(end.date())],
            "outer_period": [str(start.date()), str(end.date())],
            "initial_nested_warmup": [
                str(research_start.date()),
                str((start - pd.Timedelta(days=1)).date()),
            ],
            "development_period": [str(start.date()), str(development_end.date())],
            "known_stress_period": [str(stress_start.date()), str(end.date())],
            "horizons": horizons,
            "embargo_trading_days": int(research["embargo_trading_days"]),
            "candidate_selection_mode": str(
                research.get("candidate_selection_mode", "nested")
            ),
            "fixed_candidate_name": research.get("fixed_candidate_name"),
            "inner_candidate_selection_only": str(
                research.get("candidate_selection_mode", "nested")
            )
            == "nested",
            "outer_year_never_selects_its_own_candidate": True,
            "2024_2026_is_not_pristine_holdout": True,
            "training_history_equal_weight": True,
            "sample_frequency": sample_frequency,
            "score_calibration": research.get("score_calibration"),
        },
        "candidate_budget": research["candidates"],
        "corporate_actions": research.get("corporate_actions", []),
        "terminal_events": research.get("terminal_events", []),
        "known_suspension_records": len(suspensions),
        "folds": fold_reports,
        "selection_frequency": predictions["selected_candidate"].value_counts().to_dict(),
        "score_metrics": {
            "development": development_score,
            "known_stress": stress_score,
        },
        "portfolio_metrics": portfolio_metrics,
        "feature_importance": {
            str(key): float(value)
            for key, value in importance_total.sort_values(ascending=False).items()
        },
        "acceptance": {
            "passed": all(checks.values()),
            "checks": checks,
            "rules": acceptance,
        },
        "input_versions": {
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "base_config": (
                None if base_config_path is None else str(base_config_path)
            ),
            "base_config_sha256": (
                None
                if base_config_path is None
                else _sha256(base_config_path)
            ),
            "features": str(features_path),
            "features_sha256": _sha256(features_path),
            "bars": str(bars_path),
            "bars_sha256": _sha256(bars_path),
            "metadata": str(metadata_path),
            "metadata_sha256": _sha256(metadata_path),
            "baseline": str(baseline_path),
            "baseline_sha256": _sha256(baseline_path),
        },
        "predictions": str(prediction_path),
        "predictions_sha256": _sha256(prediction_path),
        "simulation_only": True,
        "no_broker_connection": True,
    }
    report_path = immutable_write_json(report_target, report)
    return {**report, "report": str(report_path)}

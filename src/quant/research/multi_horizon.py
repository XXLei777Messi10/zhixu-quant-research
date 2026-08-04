from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.config import ProjectPaths, load_config
from quant.models.splits import quarterly_splits
from quant.models.training import fit_models, predict_models
from quant.research.model_components import (
    build_forward_labels,
    candidate_metrics,
)
from quant.signals.archive import immutable_write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _suffix_predictions(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    keep = [
        "symbol",
        "trade_date",
        "market_state",
        "sector_name",
        "sector_state",
        "sector_return_20",
        "sector_volatility_20",
        "sector_member_count",
        "sector_mapping_as_of",
        "sector_classification",
    ]
    rename = {
        column: f"{column}_{horizon}d"
        for column in frame.columns
        if column not in keep
    }
    return frame.rename(columns=rename)


def _train_horizon(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    model_config: dict[str, Any],
    horizon: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    regression_label = f"label_regression_{horizon}d"
    classification_label = f"label_classification_{horizon}d"
    frame = features.drop(
        columns=["label_regression", "label_classification"],
        errors="ignore",
    ).merge(
        labels.rename(
            columns={
                "label_regression": regression_label,
                "label_classification": classification_label,
            }
        ),
        on=["symbol", "trade_date"],
        how="left",
        validate="one_to_one",
    )
    horizon_config = copy.deepcopy(model_config)
    horizon_config["label_horizon"] = horizon
    horizon_config["label_exit_offset"] = (
        int(horizon_config.get("label_entry_offset", 1)) + horizon
    )
    horizon_config["label_columns"] = {
        "regression": regression_label,
        "classification": classification_label,
    }
    # This experiment intentionally keeps all historical rows equally weighted.
    horizon_config.pop("training_window_years", None)
    horizon_config.pop("training_sample_weight", None)
    calendar = pd.DatetimeIndex(sorted(frame["trade_date"].unique()))
    embargo = max(int(model_config["embargo_trading_days"]), horizon)
    splits = [
        split
        for split in quarterly_splits(
            calendar,
            model_config["first_oos"],
            int(model_config["validation_months"]),
            embargo,
            int(model_config["minimum_training_years"]),
        )
        if split.test_start >= start and split.test_end <= end
    ]
    if not splits:
        raise RuntimeError(f"No development folds for {horizon}-day horizon")
    predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    for index, split in enumerate(splits, start=1):
        models, metrics, _ = fit_models(frame, split, horizon_config)
        test = frame[frame["trade_date"].between(split.test_start, split.test_end)]
        prediction = predict_models(models, test)
        prediction["fold"] = index
        predictions.append(_suffix_predictions(prediction, horizon))
        fold_metrics.append(
            {
                "horizon": horizon,
                "fold": index,
                "test_start": split.test_start.date().isoformat(),
                "test_end": split.test_end.date().isoformat(),
                "embargo_trading_days": embargo,
                **metrics,
            }
        )
        del models, test, prediction
        gc.collect()
    return pd.concat(predictions, ignore_index=True), fold_metrics


def _adaptive_scores(
    predictions: pd.DataFrame,
    horizons: list[int],
    round_trip_cost: float,
) -> pd.DataFrame:
    frame = predictions.copy()
    utility_columns: list[str] = []
    for horizon in horizons:
        prediction = pd.to_numeric(
            frame[f"predicted_excess_return_{horizon}d"],
            errors="coerce",
        )
        probability = pd.to_numeric(
            frame[f"outperform_probability_{horizon}d"],
            errors="coerce",
        ).clip(0.0, 1.0)
        confidence = (2.0 * (probability - 0.5)).clip(0.0, 1.0)
        column = f"close_utility_{horizon}d"
        frame[column] = (
            (prediction - round_trip_cost)
            * (1.0 + confidence)
            / float(horizon)
        )
        utility_columns.append(column)
    utility = frame[utility_columns].to_numpy(dtype=float)
    valid = np.isfinite(utility)
    safe = np.where(valid, utility, -np.inf)
    selected_index = safe.argmax(axis=1)
    no_valid = ~valid.any(axis=1)
    horizon_values = np.asarray(horizons, dtype=int)[selected_index]
    selected_utility = safe[np.arange(len(frame)), selected_index]
    horizon_values = horizon_values.astype("float64")
    horizon_values[no_valid] = np.nan
    selected_utility[no_valid] = np.nan
    frame["selected_horizon_close"] = horizon_values
    frame["adaptive_close_utility"] = selected_utility
    frame["adaptive_utility_rank"] = frame.groupby("trade_date")[
        "adaptive_close_utility"
    ].rank(pct=True)
    selected_probability = np.full(len(frame), np.nan)
    for horizon in horizons:
        mask = frame["selected_horizon_close"].eq(horizon)
        selected_probability[mask] = frame.loc[
            mask,
            f"outperform_probability_{horizon}d",
        ]
    frame["adaptive_probability"] = selected_probability
    frame["adaptive_probability_rank"] = frame.groupby("trade_date")[
        "adaptive_probability"
    ].rank(pct=True)
    frame["adaptive_model_score"] = (
        0.5 * frame["adaptive_utility_rank"]
        + 0.5 * frame["adaptive_probability_rank"]
    )
    return frame


def _adaptive_labels(
    frame: pd.DataFrame,
    labels: dict[int, pd.DataFrame],
    horizons: list[int],
) -> pd.DataFrame:
    output = frame.copy()
    for horizon in horizons:
        output = output.merge(
            labels[horizon][["symbol", "trade_date", "label_regression"]].rename(
                columns={"label_regression": f"label_regression_{horizon}d"}
            ),
            on=["symbol", "trade_date"],
            how="left",
            validate="one_to_one",
        )
    output["label_regression"] = np.nan
    for horizon in horizons:
        mask = output["selected_horizon_close"].eq(horizon)
        output.loc[mask, "label_regression"] = output.loc[
            mask,
            f"label_regression_{horizon}d",
        ]
    return output


def run_multi_horizon_research(
    paths: ProjectPaths,
    config_name: str = "multi_horizon_research",
) -> dict[str, Any]:
    research = load_config(paths, config_name)
    model = load_config(paths, "model")
    data = load_config(paths, "data")
    horizons = [int(value) for value in research["horizons"]]
    reused = {int(value) for value in research["reuse_existing_horizons"]}
    start = pd.Timestamp(research["development_start"])
    end = pd.Timestamp(research["development_end"])
    features_path = paths.curated / "features.parquet"
    bars_path = paths.curated / "bars.parquet"
    baseline_path = paths.reports / "research" / str(
        research["baseline_prediction_file"]
    )
    output_path = paths.reports / "research" / str(
        research["candidate_prediction_file"]
    )
    if output_path.exists():
        raise FileExistsError(f"{output_path} already exists; use a revision name")

    features = pd.read_parquet(features_path)
    features["trade_date"] = pd.to_datetime(features["trade_date"]).dt.normalize()
    features = features[features["in_universe"].fillna(False)].copy()
    bars = pd.read_parquet(
        bars_path,
        columns=["symbol", "trade_date", "hfq_open"],
    )
    labels = {
        horizon: build_forward_labels(
            bars,
            str(data["benchmark_symbol"]),
            horizon=horizon,
            entry_offset=int(model["label_entry_offset"]),
        )
        for horizon in horizons
    }

    combined: pd.DataFrame | None = None
    fold_metrics: list[dict[str, Any]] = []
    baseline = pd.read_parquet(
        baseline_path,
        filters=[("trade_date", ">=", start), ("trade_date", "<=", end)],
    )
    baseline["trade_date"] = pd.to_datetime(baseline["trade_date"]).dt.normalize()
    for horizon in horizons:
        if horizon in reused:
            horizon_predictions = _suffix_predictions(baseline, horizon)
        else:
            horizon_predictions, metrics = _train_horizon(
                features,
                labels[horizon],
                model,
                horizon,
                start,
                end,
            )
            fold_metrics.extend(metrics)
        keys = ["symbol", "trade_date"]
        combined = (
            horizon_predictions
            if combined is None
            else combined.merge(
                horizon_predictions,
                on=keys,
                how="inner",
                validate="one_to_one",
            )
        )
    if combined is None or combined.empty:
        raise RuntimeError("No multi-horizon predictions were created")
    combined = _adaptive_scores(
        combined,
        horizons,
        float(research["estimated_round_trip_rate"]),
    )
    combined.to_parquet(output_path, index=False, compression="zstd")

    adaptive_with_labels = _adaptive_labels(combined, labels, horizons)
    adaptive_metrics = candidate_metrics(
        adaptive_with_labels.rename(columns={"adaptive_model_score": "model_score"}),
        {"model_score": 1.0},
        int(research["top_k"]),
    )
    baseline_with_labels = baseline.merge(
        labels[5][["symbol", "trade_date", "label_regression"]],
        on=["symbol", "trade_date"],
        how="inner",
        validate="one_to_one",
    )
    baseline_metrics = candidate_metrics(
        baseline_with_labels,
        {"model_score": 1.0},
        int(research["top_k"]),
    )
    acceptance = research["acceptance"]
    checks = {
        "rank_ic_improvement": (
            adaptive_metrics["mean_daily_rank_ic"]
            - baseline_metrics["mean_daily_rank_ic"]
            >= float(acceptance["minimum_rank_ic_improvement"])
        ),
        "top_k_return_improvement": (
            adaptive_metrics["top_k_mean_excess_return"]
            - baseline_metrics["top_k_mean_excess_return"]
            >= float(acceptance["minimum_top_k_return_improvement"])
        ),
        "positive_years": adaptive_metrics["positive_years"]
        >= int(acceptance["minimum_positive_years"]),
    }
    accepted = all(checks.values())
    report = {
        "status": (
            "MULTI_HORIZON_DEVELOPMENT_ACCEPTED"
            if accepted
            else "MULTI_HORIZON_DEVELOPMENT_REJECTED"
        ),
        "created_at": datetime.now(UTC).isoformat(),
        "development_period": [start.date().isoformat(), end.date().isoformat()],
        "horizons": horizons,
        "reused_existing_horizons": sorted(reused),
        "training_history": "expanding_full_history_equal_weight",
        "horizon_specific_embargo": True,
        "baseline_5d": baseline_metrics,
        "adaptive_multi_horizon": adaptive_metrics,
        "fold_metrics": fold_metrics,
        "acceptance": {
            "passed": accepted,
            "checks": checks,
            "rules": acceptance,
        },
        "locked_period_read": False,
        "candidate_predictions": str(output_path),
        "candidate_predictions_sha256": _sha256(output_path),
        "input_versions": {
            "features": str(features_path),
            "features_sha256": _sha256(features_path),
            "bars": str(bars_path),
            "bars_sha256": _sha256(bars_path),
            "baseline_predictions": str(baseline_path),
            "baseline_predictions_sha256": _sha256(baseline_path),
        },
        "simulation_only": True,
    }
    report_path = immutable_write_json(
        paths.reports / "research" / str(research["report_file"]),
        report,
    )
    return {**report, "report": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", default="multi_horizon_research")
    args = parser.parse_args()
    result = run_multi_horizon_research(
        ProjectPaths(args.root.resolve()),
        str(args.config),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

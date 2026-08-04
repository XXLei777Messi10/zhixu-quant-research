from __future__ import annotations

import argparse
import gc
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant.config import ProjectPaths, load_config
from quant.models.splits import quarterly_splits
from quant.models.training import fit_models, predict_models
from quant.research.model_components import candidate_metrics
from quant.signals.archive import immutable_write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    top_k: int,
) -> dict[str, Any]:
    merged = predictions.merge(
        labels[["symbol", "trade_date", "label_regression"]],
        on=["symbol", "trade_date"],
        how="inner",
        validate="one_to_one",
    )
    return candidate_metrics(merged, {"model_score": 1.0}, top_k)


def run_time_decay_research(
    paths: ProjectPaths,
    config_name: str = "model_time_decay_research",
) -> dict[str, Any]:
    research_config = load_config(paths, config_name)
    model_config = load_config(paths, "model")
    weighting = dict(research_config["training_sample_weight"])
    model_config["training_sample_weight"] = weighting
    start = pd.Timestamp(research_config["development_start"])
    end = pd.Timestamp(research_config["development_end"])
    feature_path = paths.curated / "features.parquet"
    frame = pd.read_parquet(feature_path)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    frame = frame[frame["in_universe"].fillna(False)].copy()
    calendar = pd.DatetimeIndex(sorted(frame["trade_date"].unique()))
    splits = [
        split
        for split in quarterly_splits(
            calendar,
            model_config["first_oos"],
            int(model_config["validation_months"]),
            int(model_config["embargo_trading_days"]),
            int(model_config["minimum_training_years"]),
        )
        if split.test_start >= start and split.test_end <= end
    ]
    if not splits:
        raise RuntimeError("No development folds matched the time-decay period")

    predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    for index, split in enumerate(splits, start=1):
        models, metrics, _ = fit_models(frame, split, model_config)
        test = frame[frame["trade_date"].between(split.test_start, split.test_end)]
        fold = predict_models(models, test)
        fold["fold"] = index
        predictions.append(fold)
        fold_metrics.append(
            {
                "fold": index,
                "test_start": split.test_start.date().isoformat(),
                "test_end": split.test_end.date().isoformat(),
                **metrics,
            }
        )
        del models, test, fold
        gc.collect()
    candidate = pd.concat(predictions, ignore_index=True)
    candidate_path = (
        paths.reports
        / "research"
        / str(research_config["candidate_prediction_file"])
    )
    if candidate_path.exists():
        raise FileExistsError(
            f"{candidate_path} already exists; use a revision file name"
        )
    candidate.to_parquet(candidate_path, index=False, compression="zstd")

    labels = frame[frame["trade_date"].between(start, end)][
        ["symbol", "trade_date", "label_regression"]
    ].copy()
    baseline_path = (
        paths.reports
        / "research"
        / str(research_config["baseline_prediction_file"])
    )
    baseline = pd.read_parquet(
        baseline_path,
        filters=[("trade_date", ">=", start), ("trade_date", "<=", end)],
    )
    top_k = int(research_config["top_k"])
    baseline_metrics = _metrics(baseline, labels, top_k)
    candidate_metrics_value = _metrics(candidate, labels, top_k)
    acceptance_config = research_config["acceptance"]
    checks = {
        "rank_ic_improvement": (
            candidate_metrics_value["mean_daily_rank_ic"]
            - baseline_metrics["mean_daily_rank_ic"]
            >= float(acceptance_config["minimum_rank_ic_improvement"])
        ),
        "top_k_return_improvement": (
            candidate_metrics_value["top_k_mean_excess_return"]
            - baseline_metrics["top_k_mean_excess_return"]
            >= float(acceptance_config["minimum_top_k_return_improvement"])
        ),
        "positive_years": candidate_metrics_value["positive_years"]
        >= int(acceptance_config["minimum_positive_years"]),
        "negative_years": candidate_metrics_value["negative_years"]
        <= int(acceptance_config["maximum_negative_years"]),
    }
    accepted = all(checks.values())
    report = {
        "status": (
            "TIME_DECAY_DEVELOPMENT_ACCEPTED"
            if accepted
            else "TIME_DECAY_DEVELOPMENT_REJECTED"
        ),
        "created_at": datetime.now(UTC).isoformat(),
        "development_period": [start.date().isoformat(), end.date().isoformat()],
        "training_sample_weight": weighting,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics_value,
        "fold_metrics": fold_metrics,
        "acceptance": {
            "passed": accepted,
            "checks": checks,
            "rules": acceptance_config,
        },
        "locked_period_read": False,
        "input_versions": {
            "features": str(feature_path),
            "features_sha256": _sha256(feature_path),
            "baseline_predictions": str(baseline_path),
            "baseline_predictions_sha256": _sha256(baseline_path),
        },
        "candidate_predictions": str(candidate_path),
        "candidate_predictions_sha256": _sha256(candidate_path),
        "simulation_only": True,
    }
    report_path = immutable_write_json(
        paths.reports / "research" / "model-time-decay-development.json",
        report,
    )
    return {**report, "report": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", default="model_time_decay_research")
    args = parser.parse_args()
    report = run_time_decay_research(
        ProjectPaths(args.root.resolve()),
        config_name=str(args.config),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

from __future__ import annotations

import gc
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.config import ProjectPaths, load_config
from quant.research.cross_section_rank import _known_suspensions
from quant.research.model_components import build_forward_labels, candidate_metrics
from quant.research.rank_buffer_strategy import RankBufferSimulator, _period_metrics
from quant.signals.archive import immutable_write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def topk_and_calibration_diagnostics(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    topk_values: list[int],
) -> dict[str, Any]:
    scored = predictions[
        pd.to_datetime(predictions["trade_date"]).between(start, end)
    ].merge(
        labels[["symbol", "trade_date", "label_regression"]],
        on=["symbol", "trade_date"],
        how="inner",
        validate="one_to_one",
    )
    topk = {
        str(value): candidate_metrics(
            scored,
            {"model_score": 1.0},
            int(value),
        )
        for value in topk_values
    }
    percentile = scored.groupby("trade_date")["model_score"].rank(
        pct=True,
        method="average",
    )
    scored["score_decile"] = np.floor(
        percentile.clip(0.0, 1.0 - 1e-12) * 10
    ).astype(int)
    deciles = (
        scored.groupby("score_decile")
        .agg(
            mean_excess_return=("label_regression", "mean"),
            hit_rate=("label_regression", lambda value: float(value.gt(0).mean())),
            observations=("label_regression", "size"),
            dates=("trade_date", "nunique"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    calibration: list[dict[str, Any]] = []
    if {
        "calibration_bin",
        "calibrated_excess_return",
        "calibrated_outperform_probability",
    }.issubset(scored.columns):
        calibration = (
            scored.groupby("calibration_bin")
            .agg(
                predicted_excess=("calibrated_excess_return", "mean"),
                realized_excess=("label_regression", "mean"),
                predicted_probability=(
                    "calibrated_outperform_probability",
                    "mean",
                ),
                realized_hit_rate=(
                    "label_regression",
                    lambda value: float(value.gt(0).mean()),
                ),
                observations=("label_regression", "size"),
                dates=("trade_date", "nunique"),
            )
            .reset_index()
            .to_dict(orient="records")
        )
    return {
        "rows": len(scored),
        "dates": int(scored["trade_date"].nunique()),
        "topk": topk,
        "score_deciles": deciles,
        "calibration_bins": calibration,
    }


def _eligible(
    design: dict[str, Any],
    validation: dict[str, Any],
    rules: dict[str, Any],
) -> bool:
    for metrics in (design, validation):
        if float(metrics["excess_cumulative_return"]) < float(
            rules["minimum_excess_cumulative_return"]
        ):
            return False
        if float(metrics["max_drawdown"]) < float(
            rules["maximum_drawdown_floor"]
        ):
            return False
        if float(metrics["annual_turnover_gross"]) > float(
            rules["maximum_annual_turnover_gross"]
        ):
            return False
        if int(metrics["maximum_missing_data_positions"]) > 0:
            return False
    return True


def _selection_key(record: dict[str, Any]) -> tuple[float, float, float, float]:
    design = record["design"]
    validation = record["validation"]
    return (
        min(
            float(design["sharpe_ratio"]),
            float(validation["sharpe_ratio"]),
        ),
        min(
            float(design["excess_cumulative_return"]),
            float(validation["excess_cumulative_return"]),
        ),
        -max(
            abs(float(design["max_drawdown"])),
            abs(float(validation["max_drawdown"])),
        ),
        (
            float(design["annualized_return"])
            + float(validation["annualized_return"])
        )
        / 2.0,
    )


def _load_bars(
    paths: ProjectPaths,
    history_start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    frame = pd.read_parquet(
        paths.curated / "bars.parquet",
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
            (
                "trade_date",
                ">=",
                history_start - pd.Timedelta(days=200),
            ),
            ("trade_date", "<=", end),
        ],
    )
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str)
    return frame


def _load_inherited_config(
    paths: ProjectPaths,
    config_name: str,
    seen: set[str] | None = None,
) -> dict[str, Any]:
    visited = set() if seen is None else set(seen)
    if config_name in visited:
        raise ValueError(f"Config inheritance cycle at {config_name!r}")
    visited.add(config_name)
    current = load_config(paths, config_name)
    base_name = current.get("base_config")
    if not base_name:
        return current
    return {
        **_load_inherited_config(paths, str(base_name), visited),
        **current,
    }


def run_portfolio_baseline(
    paths: ProjectPaths,
    config_name: str = "portfolio_baseline",
) -> dict[str, Any]:
    research = _load_inherited_config(paths, config_name)
    execution = load_config(paths, "execution")
    report_target = paths.reports / "research" / str(research["report_file"])
    artifact_root = paths.reports / "research" / str(
        research["artifact_directory"]
    )
    if report_target.exists() or artifact_root.exists():
        raise FileExistsError("Portfolio baseline outputs already exist; use a revision")

    design_start, design_end = map(pd.Timestamp, research["design_period"])
    validation_start, validation_end = map(
        pd.Timestamp,
        research["validation_period"],
    )
    stress_start, stress_end = map(
        pd.Timestamp,
        research["known_stress_period"],
    )
    history_start = pd.Timestamp(research["history_start"])
    research_root = paths.reports / "research"
    prediction_paths = {
        "baseline": research_root / str(research["baseline_prediction_file"]),
        "calibrated": research_root / str(research["calibrated_prediction_file"]),
    }
    variants = {
        name: pd.read_parquet(path)
        for name, path in prediction_paths.items()
    }
    for frame in variants.values():
        frame["trade_date"] = pd.to_datetime(
            frame["trade_date"]
        ).dt.normalize()
        frame["symbol"] = frame["symbol"].astype(str)

    development_bars = _load_bars(paths, history_start, validation_end)
    benchmark_development = development_bars[
        development_bars["symbol"].eq(str(research["benchmark_symbol"]))
    ][["trade_date", "close"]]
    development_suspensions = _known_suspensions(
        paths,
        history_start,
        validation_end,
    )
    records: list[dict[str, Any]] = []
    for candidate in research["portfolio_candidates"]:
        result = RankBufferSimulator(
            research,
            execution,
            candidate,
        ).run(
            development_bars,
            variants[str(candidate["model_variant"])],
            simulation_start=design_start,
            known_suspensions=development_suspensions,
        )
        design = _period_metrics(
            result,
            benchmark_development,
            design_start,
            design_end,
        )
        validation = _period_metrics(
            result,
            benchmark_development,
            validation_start,
            validation_end,
        )
        record = {
            "name": str(candidate["name"]),
            "model_variant": str(candidate["model_variant"]),
            "design": design,
            "validation": validation,
            "eligible": _eligible(
                design,
                validation,
                research["portfolio_selection"],
            ),
        }
        record["selection_key"] = _selection_key(record)
        records.append(record)
        del result
        gc.collect()

    eligible = [record for record in records if record["eligible"]]
    selected_record = max(
        eligible if eligible else records,
        key=_selection_key,
    )
    selected = next(
        candidate
        for candidate in research["portfolio_candidates"]
        if str(candidate["name"]) == str(selected_record["name"])
    )

    full_bars = _load_bars(paths, history_start, stress_end)
    full_suspensions = _known_suspensions(paths, history_start, stress_end)
    selected_result = RankBufferSimulator(
        research,
        execution,
        selected,
    ).run(
        full_bars,
        variants[str(selected["model_variant"])],
        simulation_start=design_start,
        known_suspensions=full_suspensions,
    )
    benchmark_full = full_bars[
        full_bars["symbol"].eq(str(research["benchmark_symbol"]))
    ][["trade_date", "close"]]
    selected_stress = _period_metrics(
        selected_result,
        benchmark_full,
        stress_start,
        stress_end,
    )

    labels = build_forward_labels(
        full_bars[["symbol", "trade_date", "hfq_open"]],
        str(research["benchmark_symbol"]),
        horizon=int(research["diagnostic_horizon"]),
        entry_offset=1,
    )
    calibrated_diagnostics = {
        "design": topk_and_calibration_diagnostics(
            variants["calibrated"],
            labels,
            design_start,
            design_end,
            [int(value) for value in research["diagnostic_topk"]],
        ),
        "validation": topk_and_calibration_diagnostics(
            variants["calibrated"],
            labels,
            validation_start,
            validation_end,
            [int(value) for value in research["diagnostic_topk"]],
        ),
        "known_stress": topk_and_calibration_diagnostics(
            variants["calibrated"],
            labels,
            stress_start,
            stress_end,
            [int(value) for value in research["diagnostic_topk"]],
        ),
    }

    artifact_root.mkdir(parents=True)
    selected_result.nav.to_parquet(artifact_root / "nav.parquet", index=False)
    selected_result.holdings.to_parquet(
        artifact_root / "holdings.parquet",
        index=False,
    )
    selected_result.trades.to_parquet(
        artifact_root / "trades.parquet",
        index=False,
    )
    selected_result.orders.to_parquet(
        artifact_root / "orders.parquet",
        index=False,
    )
    report = {
        "status": (
            "PORTFOLIO_BASELINE_RESEARCH_COMPLETE"
            if eligible
            else "PORTFOLIO_BASELINE_NO_ELIGIBLE_DEVELOPMENT_CANDIDATE"
        ),
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": str(research["run_id"]),
        "protocol": {
            "design_period": [str(design_start.date()), str(design_end.date())],
            "validation_period": [
                str(validation_start.date()),
                str(validation_end.date()),
            ],
            "known_stress_period": [
                str(stress_start.date()),
                str(stress_end.date()),
            ],
            "stress_not_used_for_selection": True,
            "2024_2026_is_not_pristine_holdout": True,
            "dynamic_breadth_uses_prior_fold_calibration_only": True,
            "ranking_and_portfolio_selection_are_separate": True,
        },
        "candidate_budget": research["portfolio_candidates"],
        "development_results": records,
        "selected_candidate": selected,
        "selected_development_record": selected_record,
        "selected_known_stress": selected_stress,
        "ranking_diagnostics": calibrated_diagnostics,
        "known_suspension_records": len(full_suspensions),
        "terminal_events": research.get("terminal_events", []),
        "input_versions": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for name, path in prediction_paths.items()
        },
        "simulation_only": True,
        "no_broker_connection": True,
    }
    immutable_write_json(report_target, report)
    return {**report, "report": str(report_target)}

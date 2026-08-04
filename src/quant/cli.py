from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import shutil
import sys
import time
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant.backtest.engine import AShareSimulator
from quant.backtest.metrics import performance_metrics, signal_metrics
from quant.config import ProjectPaths, load_config
from quant.data.akshare_adapter import AKShareAdapter
from quant.data.baostock_adapter import BaostockAdapter
from quant.data.industry import (
    load_industry_snapshot,
    synthetic_industry_snapshot,
    write_industry_snapshot,
)
from quant.data.raw_archive import RawArchive
from quant.data.store import ParquetStore
from quant.data.universe import dynamic_universe, write_constituents
from quant.data.validate import (
    DataValidator,
    combine_reports,
    write_quality_report,
    write_quarantine_records,
)
from quant.execution.archive import archive_execution_plans
from quant.execution.backtest import ProductionMirrorSimulator
from quant.execution.calibration import apply_calibration, calibrate_from_oos_signals
from quant.execution.daily import settle_daily_accounts
from quant.execution.journal import AppendOnlyJournal
from quant.execution.manual import parse_manual_trade, settle_manual_account
from quant.execution.monitor import finalize_auction_day, monitor_auction_session
from quant.execution.planner import (
    build_execution_plans,
    price_contexts_from_bars,
    price_range_fields,
)
from quant.execution.policy import execution_signal, policy_snapshot, portfolio_policy
from quant.execution.replay import replay_execution_day
from quant.models.splits import current_training_split, quarterly_splits
from quant.models.training import (
    fit_models,
    load_current_models,
    predict_models,
    save_model_bundle,
)
from quant.ops.verify_daily import verify_daily_run
from quant.qlib_workflow.dataset import build_qlib_dataset, smoke_test_qlib
from quant.qlib_workflow.features import FEATURE_COLUMNS, add_labels, build_features
from quant.reports.daily_strategy import generate_daily_strategy_summary
from quant.reports.generate import generate_research_report
from quant.research.cross_section_rank import run_cross_section_rank_research
from quant.research.multi_horizon import run_multi_horizon_research
from quant.research.portfolio_baseline import run_portfolio_baseline
from quant.research.rank_buffer_strategy import run_rank_buffer_research
from quant.signals.archive import archive_signals, immutable_write_json
from quant.signals.sector import compute_sector_states
from quant.state import ProcessLock, RunLedger
from quant.synthetic import synthetic_normalized

LOGGER = logging.getLogger("quant")

CURATED_RESEARCH_COLUMNS = [
    "symbol",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "hfq_open",
    "hfq_high",
    "hfq_low",
    "hfq_close",
    "volume",
    "amount",
    "turnover",
    "adjust_factor",
    "is_trading",
    "is_st",
]
CURATED_FLOAT_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "hfq_open",
    "hfq_high",
    "hfq_low",
    "hfq_close",
    "volume",
    "amount",
    "turnover",
    "adjust_factor",
]


def _read_compact_curated(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=CURATED_RESEARCH_COLUMNS)
    frame["symbol"] = frame["symbol"].astype("category")
    frame[CURATED_FLOAT_COLUMNS] = frame[CURATED_FLOAT_COLUMNS].astype("float32")
    return frame


def _feature_research_bars(
    curated: pd.DataFrame,
    universe: pd.DataFrame,
    benchmark_symbol: str,
) -> pd.DataFrame:
    """Keep execution tails out of the feature/QLib working set."""
    membership = universe[["symbol", "trade_date"]].copy()
    membership["trade_date"] = pd.to_datetime(
        membership["trade_date"]
    ).dt.normalize()
    last_membership = (
        membership.groupby("symbol", observed=True)["trade_date"]
        .max()
        .rename("_last_membership")
    )
    scoped = curated.merge(
        last_membership,
        left_on="symbol",
        right_index=True,
        how="left",
        validate="many_to_one",
    )
    dates = pd.to_datetime(scoped["trade_date"]).dt.normalize()
    keep = scoped["symbol"].astype(str).eq(benchmark_symbol) | (
        scoped["_last_membership"].notna()
        & dates.le(scoped["_last_membership"])
    )
    return scoped.loc[keep].drop(columns="_last_membership").reset_index(drop=True)


def _configure_logging(paths: ProjectPaths, level: str = "INFO") -> None:
    paths.logs.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(paths.logs / "quant.log", encoding="utf-8"),
        ],
        force=True,
    )


def _date(value: str) -> date:
    return date.fromisoformat(value)


def _latest_immutable_revision(directory: Path, stem: str, suffix: str) -> Path | None:
    candidates = list(directory.glob(f"{stem}*{suffix}"))

    def revision(path: Path) -> int:
        if path.stem == stem:
            return 1
        marker = f"{stem}__r"
        if path.stem.startswith(marker):
            try:
                return int(path.stem[len(marker) :])
            except ValueError:
                return 0
        return 0

    return max(candidates, key=revision) if candidates else None


def _symbols(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def _check_resources(paths: ProjectPaths, config: dict[str, Any]) -> None:
    usage = shutil.disk_usage(paths.root)
    storage = config["storage"]
    if usage.free < int(storage["min_free_bytes"]) or usage.free / usage.total < float(
        storage["min_free_ratio"]
    ):
        raise RuntimeError(
            f"Disk safety gate failed: {usage.free} bytes free ({usage.free / usage.total:.1%})"
        )


def _store(paths: ProjectPaths) -> ParquetStore:
    return ParquetStore(paths.normalized / "bars")


def fetch_industry_data(
    paths: ProjectPaths,
    as_of: date,
    *,
    synthetic_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    output = paths.normalized / "industry" / "snapshots"
    if synthetic_frame is not None:
        dates = pd.DatetimeIndex(
            sorted(
                synthetic_frame.loc[
                    synthetic_frame["symbol"].eq("SH000300"),
                    "trade_date",
                ].unique()
            )
        )
        snapshot = synthetic_industry_snapshot(
            synthetic_frame,
            pd.DatetimeIndex(sorted(AShareSimulator._weekly_signal_dates(dates))),
        )
    else:
        snapshot = BaostockAdapter(RawArchive(paths.raw)).fetch_stock_industry(as_of)
    inserted = write_industry_snapshot(snapshot, output)
    available = int(snapshot["industry_name"].notna().sum())
    return {
        "as_of": pd.Timestamp(snapshot["as_of_date"].max()).date().isoformat(),
        "rows": len(snapshot),
        "classified_rows": available,
        "coverage": available / max(len(snapshot), 1),
        "inserted": inserted,
        "path": str(output),
        "source": str(snapshot["source"].iloc[0]),
    }


def backfill_industry_data(
    paths: ProjectPaths,
    start: date,
    end: date,
) -> dict[str, Any]:
    data_config = load_config(paths, "data")
    bars_path = paths.curated / "bars.parquet"
    if not bars_path.exists():
        raise FileNotFoundError("Curated bars are required before industry backfill")
    benchmark_dates = pd.read_parquet(
        bars_path,
        columns=["trade_date"],
        filters=[
            ("symbol", "==", data_config["benchmark_symbol"]),
            ("trade_date", ">=", pd.Timestamp(start)),
            ("trade_date", "<=", pd.Timestamp(end)),
        ],
    )
    calendar = pd.DatetimeIndex(
        sorted(pd.to_datetime(benchmark_dates["trade_date"]).dt.normalize().unique())
    )
    target_dates = sorted(AShareSimulator._weekly_signal_dates(calendar))
    output = paths.normalized / "industry" / "snapshots"
    adapter = BaostockAdapter(RawArchive(paths.raw))
    checkpoints = AppendOnlyJournal(paths.state / "industry-checkpoints.jsonl", "checkpoint_id")
    max_retries = int(data_config["fetch"]["max_retries"])
    requests_per_second = float(data_config["fetch"]["industry_requests_per_second"])
    fetched = 0
    skipped = 0
    with adapter.session():
        for index, timestamp in enumerate(target_dates, start=1):
            target = pd.Timestamp(timestamp).date()
            partition = output / f"as_of_date={target.isoformat()}" / "membership.parquet"
            checkpoint_id = f"industry-v1|{target.isoformat()}"
            if partition.exists() and checkpoints.contains(checkpoint_id):
                skipped += 1
                continue
            for attempt in range(max_retries + 1):
                try:
                    snapshot = adapter.fetch_stock_industry(target)
                    minimum_rows = int(
                        data_config["sector_state"]["minimum_historical_snapshot_rows"]
                    )
                    if len(snapshot) < minimum_rows:
                        raise RuntimeError(
                            f"Industry snapshot has only {len(snapshot)} rows for {target}; "
                            f"minimum={minimum_rows}"
                        )
                    write_industry_snapshot(snapshot, output)
                    checkpoints.append(
                        {
                            "checkpoint_id": checkpoint_id,
                            "as_of": target.isoformat(),
                            "rows": len(snapshot),
                            "classified_rows": int(snapshot["industry_name"].notna().sum()),
                            "completed_at": datetime.now().astimezone().isoformat(),
                        }
                    )
                    fetched += 1
                    break
                except Exception as error:
                    if attempt >= max_retries:
                        raise RuntimeError(
                            f"Industry backfill failed for {target}: {error}"
                        ) from error
                    delay = min(30, 2**attempt)
                    LOGGER.warning(
                        "Industry retry %s/%s for %s after %ss: %s",
                        attempt + 1,
                        max_retries,
                        target,
                        delay,
                        error,
                    )
                    time.sleep(delay)
            if requests_per_second > 0:
                time.sleep(1.0 / requests_per_second)
            if index % 10 == 0 or index == len(target_dates):
                LOGGER.info(
                    "Industry backfill progress %s/%s (fetched=%s skipped=%s)",
                    index,
                    len(target_dates),
                    fetched,
                    skipped,
                )
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "target_dates": len(target_dates),
        "fetched": fetched,
        "skipped": skipped,
        "path": str(output),
    }


def _bootstrap_historical_universe(
    paths: ProjectPaths,
    adapter: BaostockAdapter,
    calendar: pd.DatetimeIndex,
    max_retries: int,
) -> list[str]:
    output = paths.normalized / "universe" / "csi300_daily.parquet"
    existing_dates: set[pd.Timestamp] = set()
    if output.exists():
        existing = pd.read_parquet(output)
        existing_dates = set(pd.to_datetime(existing["trade_date"]).dt.normalize())
    frames: list[pd.DataFrame] = []
    missing = [
        pd.Timestamp(timestamp).normalize()
        for timestamp in calendar
        if pd.Timestamp(timestamp).normalize() not in existing_dates
    ]
    for batch_start in range(0, len(missing), 100):
        batch = missing[batch_start : batch_start + 100]
        with adapter.session():
            for offset, normalized in enumerate(batch, start=1):
                for attempt in range(max_retries + 1):
                    try:
                        raw = adapter.fetch_historical_csi300(normalized.date())
                        if raw.empty or len(raw) != 300:
                            raise RuntimeError(
                                f"Historical CSI300 query returned {len(raw)} rows for {normalized.date()}"
                            )
                        break
                    except Exception:
                        if attempt >= max_retries:
                            raise
                        delay = min(30, 2**attempt)
                        LOGGER.warning(
                            "Historical universe retry %s/%s for %s after %ss",
                            attempt + 1,
                            max_retries,
                            normalized.date(),
                            delay,
                        )
                        time.sleep(delay)
                frames.append(raw[["trade_date", "symbol"]])
                if len(frames) >= 20:
                    write_constituents(pd.concat(frames, ignore_index=True), output)
                    frames.clear()
                completed = batch_start + offset
                if completed % 100 == 0:
                    LOGGER.info("Historical universe progress %s/%s", completed, len(missing))
    if frames:
        write_constituents(pd.concat(frames, ignore_index=True), output)
    universe = pd.read_parquet(output)
    return sorted(universe["symbol"].unique().tolist())


def _fetch_checkpoint_id(symbol: str, start: date, end: date) -> str:
    payload = f"bars-v1|{symbol}|{start.isoformat()}|{end.isoformat()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _historical_symbol_fetch_range(
    start: date,
    end: date,
    membership: pd.DataFrame,
) -> tuple[date, date]:
    first_member = pd.Timestamp(membership["trade_date"].min()).date()
    # Membership controls feature/training eligibility, not price availability.
    # Continue downloading a former constituent through the research cutoff so
    # an already-held position can still be valued and exited after index removal.
    return (
        max(start, first_member - pd.Timedelta(days=400).to_pytimedelta()),
        end,
    )


def _select_symbol_shard(
    symbols: list[str],
    shard_index: int,
    shard_count: int,
) -> list[str]:
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")
    return symbols[shard_index::shard_count]


def fetch_data(
    paths: ProjectPaths,
    start: date,
    end: date,
    symbols: list[str] | None,
    synthetic: bool = False,
    bootstrap_universe: bool = True,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    paths.ensure_runtime_dirs()
    data_config = load_config(paths, "data")
    _check_resources(paths, data_config)
    store = _store(paths)
    if synthetic:
        frame = synthetic_normalized()
        inserted = store.upsert(frame)
        industry = fetch_industry_data(paths, end, synthetic_frame=frame)
        return {
            "mode": "synthetic",
            "rows": len(frame),
            "inserted": inserted,
            "industry": industry,
        }

    archive = RawArchive(paths.raw)
    ak = AKShareAdapter(
        archive,
        int(data_config["fetch"]["timeout_seconds"]),
        float(data_config["fetch"]["akshare_requests_per_second"]),
    )
    bs = BaostockAdapter(archive)
    industry = (
        fetch_industry_data(paths, end)
        if shard_index == 0
        else {"status": "SKIPPED_NONZERO_SHARD"}
    )
    max_retries = int(data_config["fetch"]["max_retries"])
    checkpoints = AppendOnlyJournal(paths.state / "fetch-checkpoints.jsonl", "checkpoint_id")
    benchmark_symbol = data_config["benchmark_symbol"]
    benchmark = ak.fetch_index_bars(benchmark_symbol, start, end)
    benchmark_raw = benchmark.copy()
    benchmark_raw["adjustment"] = "raw"
    store.upsert(pd.concat([benchmark, benchmark_raw], ignore_index=True))
    calendar = pd.DatetimeIndex(sorted(benchmark["trade_date"].dropna().unique()))
    symbol_ranges: dict[str, tuple[date, date]] = {}
    if symbols is None:
        universe_path = paths.normalized / "universe" / "csi300_daily.parquet"
        if bootstrap_universe:
            symbols = _bootstrap_historical_universe(paths, bs, calendar, max_retries)
            universe = pd.read_parquet(universe_path)
            universe["trade_date"] = pd.to_datetime(universe["trade_date"]).dt.normalize()
            for symbol, membership in universe.groupby("symbol"):
                symbol_ranges[str(symbol)] = _historical_symbol_fetch_range(
                    start,
                    end,
                    membership,
                )
        elif universe_path.exists():
            current = ak.fetch_current_csi300()
            code_column = "成分券代码" if "成分券代码" in current else "品种代码"
            symbols = sorted(current[code_column].astype(str).tolist())
            write_constituents(
                pd.DataFrame({"trade_date": pd.Timestamp(end), "symbol": symbols}),
                universe_path,
            )
        else:
            current = ak.fetch_current_csi300()
            code_column = "成分券代码" if "成分券代码" in current else "品种代码"
            symbols = sorted(current[code_column].astype(str).tolist())
            write_constituents(
                pd.DataFrame({"trade_date": pd.Timestamp(end), "symbol": symbols}),
                universe_path,
            )
    total_symbols = len(symbols)
    symbols = _select_symbol_shard(symbols, shard_index, shard_count)
    failures: list[str] = []
    fetched = 0
    skipped = 0
    for batch_start in range(0, len(symbols), 25):
        batch = symbols[batch_start : batch_start + 25]
        with bs.session():
            for offset, symbol in enumerate(batch, start=1):
                symbol_start, symbol_end = symbol_ranges.get(symbol, (start, end))
                checkpoint_id = _fetch_checkpoint_id(symbol, symbol_start, symbol_end)
                if checkpoints.contains(checkpoint_id):
                    skipped += 1
                    continue
                for attempt in range(max_retries + 1):
                    try:
                        raw = ak.fetch_stock_bars(symbol, symbol_start, symbol_end, "raw")
                        hfq = ak.fetch_stock_bars(symbol, symbol_start, symbol_end, "hfq")
                        backup = bs.fetch_stock_bars(symbol, symbol_start, symbol_end, "raw")
                        store.upsert(pd.concat([raw, hfq, backup], ignore_index=True))
                        checkpoints.append(
                            {
                                "checkpoint_id": checkpoint_id,
                                "symbol": symbol,
                                "start": symbol_start.isoformat(),
                                "end": symbol_end.isoformat(),
                                "completed_at": datetime.now().astimezone().isoformat(),
                            }
                        )
                        fetched += 1
                        break
                    except Exception as error:
                        if attempt >= max_retries:
                            LOGGER.exception("Fetch failed for %s", symbol)
                            failures.append(f"{symbol}: {error}")
                            break
                        delay = min(30, 2**attempt)
                        LOGGER.warning(
                            "Bar retry %s/%s for %s after %ss: %s",
                            attempt + 1,
                            max_retries,
                            symbol,
                            delay,
                            error,
                        )
                        time.sleep(delay)
                index = batch_start + offset
                if index % 25 == 0:
                    LOGGER.info(
                        "Bar fetch progress %s/%s (fetched=%s skipped=%s failures=%s)",
                        index,
                        len(symbols),
                        fetched,
                        skipped,
                        len(failures),
                    )
    if failures:
        raise RuntimeError("One or more symbols failed: " + "; ".join(failures[:20]))
    return {
        "mode": "live",
        "symbols": fetched,
        "skipped": skipped,
        "total_universe_symbols": total_symbols,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "industry": industry,
    }


def validate_data(paths: ProjectPaths, as_of: date | None = None) -> dict[str, Any]:
    data_config = load_config(paths, "data")
    store = _store(paths)
    ak = store.read("akshare", "raw")
    bs = store.read("baostock", "raw")
    validator = DataValidator(data_config)
    clean, quarantined, quarantine_report = validator.apply_configured_quarantine(
        pd.concat([ak, bs], ignore_index=True),
        as_of,
    )
    quarantine_path = write_quarantine_records(
        quarantined,
        paths.quarantine / "provider-records",
    )
    clean_ak = clean[clean["source"].eq("akshare")]
    clean_bs = clean[clean["source"].eq("baostock")]
    reports = [
        validator.validate_single(clean_ak, as_of),
        validator.validate_single(clean_bs, as_of),
        quarantine_report,
    ]
    comparable_ak = clean_ak[clean_ak["symbol"].ne(data_config["benchmark_symbol"])]
    reports.append(validator.compare_sources(comparable_ak, clean_bs, as_of))
    combined = combine_reports(*reports)
    report_path = write_quality_report(
        combined,
        paths.reports / "data-quality" / f"{combined.as_of.isoformat()}.json",
    )
    if combined.failed:
        raise RuntimeError(f"Data quality gate failed; see {report_path}")
    curated = store.build_curated(
        paths.curated / "bars.parquet",
        excluded_records=quarantined,
    )
    return {
        "status": combined.status.value,
        "issues": len(combined.issues),
        "curated_rows": len(curated),
        "report": str(report_path),
        "quarantine": str(quarantine_path) if quarantine_path else None,
    }


def _research_universe(
    paths: ProjectPaths,
    stock_bars: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    data_config: dict[str, Any],
) -> tuple[pd.DataFrame, str]:
    historical_path = paths.normalized / "universe" / "csi300_daily.parquet"
    if historical_path.exists():
        historical = pd.read_parquet(historical_path)[["trade_date", "symbol"]]
        historical["trade_date"] = pd.to_datetime(historical["trade_date"]).dt.normalize()
        counts = historical.groupby("trade_date")["symbol"].nunique()
        covered = set(counts[counts.eq(300)].index)
        required = {pd.Timestamp(value).normalize() for value in trading_dates}
        if required and required.issubset(covered):
            return (
                historical[historical["trade_date"].isin(required)].drop_duplicates(["trade_date", "symbol"]),
                "historical_csi300",
            )
        LOGGER.warning(
            "Historical CSI300 coverage is incomplete (%s/%s dates); using point-in-time dynamic fallback",
            len(required & covered),
            len(required),
        )

    fallback = data_config["fallback_universe"]
    dynamic = dynamic_universe(
        stock_bars[["symbol", "trade_date", "amount", "is_trading"]],
        min_history_days=int(fallback["min_history_days"]),
        liquidity_window=int(fallback["liquidity_window"]),
        min_median_amount=float(fallback["min_median_amount_cny"]),
        max_symbols=int(fallback["max_symbols"]),
    )
    if dynamic.empty:
        raise RuntimeError("No point-in-time universe could be constructed")
    return dynamic[["trade_date", "symbol"]], "dynamic_liquidity_fallback"


def build_dataset(paths: ProjectPaths, qlib_smoke: bool = True) -> dict[str, Any]:
    curated_path = paths.curated / "bars.parquet"
    if not curated_path.exists():
        raise FileNotFoundError("Curated bars are absent; run validate-data first")
    curated = _read_compact_curated(curated_path)
    data_config = load_config(paths, "data")
    benchmark = curated[curated["symbol"].eq(data_config["benchmark_symbol"])]
    stock_bars = curated.loc[
        curated["symbol"].ne(data_config["benchmark_symbol"]),
        ["symbol", "trade_date", "amount", "is_trading"],
    ]
    trading_dates = pd.DatetimeIndex(sorted(benchmark["trade_date"].dropna().unique()))
    universe, universe_source = _research_universe(
        paths,
        stock_bars,
        trading_dates,
        data_config,
    )
    universe.to_parquet(paths.curated / "universe.parquet", index=False, compression="zstd")
    del stock_bars
    research_bars = _feature_research_bars(
        curated,
        universe,
        str(data_config["benchmark_symbol"]),
    )
    qlib_path = build_qlib_dataset(research_bars, paths.qlib)
    if qlib_smoke:
        smoke_test_qlib(qlib_path)
    gc.collect()
    features = build_features(
        research_bars,
        data_config["benchmark_symbol"],
        universe,
    )
    model_config = load_config(paths, "model")
    labeled = add_labels(
        features,
        benchmark,
        horizon=int(model_config["label_horizon"]),
        entry_offset=int(model_config["label_entry_offset"]),
    )
    feature_path = paths.curated / "features.parquet"
    labeled.to_parquet(feature_path, index=False, compression="zstd")
    return {
        "rows": len(labeled),
        "features": len([column for column in FEATURE_COLUMNS if column in labeled]),
        "universe_source": universe_source,
        "universe_rows": len(universe),
        "research_bar_rows": len(research_bars),
        "execution_bar_rows": len(curated),
        "qlib_path": str(qlib_path),
    }


def train_current(paths: ProjectPaths) -> dict[str, Any]:
    feature_path = paths.curated / "features.parquet"
    frame = pd.read_parquet(feature_path)
    frame = frame[frame["in_universe"].fillna(False)].copy()
    model_config = load_config(paths, "model")
    calendar = pd.DatetimeIndex(sorted(frame["trade_date"].unique()))
    split = current_training_split(
        calendar,
        int(model_config["validation_months"]),
        int(model_config["embargo_trading_days"]),
        int(model_config["minimum_training_years"]),
    )
    models, metrics, importance = fit_models(frame, split, model_config)
    quality_summary = _latest_quality_summary(paths, str(split.valid_end.date()))
    bundle = save_model_bundle(
        models,
        split,
        model_config,
        metrics,
        importance,
        paths.models,
        data_quality_summary=quality_summary,
    )
    return {
        "bundle": str(bundle),
        "metrics": metrics,
        "split": {key: str(value.date()) for key, value in split.__dict__.items()},
    }


def run_backtest(paths: ProjectPaths, max_folds: int | None = None) -> dict[str, Any]:
    frame = pd.read_parquet(paths.curated / "features.parquet")
    frame = frame[frame["in_universe"].fillna(False)].copy()
    model_config = load_config(paths, "model")
    backtest_config = load_config(paths, "backtest")
    calendar = pd.DatetimeIndex(sorted(frame["trade_date"].unique()))
    splits = quarterly_splits(
        calendar,
        model_config["first_oos"],
        int(model_config["validation_months"]),
        int(model_config["embargo_trading_days"]),
        int(model_config["minimum_training_years"]),
    )
    if max_folds:
        splits = splits[-max_folds:]
    predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    for index, split in enumerate(splits, start=1):
        models, metrics, _ = fit_models(frame, split, model_config)
        test = frame[frame["trade_date"].between(split.test_start, split.test_end)].copy()
        fold_prediction = predict_models(models, test)
        fold_prediction["momentum_score"] = test["ret_20"].to_numpy()
        fold_prediction["momentum_rank"] = fold_prediction.groupby("trade_date")["momentum_score"].rank(
            pct=True
        )
        fold_prediction["ridge_rank"] = fold_prediction.groupby("trade_date")["ridge_prediction"].rank(
            pct=True
        )
        fold_prediction["logistic_rank"] = fold_prediction.groupby("trade_date")["logistic_probability"].rank(
            pct=True
        )
        fold_prediction["fold"] = index
        predictions.append(fold_prediction)
        fold_metrics.append(
            {
                "fold": index,
                **metrics,
                "test_start": str(split.test_start.date()),
                "test_end": str(split.test_end.date()),
            }
        )
        LOGGER.info("Backtest fold %s/%s complete", index, len(splits))
        del models, test, fold_prediction
        gc.collect()
    if not predictions:
        raise RuntimeError("No valid rolling folds were produced")
    prediction_frame = pd.concat(predictions, ignore_index=True)
    market_states = _historical_market_states(frame)
    prediction_frame = prediction_frame.merge(
        market_states,
        on="trade_date",
        how="left",
        validate="many_to_one",
    )
    labels = frame[
        [
            "symbol",
            "trade_date",
            "label_regression",
            "label_classification",
            "in_universe",
        ]
    ].copy()
    del frame, predictions
    gc.collect()

    oos_start = pd.Timestamp(prediction_frame["trade_date"].min())
    oos_end = pd.Timestamp(prediction_frame["trade_date"].max())
    curated = pd.read_parquet(
        paths.curated / "bars.parquet",
        columns=[
            "symbol",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "hfq_open",
            "hfq_high",
            "hfq_low",
            "hfq_close",
            "volume",
            "adjust_factor",
            "is_trading",
            "is_st",
        ],
        filters=[
            ("trade_date", ">=", oos_start - pd.Timedelta(days=90)),
            ("trade_date", "<=", oos_end),
        ],
    )
    curated["symbol"] = curated["symbol"].astype("category")
    curated[
        [
            "open",
            "high",
            "low",
            "close",
            "hfq_open",
            "hfq_high",
            "hfq_low",
            "hfq_close",
            "volume",
            "adjust_factor",
        ]
    ] = curated[
        [
            "open",
            "high",
            "low",
            "close",
            "hfq_open",
            "hfq_high",
            "hfq_low",
            "hfq_close",
            "volume",
            "adjust_factor",
        ]
    ].astype("float32")
    sector_bars = curated[
        curated["symbol"].ne(load_config(paths, "data")["benchmark_symbol"])
    ].copy()
    bars = sector_bars[sector_bars["trade_date"].ge(oos_start)].copy()
    prediction_dates = {
        pd.Timestamp(value)
        for value in pd.to_datetime(prediction_frame["trade_date"]).dt.normalize().unique()
    }
    signal_dates = prediction_dates
    historical_sectors = _historical_sector_states(
        paths,
        sector_bars,
        prediction_frame,
        signal_dates,
    )
    prediction_frame = prediction_frame.merge(
        historical_sectors[
            [
                "symbol",
                "trade_date",
                "sector_name",
                "sector_state",
                "sector_return_20",
                "sector_volatility_20",
                "sector_member_count",
                "sector_mapping_as_of",
                "sector_classification",
            ]
        ],
        on=["symbol", "trade_date"],
        how="left",
        validate="one_to_one",
    )
    prediction_path = paths.reports / "research" / "rolling_predictions.parquet"
    prediction_frame.to_parquet(prediction_path, index=False, compression="zstd")
    legacy_result = AShareSimulator(backtest_config).run(bars, prediction_frame)
    execution_config = load_config(paths, "execution")
    calibration_cache: dict[pd.Timestamp, dict[str, Any]] = {}
    gated_result = ProductionMirrorSimulator(
        execution_config,
        apply_risk_gate=True,
        calibration_cache=calibration_cache,
    ).run(bars, prediction_frame)
    ungated_result = ProductionMirrorSimulator(
        execution_config,
        apply_risk_gate=False,
        calibration_cache=calibration_cache,
    ).run(bars, prediction_frame)
    result = gated_result
    result.nav.to_parquet(paths.reports / "research" / "nav.parquet", index=False)
    result.holdings.to_parquet(paths.reports / "research" / "holdings.parquet", index=False)
    result.trades.to_parquet(paths.reports / "research" / "trades.parquet", index=False)
    result.orders.to_parquet(paths.reports / "research" / "orders.parquet", index=False)
    legacy_result.nav.to_parquet(
        paths.reports / "research" / "nav-legacy-weekly-top20.parquet",
        index=False,
    )
    ungated_result.nav.to_parquet(
        paths.reports / "research" / "nav-production-ungated.parquet",
        index=False,
    )
    ungated_result.holdings.to_parquet(
        paths.reports / "research" / "holdings-production-ungated.parquet",
        index=False,
    )
    ungated_result.trades.to_parquet(
        paths.reports / "research" / "trades-production-ungated.parquet",
        index=False,
    )
    ungated_result.orders.to_parquet(
        paths.reports / "research" / "orders-production-ungated.parquet",
        index=False,
    )
    benchmark = curated[
        curated["symbol"].eq(load_config(paths, "data")["benchmark_symbol"])
        & curated["trade_date"].ge(oos_start)
    ]
    metrics = performance_metrics(result.nav, benchmark, result.trades)
    metrics["authoritative_strategy"] = {
        "name": "production_mirror_gated_primary",
        "policy": policy_snapshot(execution_config),
        "uses_shared_daily_plan_evaluation": True,
        "uses_shared_daily_fill_simulator": True,
        "historical_intraday_policy": "OPEN_ONLY",
    }
    metrics["strategy_comparison"] = {
        "gated_primary_production_mirror": performance_metrics(
            gated_result.nav,
            benchmark,
            gated_result.trades,
        ),
        "ungated_shadow_production_mirror": performance_metrics(
            ungated_result.nav,
            benchmark,
            ungated_result.trades,
        ),
        "legacy_weekly_top20_fully_invested_model_baseline": performance_metrics(
            legacy_result.nav,
            benchmark,
            legacy_result.trades,
        ),
    }
    strategy_columns = {
        "momentum_20d": "momentum_rank",
        "ridge_regression": "ridge_rank",
        "logistic_regression": "logistic_rank",
        "lightgbm_regression": "regression_rank",
        "lightgbm_classification": "classification_rank",
    }
    strategy_metrics: dict[str, Any] = {}
    gate_columns = ["market_state", "sector_state"]
    for name, score_column in strategy_columns.items():
        baseline_predictions = prediction_frame[
            ["symbol", "trade_date", score_column, *gate_columns]
        ].rename(
            columns={score_column: "model_score"}
        )
        baseline_result = AShareSimulator(backtest_config).run(bars, baseline_predictions)
        strategy_metrics[name] = performance_metrics(
            baseline_result.nav,
            benchmark,
            baseline_result.trades,
        )
    initial_cash = float(backtest_config["initial_cash"])
    benchmark_oos = benchmark[benchmark["trade_date"].between(oos_start, oos_end)]
    pool_oos = bars.merge(
        labels[["symbol", "trade_date"]].drop_duplicates(),
        on=["symbol", "trade_date"],
        how="inner",
    )
    benchmark_nav = _close_series_nav(
        benchmark_oos.set_index("trade_date")["close"],
        initial_cash,
    )
    pool_returns = (
        pool_oos.pivot(index="trade_date", columns="symbol", values="close")
        .sort_index()
        .pct_change(fill_method=None)
        .mean(axis=1, skipna=True)
    )
    pool_nav = _returns_nav(pool_returns, initial_cash)
    metrics["baselines"] = {
        "hs300_buy_and_hold": performance_metrics(benchmark_nav),
        "pool_equal_weight_gross": performance_metrics(pool_nav, benchmark_oos),
        "ensemble_without_risk_gates": performance_metrics(
            AShareSimulator(backtest_config)
            .run(
                bars,
                prediction_frame[["symbol", "trade_date", "model_score"]],
            )
            .nav,
            benchmark_oos,
        ),
        **strategy_metrics,
    }
    metrics["signals"] = signal_metrics(prediction_frame, labels)
    metrics["folds"] = fold_metrics
    weekly_predictions = prediction_frame[
        prediction_frame["trade_date"].isin(signal_dates)
    ]
    metrics["risk_gates"] = {
        "signal_dates": len(signal_dates),
        "market_state_counts": weekly_predictions[
            ["trade_date", "market_state"]
        ]
        .drop_duplicates("trade_date")["market_state"]
        .value_counts(dropna=False)
        .to_dict(),
        "sector_state_counts": weekly_predictions["sector_state"]
        .fillna("NOT_A_SIGNAL_DATE")
        .value_counts()
        .to_dict(),
        "risk_filter_rejections": (
            int(
                result.orders["trigger_rule"]
                .eq("PLAN_PREVIOUSLY_CANCELLED")
                .sum()
            )
            if not result.orders.empty
            else 0
        ),
        "industry_snapshot_mode": "exact_weekly_point_in_time",
        "primary_account": "gated",
        "shadow_account": "ungated",
        "manual_account": "excluded_from_model_backtest_user_reported_only",
    }
    stability: dict[str, Any] = {}
    for top_k in (15, 20, 25):
        variant = dict(backtest_config)
        variant["top_k"] = top_k
        variant_result = AShareSimulator(variant).run(bars, prediction_frame)
        stability[f"top_k_{top_k}"] = performance_metrics(variant_result.nav)["annualized_return"]
    metrics["stability"] = stability
    metrics["periods"] = {}
    for name, start, end in (
        ("development_2014_2023", pd.Timestamp("2014-01-01"), pd.Timestamp("2023-12-31")),
        (
            "locked_oos_2024_onward",
            pd.Timestamp(model_config["locked_test_start"]),
            oos_end,
        ),
    ):
        period_nav = result.nav[result.nav["trade_date"].between(start, end)]
        if period_nav.empty:
            continue
        period_benchmark = benchmark[benchmark["trade_date"].between(start, end)]
        period_trades = (
            result.trades[pd.to_datetime(result.trades["trade_date"]).between(start, end)]
            if not result.trades.empty
            else result.trades
        )
        period_predictions = prediction_frame[prediction_frame["trade_date"].between(start, end)]
        period_labels = labels[labels["trade_date"].between(start, end)]
        metrics["periods"][name] = {
            "performance": performance_metrics(
                period_nav,
                period_benchmark,
                period_trades,
            ),
            "signals": signal_metrics(period_predictions, period_labels),
        }
    generate_research_report(metrics, paths.reports / "research", "backtest")
    return {"folds": len(splits), "predictions": len(prediction_frame), "metrics": metrics}


def _returns_nav(returns: pd.Series, initial_cash: float) -> pd.DataFrame:
    clean = returns.sort_index().fillna(0.0).astype(float)
    values = initial_cash * (1.0 + clean).cumprod()
    return pd.DataFrame({"trade_date": values.index, "nav": values.to_numpy()})


def _close_series_nav(close: pd.Series, initial_cash: float) -> pd.DataFrame:
    clean = close.sort_index().dropna().astype(float)
    if clean.empty:
        raise ValueError("Cannot construct a benchmark NAV from an empty close series")
    values = initial_cash * clean / clean.iloc[0]
    return pd.DataFrame({"trade_date": values.index, "nav": values.to_numpy()})


def _latest_quality_summary(paths: ProjectPaths, trade_date: str) -> dict[str, Any]:
    candidates = sorted((paths.reports / "data-quality").glob("*.json"))
    if not candidates:
        return {"status": "UNKNOWN", "report": None}
    cutoff = date.fromisoformat(trade_date)
    selected: tuple[Path, dict[str, Any]] | None = None
    for candidate in candidates:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        covered_dates = [
            date.fromisoformat(str(value["end"]))
            for value in payload.get("stats", {}).values()
            if isinstance(value, dict) and value.get("end")
        ]
        if covered_dates and max(covered_dates) >= cutoff:
            selected = (candidate, payload)
    if selected is None:
        selected = (
            candidates[-1],
            json.loads(candidates[-1].read_text(encoding="utf-8")),
        )
    report, payload = selected
    return {
        "status": payload.get("status", "UNKNOWN"),
        "report": str(report),
        "issue_count": len(payload.get("issues", [])),
        "stats": payload.get("stats", {}),
    }


def _latest_quality_status(paths: ProjectPaths, trade_date: str) -> str:
    candidates = sorted((paths.reports / "data-quality").glob(f"{trade_date}*.json"))
    if candidates:
        payload = json.loads(candidates[-1].read_text(encoding="utf-8"))
        return payload.get("status", "UNKNOWN")
    cutoff = date.fromisoformat(trade_date)
    for candidate in reversed(sorted((paths.reports / "data-quality").glob("*.json"))):
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        covered_dates = []
        for value in payload.get("stats", {}).values():
            if isinstance(value, dict) and value.get("end"):
                covered_dates.append(date.fromisoformat(str(value["end"])))
        if covered_dates and max(covered_dates) >= cutoff:
            return payload.get("status", "UNKNOWN")
    return "UNKNOWN"


def _latest_position_weights(
    paths: ProjectPaths,
    variant: str = "gated",
) -> dict[str, float]:
    simulation_state = paths.state / "simulation" / variant / "current.json"
    if simulation_state.exists():
        payload = json.loads(simulation_state.read_text(encoding="utf-8"))
        return {
            str(item["symbol"]): float(
                item.get("current_position_weight", item.get("weight", 0.0))
            )
            for item in payload.get("positions", [])
            if "symbol" in item
        }
    if variant != "gated":
        return {}
    candidates = sorted((paths.reports / "positions").glob("*.json"))
    if not candidates:
        return {}
    payload = json.loads(candidates[-1].read_text(encoding="utf-8"))
    positions = payload.get("positions", [])
    return {
        str(item["symbol"]): float(item.get("current_position_weight", item.get("weight", 0.0)))
        for item in positions
        if "symbol" in item
    }


def _latest_market_state(frame: pd.DataFrame, latest_date: pd.Timestamp) -> str:
    market = (
        frame[["trade_date", "market_ret_20", "market_vol_20"]]
        .drop_duplicates("trade_date")
        .sort_values("trade_date")
    )
    history = market[market["trade_date"].le(latest_date)].dropna()
    if len(history) < 60:
        return "DATA_UNAVAILABLE"
    latest = history.iloc[-1]
    weak_return = history["market_ret_20"].quantile(0.20)
    high_volatility = history["market_vol_20"].quantile(0.80)
    if latest["market_ret_20"] <= weak_return and latest["market_vol_20"] >= high_volatility:
        return "HIGH_RISK"
    if latest["market_ret_20"] <= weak_return or latest["market_vol_20"] >= high_volatility:
        return "CAUTIOUS"
    return "NORMAL"


def _latest_sector_states(
    paths: ProjectPaths,
    latest_date: pd.Timestamp,
    prediction_symbols: set[str],
) -> tuple[pd.DataFrame, Path]:
    data_config = load_config(paths, "data")
    sector_config = data_config["sector_state"]
    membership = load_industry_snapshot(
        paths.normalized / "industry" / "snapshots",
        latest_date,
        int(sector_config["max_mapping_age_days"]),
    )
    if membership.empty:
        states = pd.DataFrame(
            {
                "symbol": sorted(prediction_symbols),
                "stock_name": pd.NA,
                "sector_name": pd.NA,
                "sector_state": "DATA_UNAVAILABLE",
                "sector_return_20": pd.NA,
                "sector_volatility_20": pd.NA,
                "sector_member_count": pd.NA,
                "sector_mapping_as_of": pd.NaT,
                "sector_classification": pd.NA,
            }
        )
    else:
        bars = pd.read_parquet(
            paths.curated / "bars.parquet",
            columns=["symbol", "trade_date", "hfq_close", "is_trading"],
            filters=[
                ("trade_date", ">=", latest_date - pd.Timedelta(days=90)),
                ("trade_date", "<=", latest_date),
            ],
        )
        states = compute_sector_states(bars, membership, latest_date, sector_config)
    relevant = states[states["symbol"].isin(prediction_symbols)]
    mapping_coverage = float(relevant["sector_name"].notna().mean()) if len(relevant) else 0.0
    state_coverage = (
        float(relevant["sector_state"].ne("DATA_UNAVAILABLE").mean()) if len(relevant) else 0.0
    )
    minimum = float(sector_config["minimum_signal_coverage"])
    report = {
        "as_of": latest_date.date().isoformat(),
        "status": "PASS" if state_coverage >= minimum else "WARN",
        "prediction_symbols": len(prediction_symbols),
        "mapped_symbols": int(relevant["sector_name"].notna().sum()),
        "available_sector_states": int(
            relevant["sector_state"].ne("DATA_UNAVAILABLE").sum()
        ),
        "mapping_coverage": mapping_coverage,
        "state_coverage": state_coverage,
        "minimum_signal_coverage": minimum,
        "mapping_snapshot": (
            pd.Timestamp(membership["as_of_date"].max()).date().isoformat()
            if not membership.empty
            else None
        ),
        "classification": (
            sorted(membership["industry_classification"].dropna().astype(str).unique())
            if not membership.empty
            else []
        ),
        "basis": "point_in_time_membership_and_equal_weight_member_daily_returns",
    }
    report_path = immutable_write_json(
        paths.reports
        / "data-quality"
        / f"industry-{latest_date.date().isoformat()}.json",
        report,
    )
    return states, report_path


def _historical_market_states(frame: pd.DataFrame) -> pd.DataFrame:
    market = (
        frame[["trade_date", "market_ret_20", "market_vol_20"]]
        .drop_duplicates("trade_date")
        .sort_values("trade_date")
        .copy()
    )
    market["weak_return"] = (
        market["market_ret_20"].expanding(min_periods=60).quantile(0.20)
    )
    market["high_volatility"] = (
        market["market_vol_20"].expanding(min_periods=60).quantile(0.80)
    )
    market["market_state"] = "DATA_UNAVAILABLE"
    available = market[["weak_return", "high_volatility"]].notna().all(axis=1)
    cautious = available & (
        market["market_ret_20"].le(market["weak_return"])
        | market["market_vol_20"].ge(market["high_volatility"])
    )
    high_risk = available & (
        market["market_ret_20"].le(market["weak_return"])
        & market["market_vol_20"].ge(market["high_volatility"])
    )
    market.loc[available, "market_state"] = "NORMAL"
    market.loc[cautious, "market_state"] = "CAUTIOUS"
    market.loc[high_risk, "market_state"] = "HIGH_RISK"
    return market[["trade_date", "market_state"]]


def _historical_sector_states(
    paths: ProjectPaths,
    bars: pd.DataFrame,
    predictions: pd.DataFrame,
    signal_dates: set[pd.Timestamp],
) -> pd.DataFrame:
    data_config = load_config(paths, "data")
    sector_config = data_config["sector_state"]
    snapshots = paths.normalized / "industry" / "snapshots"
    frames: list[pd.DataFrame] = []
    missing_dates: list[str] = []
    for index, signal_date in enumerate(sorted(signal_dates), start=1):
        membership = load_industry_snapshot(
            snapshots,
            signal_date,
            max_age_days=int(sector_config["max_mapping_age_days"]),
        )
        if membership.empty:
            missing_dates.append(signal_date.date().isoformat())
            symbols = sorted(
                set(
                    predictions.loc[
                        predictions["trade_date"].eq(signal_date),
                        "symbol",
                    ].astype(str)
                )
            )
            frames.append(
                pd.DataFrame(
                    {
                        "symbol": symbols,
                        "stock_name": pd.NA,
                        "sector_name": pd.NA,
                        "sector_state": "DATA_UNAVAILABLE",
                        "sector_return_20": pd.NA,
                        "sector_volatility_20": pd.NA,
                        "sector_member_count": pd.NA,
                        "sector_mapping_as_of": pd.NaT,
                        "sector_classification": pd.NA,
                        "trade_date": signal_date,
                    }
                )
            )
            continue
        trailing = bars[
            bars["trade_date"].between(
                signal_date - pd.Timedelta(days=90),
                signal_date,
            )
        ]
        states = compute_sector_states(
            trailing[["symbol", "trade_date", "hfq_close", "is_trading"]],
            membership,
            signal_date,
            sector_config,
        )
        symbols = set(
            predictions.loc[
                predictions["trade_date"].eq(signal_date),
                "symbol",
            ].astype(str)
        )
        states = states[states["symbol"].isin(symbols)].copy()
        states["trade_date"] = signal_date
        frames.append(states)
        if index % 50 == 0:
            LOGGER.info(
                "Historical sector-state progress %s/%s",
                index,
                len(signal_dates),
            )
    if missing_dates:
        LOGGER.warning(
            "Point-in-time industry snapshots unavailable for %s dates; "
            "those rows are explicitly DATA_UNAVAILABLE: %s",
            len(missing_dates),
            ",".join(missing_dates[:20]),
        )
    if not frames:
        raise RuntimeError("No historical sector states were produced")
    output = pd.concat(frames, ignore_index=True)
    output.to_parquet(
        paths.reports / "research" / "historical_sector_states.parquet",
        index=False,
        compression="zstd",
    )
    return output


def _execution_signal(
    rank: int,
    current_weight: float,
    target_weight: float,
    selected_count: int,
) -> str:
    return execution_signal(rank, current_weight, target_weight, selected_count)


def _candidate_pool(
    paths: ProjectPaths,
    eligible_prediction: pd.DataFrame,
    signal_date: pd.Timestamp,
    next_execution_date: date,
    execution_config: dict[str, Any],
    model_version: str,
) -> tuple[set[str], dict[str, Any]]:
    policy = portfolio_policy(execution_config)
    state_path = paths.state / "strategy" / "candidate-pool.json"
    refresh_mode = str(policy["candidate_refresh"])
    if refresh_mode != "weekly_last_trading_day":
        raise ValueError(f"Unsupported candidate refresh mode: {refresh_mode}")
    signal_iso = signal_date.date().isocalendar()
    execution_iso = next_execution_date.isocalendar()
    weekly_refresh = (signal_iso.year, signal_iso.week) != (
        execution_iso.year,
        execution_iso.week,
    )
    existing: dict[str, Any] | None = None
    if state_path.exists():
        existing = json.loads(state_path.read_text(encoding="utf-8"))
    should_refresh = weekly_refresh or (
        existing is None and bool(policy["bootstrap_when_pool_missing"])
    )
    if should_refresh:
        selected_rows = eligible_prediction.nsmallest(
            int(policy["candidate_count"]),
            "signal_rank",
        )
        payload = {
            "selected_on": signal_date.date().isoformat(),
            "valid_from": next_execution_date.isoformat(),
            "refresh_reason": (
                "WEEKLY_LAST_TRADING_DAY"
                if weekly_refresh
                else "BOOTSTRAP_MISSING_POOL"
            ),
            "candidate_refresh": refresh_mode,
            "model_version": model_version,
            "symbols": selected_rows["symbol"].astype(str).tolist(),
            "ranks_at_selection": {
                str(row["symbol"]): int(row["signal_rank"])
                for row in selected_rows.to_dict("records")
            },
        }
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temp = state_path.with_suffix(".tmp")
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp.replace(state_path)
        existing = payload
    if existing is None:
        raise RuntimeError("Weekly candidate pool is missing and bootstrap is disabled")
    eligible = set(eligible_prediction["symbol"].astype(str))
    selected = set(map(str, existing["symbols"])) & eligible
    return selected, existing


def _price_contexts_with_calibration(
    paths: ProjectPaths,
    symbols: list[str],
    signal_date: pd.Timestamp,
) -> tuple[dict[str, dict[str, Any]], Path | None]:
    bars_path = paths.curated / "bars.parquet"
    trailing = pd.read_parquet(
        bars_path,
        columns=["symbol", "trade_date", "high", "low", "close", "adjust_factor"],
        filters=[
            ("symbol", "in", symbols),
            ("trade_date", ">=", signal_date - pd.Timedelta(days=90)),
            ("trade_date", "<=", signal_date),
        ],
    )
    contexts = price_contexts_from_bars(trailing)
    predictions_path = paths.reports / "research" / "rolling_predictions.parquet"
    if not predictions_path.exists():
        return contexts, None

    execution_config = load_config(paths, "execution")
    lookback_years = int(execution_config["calibration"]["lookback_years"])
    calibration_start = signal_date - pd.DateOffset(years=lookback_years) - pd.Timedelta(days=40)
    calibration_bars = pd.read_parquet(
        bars_path,
        columns=[
            "symbol",
            "trade_date",
            "hfq_open",
            "hfq_high",
            "hfq_low",
            "hfq_close",
        ],
        filters=[
            ("trade_date", ">=", calibration_start),
            ("trade_date", "<=", signal_date),
        ],
    )
    oos_predictions = pd.read_parquet(
        predictions_path,
        columns=["symbol", "trade_date", "model_score"],
        filters=[
            ("trade_date", ">=", signal_date - pd.DateOffset(years=lookback_years)),
            ("trade_date", "<=", signal_date),
        ],
    )
    calibration = calibrate_from_oos_signals(
        calibration_bars,
        oos_predictions,
        signal_date,
        execution_config,
    )
    artifact = immutable_write_json(
        paths.reports
        / "research"
        / f"price-calibration-{signal_date.date().isoformat()}.json",
        calibration,
    )
    return apply_calibration(contexts, calibration), artifact


def create_execution_plan_command(
    paths: ProjectPaths,
    execution_date: date | None = None,
    signals: pd.DataFrame | None = None,
    price_contexts: dict[str, dict[str, Any]] | None = None,
    *,
    apply_risk_gate: bool = True,
    archive_directory: str = "execution-plans",
) -> dict[str, Any]:
    if signals is None:
        signal_dates = sorted(
            {
                path.stem.split("__r", maxsplit=1)[0]
                for path in (paths.reports / "signals").glob("*.csv")
            }
        )
        if not signal_dates:
            raise FileNotFoundError("No archived signal CSV is available")
        signal_path = _latest_immutable_revision(
            paths.reports / "signals",
            signal_dates[-1],
            ".csv",
        )
        if signal_path is None:
            raise FileNotFoundError("No archived signal CSV is available")
        signals = pd.read_csv(signal_path, encoding="utf-8-sig")
    signal_date = pd.Timestamp(signals["trade_date"].max())
    target_date = execution_date or (signal_date + pd.offsets.BDay(1)).date()
    eligible = signals[signals["model_signal"].ne("AVOID") | signals["current_position_weight"].gt(0)].copy()
    symbols = sorted(set(eligible["symbol"].astype(str)))
    contexts = price_contexts
    if contexts is None:
        contexts = (
            _price_contexts_with_calibration(paths, symbols, signal_date)[0] if symbols else {}
        )
    plans = build_execution_plans(
        eligible,
        contexts,
        target_date,
        load_config(paths, "execution"),
        apply_risk_gate=apply_risk_gate,
    )
    if not plans and not eligible.empty:
        raise RuntimeError("No execution plans were created; trailing price context is incomplete")
    json_path, csv_path = archive_execution_plans(
        plans,
        paths.reports,
        target_date.isoformat(),
        directory=archive_directory,
    )
    return {
        "execution_date": target_date.isoformat(),
        "plans": len(plans),
        "risk_gate_mode": "ENABLED" if apply_risk_gate else "DISABLED_SHADOW",
        "json": str(json_path),
        "csv": str(csv_path),
        "simulation_only": True,
    }


def _eligible_prediction_symbols(
    frame: pd.DataFrame,
    latest_date: pd.Timestamp,
    data_config: dict[str, Any],
) -> set[str]:
    fallback = data_config["fallback_universe"]
    liquidity_window = int(fallback["liquidity_window"])
    ordered = frame.sort_values(["symbol", "trade_date"])[
        ["symbol", "trade_date", "amount", "is_trading", "in_universe"]
    ].copy()
    if isinstance(ordered["symbol"].dtype, pd.CategoricalDtype):
        ordered["symbol"] = ordered["symbol"].cat.remove_unused_categories()
    grouped = ordered.groupby("symbol", group_keys=False, observed=True)
    ordered["history_days"] = grouped.cumcount() + 1
    ordered["median_amount"] = (
        grouped["amount"].rolling(liquidity_window).median().reset_index(level=0, drop=True)
    )
    latest = ordered[pd.to_datetime(ordered["trade_date"]).eq(latest_date.normalize())]
    eligible = latest[
        latest["in_universe"].fillna(False)
        & latest["history_days"].ge(int(fallback["min_history_days"]))
        & latest["median_amount"].ge(float(fallback["min_median_amount_cny"]))
        & latest["is_trading"].fillna(False)
    ]
    return set(eligible["symbol"])


def predict_latest(paths: ProjectPaths, execution_date: date | None = None) -> dict[str, Any]:
    feature_path = paths.curated / "features.parquet"
    context = pd.read_parquet(
        feature_path,
        columns=[
            "symbol",
            "trade_date",
            "amount",
            "is_trading",
            "in_universe",
            "market_ret_20",
            "market_vol_20",
        ],
    )
    latest_date = pd.Timestamp(context["trade_date"].max())
    latest = pd.read_parquet(
        feature_path,
        filters=[("trade_date", "==", latest_date)],
    )
    data_config = load_config(paths, "data")
    eligible_symbols = _eligible_prediction_symbols(context, latest_date, data_config)
    positions = _latest_position_weights(paths)
    shadow_positions = _latest_position_weights(paths, "ungated")
    latest = latest[
        latest["symbol"].isin(
            eligible_symbols | set(positions) | set(shadow_positions)
        )
    ].copy()
    if latest.empty or not eligible_symbols:
        raise RuntimeError(
            "No stocks passed the point-in-time history and liquidity gates; "
            "prediction stopped instead of scoring incomplete data"
        )
    models, manifest = load_current_models(paths.models)
    prediction = predict_models(models, latest)
    importance_path = (
        Path(json.loads((paths.models / "current.json").read_text())["path"]) / "feature_importance.json"
    )
    importance = json.loads(importance_path.read_text(encoding="utf-8"))
    main_features = ",".join(sorted(importance, key=importance.get, reverse=True)[:3])
    execution_config = load_config(paths, "execution")
    policy = portfolio_policy(execution_config)
    eligible_prediction = prediction[prediction["symbol"].isin(eligible_symbols)].copy()
    eligible_prediction["signal_rank"] = (
        eligible_prediction["model_score"].rank(ascending=False, method="first").astype(int)
    )
    ranks = eligible_prediction.set_index("symbol")["signal_rank"]
    prediction["signal_rank"] = (
        prediction["symbol"].map(ranks).fillna(len(eligible_prediction) + 1).astype(int)
    )
    next_execution_date = execution_date or (latest_date + pd.offsets.BDay(1)).date()
    selected, candidate_pool = _candidate_pool(
        paths,
        eligible_prediction,
        latest_date,
        next_execution_date,
        execution_config,
        str(manifest["version"]),
    )
    selected_weight = float(policy["target_position_weight"])
    sector_states, sector_report = _latest_sector_states(
        paths,
        latest_date,
        set(prediction["symbol"].astype(str)),
    )
    prediction = prediction.merge(
        sector_states,
        on="symbol",
        how="left",
        validate="one_to_one",
    )
    prediction["name"] = prediction["stock_name"].fillna(prediction["symbol"])
    prediction["sector_state"] = prediction["sector_state"].fillna("DATA_UNAVAILABLE")
    prediction["current_position_weight"] = prediction["symbol"].map(positions).fillna(0.0)
    prediction["target_position_weight"] = prediction["symbol"].map(
        lambda symbol: selected_weight if symbol in selected else 0.0
    )
    prediction["model_signal"] = prediction.apply(
        lambda row: (
            _execution_signal(
                int(row["signal_rank"]),
                float(row["current_position_weight"]),
                float(row["target_position_weight"]),
                len(selected),
            )
            if row["symbol"] in eligible_symbols
            else ("EXIT" if float(row["current_position_weight"]) > 0 else "AVOID")
        ),
        axis=1,
    )
    prediction["market_state"] = _latest_market_state(context, latest_date)
    prediction["signal_valid_until"] = next_execution_date.isoformat()
    prediction["predicted_excess_return_5d"] = prediction["predicted_excess_return"]
    prediction["trade_date"] = latest_date.date().isoformat()
    prediction["stock_name"] = prediction["name"]
    prediction["currently_held"] = prediction["current_position_weight"].gt(0)
    prediction["suggested_action"] = prediction["model_signal"]
    prediction["target_weight"] = prediction["target_position_weight"]
    prediction["main_features"] = main_features
    prediction["model_version"] = manifest["version"]
    prediction["data_cutoff"] = latest_date.date().isoformat()
    prediction["data_quality_status"] = _latest_quality_status(paths, latest_date.date().isoformat())
    price_contexts, calibration_artifact = _price_contexts_with_calibration(
        paths,
        sorted(set(prediction["symbol"].astype(str))),
        latest_date,
    )
    range_records = {
        symbol: price_range_fields(price_context, execution_config)
        for symbol, price_context in price_contexts.items()
    }
    if missing_ranges := sorted(set(prediction["symbol"].astype(str)) - set(range_records)):
        raise RuntimeError(
            "Trailing daily bars are insufficient for price ranges: "
            + ",".join(missing_ranges[:10])
        )
    ranges = pd.DataFrame.from_dict(range_records, orient="index")
    ranges.index.name = "symbol"
    prediction = prediction.merge(ranges.reset_index(), on="symbol", how="left", validate="one_to_one")
    signal_columns = [
        "trade_date",
        "symbol",
        "name",
        "stock_name",
        "model_version",
        "model_signal",
        "model_score",
        "outperform_probability",
        "predicted_excess_return_5d",
        "predicted_excess_return",
        "signal_rank",
        "current_position_weight",
        "target_position_weight",
        "market_state",
        "sector_state",
        "sector_name",
        "sector_return_20",
        "sector_volatility_20",
        "sector_member_count",
        "sector_mapping_as_of",
        "sector_classification",
        "signal_valid_until",
        "data_quality_status",
        "currently_held",
        "suggested_action",
        "target_weight",
        "main_features",
        "data_cutoff",
        "reference_close",
        "buy_price_low",
        "buy_price_high",
        "sell_price_low",
        "sell_price_high",
        "avoid_chasing_above",
        "signal_invalid_below",
        "hard_exit_below",
        "price_range_status",
        "price_range_basis",
        "price_range_explanation",
    ]
    csv_path, html_path = archive_signals(
        prediction[signal_columns], paths.reports / "signals", latest_date.date().isoformat()
    )
    portfolio = {
        "trade_date": latest_date.date().isoformat(),
        "cash": None,
        "account_role": "gated_primary_model_account",
        "candidate_pool": candidate_pool,
        "portfolio_policy": policy_snapshot(execution_config),
        "positions": [
            {"symbol": symbol, "target_weight": selected_weight}
            for symbol in sorted(selected)
        ],
        "status": "TARGETS_CREATED_NO_REAL_ORDERS",
        "model_version": manifest["version"],
    }
    portfolio_path = immutable_write_json(
        paths.reports / "portfolio" / f"{latest_date.date().isoformat()}.json", portfolio
    )
    execution = create_execution_plan_command(
        paths,
        execution_date=next_execution_date,
        signals=prediction[signal_columns],
        price_contexts=price_contexts,
    )
    shadow_prediction = prediction[signal_columns].copy()
    shadow_prediction["current_position_weight"] = (
        shadow_prediction["symbol"].map(shadow_positions).fillna(0.0)
    )
    shadow_prediction["model_signal"] = shadow_prediction.apply(
        lambda row: (
            _execution_signal(
                int(row["signal_rank"]),
                float(row["current_position_weight"]),
                float(row["target_position_weight"]),
                len(selected),
            )
            if row["symbol"] in eligible_symbols
            else (
                "EXIT"
                if float(row["current_position_weight"]) > 0
                else "AVOID"
            )
        ),
        axis=1,
    )
    shadow_execution = create_execution_plan_command(
        paths,
        execution_date=next_execution_date,
        signals=shadow_prediction,
        price_contexts=price_contexts,
        apply_risk_gate=False,
        archive_directory="shadow-execution-plans",
    )
    daily_strategy = generate_daily_strategy_summary(
        paths,
        prediction[signal_columns],
        latest_date.date().isoformat(),
        next_execution_date.isoformat(),
        Path(execution["json"]),
        Path(shadow_execution["json"]),
        candidate_pool,
        policy_snapshot(execution_config),
    )
    return {
        "signals_csv": str(csv_path),
        "signals_html": str(html_path),
        "portfolio": str(portfolio_path),
        "price_calibration": str(calibration_artifact) if calibration_artifact else None,
        "sector_quality": str(sector_report),
        "execution_plan": execution,
        "shadow_execution_plan": shadow_execution,
        "candidate_pool": candidate_pool,
        "portfolio_policy": policy_snapshot(execution_config),
        "daily_strategy": daily_strategy,
    }


def generate_report_command(paths: ProjectPaths) -> dict[str, Any]:
    metrics_path = _latest_immutable_revision(
        paths.reports / "research",
        "backtest",
        ".json",
    )
    if metrics_path is None:
        raise FileNotFoundError("Backtest report is absent; run backtest first")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    json_path, html_path = generate_research_report(
        payload.get("metrics", payload), paths.reports / "research", "latest"
    )
    return {"json": str(json_path), "html": str(html_path)}


def run_daily(paths: ProjectPaths, synthetic: bool = False) -> dict[str, Any]:
    paths.ensure_runtime_dirs()
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    ledger = RunLedger(paths.state / "runs.duckdb")
    with ProcessLock(paths.state / "run-daily.lock.pid"):
        ledger.start(run_id, "run-daily")
        try:
            today = date.today()
            calendar_adapter = BaostockAdapter(RawArchive(paths.raw))
            if not synthetic and (today.weekday() >= 5 or not calendar_adapter.is_trading_day(today)):
                result = {"status": "SKIPPED_NON_TRADING_DAY", "date": today.isoformat()}
            else:
                if synthetic:
                    fetch_data(paths, today, today, None, synthetic=True)
                else:
                    fetch_data(paths, today, today, None, synthetic=False, bootstrap_universe=False)
                validate_data(paths, None)
                build_dataset(paths)
                settlement_date = today
                if synthetic:
                    settlement_date = pd.Timestamp(
                        pd.read_parquet(
                            paths.curated / "bars.parquet",
                            columns=["trade_date"],
                        )["trade_date"].max()
                    ).date()
                settlement = settle_daily_accounts(paths, settlement_date)
                settlement["manual"] = settle_manual_account(
                    paths,
                    settlement_date,
                )
                if (
                    not (paths.models / "current.json").exists()
                    or today.month in {1, 4, 7, 10}
                    and today.day <= 7
                ):
                    train_current(paths)
                next_execution_date = None if synthetic else calendar_adapter.next_trading_day(today)
                prediction = predict_latest(paths, next_execution_date)
                result = {
                    "status": "SUCCESS",
                    "simulation_settlement": settlement,
                    **prediction,
                }
            ledger.finish(run_id, "SUCCESS", json.dumps(result, ensure_ascii=False))
            return result
        except Exception as error:
            ledger.finish(run_id, "FAILED", str(error))
            raise
        finally:
            ledger.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m quant")
    parser.add_argument("--root", default=".", help="Project root")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser("fetch")
    fetch.add_argument("--start")
    fetch.add_argument("--end")
    fetch.add_argument("--symbols")
    fetch.add_argument("--synthetic", action="store_true")
    fetch.add_argument("--no-bootstrap-universe", action="store_true")
    fetch.add_argument("--shard-index", type=int, default=0)
    fetch.add_argument("--shard-count", type=int, default=1)
    industry = subparsers.add_parser("fetch-industry")
    industry.add_argument("--date")
    industry_backfill = subparsers.add_parser("backfill-industry")
    industry_backfill.add_argument("--start", default="2014-01-01")
    industry_backfill.add_argument("--end")
    validate = subparsers.add_parser("validate-data")
    validate.add_argument("--as-of")
    build = subparsers.add_parser("build-dataset")
    build.add_argument("--no-qlib-smoke", action="store_true")
    subparsers.add_parser("train")
    backtest = subparsers.add_parser("backtest")
    backtest.add_argument("--max-folds", type=int)
    subparsers.add_parser("research-multi-horizon")
    rank_research = subparsers.add_parser("research-rank")
    rank_research.add_argument("--config", default="rank_research")
    portfolio_research = subparsers.add_parser("research-portfolio")
    portfolio_research.add_argument("--config", default="portfolio_baseline")
    rank_buffer = subparsers.add_parser("research-rank-buffer")
    rank_buffer.add_argument("--config", default="rank_buffer_research")
    subparsers.add_parser("predict")
    execution_plan = subparsers.add_parser("build-execution-plan")
    execution_plan.add_argument("--execution-date")
    replay = subparsers.add_parser("replay-execution")
    replay.add_argument("--date", required=True)
    subparsers.add_parser("auction-monitor")
    auction_finalize = subparsers.add_parser("auction-finalize")
    auction_finalize.add_argument("--date", required=True)
    subparsers.add_parser("report")
    daily = subparsers.add_parser("run-daily")
    daily.add_argument("--synthetic", action="store_true")
    verify_daily = subparsers.add_parser("verify-daily")
    verify_daily.add_argument("--date")
    manual = subparsers.add_parser("record-manual-trades")
    manual.add_argument("--date", required=True)
    manual.add_argument(
        "--trade",
        action="append",
        required=True,
        help="SIDE,SYMBOL,QUANTITY,PRICE[,NAME]; may be repeated",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    paths = ProjectPaths(Path(args.root).resolve())
    paths.ensure_runtime_dirs()
    operations = load_config(paths, "operations")
    _configure_logging(paths, operations["log_level"])
    data_config = load_config(paths, "data")
    if args.command == "fetch":
        start = _date(args.start) if args.start else _date(data_config["research_start"])
        end = _date(args.end) if args.end else date.today()
        result = fetch_data(
            paths,
            start,
            end,
            _symbols(args.symbols),
            args.synthetic,
            not args.no_bootstrap_universe,
            args.shard_index,
            args.shard_count,
        )
    elif args.command == "fetch-industry":
        result = fetch_industry_data(
            paths,
            _date(args.date) if args.date else date.today(),
        )
    elif args.command == "backfill-industry":
        result = backfill_industry_data(
            paths,
            _date(args.start),
            _date(args.end) if args.end else date.today(),
        )
    elif args.command == "validate-data":
        result = validate_data(paths, _date(args.as_of) if args.as_of else None)
    elif args.command == "build-dataset":
        result = build_dataset(paths, not args.no_qlib_smoke)
    elif args.command == "train":
        result = train_current(paths)
    elif args.command == "backtest":
        result = run_backtest(paths, args.max_folds)
    elif args.command == "research-multi-horizon":
        result = run_multi_horizon_research(paths)
    elif args.command == "research-rank":
        result = run_cross_section_rank_research(paths, args.config)
    elif args.command == "research-portfolio":
        result = run_portfolio_baseline(paths, args.config)
    elif args.command == "research-rank-buffer":
        result = run_rank_buffer_research(paths, args.config)
    elif args.command == "predict":
        result = predict_latest(paths)
    elif args.command == "build-execution-plan":
        result = create_execution_plan_command(
            paths,
            _date(args.execution_date) if args.execution_date else None,
        )
    elif args.command == "replay-execution":
        execution_day = args.date
        state_root = paths.state / "execution" / execution_day
        result = replay_execution_day(
            paths.reports / "execution-plans" / f"{execution_day}.json",
            state_root / "decisions.jsonl",
            state_root / "orders.jsonl",
        )
    elif args.command == "auction-monitor":
        result = monitor_auction_session(paths)
    elif args.command == "auction-finalize":
        result = finalize_auction_day(paths, _date(args.date))
    elif args.command == "report":
        result = generate_report_command(paths)
    elif args.command == "run-daily":
        result = run_daily(paths, args.synthetic)
    elif args.command == "verify-daily":
        result = verify_daily_run(
            paths,
            _date(args.date) if args.date else None,
        )
    elif args.command == "record-manual-trades":
        result = settle_manual_account(
            paths,
            _date(args.date),
            [parse_manual_trade(value) for value in args.trade],
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

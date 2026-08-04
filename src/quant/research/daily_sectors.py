from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant.cli import _historical_sector_states
from quant.config import ProjectPaths, load_config
from quant.signals.archive import immutable_write_json

SECTOR_COLUMNS = [
    "stock_name",
    "sector_name",
    "sector_state",
    "sector_return_20",
    "sector_volatility_20",
    "sector_member_count",
    "sector_mapping_as_of",
    "sector_classification",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_daily_sector_predictions(
    paths: ProjectPaths,
    *,
    input_name: str = "rolling_predictions.parquet",
    output_name: str = "rolling_predictions_daily_sector.parquet",
) -> dict[str, Any]:
    for value in (input_name, output_name):
        if Path(value).name != value:
            raise ValueError("Prediction paths must be file names inside reports/research")
    research_root = paths.reports / "research"
    input_path = research_root / input_name
    output_path = research_root / output_name
    bars_path = paths.curated / "bars.parquet"
    predictions = pd.read_parquet(input_path)
    predictions["trade_date"] = pd.to_datetime(
        predictions["trade_date"]
    ).dt.normalize()
    bars = pd.read_parquet(
        bars_path,
        columns=["symbol", "trade_date", "hfq_close", "is_trading"],
    )
    bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.normalize()
    benchmark_symbol = str(load_config(paths, "data")["benchmark_symbol"])
    bars = bars[~bars["symbol"].astype(str).eq(benchmark_symbol)].copy()
    target_dates = {
        pd.Timestamp(value)
        for value in predictions["trade_date"].dropna().unique()
    }
    states = _historical_sector_states(
        paths,
        bars,
        predictions,
        target_dates,
    )
    merged = predictions.drop(
        columns=[column for column in SECTOR_COLUMNS if column in predictions],
    ).merge(
        states[["symbol", "trade_date", *SECTOR_COLUMNS]],
        on=["symbol", "trade_date"],
        how="left",
        validate="one_to_one",
    )
    merged["sector_state"] = merged["sector_state"].fillna("DATA_UNAVAILABLE")
    coverage = float(merged["sector_state"].ne("DATA_UNAVAILABLE").mean())
    minimum = float(load_config(paths, "data")["sector_state"]["minimum_signal_coverage"])
    if coverage < minimum:
        raise RuntimeError(
            f"Daily sector-state coverage {coverage:.2%} is below {minimum:.2%}"
        )
    research_root.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".tmp.parquet")
    merged.to_parquet(temp_path, index=False, compression="zstd")
    output_hash = _sha256(temp_path)
    if output_path.exists():
        if _sha256(output_path) != output_hash:
            temp_path.unlink()
            raise FileExistsError(
                f"{output_path} already exists with different content; "
                "use an explicit revision file name"
            )
        temp_path.unlink()
    else:
        temp_path.replace(output_path)
    report = {
        "status": "PASS",
        "created_at": datetime.now(UTC).isoformat(),
        "input": str(input_path),
        "input_sha256": _sha256(input_path),
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "rows": int(len(merged)),
        "dates": int(merged["trade_date"].nunique()),
        "sector_state_coverage": coverage,
        "minimum_sector_state_coverage": minimum,
        "mapping_is_point_in_time": True,
        "uses_future_mapping": False,
        "simulation_only": True,
    }
    report_path = immutable_write_json(
        research_root / "daily-sector-predictions.json",
        report,
    )
    return {**report, "report": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input", default="rolling_predictions.parquet")
    parser.add_argument(
        "--output",
        default="rolling_predictions_daily_sector.parquet",
    )
    args = parser.parse_args()
    report = build_daily_sector_predictions(
        ProjectPaths(args.root.resolve()),
        input_name=str(args.input),
        output_name=str(args.output),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

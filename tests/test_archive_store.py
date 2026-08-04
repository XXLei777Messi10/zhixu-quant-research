from datetime import UTC, datetime

import pandas as pd

from quant.data.raw_archive import RawArchive
from quant.data.store import ParquetStore
from quant.data.validate import write_quarantine_records
from quant.synthetic import synthetic_normalized


def test_raw_archive_content_dedup_and_attempt_manifest(tmp_path) -> None:
    archive = RawArchive(tmp_path)
    frame = pd.DataFrame({"原始列": [1, 2]})
    first = archive.save("test", "endpoint", frame, {"x": 1}, "1", datetime.now(UTC))
    second = archive.save("test", "endpoint", frame, {"x": 1}, "1", datetime.now(UTC))
    assert first.data_path == second.data_path
    assert first.metadata_path != second.metadata_path
    assert len((tmp_path / "test" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_parquet_upsert_is_idempotent(tmp_path) -> None:
    store = ParquetStore(tmp_path)
    frame = synthetic_normalized(symbols=1, periods=10)
    store.upsert(frame)
    first = store.read()
    store.upsert(frame)
    second = store.read()
    assert len(first) == len(second)
    assert not second.duplicated(["source", "adjustment", "symbol", "trade_date"]).any()


def test_curated_data_excludes_quarantined_primary_record(tmp_path) -> None:
    store = ParquetStore(tmp_path / "normalized")
    frame = synthetic_normalized(symbols=1, periods=10)
    store.upsert(frame)
    raw = frame[(frame["source"] == "akshare") & (frame["adjustment"] == "raw")]
    excluded = raw.iloc[[0]].copy()
    curated = store.build_curated(
        tmp_path / "curated.parquet",
        excluded_records=excluded,
    )
    assert len(curated) == len(raw) - 1
    key = excluded.iloc[0]
    assert not (curated["symbol"].eq(key["symbol"]) & curated["trade_date"].eq(key["trade_date"])).any()


def test_quarantine_artifact_is_content_addressed(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["SH600000"],
            "trade_date": [pd.Timestamp("2020-01-02")],
            "reason": ["verified provider defect"],
        }
    )
    first = write_quarantine_records(frame, tmp_path)
    second = write_quarantine_records(frame, tmp_path)
    assert first == second
    assert first is not None and first.exists()
    assert len(list(tmp_path.glob("*.json"))) == 1

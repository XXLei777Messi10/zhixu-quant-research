from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

QLIB_FIELDS = {
    "open": "hfq_open",
    "high": "hfq_high",
    "low": "hfq_low",
    "close": "hfq_close",
    "volume": "volume",
    "amount": "amount",
    "factor": "adjust_factor",
}


def _write_bin(path: Path, values: np.ndarray, start_index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = np.concatenate([np.array([start_index], dtype="<f4"), values.astype("<f4", copy=False)])
    with path.open("wb") as handle:
        output.tofile(handle)


def build_qlib_dataset(curated: pd.DataFrame, root: Path) -> Path:
    required = {"symbol", "trade_date", *QLIB_FIELDS.values()}
    missing = required - set(curated.columns)
    if missing:
        raise ValueError(f"Cannot build Qlib data, missing: {sorted(missing)}")
    frame = curated.sort_values(["trade_date", "symbol"])
    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(frame["trade_date"]).unique()))
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(frame[["symbol", "trade_date", "hfq_close"]], index=False).values.tobytes()
    ).hexdigest()[:16]
    version = root / "versions" / digest
    if not version.exists():
        version.mkdir(parents=True)
        calendars = version / "calendars"
        instruments = version / "instruments"
        calendars.mkdir()
        instruments.mkdir()
        (calendars / "day.txt").write_text(
            "\n".join(pd.Timestamp(value).strftime("%Y-%m-%d") for value in calendar) + "\n",
            encoding="utf-8",
        )
        instrument_lines: list[str] = []
        calendar_positions = {pd.Timestamp(value): index for index, value in enumerate(calendar)}
        for symbol, bars in frame.groupby("symbol", sort=True):
            ordered = bars.sort_values("trade_date").set_index(pd.to_datetime(bars["trade_date"]))
            start = calendar_positions[pd.Timestamp(ordered.index.min())]
            end = calendar_positions[pd.Timestamp(ordered.index.max())]
            aligned_index = calendar[start : end + 1]
            instrument_lines.append(
                f"{symbol}\t{pd.Timestamp(aligned_index[0]).date()}\t{pd.Timestamp(aligned_index[-1]).date()}"
            )
            for qlib_name, source_name in QLIB_FIELDS.items():
                values = ordered[source_name].reindex(aligned_index).to_numpy(dtype="float32")
                _write_bin(version / "features" / symbol.lower() / f"{qlib_name}.day.bin", values, start)
        (instruments / "all.txt").write_text("\n".join(instrument_lines) + "\n", encoding="utf-8")
        manifest = {
            "version": digest,
            "rows": len(frame),
            "symbols": int(frame["symbol"].nunique()),
            "calendar_days": len(calendar),
            "start": str(calendar.min().date()),
            "end": str(calendar.max().date()),
        }
        (version / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
    pointer = root / "current.json"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    temp = pointer.with_name(f".current.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps({"version": digest, "path": str(version)}, indent=2), encoding="utf-8")
    os.replace(temp, pointer)
    return version


def current_qlib_path(root: Path) -> Path:
    pointer = root / "current.json"
    if not pointer.exists():
        raise FileNotFoundError("Qlib current dataset pointer is absent")
    return Path(json.loads(pointer.read_text(encoding="utf-8"))["path"])


def smoke_test_qlib(path: Path, symbol: str | None = None) -> None:
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    qlib.init(provider_uri=str(path), region=REG_CN)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if symbol is None:
        symbol = (path / "instruments" / "all.txt").read_text(encoding="utf-8").split("\t", 1)[0]
    result = D.features(
        [symbol],
        ["$close", "Ref($close, 1)", "Mean($close, 5)"],
        start_time=manifest["start"],
        end_time=manifest["end"],
        freq="day",
    )
    if result.empty or result["$close"].notna().sum() == 0:
        raise RuntimeError("Qlib smoke test returned no valid close data")

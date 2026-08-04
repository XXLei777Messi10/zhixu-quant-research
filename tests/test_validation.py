import pandas as pd

from quant.data.validate import DataValidator
from quant.synthetic import synthetic_normalized
from quant.types import QualityStatus

CONFIG = {
    "price_warn_relative": 0.001,
    "price_fail_relative": 0.005,
    "volume_warn_relative": 0.005,
    "volume_fail_relative": 0.02,
    "amount_warn_relative": 0.01,
    "amount_fail_relative": 0.03,
    "adjusted_jump_fail_relative": 0.35,
}


def test_valid_ohlc_and_cross_source() -> None:
    frame = synthetic_normalized(symbols=2, periods=20)
    validator = DataValidator(CONFIG)
    assert validator.validate_single(frame).status == QualityStatus.PASS
    ak = frame[(frame.source == "akshare") & (frame.adjustment == "raw")]
    bs = frame[(frame.source == "baostock") & (frame.adjustment == "raw")]
    assert validator.compare_sources(ak, bs).status in {QualityStatus.PASS, QualityStatus.WARN}


def test_invalid_ohlc_fails() -> None:
    frame = synthetic_normalized(symbols=1, periods=5)
    frame.loc[0, "high"] = frame.loc[0, "low"] - 1
    report = DataValidator(CONFIG).validate_single(frame)
    assert report.status == QualityStatus.FAIL
    assert any(issue.code == "INVALID_OHLC" for issue in report.issues)


def test_missing_calendar_day_fails() -> None:
    frame = synthetic_normalized(symbols=1, periods=5)
    stock = frame[(frame.symbol == "SH600000") & (frame.source == "akshare") & (frame.adjustment == "raw")]
    calendar = pd.DatetimeIndex(stock.trade_date.unique())
    missing = stock.iloc[1:].copy()
    report = DataValidator(CONFIG).validate_calendar(missing, calendar, ["SH600000"])
    assert report.status == QualityStatus.FAIL
    assert report.issues[0].code == "MISSING_TRADING_DATE"


def test_adjustment_discontinuity_fails() -> None:
    frame = synthetic_normalized(symbols=1, periods=10)
    mask = (frame.symbol == "SH600000") & (frame.adjustment == "hfq")
    target = frame.index[mask][-1]
    frame.loc[target, ["open", "high", "low", "close"]] *= 2
    report = DataValidator(CONFIG).validate_single(frame)
    assert any(issue.code == "ADJUSTED_DISCONTINUITY" for issue in report.issues)


def test_configured_quarantine_requires_exact_values() -> None:
    frame = synthetic_normalized(symbols=1, periods=5)
    target = frame[(frame["source"] == "baostock") & (frame["adjustment"] == "raw")].iloc[[0]].copy()
    row = target.iloc[0]
    config = {
        **CONFIG,
        "quarantined_records": [
            {
                "source": "baostock",
                "adjustment": "raw",
                "symbol": row["symbol"],
                "trade_date": pd.Timestamp(row["trade_date"]).date().isoformat(),
                "expected": {"open": float(row["open"])},
                "reason": "verified provider defect",
            }
        ],
    }
    clean, rejected, report = DataValidator(config).apply_configured_quarantine(target)
    assert clean.empty
    assert len(rejected) == 1
    assert report.status == QualityStatus.WARN
    assert report.issues[0].code == "QUARANTINED_PROVIDER_RECORD"

    changed = target.copy()
    changed["open"] += 0.01
    clean, rejected, report = DataValidator(config).apply_configured_quarantine(changed)
    assert len(clean) == 1
    assert rejected.empty
    assert report.status == QualityStatus.FAIL
    assert report.issues[0].code == "QUARANTINE_RULE_MISMATCH"


def test_cross_source_issue_contains_compared_values() -> None:
    frame = synthetic_normalized(symbols=1, periods=5)
    ak = frame[(frame.source == "akshare") & (frame.adjustment == "raw")].copy()
    bs = frame[(frame.source == "baostock") & (frame.adjustment == "raw")].copy()
    bs.loc[bs.index[0], "open"] *= 0.9
    issue = DataValidator(CONFIG).compare_sources(ak, bs).issues[0]
    assert issue.details["relative_difference"] > CONFIG["price_fail_relative"]
    assert issue.details["akshare_value"] != issue.details["baostock_value"]

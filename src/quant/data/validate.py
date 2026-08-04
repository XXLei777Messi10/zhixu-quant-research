from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.types import QualityIssue, QualityReport, QualityStatus

PRICE_COLUMNS = ["open", "high", "low", "close"]


def _relative_difference(left: pd.Series, right: pd.Series) -> pd.Series:
    denominator = pd.concat([left.abs(), right.abs()], axis=1).max(axis=1).replace(0, np.nan)
    return (left - right).abs() / denominator


class DataValidator:
    def __init__(self, config: dict[str, Any]):
        self.config = config["quality"] if "quality" in config else config

    def validate_single(self, frame: pd.DataFrame, as_of: date | None = None) -> QualityReport:
        if frame.empty:
            target = as_of or date.today()
            return QualityReport(
                target,
                QualityStatus.FAIL,
                [QualityIssue("EMPTY_DATA", QualityStatus.FAIL, "Dataset is empty")],
            )
        target = as_of or pd.Timestamp(frame["trade_date"].max()).date()
        issues: list[QualityIssue] = []
        duplicates = frame.duplicated(["source", "adjustment", "symbol", "trade_date"], keep=False)
        if duplicates.any():
            issues.append(
                QualityIssue(
                    "DUPLICATE_KEY",
                    QualityStatus.FAIL,
                    f"{int(duplicates.sum())} duplicate provider rows",
                )
            )
        trading = frame[frame["is_trading"].fillna(False)].copy()
        nonpositive = trading[PRICE_COLUMNS].le(0).any(axis=1)
        for row in trading.loc[nonpositive, ["symbol", "trade_date"]].itertuples(index=False):
            issues.append(
                QualityIssue(
                    "NON_POSITIVE_PRICE",
                    QualityStatus.FAIL,
                    "Trading row has non-positive price",
                    row.symbol,
                    pd.Timestamp(row.trade_date).date(),
                )
            )
        invalid_ohlc = trading["high"].lt(trading[["open", "close", "low"]].max(axis=1)) | trading["low"].gt(
            trading[["open", "close", "high"]].min(axis=1)
        )
        for row in trading.loc[invalid_ohlc, ["symbol", "trade_date"]].itertuples(index=False):
            issues.append(
                QualityIssue(
                    "INVALID_OHLC",
                    QualityStatus.FAIL,
                    "OHLC relationship is invalid",
                    row.symbol,
                    pd.Timestamp(row.trade_date).date(),
                )
            )
        adjusted = frame[frame["adjustment"].eq("hfq") & frame["is_trading"].fillna(False)].copy()
        if not adjusted.empty:
            adjusted = adjusted.sort_values(["symbol", "trade_date"])
            jumps = adjusted.groupby("symbol")["close"].pct_change(fill_method=None).abs()
            bad_jumps = jumps.gt(float(self.config["adjusted_jump_fail_relative"]))
            for row in adjusted.loc[bad_jumps, ["symbol", "trade_date"]].itertuples(index=False):
                issues.append(
                    QualityIssue(
                        "ADJUSTED_DISCONTINUITY",
                        QualityStatus.FAIL,
                        "Adjusted close has an implausible discontinuity",
                        row.symbol,
                        pd.Timestamp(row.trade_date).date(),
                    )
                )
        status = (
            QualityStatus.FAIL
            if any(issue.severity == QualityStatus.FAIL for issue in issues)
            else QualityStatus.PASS
        )
        return QualityReport(
            target,
            status,
            issues,
            {
                "rows": len(frame),
                "symbols": int(frame["symbol"].nunique()),
                "start": str(pd.Timestamp(frame["trade_date"].min()).date()),
                "end": str(pd.Timestamp(frame["trade_date"].max()).date()),
            },
        )

    def validate_calendar(
        self,
        frame: pd.DataFrame,
        calendar: pd.DatetimeIndex,
        symbols: list[str],
        as_of: date | None = None,
    ) -> QualityReport:
        target = as_of or pd.Timestamp(calendar.max()).date()
        expected = pd.MultiIndex.from_product(
            [symbols, pd.DatetimeIndex(calendar).normalize()], names=["symbol", "trade_date"]
        )
        available = frame.copy()
        available["trade_date"] = pd.to_datetime(available["trade_date"]).dt.normalize()
        present = pd.MultiIndex.from_frame(available[["symbol", "trade_date"]].drop_duplicates())
        missing = expected.difference(present)
        issues = [
            QualityIssue(
                "MISSING_TRADING_DATE",
                QualityStatus.FAIL,
                "Expected calendar row is absent and no explicit suspension record exists",
                str(symbol),
                pd.Timestamp(trade_date).date(),
            )
            for symbol, trade_date in missing
        ]
        status = QualityStatus.FAIL if issues else QualityStatus.PASS
        return QualityReport(target, status, issues, {"expected": len(expected), "missing": len(missing)})

    def compare_sources(
        self, akshare: pd.DataFrame, baostock: pd.DataFrame, as_of: date | None = None
    ) -> QualityReport:
        target = as_of or date.today()
        keys = ["symbol", "trade_date"]
        left = akshare[akshare["adjustment"].eq("raw")].copy()
        right = baostock[baostock["adjustment"].eq("raw")].copy()
        common = left.merge(right, on=keys, suffixes=("_ak", "_bs"), how="inner")
        issues: list[QualityIssue] = []
        if common.empty:
            issues.append(QualityIssue("NO_CROSS_SOURCE_OVERLAP", QualityStatus.FAIL, "No comparable rows"))
        thresholds = {
            **{
                column: (self.config["price_warn_relative"], self.config["price_fail_relative"])
                for column in PRICE_COLUMNS
            },
            "volume": (self.config["volume_warn_relative"], self.config["volume_fail_relative"]),
            "amount": (self.config["amount_warn_relative"], self.config["amount_fail_relative"]),
        }
        stats: dict[str, Any] = {"compared_rows": len(common)}
        for column, (warn, fail) in thresholds.items():
            difference = _relative_difference(common[f"{column}_ak"], common[f"{column}_bs"])
            stats[f"{column}_max_relative_difference"] = float(difference.max()) if len(difference) else None
            failures = difference.gt(fail)
            warnings = difference.gt(warn) & ~failures
            for severity, mask in ((QualityStatus.FAIL, failures), (QualityStatus.WARN, warnings)):
                for index in common.index[mask]:
                    row = common.loc[index]
                    issues.append(
                        QualityIssue(
                            f"CROSS_SOURCE_{column.upper()}",
                            severity,
                            f"{column} differs across providers",
                            str(row["symbol"]),
                            pd.Timestamp(row["trade_date"]).date(),
                            {
                                "relative_difference": float(difference.loc[index]),
                                "akshare_value": float(row[f"{column}_ak"]),
                                "baostock_value": float(row[f"{column}_bs"]),
                            },
                        )
                    )
        if any(issue.severity == QualityStatus.FAIL for issue in issues):
            status = QualityStatus.FAIL
        elif issues:
            status = QualityStatus.WARN
        else:
            status = QualityStatus.PASS
        return QualityReport(target, status, issues, stats)

    def apply_configured_quarantine(
        self,
        frame: pd.DataFrame,
        as_of: date | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, QualityReport]:
        """Remove only exact, independently verified provider defects.

        A rule stops matching if any configured expected value changes. In that
        case the record remains in the validation input and a hard issue is
        emitted, so an obsolete exception can never silently admit data.
        """
        target = as_of or date.today()
        clean = frame.copy()
        rejected: list[pd.DataFrame] = []
        issues: list[QualityIssue] = []
        rules = self.config.get("quarantined_records", [])
        for rule in rules:
            trade_date = pd.Timestamp(rule["trade_date"]).normalize()
            mask = (
                clean["source"].eq(rule["source"])
                & clean["adjustment"].eq(rule["adjustment"])
                & clean["symbol"].eq(rule["symbol"])
                & pd.to_datetime(clean["trade_date"]).dt.normalize().eq(trade_date)
            )
            matches = clean.loc[mask].copy()
            if matches.empty:
                continue

            mismatches: dict[str, dict[str, float]] = {}
            for column, expected in rule.get("expected", {}).items():
                actual = pd.to_numeric(matches[column], errors="coerce")
                if actual.isna().any() or not np.allclose(
                    actual.to_numpy(dtype=float),
                    float(expected),
                    rtol=0.0,
                    atol=1e-9,
                ):
                    mismatches[column] = {
                        "expected": float(expected),
                        "actual": float(actual.iloc[0]),
                    }
            if mismatches:
                issues.append(
                    QualityIssue(
                        "QUARANTINE_RULE_MISMATCH",
                        QualityStatus.FAIL,
                        "Configured quarantine no longer matches the provider record",
                        str(rule["symbol"]),
                        trade_date.date(),
                        {"source": rule["source"], "mismatches": mismatches},
                    )
                )
                continue

            matches["quarantine_reason"] = rule["reason"]
            matches["quarantine_evidence_url"] = rule.get("evidence_url")
            rejected.append(matches)
            clean = clean.loc[~mask].copy()
            issues.append(
                QualityIssue(
                    "QUARANTINED_PROVIDER_RECORD",
                    QualityStatus.WARN,
                    str(rule["reason"]),
                    str(rule["symbol"]),
                    trade_date.date(),
                    {
                        "source": rule["source"],
                        "adjustment": rule["adjustment"],
                        "evidence_url": rule.get("evidence_url"),
                        "verified_value": rule.get("verified_value", {}),
                    },
                )
            )

        quarantined = pd.concat(rejected, ignore_index=True) if rejected else pd.DataFrame()
        status = (
            QualityStatus.FAIL
            if any(issue.severity == QualityStatus.FAIL for issue in issues)
            else QualityStatus.WARN
            if issues
            else QualityStatus.PASS
        )
        report = QualityReport(
            target,
            status,
            issues,
            {"configured_rules": len(rules), "quarantined_rows": len(quarantined)},
        )
        return clean, quarantined, report


def combine_reports(*reports: QualityReport) -> QualityReport:
    issues = [issue for report in reports for issue in report.issues]
    stats = {f"report_{index}": report.stats for index, report in enumerate(reports)}
    if any(report.status == QualityStatus.FAIL for report in reports):
        status = QualityStatus.FAIL
    elif any(report.status == QualityStatus.WARN for report in reports):
        status = QualityStatus.WARN
    else:
        status = QualityStatus.PASS
    return QualityReport(max(report.as_of for report in reports), status, issues, stats)


def write_quality_report(report: QualityReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    payload["status"] = report.status.value
    for issue in payload["issues"]:
        issue["severity"] = issue["severity"].value
        if issue["trade_date"] is not None:
            issue["trade_date"] = issue["trade_date"].isoformat()
    payload["as_of"] = report.as_of.isoformat()
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return path
        revision = 2
        while path.with_name(f"{path.stem}__r{revision}{path.suffix}").exists():
            revision += 1
        path = path.with_name(f"{path.stem}__r{revision}{path.suffix}")
    path.write_text(text, encoding="utf-8")
    return path


def write_quarantine_records(frame: pd.DataFrame, root: Path) -> Path | None:
    if frame.empty:
        return None
    root.mkdir(parents=True, exist_ok=True)
    records = json.loads(frame.to_json(orient="records", date_format="iso"))
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    path = root / f"{digest}.json"
    if not path.exists():
        path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return path
